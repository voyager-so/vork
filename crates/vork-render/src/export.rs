//! Offline glTF snapshots and native clips use the renderer's loader;
//! stages requiring a GPU publication are refused before any asset is published.
use crate::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use vork_scene::v2::{self, Input};

/// One evaluated snapshot or inclusive frame-range clip. Loss permission names a diagnostic code.
#[derive(Default)]
pub struct Export {
    pub scene: PathBuf,
    pub out: PathBuf,
    pub root: Option<String>,
    pub at: Option<vork_time::Clock>,
    pub frame_start: Option<u32>,
    pub frame_end: Option<u32>,
    pub allow_losses: Vec<String>,
    /// Executable provenance supplied by the CLI; crate version for internal callers.
    pub vork_version: String,
}
#[derive(Default)]
struct Writer {
    input_hash: blake3::Hasher,
    vertex_count: u64,
    triangle_count: u64,
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    meshes: Vec<Value>,
    materials: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
    samplers: Vec<Value>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    extensions: BTreeSet<String>,
    issues: Vec<Value>,
    dispositions: Vec<Value>,
}
impl Writer {
    fn issue(
        &mut self,
        code: &str,
        source: &str,
        requested: &str,
        delivered: &str,
        fix: &str,
        refused: bool,
    ) {
        self.issues.push(json!({"code":code,"source":source,"stage":"export","severity":if refused {"error"} else {"warning"},"requested":requested,"dropped":delivered,"remediation":fix}));
    }
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let id = self.views.len();
        let mut view = json!({"buffer":0,"byteOffset":self.bin.len(),"byteLength":bytes.len()});
        if let Some(target) = target {
            view["target"] = target.into();
        }
        self.views.push(view);
        self.bin.extend_from_slice(bytes);
        id
    }
    fn floats(&mut self, data: &[Vec<f32>], kind: &str, bounds: bool) -> Result<usize> {
        ensure!(
            !data.is_empty() && data.iter().flatten().all(|x| x.is_finite()),
            "export attribute is empty or nonfinite"
        );
        let bytes: Vec<_> = data
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let view = self.view(&bytes, Some(34962));
        let id = self.accessors.len();
        let mut a = json!({"bufferView":view,"componentType":5126,"count":data.len(),"type":kind});
        if bounds {
            let width = data[0].len();
            a["min"] = json!(
                (0..width)
                    .map(|i| data.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min))
                    .collect::<Vec<_>>()
            );
            a["max"] = json!(
                (0..width)
                    .map(|i| data.iter().map(|v| v[i]).fold(f32::NEG_INFINITY, f32::max))
                    .collect::<Vec<_>>()
            );
        }
        self.accessors.push(a);
        Ok(id)
    }
    fn indices(&mut self, data: &[u32]) -> usize {
        let bytes: Vec<_> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(34963));
        let id = self.accessors.len();
        self.accessors.push(
            json!({"bufferView":view,"componentType":5125,"count":data.len(),"type":"SCALAR"}),
        );
        id
    }
    fn mesh(
        &mut self,
        source: &v2::Geometry,
        geometry: &geometry::Geometry,
        slots: &[usize],
        skinned: bool,
    ) -> Result<usize> {
        let vertices = &geometry.vertices;
        self.vertex_count += vertices.len() as u64;
        self.triangle_count += (geometry.indices.len() / 3) as u64;
        ensure!(
            !vertices.is_empty() && !geometry.indices.is_empty(),
            "cannot export an empty mesh"
        );
        ensure!(
            geometry
                .indices
                .iter()
                .all(|&i| (i as usize) < vertices.len()),
            "export index outside vertex range"
        );
        let mut attributes = json!({});
        for (name, width, kind) in [
            ("POSITION", 3, "VEC3"),
            ("NORMAL", 3, "VEC3"),
            ("TANGENT", 4, "VEC4"),
            ("COLOR_0", 4, "VEC4"),
            ("TEXCOORD_0", 2, "VEC2"),
            ("TEXCOORD_1", 2, "VEC2"),
        ] {
            if name == "NORMAL" && skinned && !source.attributes.contains_key("NORMAL") {
                // Preserve flat shading after deformation. A rest-pose NORMAL
                // would instead request interpolated skinned vertex normals.
                continue;
            }
            if name == "COLOR_0" && vertices.iter().any(|v| v.COLOR_0 != [1.; 4]) {
                // Native OpenPBR does not implicitly multiply COLOR_0. Retain it
                // as a custom channel so glTF does not introduce that multiply.
                continue;
            }
            if name == "TANGENT" && !source.attributes.contains_key("TANGENT") {
                continue;
            }
            if name.starts_with("TEXCOORD")
                && geometry.uv_sets & (1 << usize::from(name.ends_with('1'))) == 0
            {
                continue;
            }
            let data: Vec<_> = vertices
                .iter()
                .map(|v| match name {
                    "POSITION" => v.POSITION.to_vec(),
                    "NORMAL" => v.NORMAL.to_vec(),
                    "TANGENT" => v.TANGENT.to_vec(),
                    "COLOR_0" => v.COLOR_0.to_vec(),
                    "TEXCOORD_0" => v.TEXCOORD_0.to_vec(),
                    _ => v.TEXCOORD_1.to_vec(),
                })
                .collect();
            ensure!(
                data.iter().all(|v| v.len() == width),
                "invalid attribute width"
            );
            attributes[name] = self.floats(&data, kind, name == "POSITION")?.into();
        }
        if vertices.iter().any(|v| v.COLOR_0 != [1.; 4]) {
            attributes["_VORK_COLOR_0"] = self
                .floats(
                    &vertices
                        .iter()
                        .map(|v| v.COLOR_0.to_vec())
                        .collect::<Vec<_>>(),
                    "VEC4",
                    false,
                )?
                .into();
            self.dispositions.push(json!({"kind":"attribute","source":"COLOR_0","disposition":"preserved","meaning":"retained as _VORK_COLOR_0; native OpenPBR has no implicit vertex-color multiplication"}));
        }
        ensure!(
            vertices
                .iter()
                .all(|v| (Vec3::from_array(v.NORMAL).length_squared() - 1.).abs() < 1e-4),
            "mesh contains degenerate normals; remove zero-area triangles and supply unit normals"
        );
        if source.attributes.contains_key("TANGENT") {
            ensure!(
                vertices
                    .iter()
                    .all(|v| (Vec3::new(v.TANGENT[0], v.TANGENT[1], v.TANGENT[2])
                        .length_squared()
                        - 1.)
                        .abs()
                        < 1e-4
                        && v.TANGENT[3].abs() == 1.
                        && Vec3::new(v.TANGENT[0], v.TANGENT[1], v.TANGENT[2])
                            .dot(Vec3::from_array(v.NORMAL))
                            .abs()
                            < 1e-4),
                "mesh tangents must be unit XYZ orthogonal to NORMAL with handedness +1 or -1"
            );
        }
        let mut primitives = Vec::new();
        for (range, slot) in buffers::groups(source, geometry.indices.len(), slots.len())? {
            if range.is_empty() {
                continue;
            }
            let indices = self.indices(&geometry.indices[range]);
            primitives.push(
                json!({"attributes":attributes,"indices":indices,"material":slots[slot],"mode":4}),
            );
        }
        let id = self.meshes.len();
        self.meshes.push(json!({"primitives":primitives}));
        Ok(id)
    }
    fn texture(&mut self, scene: &Scene, base: &Path, name: &str, srgb: bool) -> Result<usize> {
        let image = &scene.images[name];
        ensure!(
            image.render.is_none() && image.time.is_none(),
            "image {name}: export requires a still image file; replace rendered/animated input with a still PNG"
        );
        ensure!(
            image.is_srgb() == srgb,
            "image {name}: color space does not match the glTF channel; supply {} data",
            if srgb { "sRGB" } else { "Non-Color" }
        );
        let path = base.join(image.file.as_ref().context("image has no file")?);
        ensure!(
            !path.to_string_lossy().contains('%'),
            "image {name}: freeze the sequence to one PNG before exporting"
        );
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read export image {}", path.display()))?;
        self.input_hash.update(name.as_bytes());
        self.input_hash.update(&bytes);
        // Precision is the file's own; colour conversion works in float and is judged by its range instead.
        let (image_data, stored) = crate::texture_source::decode_color_stored(&path,&bytes,u32::MAX,image.color_space,crate::texture_source::upload(srgb))
            .map_err(|error| crate::texture_source::for_image(error, name))?;
        ensure!(
            stored.bits_per_pixel() / u16::from(stored.channel_count())
                <= 8,
            "image {name}: higher precision textures are not admitted; explicitly prepare an 8-bit PNG"
        );
        // Reject HDR rather than silently clamp it to a PNG.
        ensure!(
            !matches!(stored, image::ColorType::Rgb32F | image::ColorType::Rgba32F),
            "image {name}: HDR texture requires factor extraction; supply a bounded PNG and an emission factor"
        );
        ensure!(
            !matches!(image_data, image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_))
                || image_data.to_rgb32f().pixels().all(|pixel| pixel.0.iter().all(|value| (-0.5 / 255. ..=1. + 0.5 / 255.).contains(value))),
            "image {name}: its colours fall outside the exported 8-bit range after colour conversion; convert it to sRGB before exporting"
        );
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image_data.to_rgb8())
            .write_to(&mut encoded, image::ImageFormat::Png)?;
        let view = self.view(encoded.get_ref(), None);
        let image_id = self.images.len();
        self.images
            .push(json!({"name":name,"bufferView":view,"mimeType":"image/png"}));
        let (s, t, min, mag) = image.sampler().gltf_codes();
        let sampler = self.samplers.len();
        self.samplers
            .push(json!({"wrapS":s,"wrapT":t,"minFilter":min,"magFilter":mag}));
        let id = self.textures.len();
        self.textures
            .push(json!({"source":image_id,"sampler":sampler}));
        Ok(id)
    }
    fn metallic_roughness_texture(
        &mut self,
        scene: &Scene,
        base: &Path,
        material: &v2::Material,
    ) -> Result<Option<usize>> {
        let mut inputs = Vec::new();
        for (key, channel) in [("specular_roughness", 1usize), ("base_metalness", 2usize)] {
            let Some(Input::Texture(binding)) = material.inputs.get(key) else {
                continue;
            };
            let definition = &scene.images[&binding.image];
            ensure!(
                definition.render.is_none() && definition.time.is_none() && !definition.is_srgb(),
                "image {}: metallic/roughness needs a still Non-Color image",
                binding.image
            );
            let file = definition
                .file
                .as_ref()
                .context("metallic/roughness image has no file")?;
            ensure!(
                !file.contains('%'),
                "image {}: freeze the sequence to a still PNG",
                binding.image
            );
            let bytes = std::fs::read(base.join(file))?;
            self.input_hash.update(&bytes);
            // Precision is the file's own; colour conversion works in float.
            let (image, stored) = crate::texture_source::decode_color_stored(&base.join(file),&bytes,u32::MAX,definition.color_space,crate::texture_source::upload(false))
                .map_err(|error| crate::texture_source::for_image(error, &binding.image))?;
            ensure!(
                stored.bits_per_pixel() / u16::from(stored.channel_count()) <= 8,
                "image {}: prepare an 8-bit Non-Color PNG before packing",
                binding.image
            );
            inputs.push((image.to_rgb8(), channel, definition.sampler()));
        }
        let Some((first, _, sampler)) = inputs.first() else {
            return Ok(None);
        };
        let sampler = *sampler;
        let dimensions = first.dimensions();
        ensure!(
            inputs
                .iter()
                .all(|(image, _, s)| image.dimensions() == dimensions && *s == sampler),
            "metallic/roughness textures require matching dimensions and samplers; repack them consistently before exporting"
        );
        let mut packed =
            image::RgbImage::from_pixel(dimensions.0, dimensions.1, image::Rgb([255, 255, 255]));
        for (source, channel, _) in inputs {
            for (target, pixel) in packed.pixels_mut().zip(source.pixels()) {
                target[channel] = pixel[0];
            }
        }
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(packed).write_to(&mut encoded, image::ImageFormat::Png)?;
        let view = self.view(encoded.get_ref(), None);
        let image = self.images.len();
        self.images
            .push(json!({"bufferView":view,"mimeType":"image/png"}));
        let (s, t, min, mag) = sampler.gltf_codes();
        let sid = self.samplers.len();
        self.samplers
            .push(json!({"wrapS":s,"wrapT":t,"minFilter":min,"magFilter":mag}));
        let id = self.textures.len();
        self.textures.push(json!({"source":image,"sampler":sid}));
        Ok(Some(id))
    }
    fn material(&mut self, scene: &Scene, base: &Path, name: Option<&str>) -> Result<usize> {
        let default: v2::Material = serde_json::from_value(json!({}))?;
        let material = name.map_or(&default, |n| &scene.materials[n]);
        let address = name.unwrap_or("<default>");
        if material.program != "std/openpbr" {
            self.issue(
                "material_program",
                address,
                "compiled surface closure",
                "refused; no qualified UV closure baker",
                "replace with std/openpbr numeric inputs or bake PBR maps in the authoring tool",
                true,
            );
            return Ok(0);
        }
        let supported = [
            "base_color",
            "base_metalness",
            "specular_roughness",
            "emission_color",
            "emission_luminance",
            "geometry_opacity",
            "geometry_normal",
            "coat_weight",
            "coat_roughness",
            "coat_ior",
        ];
        for (key, value) in &material.inputs {
            if supported.contains(&key.as_str()) {
                continue;
            }
            let equal = v2::OpenPbrParam::find(key).is_some_and(|p| match value {
                Input::Number(v) => p.default == [*v],
                Input::Vector(v) => p.default == v.as_slice(),
                Input::Bool(v) => p.default == [if *v { 1. } else { 0. }],
                _ => false,
            });
            if !equal {
                self.issue(
                    "material_parameter",
                    &format!("{address}.inputs.{key}"),
                    "OpenPBR parameter",
                    "refused; no qualified glTF conversion",
                    "remove this input or supply directly representable PBR parameters",
                    true,
                );
            }
        }
        if material.opacity == Some(v2::Opacity::Transmittance) {
            self.issue(
                "material_transmittance",
                address,
                "physical transmittance",
                "refused; glTF alpha is coverage",
                "use an opaque material or coverage opacity",
                true,
            );
        }
        let scalar = |key: &str| -> Result<f32> {
            match material.inputs.get(key) {
                Some(Input::Number(v)) => {
                    ensure!(
                        v2::OpenPbrParam::find(key).is_some_and(|p| p.allows(*v)),
                        "material {address}.{key}: sampled value is outside the parameter range; adjust animation keys/tangents or the numeric input"
                    );
                    Ok(*v)
                }
                Some(Input::Texture(_))
                    if matches!(key, "base_metalness" | "specular_roughness") =>
                {
                    Ok(1.)
                }
                None => Ok(v2::OpenPbrParam::find(key)
                    .context("unknown OpenPBR input")?
                    .default[0]),
                _ => bail!(
                    "material {address}.{key}: requires a numeric factor; supply numeric coverage/coat/emission values"
                ),
            }
        };
        let mut out = json!({"name":address,"doubleSided":material.double_sided,"pbrMetallicRoughness":{"metallicFactor":scalar("base_metalness")?,"roughnessFactor":scalar("specular_roughness")?}});
        if let Some(texture) = self.metallic_roughness_texture(scene, base, material)? {
            out["pbrMetallicRoughness"]["metallicRoughnessTexture"] = json!({"index":texture});
        }
        for (key, factor, texture) in [
            ("base_color", "baseColorFactor", "baseColorTexture"),
            ("emission_color", "emissiveFactor", "emissiveTexture"),
        ] {
            let input = material.inputs.get(key);
            let mut color = match input {
                Some(Input::Vector(v)) => v.clone(),
                None => v2::OpenPbrParam::find(key).unwrap().default.to_vec(),
                Some(Input::Texture(_)) => vec![1.; 3],
                _ => bail!("material {address}.{key}: requires RGB or a still image"),
            };
            ensure!(
                color
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0. && (key != "base_color" || *v <= 1.)),
                "material {address}.{key}: sampled color is outside glTF's range; adjust animation keys/tangents or prepare bounded color data"
            );
            let tex = if let Some(Input::Texture(binding)) = input {
                Some(self.texture(scene, base, &binding.image, true)?)
            } else {
                None
            };
            if key == "base_color" {
                color.push(scalar("geometry_opacity")?);
                out["pbrMetallicRoughness"][factor] = json!(color);
                if let Some(id) = tex {
                    out["pbrMetallicRoughness"][texture] = json!({"index":id});
                }
            } else {
                // Match the renderer's admitted glTF/OpenPBR emission convention.
                let scale = scalar("emission_luminance")?;
                for v in &mut color {
                    *v *= scale;
                }
                ensure!(
                    color.iter().all(|v| v.is_finite()),
                    "material {address}: emission exceeds finite delivery range; lower emission_luminance or emission_color"
                );
                let strength = color.iter().copied().fold(1., f32::max);
                out[factor] = json!(color.iter().map(|v| v / strength).collect::<Vec<_>>());
                if strength > 1. {
                    self.extensions
                        .insert("KHR_materials_emissive_strength".into());
                    out["extensions"] =
                        json!({"KHR_materials_emissive_strength":{"emissiveStrength":strength}});
                }
                if let Some(id) = tex {
                    out[texture] = json!({"index":id});
                }
            }
        }
        if let Some(normal) = material.inputs.get("geometry_normal") {
            if let Input::Texture(binding) = normal {
                let texture = self.texture(scene, base, &binding.image, false)?;
                out["normalTexture"] = json!({"index":texture});
            } else {
                self.issue(
                    "material_normal",
                    address,
                    "authored normal input",
                    "refused; only tangent-space normal images admitted",
                    "supply a Non-Color normal PNG and authored mesh tangents",
                    true,
                );
            }
        }
        if scalar("coat_weight")? > 0. {
            if scalar("coat_ior")? != 1.5 {
                self.issue(
                    "clearcoat_ior",
                    address,
                    "OpenPBR coat IOR",
                    "glTF clearcoat uses fixed IOR 1.5",
                    "set coat_ior to 1.5 or acknowledge clearcoat_ior",
                    false,
                );
            }
            self.extensions.insert("KHR_materials_clearcoat".into());
            out["extensions"]["KHR_materials_clearcoat"] = json!({"clearcoatFactor":scalar("coat_weight")?,"clearcoatRoughnessFactor":scalar("coat_roughness")?});
        }
        out["alphaMode"] = json!(if scalar("geometry_opacity")? < 1. {
            "BLEND"
        } else {
            "OPAQUE"
        });
        self.dispositions.push(json!({"source":address,"kind":"material","disposition":"approximated","meaning":"OpenPBR factors mapped to glTF metallic/roughness; BRDF highlights may differ"}));
        let id = self.materials.len();
        self.materials.push(out);
        Ok(id)
    }
}

#[derive(Debug)]
struct Refused;
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("asset export refused")
    }
}
impl std::error::Error for Refused {}

impl Export {
    pub fn run(&self) -> Result<Value> {
        if !matches!(
            self.out.extension().and_then(|e| e.to_str()),
            Some("glb" | "gltf")
        ) {
            return Err(anyhow::anyhow!("export --out requires .glb or .gltf")
                .context(failure::FailureKind::Usage));
        }
        let result = self.run_inner();
        if let Err(error) = &result {
            if error.downcast_ref::<Refused>().is_none() {
                let report = json!({"version":1,"status":"failed","source":self.scene,"selection":{"root":self.root},"issues":[{"code":"export_input","source":self.scene,"stage":"export","severity":"error","requested":"evaluated asset","dropped":"refused; no new asset published","remediation":format!("{error:#}"),"acknowledged":false}]});
                // Preserve the original failure category if its destination is unwritable.
                let _ = publish(
                    &self.out.with_extension("export.json"),
                    &serde_json::to_vec_pretty(&report)?,
                );
            }
        }
        // Export reads and converts the document on the CPU: no device is opened.
        result.map_err(failure::scene)
    }
    fn run_inner(&self) -> Result<Value> {
        let extension = self.out.extension().and_then(|e| e.to_str()).unwrap_or("");
        ensure!(
            matches!(extension, "glb" | "gltf"),
            "export --out requires .glb or .gltf"
        );
        let bytes = std::fs::read(&self.scene)?;
        let mut document = v2::diagnostics::read_document(&bytes)?;
        vork_std::expand_scene(&mut document, self.scene.parent().unwrap_or(Path::new(".")))?;
        let mut scene = Scene::from_json_for_export(
            &serde_json::to_vec(&document)?,
            &bytes,
            Some(&self.scene),
        )?;
        let base = self.scene.parent().unwrap_or(Path::new("."));
        let animated = self.frame_start.is_some() || self.frame_end.is_some();
        ensure!(
            !animated || self.at.is_none(),
            "--at conflicts with animation frame ranges"
        );
        let start = self.frame_start.unwrap_or(0);
        let end = if !animated {
            0
        } else if let Some(end) = self.frame_end {
            end
        } else {
            let count = scene.render.duration.exact()
                * vork_time::exact::decimal(&scene.render.fps.to_string())?;
            let count: u32 = count
                .ceil()
                .to_integer()
                .try_into()
                .context("animation frame count exceeds u32; supply --frame-end")?;
            count
                .checked_sub(1)
                .context("animation requires a positive scene duration")?
        };
        ensure!(
            !animated || end >= start,
            "--frame-end must be at least --frame-start"
        );
        ensure!(
            !animated || end - start <= 100_000,
            "animation exceeds 100001 samples; export a shorter range"
        );
        let initial_clock = if animated {
            format!(
                "{}/{}",
                u64::from(start) * scene.render.fps.denominator() as u64,
                scene.render.fps.numerator()
            )
            .parse()?
        } else {
            self.at
                .clone()
                .unwrap_or_else(|| vork_time::Clock::from(0.0))
        };
        let request: vork_time::Request = if animated {
            vork_time::Request::Frame(start, scene.render.fps)
        } else {
            (&initial_clock).into()
        };
        let time = request.numeric()?;
        let mut w = Writer::default();
        w.input_hash.update(&bytes);
        if self.root.is_none() && scene.propagation.is_some() {
            w.issue(
                "propagation",
                "scene.propagation",
                "curved transport",
                "omitted",
                "remove propagation for asset export",
                false,
            );
        }
        if self.root.is_none() && !scene.compositor().is_empty() {
            w.issue(
                "compositor",
                "composition.compositor",
                "image compositing",
                "omitted",
                "export without the compositor or acknowledge compositor",
                false,
            );
        }
        let buffers = buffers::SceneBuffers::decode(&scene, base)?;
        for (name, data) in &buffers.data {
            w.input_hash.update(name.as_bytes());
            for word in &data.words {
                w.input_hash.update(&word.to_le_bytes());
            }
        }
        let mut nodes = Vec::<EvalNode>::new();
        fn flatten(input: &[Node], parent: Option<usize>, out: &mut Vec<EvalNode>) {
            for n in input {
                let id = out.len();
                let world = parent.map_or(Mat4::IDENTITY, |p| out[p].world) * n.transform.matrix();
                out.push(EvalNode {
                    node: n.without_children(),
                    parent,
                    world,
                    dynamic: true,
                    camera: None,
                    embedded: None,
                });
                flatten(&n.children, Some(id), out);
            }
        }
        flatten(&scene.nodes, None, &mut nodes);
        let root = self
            .root
            .as_ref()
            .map(|name| {
                nodes
                    .iter()
                    .position(|n| &n.node.name == name)
                    .with_context(|| format!("export root {name:?}: use an existing node name"))
            })
            .transpose()?;
        if self.root.is_none()
            && !nodes.iter().any(|n| {
                matches!(
                    n.node.kind,
                    NodeKind::Light {
                        shape: vork_scene::LightShape::Environment,
                        ..
                    }
                )
            })
        {
            w.issue(
                "environment_lighting",
                "scene",
                "default environment illumination",
                "omitted; glTF has no environment light",
                "add environment lighting in the consumer or acknowledge environment_lighting",
                false,
            );
        }
        let mut plan = animation::Plan::compile_with_buffers(&scene, base, Some(&buffers.data))?;
        let material_keys = plan.material_inputs();
        for (slot, (material, input)) in material_keys.iter().enumerate() {
            let default = match scene.materials[material].inputs.get(input) {
                Some(Input::Number(v)) => animation::Sample::Numeric(vec![*v]),
                Some(Input::Vector(v)) => animation::Sample::Numeric(v.clone()),
                Some(Input::Bool(v)) => animation::Sample::Bool(*v),
                None => {
                    let p = v2::OpenPbrParam::find(input).with_context(|| {
                        format!(
                            "material {material}.{input}: author a numeric base value before export"
                        )
                    })?;
                    animation::Sample::Numeric(vec![0.; p.default.len().max(1)])
                }
                _ => bail!(
                    "material {material}.{input}: freeze resource/enum animation to a numeric value before exporting"
                ),
            };
            plan.bind_material(
                &(material.clone(), input.clone()),
                vec![animation::MaterialBinding {
                    slot,
                    parameter: slot,
                    storage: None,
                    enum_tokens: None,
                    default,
                    offset: 0.,
                }],
            )?;
        }
        plan.mark_nodes(&mut nodes, 0)?;
        let targets = targets::Plan::compile(&mut nodes)?;
        let mut bind_nodes = Vec::new();
        flatten(&scene.nodes, None, &mut bind_nodes);
        plan.mark_nodes(&mut bind_nodes, 0)?;
        let mut samples = plan.samples();
        plan.sample_request_into(time, &request, &mut samples)?;
        for (binding, value) in plan.material_values(&samples, 0.) {
            let (material, input) = &material_keys[binding.parameter];
            let value = match value {
                animation::Sample::Numeric(v) if v.len() == 1 => Input::Number(v[0]),
                animation::Sample::Numeric(v) => Input::Vector(v.clone()),
                animation::Sample::Bool(v) => Input::Bool(*v),
                animation::Sample::Style(_) => unreachable!("CSS/SVG channels do not bind material inputs"),
                animation::Sample::String(_) => bail!("export does not admit animated enum inputs"),
            };
            scene
                .materials
                .get_mut(material)
                .unwrap()
                .inputs
                .insert(input.clone(), value);
        }
        let mut sampled = values::sample_storage(&nodes);
        plan.apply_nodes(&samples, &mut sampled);
        for (n, value) in nodes.iter_mut().zip(sampled) {
            if let Some(value) = value {
                *n = value;
            }
        }
        let mut worlds: Vec<_> = nodes.iter().map(|n| n.world).collect();
        let order = if targets.order.is_empty() {
            (0..nodes.len()).collect()
        } else {
            targets.order.clone()
        };
        for i in order {
            let world = values::node_world(
                &nodes[i].node,
                nodes[i].parent.map_or(Mat4::IDENTITY, |p| worlds[p]),
            )?;
            worlds[i] = targets.aim(i, world, &worlds, &nodes[i].node, |_| false)?;
        }
        let mut selected: Vec<_> = (0..nodes.len())
            .filter(|&i| {
                root.is_none_or(|root| {
                    let mut p = Some(i);
                    while let Some(id) = p {
                        if id == root {
                            return true;
                        }
                        p = nodes[id].parent;
                    }
                    false
                })
            })
            .collect();
        let payload: BTreeSet<_> = selected.iter().copied().collect();
        if animated {
            // A skin may reference joints outside the selected subtree. Include
            // their ancestors, keeping one hierarchy and their evaluated placement.
            let mut dependencies = BTreeSet::new();
            for &i in &selected {
                if let NodeKind::Mesh {
                    skin: Some(skin), ..
                } = &nodes[i].node.kind
                {
                    for name in &skin.joints {
                        let matches: Vec<_> = nodes
                            .iter()
                            .enumerate()
                            .filter(|(_, n)| &n.node.name == name)
                            .map(|(i, _)| i)
                            .collect();
                        ensure!(
                            matches.len() == 1,
                            "skin joint {name:?} must name exactly one node"
                        );
                        let mut p = Some(matches[0]);
                        while let Some(i) = p {
                            dependencies.insert(i);
                            p = nodes[i].parent;
                        }
                    }
                }
            }
            for &i in &dependencies {
                if !selected.contains(&i) {
                    w.dispositions.push(json!({"source":nodes[i].node.name,"disposition":"preserved","meaning":"external skin joint/ancestor dependency"}));
                }
            }
            selected.extend(dependencies);
            selected.sort_unstable();
            selected.dedup();
        }
        ensure!(
            !selected.is_empty(),
            "export selection is empty; select a scene node"
        );
        ensure!(
            !animated || (selected.len() as u64) * (u64::from(end - start) + 1) <= 1_000_000,
            "animation exceeds one million node samples; select a smaller subtree or shorter range"
        );
        let mut map = BTreeMap::new();
        let mut output = Vec::<Value>::new();
        let mut roots = Vec::new();
        let mut material_ids = BTreeMap::new();
        let mut mesh_ids = BTreeMap::new();
        let mut skins = Vec::new();
        for &i in &selected {
            let n = &nodes[i].node;
            let address = if n.name.is_empty() {
                format!("nodes[{i}]")
            } else {
                n.name.clone()
            };
            if payload.contains(&i) && n.audio.is_some() {
                w.issue("audio", &address, "node sound", "omitted; glTF sound export is not built yet",
                    "export a picture-only subtree or acknowledge audio", false);
            }
            let parent = nodes[i].parent.and_then(|p| map.get(&p).copied());
            let local = if parent.is_some() {
                worlds[nodes[i].parent.unwrap()].inverse() * worlds[i]
            } else {
                worlds[i]
            };
            let (scale, rotation, translation) = local.to_scale_rotation_translation();
            ensure!(
                local.is_finite()
                    && local.abs_diff_eq(
                        Mat4::from_scale_rotation_translation(scale, rotation, translation),
                        1e-4
                    ),
                "node {address}: singular transform or shear; bake the transform into geometry before exporting"
            );
            let mut node = json!({"name":address,"translation":translation.to_array(),"rotation":rotation.to_array(),"scale":scale.to_array(),"extras":{"vork_source":address}});
            if buffers::node_visibility_mask(&nodes, i) != constants::RAY_VISIBILITY_ALL
                || buffers::node_matte_flags(&nodes, i) != 0
            {
                w.issue(
                    "visibility",
                    &address,
                    "renderer visibility/matte controls",
                    "refused; glTF has no equivalent visibility masks",
                    "remove visibility masks and matte flags, or select a visible subtree",
                    true,
                );
            }
            if !n.light_link_tags.is_empty() {
                w.issue(
                    "light_links",
                    &address,
                    "light receiver links",
                    "omitted",
                    "remove links or acknowledge light_links",
                    false,
                );
            }
            // External skin dependencies contribute transforms, not unrelated
            // camera/light/mesh payloads (which may themselves need other joints).
            let kind = if payload.contains(&i) {
                &n.kind
            } else {
                &NodeKind::Node {}
            };
            match kind {
                NodeKind::Node {} => {}
                // A 2D canvas's text has no glTF counterpart.
                NodeKind::Text { .. } => {
                    w.issue("canvas_text", &address, "2D text", "omitted; glTF has no 2D text", "export a 3D scene", false);
                }
                // A 2D canvas's path has no glTF counterpart.
                NodeKind::Path { .. } => {
                    w.issue("canvas_path", &address, "2D path", "omitted; glTF has no 2D path", "export a 3D scene", false);
                }
                NodeKind::Mesh {
                    geometry,
                    materials,
                    instances,
                    skin,
                    morph_weights,
                } => {
                    let source = &scene.geometry[geometry];
                    let unsupported = !matches!(source.topology, v2::Topology::Triangles)
                        || (skin.is_some() && !animated)
                        || !morph_weights.is_empty()
                        || !source.morphs.is_empty()
                        || source
                            .subdivision
                            .as_ref()
                            .is_some_and(|s| s.render_levels > 0)
                        || source.face_vertex_counts.is_some()
                        || computed_geometry::dynamic(&scene, source);
                    if unsupported {
                        w.issue("geometry_stage",&address,"final evaluated surface","refused; this stage needs a qualified final snapshot","supply static triangle buffers; skin, morph, subdivision, compute-written streams, polygons and nontriangles are not admitted yet",true);
                    } else {
                        let mut slots = Vec::new();
                        for name in if materials.is_empty() {
                            vec![None]
                        } else {
                            materials.iter().map(|n| Some(n.as_str())).collect()
                        } {
                            let key = name.unwrap_or("").to_owned();
                            let id = if let Some(&id) = material_ids.get(&key) {
                                id
                            } else {
                                let id = w.material(&scene, base, name)?;
                                material_ids.insert(key, id);
                                id
                            };
                            slots.push(id);
                        }
                        ensure!(
                            source.attributes.contains_key("TEXCOORD_0")
                                || !materials.iter().any(|n| scene.materials[n]
                                    .inputs
                                    .values()
                                    .any(|v| matches!(v, Input::Texture(_)))),
                            "node {address}: textured materials require TEXCOORD_0; author UVs before exporting"
                        );
                        ensure!(
                            source.attributes.contains_key("TANGENT")
                                || !materials.iter().any(|n| scene.materials[n]
                                    .inputs
                                    .contains_key("geometry_normal")),
                            "node {address}: a normal map requires authored unit tangents; generate them in the authoring tool before export"
                        );
                        if let Some(skin) = skin {
                            ensure!(
                                skin.joints.iter().collect::<BTreeSet<_>>().len()
                                    == skin.joints.len(),
                                "node {address}: glTF skin joints must be unique; combine duplicate joint influences before export"
                            );
                            ensure!(
                                source.attributes.contains_key("NORMAL")
                                    || !source.attributes.contains_key("TANGENT"),
                                "node {address}: a flat skinned mesh cannot retain rest-pose tangents; author normals with tangents or remove the tangent/normal-map input"
                            );
                            ensure!(
                                !skin.joints.is_empty() && instances.is_none(),
                                "node {address}: skin needs joints and cannot use instances; expand skinned instances into separate meshes"
                            );
                        }
                        let key = format!(
                            "{geometry}:{slots:?}:{:?}",
                            skin.as_ref().map(|s| s.joints.len())
                        );
                        let mesh = if let Some(&mesh) = mesh_ids.get(&key) {
                            mesh
                        } else {
                            let g = buffers::geometry_v2(&buffers, source, &[], 8, None)?;
                            let mesh = w.mesh(source, &g, &slots, skin.is_some())?;
                            if let Some(skin) = skin {
                                ensure!(
                                    instances.is_none(),
                                    "node {address}: skinned instances are not qualified; expand into individual meshes"
                                );
                                ensure!(
                                    !source.attributes.keys().any(|k| (k.starts_with("JOINTS_")
                                        || k.starts_with("WEIGHTS_"))
                                        && !k.ends_with("_0")),
                                    "node {address}: additional skin influence sets are not supported; explicitly reduce authored influences with measured error or use another exporter; no influences were truncated"
                                );
                                let joints = &buffers.data[source
                                    .attributes
                                    .get("JOINTS_0")
                                    .context("skin requires JOINTS_0")?];
                                let weights = buffers.data[source
                                    .attributes
                                    .get("WEIGHTS_0")
                                    .context("skin requires WEIGHTS_0")?]
                                .floats::<4>()?;
                                let count = buffers.data[&source.attributes["POSITION"]].count();
                                ensure!(
                                    joints.count() == count && weights.len() == count,
                                    "skin attributes must match POSITION count"
                                );
                                let joint_rows: Vec<_> = joints.words.chunks_exact(4).collect();
                                ensure!(
                                    joint_rows.len() == count
                                        && joint_rows
                                            .iter()
                                            .flat_map(|row| row.iter())
                                            .all(|&j| (j as usize) < skin.joints.len()
                                                && j <= u16::MAX as u32),
                                    "joint index outside skin or glTF u16 range"
                                );
                                ensure!(
                                    weights
                                        .iter()
                                        .all(|row| row.iter().all(|x| x.is_finite() && *x >= 0.)
                                            && (row.iter().sum::<f32>() - 1.).abs() < 1e-5),
                                    "skin weights must be finite, nonnegative and sum to one; normalize the authored WEIGHTS_0 values"
                                );
                                let ids: Vec<_> = if g.vertex_ids.is_empty() {
                                    (0..g.vertices.len()).collect()
                                } else {
                                    g.vertex_ids.iter().map(|&x| x as usize).collect()
                                };
                                let bytes: Vec<_> = ids
                                    .iter()
                                    .flat_map(|&i| {
                                        joint_rows[i].iter().flat_map(|&x| (x as u16).to_le_bytes())
                                    })
                                    .collect();
                                let view = w.view(&bytes, Some(34962));
                                let ja = w.accessors.len();
                                w.accessors.push(json!({"bufferView":view,"componentType":5123,"count":ids.len(),"type":"VEC4"}));
                                let wa = w.floats(
                                    &ids.iter().map(|&i| weights[i].to_vec()).collect::<Vec<_>>(),
                                    "VEC4",
                                    false,
                                )?;
                                for primitive in
                                    w.meshes[mesh]["primitives"].as_array_mut().unwrap()
                                {
                                    primitive["attributes"]["JOINTS_0"] = ja.into();
                                    primitive["attributes"]["WEIGHTS_0"] = wa.into();
                                }
                            }
                            mesh_ids.insert(key, mesh);
                            mesh
                        };
                        if let Some(instances) = instances {
                            ensure!(
                                !instances.attributes.contains_key("time_offset")
                                    && instances.count.is_none()
                                    && !instances
                                        .attributes
                                        .values()
                                        .any(|name| crate::compute::writes_buffer(&scene, name)),
                                "node {address}: live, compute-written or time-offset instances are not qualified; supply static TRS columns without count or time_offset"
                            );
                            for transform in buffers::instance_transforms(instances, &buffers)? {
                                // Native instance visibility hides any zero scale axis.
                                // Such a matrix cannot be decomposed into a unit quaternion.
                                if [transform.x_axis, transform.y_axis, transform.z_axis]
                                    .iter()
                                    .any(|axis| axis.truncate() == Vec3::ZERO)
                                {
                                    continue;
                                }
                                let (s, r, t) = transform.to_scale_rotation_translation();
                                let child = output.len();
                                output.push(json!({"mesh":mesh,"translation":t.to_array(),"rotation":r.to_array(),"scale":s.to_array()}));
                                node["children"]
                                    .as_array_mut()
                                    .map(|a| a.push(child.into()))
                                    .unwrap_or_else(|| {
                                        node["children"] = json!([child]);
                                    });
                            }
                        } else {
                            node["mesh"] = mesh.into();
                            if skin.is_some() {
                                node["skin"] = skins.len().into();
                                skins.push(i);
                            }
                        }
                    }
                }
                NodeKind::Asset { .. } => w.issue(
                    "imported_asset",
                    &address,
                    "evaluated imported asset",
                    "refused; imported snapshot not qualified",
                    "expand the imported asset into native static triangle buffers",
                    true,
                ),
                NodeKind::Camera { material, .. } => {
                    let default: v2::Material =
                        serde_json::from_value(json!({"program":"std/camera"}))?;
                    let m = material
                        .as_ref()
                        .map_or(&default, |name| &scene.materials[name]);
                    if m.program != "std/camera" {
                        w.issue(
                            "camera_program",
                            &address,
                            "custom camera rays",
                            "refused",
                            "use std/camera with symmetric projective optics",
                            true,
                        );
                    } else {
                        let compiler = vork_slang::Compiler::discover();
                        let abi = programs::abi();
                        let program = compiler.compile_camera(
                            &abi,
                            vork_slang::module_source("std/camera").unwrap(),
                            Path::new("std/camera.slang"),
                            "exportcamera",
                        )?;
                        let inputs = m
                            .inputs
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
                            .collect::<Result<BTreeMap<_, _>>>()?;
                        let words = program.inputs.pack(&inputs, |name, _| {
                            bail!("camera {address}.{name}: numeric inputs required")
                        })?;
                        let aspect =
                            scene.render.resolution[0] as f32 / scene.render.resolution[1] as f32;
                        let state = camera_program::State {
                            program: std::sync::Arc::new(program),
                            words,
                            offset: 0,
                            focus: -1.,
                            time,
                            evaluated: Default::default(),
                        };
                        let c = state
                            .calibration(aspect)?
                            .context("camera has no projective calibration")?;
                        if c[0][2] != 0. || c[0][3] != 0. {
                            w.issue(
                                "camera_shift",
                                &address,
                                "shifted view plane",
                                "refused",
                                "set camera shift to [0,0]",
                                true,
                            );
                        }
                        if c[2][0] > 0. {
                            w.issue(
                                "camera_dof",
                                &address,
                                "depth of field",
                                "omitted; pinhole projection",
                                "disable use_dof or acknowledge camera_dof",
                                false,
                            );
                        }
                        let projection = if c[1][2] == 1. {
                            ensure!(
                                (1. / c[0][0]).is_finite() && (1. / c[0][1]).is_finite(),
                                "camera {address}: orthographic extent exceeds finite delivery range; adjust ortho_scale"
                            );
                            json!({"type":"orthographic","orthographic":{"xmag":1./c[0][0],"ymag":1./c[0][1],"znear":c[1][0],"zfar":c[1][1]}})
                        } else {
                            let yfov = 2. * (1. / c[0][1]).atan();
                            ensure!(
                                yfov > 0. && yfov < std::f32::consts::PI,
                                "camera {address}: field of view is outside glTF's open (0, pi) range; adjust lens/sensor inputs"
                            );
                            json!({"type":"perspective","perspective":{"yfov":yfov,"aspectRatio":aspect,"znear":c[1][0],"zfar":c[1][1]}})
                        };
                        node["camera"] = w.cameras.len().into();
                        w.cameras.push(projection);
                    }
                }
                NodeKind::Light {
                    shape,
                    color,
                    temperature,
                    energy,
                    angle,
                    radius,
                    spot_size,
                    spot_blend,
                    illuminates,
                    shadowed_by,
                    image,
                    medium,
                    ..
                } => {
                    if let Some(image) = image {
                        w.issue("light_profile",&address,&format!("IES image {image}"),"IES angular distribution and absolute modulation omitted; plain punctual energy used","remove light.image or acknowledge light_profile after reviewing the lighting change",false);
                    }
                    if medium.is_some() {
                        w.issue(
                            "medium",
                            &address,
                            "participating medium",
                            "refused; glTF has no equivalent transport",
                            "remove the medium or select an ordinary mesh subtree",
                            true,
                        );
                    }

                    if illuminates.is_some() || shadowed_by.is_some() {
                        w.issue(
                            "light_links",
                            &address,
                            "light links",
                            "omitted",
                            "remove illuminates/shadowed_by or acknowledge light_links",
                            false,
                        );
                    }
                    let kind = match shape {
                        vork_scene::LightShape::Sun => Some("directional"),
                        vork_scene::LightShape::Point => Some("point"),
                        vork_scene::LightShape::Spot => Some("spot"),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        if temperature.is_some() {
                            w.issue(
                                "light_temperature",
                                &address,
                                "blackbody tint",
                                "refused",
                                "replace temperature with evaluated RGB color",
                                true,
                            );
                        }
                        if (*angle > 0. && kind == "directional") || *radius > 0. {
                            w.issue(
                                "light_extent",
                                &address,
                                "finite emitter extent",
                                "punctual emitter; shadow softness changes",
                                "set angle/radius to zero or acknowledge light_extent",
                                false,
                            );
                        }
                        let energy = energy.unwrap_or(shape.default_energy());
                        let intensity = energy * constants::LUMENS_PER_WATT
                            / if kind == "directional" {
                                1.
                            } else {
                                4. * std::f32::consts::PI
                            };
                        ensure!(
                            angle.is_finite()
                                && *angle >= 0.
                                && radius.is_finite()
                                && *radius >= 0.,
                            "light {address}: sampled extent must be finite and nonnegative; adjust angle/radius keys and tangents"
                        );
                        let color = color.unwrap_or(shape.default_color(false));
                        ensure!(
                            color
                                .iter()
                                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                                && intensity.is_finite()
                                && intensity >= 0.,
                            "light {address}: sampled color/intensity is outside glTF's range; adjust keys/tangents or normalize color into energy"
                        );
                        let mut light =
                            json!({"name":address,"type":kind,"color":color,"intensity":intensity});
                        if kind == "spot" {
                            let inner = vork_scene::spot_inner_angle(*spot_size, *spot_blend);
                            let outer =
                                (f64::from(*spot_size) * 0.5).min(std::f64::consts::FRAC_PI_2);
                            ensure!(
                                inner.is_finite()
                                    && inner >= 0.
                                    && f64::from(inner) < outer
                                    && *spot_size > 0.
                                    && *spot_size <= std::f32::consts::PI,
                                "light {address}: sampled spot cone is outside glTF's range; adjust spot_size/spot_blend keys and tangents"
                            );
                            light["spot"] = json!({"innerConeAngle":inner,"outerConeAngle":outer});
                        }
                        node["extensions"] =
                            json!({"KHR_lights_punctual":{"light":w.lights.len()}});
                        w.lights.push(light);
                        w.extensions.insert("KHR_lights_punctual".into());
                    } else {
                        w.issue("light_representation",&address,"area/environment illumination","omitted","replace with punctual lights or select an asset subtree; acknowledge light_representation to omit",false);
                    }
                }
            }
            let id = output.len();
            map.insert(i, id);
            output.push(node);
            if let Some(parent) = parent {
                let p: &mut Value = &mut output[parent];
                if p.get("children").is_none() {
                    p["children"] = json!([]);
                }
                p["children"].as_array_mut().unwrap().push(id.into());
            } else {
                roots.push(id);
            }
            w.dispositions.push(
                json!({"source":address,"kind":"object","disposition":"evaluated","source_node":i}),
            );
        }
        let mut skin_output = Vec::new();
        let names = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.node.name.clone(), i))
            .collect();
        for i in skins {
            let binding = buffers::native_binding(&scene, &nodes, &names, i, &buffers)?;
            let matrices = binding
                .joints
                .iter()
                .map(|(_, m)| m.to_cols_array().to_vec())
                .collect::<Vec<_>>();
            let accessor = w.floats(&matrices, "MAT4", false)?;
            let view = w.accessors[accessor]["bufferView"].as_u64().unwrap() as usize;
            w.views[view].as_object_mut().unwrap().remove("target");
            skin_output.push(json!({"joints":binding.joints.iter().map(|(j, _)| map[j]).collect::<Vec<_>>(),"inverseBindMatrices":accessor}));
        }
        let mut animations = Vec::new();
        if animated {
            // Core glTF admits node TRS only. Never freeze other keyed values in a clip.
            for layer in &scene.animation {
                ensure!(
                    layer.clip.is_none(),
                    "imported clips require native expanded nodes before animation export"
                );
                for track in layer.tracks.iter().flatten() {
                    let trs = matches!(track.path.as_slice(), [root, _, field] if root == "nodes" && ["translation", "rotation", "scale"].contains(&field.as_str()));
                    let weight = matches!(track.path.as_slice(), [root, _, field] if root == "animation" && field == "weight");
                    ensure!(
                        trs || weight,
                        "animation {:?}: core glTF cannot animate this property; remove the track or export a snapshot",
                        track.path
                    );
                }
            }
            let mut modes = BTreeMap::new();
            let mut key_times = Vec::new();
            let mut equivalent = self.root.is_none()
                && targets.order.is_empty()
                && scene.animation.len() <= 1
                && bind_nodes
                    .iter()
                    .all(|n| n.node.transform.scale.iter().all(|s| *s > 0.));
            for layer in &scene.animation {
                equivalent &= layer.weight == 1.
                    && matches!(layer.blend, v2::Blend::Replace)
                    && layer.offset == 0.
                    && layer.speed == 1.
                    && !layer.looping;
                for track in layer.tracks.iter().flatten() {
                    let [root, name, field] = track.path.as_slice() else {
                        equivalent = false;
                        continue;
                    };
                    if root != "nodes" {
                        equivalent = false;
                        continue;
                    }
                    let mode = match track.interpolation {
                        v2::Interpolation::Step => "STEP",
                        v2::Interpolation::Linear => "LINEAR",
                        _ => {
                            equivalent = false;
                            "LINEAR"
                        }
                    };
                    if modes.insert((name.clone(), field.clone()), mode).is_some() {
                        equivalent = false;
                    }
                    match &track.values {
                        v2::Values::Inline(values) => {
                            if field == "rotation"
                                && values.iter().any(|v| v.is_object())
                                && mode != "STEP"
                            {
                                equivalent = false;
                            }
                            if field == "scale"
                                && values.iter().any(|v| {
                                    v.as_array().is_none_or(|v| {
                                        v.iter().any(|x| x.as_f64().is_none_or(|x| x <= 0.))
                                    })
                                })
                            {
                                equivalent = false;
                            }
                        }
                        _ => equivalent = false,
                    }
                    match &track.keys {
                        v2::Keys::Inline(keys) => {
                            for key in keys {
                                key_times.push((
                                    serde_json::from_value::<vork_time::Clock>(key[0].clone())?,
                                    mode == "STEP",
                                ));
                            }
                        }
                        _ => equivalent = false,
                    }
                }
            }
            if !equivalent {
                w.issue("animation_resampling", "animation", "continuous composed animation",
                    "LINEAR TRS keys at source frames; between-frame curves may differ",
                    "acknowledge animation_resampling after checking playback, or export a snapshot", false);
            }
            let origin = vork_time::Request::Frame(start, scene.render.fps)
                .exact()?
                .value()
                .clone();
            let last = vork_time::Request::Frame(end, scene.render.fps)
                .exact()?
                .value()
                .clone();
            let mut quantization_max = 0_f32;
            let mut requests: Vec<_> = (start..=end)
                .map(|frame| {
                    (
                        ((frame - start) as f64 / scene.render.fps.as_f64()) as f32,
                        vork_time::Request::Frame(frame, scene.render.fps),
                    )
                })
                .collect();
            if equivalent {
                let mut authored_times = BTreeMap::new();
                for (clock, step) in key_times {
                    let source = clock.exact();
                    if source > origin && source < last {
                        let relative = source - &origin;
                        let local: vork_time::Clock = relative.to_string().parse()?;
                        let mut t = local.as_f32();
                        // STEP must already have advanced at the authored boundary.
                        // Round toward the earlier representable timestamp rather
                        // than delaying a jump by a fraction of a float ULP.
                        if step && vork_time::exact::ieee(t)? > relative {
                            t = t.next_down();
                        }
                        let delta = relative - vork_time::exact::ieee(t)?;
                        let error: vork_time::Clock = delta.to_string().parse()?;
                        quantization_max = quantization_max.max(error.as_f32().abs());
                        if let Some(previous) = authored_times.insert(t.to_bits(), clock.exact()) {
                            ensure!(
                                previous == clock.exact(),
                                "distinct authored key times collide in glTF float seconds; shorten the range or retime the keys"
                            );
                        }
                        requests.push((t, (&clock).into()));
                    }
                }
                requests.sort_by(|a, b| {
                    a.0.total_cmp(&b.0).then_with(|| {
                        matches!(a.1, vork_time::Request::Frame(..))
                            .cmp(&matches!(b.1, vork_time::Request::Frame(..)))
                    })
                });
                requests.dedup_by(|a, b| a.0 == b.0);
            }
            ensure!(
                (requests.len() as u64) * selected.len() as u64 <= 1_000_000,
                "animation exceeds one million node samples; shorten the range"
            );
            let mut channels = Vec::new();
            let mut samplers = Vec::new();
            let mut tracks: BTreeMap<usize, [Vec<Vec<f32>>; 3]> =
                selected.iter().map(|&i| (i, Default::default())).collect();
            let mut times = Vec::new();
            for (clip_time, request) in requests {
                plan.sample_request_into(request.numeric()?, &request, &mut samples)?;
                let mut values = values::sample_storage(&bind_nodes);
                plan.apply_nodes(&samples, &mut values);
                let current: Vec<_> = values
                    .into_iter()
                    .zip(&bind_nodes)
                    .map(|(v, n)| {
                        v.unwrap_or_else(|| EvalNode {
                            node: n.node.clone(),
                            parent: n.parent,
                            world: n.world,
                            dynamic: n.dynamic,
                            camera: None,
                            embedded: None,
                        })
                    })
                    .collect();
                let mut world = vec![Mat4::IDENTITY; current.len()];
                let order: Vec<_> = if targets.order.is_empty() {
                    (0..current.len()).collect()
                } else {
                    targets.order.clone()
                };
                for i in order {
                    let m = values::node_world(
                        &current[i].node,
                        current[i].parent.map_or(Mat4::IDENTITY, |p| world[p]),
                    )?;
                    world[i] = targets.aim(i, m, &world, &current[i].node, |_| false)?;
                }
                let t = clip_time;
                ensure!(
                    times.last().is_none_or(|last: &Vec<f32>| t > last[0]),
                    "clip times collide in glTF float seconds; shorten the range"
                );
                times.push(vec![t]);
                for (&i, rows) in &mut tracks {
                    let local = current[i]
                        .parent
                        .filter(|p| map.contains_key(p))
                        .map_or(world[i], |p| world[p].inverse() * world[i]);
                    let (s, mut r, t) = if equivalent {
                        let transform = &current[i].node.transform;
                        (
                            Vec3::from_array(transform.scale),
                            glam::Quat::from_array(transform.resolved_rotation()),
                            Vec3::from_array(transform.translation),
                        )
                    } else {
                        local.to_scale_rotation_translation()
                    };
                    ensure!(
                        local.is_finite()
                            && local
                                .abs_diff_eq(Mat4::from_scale_rotation_translation(s, r, t), 1e-4),
                        "node {} at clip time {clip_time}: animated shear/singular transform cannot be exported",
                        current[i].node.name
                    );
                    if rows[1].last().is_some_and(|last| {
                        r.to_array()
                            .iter()
                            .zip(last)
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            < 0.
                    }) {
                        r = -r;
                    }
                    rows[0].push(t.to_array().to_vec());
                    rows[1].push(r.to_array().to_vec());
                    rows[2].push(s.to_array().to_vec());
                }
            }
            let input = w.floats(&times, "SCALAR", true)?;
            let view = w.accessors[input]["bufferView"].as_u64().unwrap() as usize;
            w.views[view].as_object_mut().unwrap().remove("target");
            for (i, rows) in tracks {
                for ((path, kind), data) in [
                    ("translation", "VEC3"),
                    ("rotation", "VEC4"),
                    ("scale", "VEC3"),
                ]
                .into_iter()
                .zip(rows)
                {
                    if data.iter().all(|v| v == &data[0]) {
                        continue;
                    }
                    let output = w.floats(&data, kind, false)?;
                    let view = w.accessors[output]["bufferView"].as_u64().unwrap() as usize;
                    w.views[view].as_object_mut().unwrap().remove("target");
                    channels.push(
                        json!({"sampler":samplers.len(),"target":{"node":map[&i],"path":path}}),
                    );
                    let mode = if equivalent {
                        modes
                            .get(&(nodes[i].node.name.clone(), path.to_owned()))
                            .copied()
                            .unwrap_or("LINEAR")
                    } else {
                        "LINEAR"
                    };
                    samplers.push(json!({"input":input,"output":output,"interpolation":mode}));
                }
            }
            if !channels.is_empty() {
                animations.push(json!({"name":"Scene","channels":channels,"samplers":samplers}));
            }
            w.dispositions.push(json!({"source":"animation","disposition":if equivalent {"evaluated"} else {"approximated"},"interpolation_equivalent":equivalent,"key_time_quantization_max_seconds":quantization_max,"step_time_rounding":"toward earlier float to retain authored-boundary advancement","frame_start":start,"frame_end":end,"sample_rate":scene.render.fps,"clip_time_origin":0,"meaning":if equivalent {"composed TRS at source frames and authored breakpoints; source interpolation retained within float timestamp precision"} else {"composed TRS sampled at inclusive source frames; runtime layers and between-frame curves are not reconstructed"}}));
        }
        if !animated && !scene.animation.is_empty() {
            w.dispositions.push(json!({"source":"animation","disposition":"evaluated","meaning":"frozen static snapshot; no animation clip delivered"}));
        }
        for issue in &mut w.issues {
            if let Some(source) = issue["source"].as_str() {
                let mut locations = Vec::new();
                for (name, origins) in &scene.origins {
                    let direct = name
                        .strip_prefix("materials.")
                        .or_else(|| name.strip_prefix("nodes."));
                    if direct.is_some_and(|name| {
                        source == name
                            || source
                                .strip_prefix(name)
                                .is_some_and(|rest| rest.starts_with(".inputs."))
                    }) {
                        locations.extend(origins.iter().cloned());
                    }
                }
                if !locations.is_empty() {
                    issue["source_locations"] = json!(locations);
                }
            }

            issue["acknowledged"] =
                json!(self.allow_losses.iter().any(|code| issue["code"] == *code));
        }
        let failed = w
            .issues
            .iter()
            .any(|i| i["severity"] == "error" || i["acknowledged"] != true);
        let mut report = json!({"version":1,"status":if failed{"failed"}else{"complete"},"vork_version":if self.vork_version.is_empty(){env!("CARGO_PKG_VERSION")}else{self.vork_version.as_str()},"source":self.scene,"source_hash":w.input_hash.finalize().to_hex().to_string(),"selection":{"root":self.root},"time":time,"source_clock":self.at,"fps":scene.render.fps,"target":{"specification":"glTF 2.0","extensions_used":w.extensions,"extensions_required":w.extensions},"animation_range":if animated {json!({"frame_start":start,"frame_end":end,"clip_time_origin":0})} else {Value::Null},"counts":{"skins":skin_output.len(),"animations":animations.len(),"cameras":w.cameras.len(),"lights":w.lights.len(),"joints":skin_output.iter().map(|s| s["joints"].as_array().unwrap().len()).sum::<usize>(),"nodes":output.len(),"meshes":w.meshes.len(),"materials":w.materials.len(),"images":w.images.len(),"vertices":w.vertex_count,"triangles":w.triangle_count,"buffer_bytes":w.bin.len()},"issues":w.issues,"dispositions":w.dispositions,"dependency_scope":"selected subtree; full scene transform dependencies evaluated","bake":null});
        let report_path = self.out.with_extension("export.json");
        if failed {
            publish(&report_path, &serde_json::to_vec_pretty(&report)?)?;
            return Err(anyhow::anyhow!(
                "export refused: {}; fix the named sources in {}",
                w.issues
                    .iter()
                    .filter(|i| i["severity"] == "error" || i["acknowledged"] != true)
                    .map(|i| format!("{} at {}: {}", i["code"], i["source"], i["remediation"]))
                    .collect::<Vec<_>>()
                    .join("; "),
                report_path.display()
            )
            .context(Refused));
        }
        let mut document = json!({"asset":{"version":"2.0","generator":"vork export"},"scene":0,"scenes":[{"nodes":roots}],"nodes":output});
        for (key, items) in [
            ("animations", animations),
            ("skins", skin_output),
            ("meshes", w.meshes),
            ("materials", w.materials),
            ("images", w.images),
            ("textures", w.textures),
            ("samplers", w.samplers),
            ("cameras", w.cameras),
            ("bufferViews", w.views),
            ("accessors", w.accessors),
        ] {
            if !items.is_empty() {
                document[key] = json!(items);
            }
        }
        if !w.extensions.is_empty() {
            document["extensionsUsed"] = json!(w.extensions);
            document["extensionsRequired"] = json!(w.extensions);
        }
        if !w.lights.is_empty() {
            document["extensions"] = json!({"KHR_lights_punctual":{"lights":w.lights}});
        }
        let image_bytes = if extension == "gltf" {
            externalize_images(&mut document, &mut w.bin, &self.out)?
        } else {
            0
        };
        let asset = if extension == "glb" {
            if !w.bin.is_empty() {
                document["buffers"] = json!([{"byteLength":w.bin.len()}]);
            }
            glb(&document, &w.bin)?
        } else {
            // Content-addressed sidecars keep an old asset valid until the new JSON is published.
            if !w.bin.is_empty() {
                let name = format!("{}.bin", blake3::hash(&w.bin).to_hex());
                publish(&self.out.with_file_name(&name), &w.bin)?;
                document["buffers"] = json!([{"byteLength":w.bin.len(),"uri":name}]);
            }
            serde_json::to_vec_pretty(&document)?
        };
        // Structural validation before publication; external qualification also runs Khronos Validator.
        let checked = if extension == "glb" {
            asset.clone()
        } else {
            glb(
                &{
                    let mut d = document.clone();
                    if !w.bin.is_empty() {
                        d["buffers"][0].as_object_mut().unwrap().remove("uri");
                    }
                    d
                },
                &w.bin,
            )?
        };
        import::from_slice(&checked).context("export generated invalid glTF")?;
        report["asset_bytes"] = asset.len().into();
        report["bytes"] = (asset.len()
            + if extension == "gltf" {
                w.bin.len() + image_bytes
            } else {
                0
            })
        .into();
        report["counts"]["buffer_bytes"] = w.bin.len().into();
        publish(&report_path, &serde_json::to_vec_pretty(&report)?)?;
        publish(&self.out, &asset)?;
        Ok(report)
    }
}
/// glTF uses ordinary relative PNGs; GLB retains the same images in bufferViews.
fn externalize_images(document: &mut Value, bin: &mut Vec<u8>, out: &Path) -> Result<usize> {
    let mut removed = BTreeSet::new();
    let mut resources = 0;
    let views = document["bufferViews"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if let Some(images) = document.get_mut("images").and_then(Value::as_array_mut) {
        for image in images {
            let view = image["bufferView"]
                .as_u64()
                .context("export image has no view")? as usize;
            let offset = views[view]["byteOffset"].as_u64().unwrap_or(0) as usize;
            let length = views[view]["byteLength"].as_u64().unwrap() as usize;
            let bytes = &bin[offset..offset + length];
            let name = format!("{}.png", blake3::hash(bytes).to_hex());
            publish(&out.with_file_name(&name), bytes)?;
            resources += length;
            image.as_object_mut().unwrap().remove("bufferView");
            image["uri"] = name.into();
            removed.insert(view);
        }
    }
    if removed.is_empty() {
        return Ok(0);
    }
    let mut packed = Vec::new();
    let mut retained = Vec::new();
    let mut remap = BTreeMap::new();
    for (index, mut view) in views.into_iter().enumerate() {
        if removed.contains(&index) {
            continue;
        }
        while packed.len() % 4 != 0 {
            packed.push(0);
        }
        let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let length = view["byteLength"].as_u64().unwrap() as usize;
        view["byteOffset"] = packed.len().into();
        packed.extend_from_slice(&bin[offset..offset + length]);
        remap.insert(index, retained.len());
        retained.push(view);
    }
    if let Some(accessors) = document.get_mut("accessors").and_then(Value::as_array_mut) {
        for accessor in accessors {
            if let Some(index) = accessor["bufferView"].as_u64() {
                accessor["bufferView"] = remap[&(index as usize)].into();
            }
        }
    }
    if retained.is_empty() {
        document.as_object_mut().unwrap().remove("bufferViews");
    } else {
        document["bufferViews"] = retained.into();
    }
    *bin = packed;
    Ok(resources)
}

fn glb(document: &Value, bin: &[u8]) -> Result<Vec<u8>> {
    let mut text = serde_json::to_vec(document)?;
    while text.len() % 4 != 0 {
        text.push(b' ');
    }
    let bin_size = (bin.len() + 3) & !3;
    let size = 12 + 8 + text.len() + if bin.is_empty() { 0 } else { 8 + bin_size };
    let size = u32::try_from(size)
        .context("GLB exceeds the 4 GiB format limit; select a smaller subtree")?;
    let mut out = Vec::new();
    for word in [0x46546c67, 2, size, text.len() as u32, 0x4e4f534a] {
        out.extend(word.to_le_bytes());
    }
    out.extend(text);
    if !bin.is_empty() {
        out.extend((bin_size as u32).to_le_bytes());
        out.extend(0x004e4942u32.to_le_bytes());
        out.extend(bin);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    Ok(out)
}
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let temp = path.with_extension(format!(
        "export-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result.map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Case(PathBuf);
    impl Case {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "vork-export-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn export(&self, scene: Value, root: Option<&str>, out: &str) -> Export {
            let file = self.0.join("scene.json");
            std::fs::write(&file, serde_json::to_vec(&scene).unwrap()).unwrap();
            Export {
                scene: file,
                out: self.0.join(out),
                root: root.map(str::to_owned),
                allow_losses: vec!["environment_lighting".into()],
                ..Default::default()
            }
        }
    }
    impl Drop for Case {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn triangle() -> Value {
        json!({"version":2,"nodes":[{"name":"parent","translation":[2,0,0],"children":[{"name":"mesh","type":"mesh","geometry":"g","translation":[0,3,0]}]}],"geometry":{"g":{"attributes":{"POSITION":"p"}}},"buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}}})
    }
    fn moving_skin() -> Value {
        let mut scene = triangle();
        scene["render"] = json!({"fps":"24000/1001","duration":2});
        scene["nodes"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":"joint","translation":[0,0,0]}));
        scene["nodes"][0]["children"][0]["skin"] =
            json!({"joints":["joint"],"inverse_bind_matrices":"ibm"});
        scene["buffers"]["ibm"] = json!({"type":"mat4","data":[[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]]});
        scene["buffers"]["j"] = json!({"type":"uvec4","data":[[0,0,0,0],[0,0,0,0],[0,0,0,0]]});
        scene["buffers"]["w"] = json!({"type":"vec4","data":[[1,0,0,0],[1,0,0,0],[1,0,0,0]]});
        scene["geometry"]["g"]["attributes"]["JOINTS_0"] = json!("j");
        scene["geometry"]["g"]["attributes"]["WEIGHTS_0"] = json!("w");
        scene["animation"] = json!([{"tracks":[{"path":["nodes","joint","translation"],"keys":[[0,[0,0,0]],[2,[2,0,0]]]}]}]);
        scene
    }
    #[test]
    fn skin_clip_keeps_rest_vertices_bindings_and_external_joint() {
        let c = Case::new();
        let mut e = c.export(moving_skin(), Some("mesh"), "skin.glb");
        e.frame_start = Some(12);
        e.frame_end = Some(24);
        e.allow_losses.push("animation_resampling".into());
        e.run().unwrap();
        let file = import::open(&e.out).unwrap();
        let data = import::buffers(&file.document, e.out.parent(), file.blob.clone()).unwrap();
        let skin = file.skins().next().unwrap();
        assert_eq!(
            skin.joints().map(|j| j.name().unwrap()).collect::<Vec<_>>(),
            ["joint"]
        );
        assert_eq!(
            skin.reader(|b| Some(&data[b.index()]))
                .read_inverse_bind_matrices()
                .unwrap()
                .next()
                .unwrap(),
            Mat4::IDENTITY.to_cols_array_2d()
        );
        let primitive = file.meshes().next().unwrap().primitives().next().unwrap();
        let reader = primitive.reader(|b| Some(&data[b.index()]));
        assert_eq!(
            reader.read_positions().unwrap().collect::<Vec<_>>(),
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
        );
        assert_eq!(
            reader
                .read_weights(0)
                .unwrap()
                .into_f32()
                .collect::<Vec<_>>(),
            vec![[1., 0., 0., 0.]; 3]
        );
        let channel = file.animations().next().unwrap().channels().next().unwrap();
        let reader = channel.reader(|b| Some(&data[b.index()]));
        let times: Vec<_> = reader.read_inputs().unwrap().collect();
        assert_eq!(times.len(), 13);
        assert_eq!(times[0], 0.);
        assert!((times[12] - 0.5005).abs() < 1e-6);
        let gltf::animation::util::ReadOutputs::Translations(values) =
            reader.read_outputs().unwrap()
        else {
            panic!()
        };
        let values: Vec<_> = values.collect();
        assert!((values[0][0] - 0.5005).abs() < 1e-6);
        assert!((values[12][0] - 1.001).abs() < 1e-6);
    }
    #[test]
    fn external_skin_ancestors_contribute_only_transforms() {
        let c = Case::new();
        let mut scene = moving_skin();
        // The mesh's parent is also a joint ancestor. Its separate geometry is
        // outside the selected subtree, and must not be exported or evaluated.
        scene["nodes"][0]["type"] = json!("mesh");
        scene["nodes"][0]["geometry"] = json!("unrelated");
        scene["geometry"]["unrelated"] =
            json!({"topology":"points","attributes":{"POSITION":"p"},"widths":"widths"});
        scene["buffers"]["widths"] = json!({"type":"f32","data":[0.1,0.1,0.1]});
        let joint = scene["nodes"].as_array_mut().unwrap().pop().unwrap();
        scene["nodes"][0]["children"]
            .as_array_mut()
            .unwrap()
            .push(joint);
        let mut e = c.export(scene, Some("mesh"), "skin.glb");
        e.frame_end = Some(24);
        e.allow_losses.push("animation_resampling".into());
        let report = e.run().unwrap();
        assert_eq!(report["counts"]["meshes"], 1);
        let file = import::open(&e.out).unwrap();
        let parent = file.nodes().find(|n| n.name() == Some("parent")).unwrap();
        assert!(parent.mesh().is_none());
        assert_eq!(parent.transform().decomposed().0, [2., 0., 0.]);
        assert_eq!(
            file.skins().next().unwrap().joints().next().unwrap().name(),
            Some("joint")
        );
    }
    #[test]
    fn zero_scale_instances_remain_hidden_and_meshes_remain_shared() {
        let c = Case::new();
        let mut scene = triangle();
        scene["nodes"][0]["children"][0]["instances"] = json!({"attributes":{"SCALE":"scales"}});
        scene["buffers"]["scales"] = json!({"type":"vec3","data":[[1,1,1],[0,1,1],[-1,2,1]]});
        let e = c.export(scene, None, "instances.glb");
        e.run().unwrap();
        let file = import::open(&e.out).unwrap();
        assert_eq!(file.meshes().len(), 1);
        let nodes: Vec<_> = file.nodes().filter(|n| n.mesh().is_some()).collect();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].transform().decomposed().2, [1., 1., 1.]);
        assert_eq!(nodes[1].transform().decomposed().2, [-1., 2., 1.]);
    }
    #[test]
    fn skinned_flat_shading_does_not_freeze_rest_pose_normals() {
        let c = Case::new();
        let mut e = c.export(moving_skin(), None, "flat.glb");
        e.frame_end = Some(24);
        e.run().unwrap();
        let file = import::open(&e.out).unwrap();
        let primitive = file.meshes().next().unwrap().primitives().next().unwrap();
        assert!(primitive.get(&gltf::Semantic::Normals).is_none());
        assert!(primitive.get(&gltf::Semantic::Joints(0)).is_some());
    }
    #[test]
    fn step_clip_retains_off_frame_jump_and_needs_no_approximation_permission() {
        let c = Case::new();
        let mut scene = triangle();
        scene["animation"] = json!([{"tracks":[{"path":["nodes","mesh","translation"],"interpolation":"STEP","keys":[[0,[0,0,0]],[0.37,[2,0,0]],[1,[3,0,0]]]}]}]);
        let mut e = c.export(scene, None, "step.glb");
        e.frame_end = Some(24);
        let report = e.run().unwrap();
        assert!(
            !report["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["code"] == "animation_resampling")
        );
        let file = import::open(&e.out).unwrap();
        let data = import::buffers(&file.document, e.out.parent(), file.blob.clone()).unwrap();
        let channel = file.animations().next().unwrap().channels().next().unwrap();
        assert_eq!(
            channel.sampler().interpolation(),
            gltf::animation::Interpolation::Step
        );
        let reader = channel.reader(|b| Some(&data[b.index()]));
        let times: Vec<_> = reader.read_inputs().unwrap().collect();
        let gltf::animation::util::ReadOutputs::Translations(values) =
            reader.read_outputs().unwrap()
        else {
            panic!()
        };
        let values: Vec<_> = values.collect();
        let at = times
            .iter()
            .position(|t| *t == 0.37_f32.next_down())
            .unwrap();
        assert_eq!(values[at - 1], [0., 0., 0.]);
        assert_eq!(values[at], [2., 0., 0.]);
    }
    #[test]
    fn animation_range_conflicts_and_budget_fail_before_publication() {
        for (start, end, at) in [
            (5, 4, None),
            (0, 24, Some(vork_time::Clock::from(0.))),
            (0, 100_001, None),
        ] {
            let c = Case::new();
            let mut e = c.export(triangle(), None, "a.glb");
            e.frame_start = Some(start);
            e.frame_end = Some(end);
            e.at = at;
            assert!(e.run().is_err());
            assert!(!e.out.exists());
        }
    }
    #[test]
    fn animation_refusal_preserves_previous_asset() {
        let c = Case::new();
        let mut scene = moving_skin();
        scene["animation"][0]["tracks"]
            .as_array_mut()
            .unwrap()
            .push(json!({"path":["nodes","mesh","visible"],"keys":[[0,true],[1,false]]}));
        let mut e = c.export(scene, None, "a.glb");
        e.frame_end = Some(24);
        e.allow_losses.push("animation_resampling".into());
        std::fs::write(&e.out, b"previous").unwrap();
        assert!(format!("{:#}", e.run().unwrap_err()).contains("cannot animate"));
        assert_eq!(std::fs::read(e.out).unwrap(), b"previous");
    }
    #[test]
    fn skin_rejects_nonunit_weights_and_extra_influences() {
        for extra in [false, true] {
            let c = Case::new();
            let mut scene = moving_skin();
            if extra {
                scene["geometry"]["g"]["attributes"]["WEIGHTS_1"] = json!("w");
            } else {
                scene["buffers"]["w"]["data"][0] = json!([0.5, 0, 0, 0]);
            }
            let mut e = c.export(scene, None, "a.glb");
            e.frame_end = Some(24);
            e.allow_losses.push("animation_resampling".into());
            assert!(e.run().is_err());
            assert!(!e.out.exists());
        }
    }
    #[test]
    fn analytic_triangle_subtree_preserves_world_placement_and_generated_normals() {
        let c = Case::new();
        let e = c.export(triangle(), Some("mesh"), "a.glb");
        let report = e.run().unwrap();
        assert_eq!(report["counts"]["meshes"], 1);
        let doc = import::open(&e.out).unwrap();
        let data = import::buffers(&doc.document, None, doc.blob.clone()).unwrap();
        let node = doc.nodes().next().unwrap();
        assert_eq!(node.transform().decomposed().0, [2., 3., 0.]);
        let primitive = doc.meshes().next().unwrap().primitives().next().unwrap();
        let reader = primitive.reader(|b| Some(&data[b.index()]));
        assert_eq!(
            reader.read_positions().unwrap().collect::<Vec<_>>(),
            vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
        );
        assert!(reader.read_normals().unwrap().all(|n| n == [0., 0., 1.]));
        assert_eq!(
            reader
                .read_indices()
                .unwrap()
                .into_u32()
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let first = std::fs::read(&e.out).unwrap();
        e.run().unwrap();
        assert_eq!(std::fs::read(&e.out).unwrap(), first);
    }
    #[test]
    fn shared_mesh_and_mirrored_node_keep_geometry_local() {
        let c = Case::new();
        let mut scene = triangle();
        scene["nodes"][0]["children"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":"mirror","type":"mesh","geometry":"g","scale":[-1,2,1]}));
        let e = c.export(scene, None, "a.glb");
        e.run().unwrap();
        let doc = import::open(&e.out).unwrap();
        assert_eq!(doc.meshes().len(), 1);
        assert_eq!(doc.nodes().filter(|n| n.mesh().is_some()).count(), 2);
        let node = doc.nodes().find(|n| n.name() == Some("mirror")).unwrap();
        assert_eq!(node.transform().decomposed().2, [-1., 2., 1.]);
    }
    #[test]
    fn unsupported_shader_never_overwrites_an_existing_asset() {
        let c = Case::new();
        let mut scene = triangle();
        scene["materials"] = json!({"m":{"program":"std/hair"}});
        scene["nodes"][0]["children"][0]["materials"] = json!(["m"]);
        let e = c.export(scene, None, "a.glb");
        std::fs::write(&e.out, b"previous asset").unwrap();
        assert!(e.run().is_err());
        assert_eq!(std::fs::read(&e.out).unwrap(), b"previous asset");
        let report: Value =
            serde_json::from_slice(&std::fs::read(e.out.with_extension("export.json")).unwrap())
                .unwrap();
        assert_eq!(report["status"], "failed");
        assert!(
            report["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["code"] == "material_program")
        );
    }
    #[test]
    fn gltf_sidecar_and_io_failure_categories() {
        let c = Case::new();
        let e = c.export(triangle(), None, "a.gltf");
        e.run().unwrap();
        let doc = import::open(&e.out).unwrap();
        import::buffers(&doc.document, Some(&c.0), None).unwrap();
        let mut e = c.export(triangle(), None, "blocked/a.glb");
        std::fs::write(c.0.join("blocked"), b"file").unwrap();
        let error = e.run().unwrap_err();
        assert_eq!(failure::exit_code(&error), 5);
        e.out = c.0.join("a.obj");
        assert_eq!(failure::exit_code(&e.run().unwrap_err()), 2);
    }
    #[test]
    fn emissive_hdr_and_core_factors_survive() {
        let c = Case::new();
        let mut scene = triangle();
        scene["materials"] = json!({"m":{"inputs":{"base_color":[0.2,0.4,0.6],"base_metalness":0.7,"specular_roughness":0.8,"emission_color":[1,0.5,0.25],"emission_luminance":10}}});
        scene["nodes"][0]["children"][0]["materials"] = json!(["m"]);
        let e = c.export(scene, None, "a.glb");
        e.run().unwrap();
        let doc = import::open(&e.out).unwrap();
        let m = doc.materials().next().unwrap();
        assert_eq!(m.emissive_strength(), Some(10.));
        assert_eq!(m.emissive_factor(), [1., 0.5, 0.25]);
        assert_eq!(m.pbr_metallic_roughness().metallic_factor(), 0.7);
    }
    #[test]
    fn samples_node_and_material_tracks_at_fractional_time() {
        let c = Case::new();
        let mut scene = triangle();
        scene["materials"] =
            json!({"m":{"inputs":{"base_color":[0.2,0.4,0.6],"specular_roughness":0.2}}});
        scene["nodes"][0]["children"][0]["materials"] = json!(["m"]);
        scene["animation"] = json!([{"tracks":[
            {"path":["nodes","mesh","translation"],"keys":[[0,[0,3,0]],[2,[0,7,0]]]},
            {"path":["materials","m","inputs","specular_roughness"],"keys":[[0,0.2],[2,0.8]]}
        ]}]);
        let mut e = c.export(scene, Some("mesh"), "a.glb");
        e.at = Some(vork_time::Clock::from(0.5));
        e.run().unwrap();
        let doc = import::open(&e.out).unwrap();
        assert_eq!(
            doc.nodes().next().unwrap().transform().decomposed().0,
            [2., 4., 0.]
        );
        assert!(
            (doc.materials()
                .next()
                .unwrap()
                .pbr_metallic_roughness()
                .roughness_factor()
                - 0.35)
                .abs()
                < 1e-6
        );
        assert_eq!(doc.animations().len(), 0);
    }
    #[test]
    fn embeds_srgb_png_with_sampler_and_uvs() {
        let c = Case::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([128, 64, 32]))
            .save(c.0.join("color.png"))
            .unwrap();
        let mut scene = triangle();
        scene["buffers"]["uv"] = json!({"type":"vec2","data":[[0,0],[1,0],[0,1]]});
        scene["geometry"]["g"]["attributes"]["TEXCOORD_0"] = json!("uv");
        scene["images"] =
            json!({"color":{"file":"color.png","wrap_s":"clamp_to_edge","mag_filter":"nearest"}});
        scene["materials"] = json!({"m":{"inputs":{"base_color":{"image":"color"}}}});
        scene["nodes"][0]["children"][0]["materials"] = json!(["m"]);
        let e = c.export(scene, Some("mesh"), "a.glb");
        let report = e.run().unwrap();
        assert_eq!(report["counts"]["images"], 1);
        let doc = import::open(&e.out).unwrap();
        assert!(doc.images().all(|i| matches!(
            i.source(),
            gltf::image::Source::View {
                mime_type: "image/png",
                ..
            }
        )));
        let info = doc
            .materials()
            .next()
            .unwrap()
            .pbr_metallic_roughness()
            .base_color_texture()
            .unwrap();
        assert_eq!(
            info.texture().sampler().wrap_s(),
            gltf::texture::WrappingMode::ClampToEdge
        );
    }
    #[test]
    fn punctual_light_units_and_spot_cones_invert_the_importer() {
        let c = Case::new();
        let mut scene = triangle();
        scene["nodes"].as_array_mut().unwrap().extend([
            json!({"type":"light","name":"sun","shape":"sun","angle":0,"energy":2}),
            json!({"type":"light","name":"point","shape":"point","energy":4}),
            json!({"type":"light","name":"spot","shape":"spot","spot_size":1.2,"spot_blend":0.4,"energy":4})]);
        let e = c.export(scene, None, "a.glb");
        e.run().unwrap();
        let doc = import::open(&e.out).unwrap();
        let lights = doc.lights().unwrap().collect::<Vec<_>>();
        assert_eq!(lights[0].intensity(), 2. * constants::LUMENS_PER_WATT);
        assert!(
            (lights[1].intensity() - constants::LUMENS_PER_WATT / std::f32::consts::PI).abs()
                < 1e-4
        );
        match lights[2].kind() {
            gltf::khr_lights_punctual::Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => {
                assert!((inner_cone_angle - vork_scene::spot_inner_angle(1.2, 0.4)).abs() < 1e-6);
                assert!((outer_cone_angle - 0.6).abs() < 1e-6);
            }
            _ => panic!("spot lost"),
        }
    }
    #[test]
    fn export_validation_is_camera_independent_without_weakening_render_validation() {
        let bytes = serde_json::to_vec(&triangle()).unwrap();
        assert!(
            Scene::from_json(&bytes)
                .unwrap_err()
                .to_string()
                .contains("camera")
        );
        assert!(Scene::from_json_for_export(&bytes, &bytes, None).is_ok());
    }
    #[test]
    fn subtree_cannot_drop_external_ancestor_visibility_or_matte_flags() {
        for (field, value) in [("visible", json!(false)), ("is_holdout", json!(true))] {
            let c = Case::new();
            let mut scene = triangle();
            scene["nodes"][0][field] = value;
            let e = c.export(scene, Some("mesh"), "a.glb");
            assert!(e.run().is_err());
            assert!(!e.out.exists());
            let report: Value = serde_json::from_slice(
                &std::fs::read(e.out.with_extension("export.json")).unwrap(),
            )
            .unwrap();
            assert!(
                report["issues"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|i| i["code"] == "visibility")
            );
        }
    }
    #[test]
    fn ies_distribution_is_never_silently_exported_as_a_plain_point_light() {
        let c = Case::new();
        let mut scene = triangle();
        scene["images"] = json!({"profile":{"file":"profile.ies"}});
        scene["nodes"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"light","name":"ies","shape":"point","image":"profile"}));
        let e = c.export(scene, None, "a.glb");
        assert!(e.run().is_err());
        assert!(!e.out.exists());
        let report: Value =
            serde_json::from_slice(&std::fs::read(e.out.with_extension("export.json")).unwrap())
                .unwrap();
        assert!(
            report["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["code"] == "light_profile")
        );
    }
    #[test]
    fn cubic_material_overshoot_cannot_publish_an_invalid_gltf_factor() {
        let c = Case::new();
        let mut scene = triangle();
        scene["materials"] = json!({"m":{"inputs":{"geometry_opacity":0.5}}});
        scene["nodes"][0]["children"][0]["materials"] = json!(["m"]);
        scene["animation"] = json!([{"tracks":[{"path":["materials","m","inputs","geometry_opacity"],"interpolation":"CUBICSPLINE","keys":[[0,0.5,0,4],[1,0.5,-4,0]]}]}]);
        let mut e = c.export(scene, Some("mesh"), "a.glb");
        e.at = Some(vork_time::Clock::from(0.5));
        let error = e.run().unwrap_err();
        assert!(error.to_string().contains("geometry_opacity"), "{error:#}");
        assert!(!e.out.exists());
    }
    #[test]
    fn invalid_tangent_frames_and_equal_spot_cones_are_refused() {
        let c = Case::new();
        let mut scene = triangle();
        scene["buffers"]["t"] = json!({"type":"vec4","data":[[0,0,1,1],[0,0,1,1],[0,0,1,1]]});
        scene["geometry"]["g"]["attributes"]["TANGENT"] = json!("t");
        let e = c.export(scene, Some("mesh"), "tangent.glb");
        assert!(e.run().unwrap_err().to_string().contains("orthogonal"));
        assert!(!e.out.exists());
        let mut scene = triangle();
        scene["nodes"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"light","shape":"spot","spot_size":2,"spot_blend":0}));
        let e = c.export(scene, None, "spot.glb");
        assert!(e.run().unwrap_err().to_string().contains("spot cone"));
        assert!(!e.out.exists());
    }
}
