use crate::{EvalNode, animation, buffers, import, lights, programs, targets, values};
use anyhow::{Context, Result, ensure};
use glam::Mat4;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
};
use vork_scene::{Node, NodeKind, Scene, v2};

/// The renderer's CPU inputs, with no material installation or device resources.
pub(super) struct CpuScene {
    pub scene: Scene,
    pub base: PathBuf,
    pub buffers: buffers::SceneBuffers,
    pub nodes: Vec<EvalNode>,
    pub assets: Vec<animation::Asset>,
    pub imports: Vec<Imported>,
    pub animation: animation::Plan,
    targets: targets::Plan,
    material_keys: Vec<(String, String)>,
    pub cameras: RefCell<BTreeMap<String, Arc<vork_slang::CameraProgram>>>,
    input_defaults: RefCell<BTreeMap<String, BTreeMap<String, Option<Value>>>>,
}

pub(super) struct Imported {
    pub root: usize,
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
}

pub(super) struct Pose {
    pub nodes: Vec<EvalNode>,
    pub worlds: Vec<Mat4>,
    pub samples: animation::Samples,
    pub materials: v2::Table<v2::Material>,
    pub render: v2::Render,
}

fn flatten(input: &[Node], parent: Option<usize>, out: &mut Vec<EvalNode>) {
    for source in input {
        let index = out.len();
        let world = parent.map_or(Mat4::IDENTITY, |p| out[p].world) * source.transform.matrix();
        let mut node = source.without_children();
        if let NodeKind::Light {
            shape,
            energy,
            color,
            image,
            material,
            ..
        } = &mut node.kind
        {
            energy.get_or_insert(shape.default_energy());
            color.get_or_insert(shape.default_color(image.is_some() || material.is_some()));
        }
        out.push(EvalNode {
            node,
            parent,
            world,
            dynamic: true,
            camera: None,
            embedded: None,
        });
        flatten(&source.children, Some(index), out);
    }
}

impl CpuScene {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("read scene {}", path.display()))?;
        let document = v2::diagnostics::read_document(&bytes)?;
        Self::document(
            document,
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )
    }

    pub fn document(mut document: Value, base: &Path) -> Result<Self> {
        let authored = serde_json::to_vec(&document)?;
        vork_std::expand_scene(&mut document, base)?;
        let scene = Scene::from_json_for_export(&serde_json::to_vec(&document)?, &authored, None)?;
        Self::new(scene, base)
    }

    pub fn new(mut scene: Scene, base: &Path) -> Result<Self> {
        ensure!(!scene.is_canvas(), "ask: answers about a 2D canvas are not built yet");
        let buffers = buffers::SceneBuffers::decode(&scene, base)?;
        let mut nodes = Vec::new();
        flatten(&scene.nodes, None, &mut nodes);
        let mut animation =
            animation::Plan::compile_with_buffers(&scene, base, Some(&buffers.data))?;
        let mut assets = Vec::new();
        let mut imports = Vec::new();
        let mut cache = animation::SourceCache::default();
        let authored_nodes = nodes.len();
        for root in 0..authored_nodes {
            let NodeKind::Asset {
                file,
                nodes: overrides,
                ..
            } = &nodes[root].node.kind
            else {
                continue;
            };
            let path = base.join(file);
            if crate::splats::is_file(&path) {
                ensure!(overrides.is_empty(), "Gaussian splat assets have no imported nodes; transform the asset node instead");
                crate::splats::inspect(&path)?;
                continue;
            }
            let gltf = import::open(&path)?;
            let payload = import::buffers(&gltf.document, path.parent(), gltf.blob)?;
            let clips: Vec<_> = animation.clips_for_asset(&nodes[root].node.name).collect();
            let asset =
                animation::Asset::cached(&gltf.document, &payload, &clips, base, file, &mut cache)?;
            let asset = asset.register_native(
                &mut animation,
                &gltf.document,
                &nodes[root].node.name,
                assets.len(),
                overrides,
            )?;
            // Share the renderer's imported camera and light conversion, including its names and optics.
            let mut embedded = Vec::new();
            lights::import(
                &gltf.document,
                root,
                assets.len(),
                nodes[root].world,
                &nodes[root].node.name,
                overrides,
                &mut embedded,
                &mut scene.materials,
            )?;
            let mut special: HashMap<_, _> = embedded
                .into_iter()
                .map(|n| (n.embedded.unwrap().1, n))
                .collect();
            let prefix = if nodes[root].node.name.is_empty() {
                format!("#{root}")
            } else {
                nodes[root].node.name.clone()
            };
            let mut mapping = HashMap::new();
            fn visit(
                node: gltf::Node<'_>,
                parent: usize,
                root: usize,
                asset: usize,
                prefix: &str,
                special: &mut HashMap<usize, EvalNode>,
                mapping: &mut HashMap<usize, usize>,
                out: &mut Vec<EvalNode>,
            ) -> Result<()> {
                let index = out.len();
                let name = format!(
                    "{prefix}/{}",
                    node.name()
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("#{}", node.index()))
                );
                let (translation, rotation, scale) = node.transform().decomposed();
                let mut evaluated = if let Some(n) = special.remove(&node.index()) {
                    n
                } else {
                    EvalNode {
                        node: serde_json::from_value(json!({"name":name}))?,
                        parent: Some(parent),
                        world: Mat4::IDENTITY,
                        dynamic: true,
                        camera: None,
                        embedded: Some((asset, node.index(), root, None)),
                    }
                };
                evaluated.node.transform = vork_scene::Transform {
                    translation,
                    rotation: Some(rotation),
                    scale,
                };
                evaluated.parent = Some(parent);
                evaluated.dynamic = true;
                out.push(evaluated);
                mapping.insert(node.index(), index);
                for child in node.children() {
                    visit(child, index, root, asset, prefix, special, mapping, out)?;
                }
                Ok(())
            }
            let selected = gltf
                .document
                .default_scene()
                .or_else(|| gltf.document.scenes().next())
                .context("glTF has no scene")?;
            for node in selected.nodes() {
                visit(
                    node,
                    root,
                    root,
                    assets.len(),
                    &prefix,
                    &mut special,
                    &mut mapping,
                    &mut nodes,
                )?;
            }
            assets.push(asset);
            imports.push(Imported {
                root,
                document: gltf.document,
                buffers: payload,
            });
        }
        animation.finish_asset_bindings()?;
        let material_keys = animation.material_inputs();
        let mut layouts = BTreeMap::new();
        for (slot, (name, input)) in material_keys.iter().enumerate() {
            let material = &scene.materials[name];
            let mut storage = None;
            let mut enum_tokens = None;
            let default = if material.program == "std/openpbr" {
                let p = v2::OpenPbrParam::find(input).context("unknown OpenPBR input")?;
                match material.inputs.get(input) {
                    Some(v2::Input::Number(v)) => animation::Sample::Numeric(vec![*v]),
                    Some(v2::Input::Vector(v)) => animation::Sample::Numeric(v.clone()),
                    Some(v2::Input::Attribute(binding)) => {
                        animation::Sample::Numeric(match &binding.factor {
                            Some(v2::Factor::Number(v)) => vec![*v],
                            Some(v2::Factor::Vector(v)) => v.clone(),
                            None => vec![
                                1.;
                                if matches!(
                                    p.kind,
                                    v2::OpenPbrKind::Color3 | v2::OpenPbrKind::Vector3
                                ) {
                                    3
                                } else {
                                    1
                                }
                            ],
                        })
                    }
                    None => animation::Sample::Numeric(if p.kind == v2::OpenPbrKind::Vector3 {
                        vec![0.; 3]
                    } else {
                        p.default.to_vec()
                    }),
                    _ => anyhow::bail!(
                        "ask: animated material binding {name}.{input} is not built yet"
                    ),
                }
            } else {
                if !layouts.contains_key(&material.program) {
                    let (source, origin) =
                        programs::material_source(&scene, base, &material.program)?;
                    let layout = vork_slang::Compiler::discover().reflect_inputs(
                        &programs::abi(),
                        &source,
                        &origin,
                    )?;
                    layouts.insert(material.program.clone(), layout);
                }
                let layout = &layouts[&material.program];
                let field = layout
                    .fields
                    .iter()
                    .find(|f| f.name == *input)
                    .context("unknown material input")?;
                let supplied = material
                    .inputs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
                    .collect::<Result<BTreeMap<_, _>>>()?;
                let words = layout.pack(&supplied, |_, _| Ok(0))?;
                storage = Some(field.ty);
                enum_tokens = field.enumeration.as_ref().map(|e| {
                    e.cases
                        .iter()
                        .map(|s| s.to_ascii_lowercase())
                        .collect::<Vec<_>>()
                });
                if let Some(tokens) = &enum_tokens {
                    animation::Sample::String(tokens[words[field.offset] as usize].clone().into())
                } else if field.ty == vork_slang::Type::Bool {
                    animation::Sample::Bool(words[field.offset] != 0)
                } else {
                    animation::Sample::Numeric(
                        words[field.offset..field.offset + field.ty.words()]
                            .iter()
                            .map(|&w| match field.ty {
                                vork_slang::Type::Int => w as i32 as f32,
                                vork_slang::Type::Uint => w as f32,
                                _ => f32::from_bits(w),
                            })
                            .collect(),
                    )
                }
            };
            animation.bind_material(
                &(name.clone(), input.clone()),
                vec![animation::MaterialBinding {
                    slot,
                    parameter: slot,
                    storage,
                    enum_tokens,
                    default,
                    offset: 0.,
                }],
            )?;
        }
        animation.mark_nodes(&mut nodes, 0)?;
        let targets = targets::Plan::compile(&mut nodes)?;
        Ok(Self {
            scene,
            base: base.to_owned(),
            buffers,
            nodes,
            assets,
            imports,
            animation,
            targets,
            material_keys,
            cameras: Default::default(),
            input_defaults: Default::default(),
        })
    }

    pub fn input_defaults(&self, program: &str) -> Result<BTreeMap<String, Option<Value>>> {
        let mut cache = self.input_defaults.borrow_mut();
        if !cache.contains_key(program) {
            let (source, origin) = programs::material_source(&self.scene, &self.base, program)?;
            let values = vork_slang::Compiler::discover().input_defaults(
                &programs::abi(),
                &source,
                &origin,
            )?;
            cache.insert(program.to_owned(), values);
        }
        Ok(cache[program].clone())
    }

    pub fn sample(&self, request: &vork_time::Request) -> Result<Pose> {
        let mut samples = self.animation.samples();
        let mut values = values::sample_storage(&self.nodes);
        let mut worlds = self.nodes.iter().map(|n| n.world).collect::<Vec<_>>();
        values::sample_pose_inputs(
            &self.nodes,
            &self.assets,
            &self.animation,
            &self.targets,
            request.clone(),
            &mut samples,
            &mut values,
            &mut worlds,
        )?;
        self.animation
            .apply_render(&mut samples, &self.scene.render);
        let mut render = samples
            .render
            .clone()
            .unwrap_or_else(|| self.scene.render.clone());
        render.camera = self.animation.start_camera(&render, request.clone());
        let mut nodes: Vec<_> = values
            .into_iter()
            .zip(&self.nodes)
            .map(|(n, original)| n.unwrap_or_else(|| original.clone()))
            .collect();
        for (asset, source) in self.assets.iter().enumerate() {
            let transforms = source.local_transforms_with_plan(&samples, &self.animation);
            for node in &mut nodes {
                if let Some((owner, local, _, _)) = node.embedded {
                    if owner == asset {
                        node.node.transform = transforms[local].clone();
                    }
                }
            }
        }
        let mut materials = self.scene.materials.clone();
        for (binding, sample) in self.animation.material_values(&samples, 0.) {
            let (name, input) = &self.material_keys[binding.parameter];
            let value = match sample {
                animation::Sample::Numeric(v) => {
                    let v = v
                        .iter()
                        .map(|&v| match binding.storage {
                            Some(vork_slang::Type::Int) => v as i32 as f32,
                            Some(vork_slang::Type::Uint) => v as u32 as f32,
                            _ => v,
                        })
                        .collect::<Vec<_>>();
                    if v.len() == 1 {
                        v2::Input::Number(v[0])
                    } else {
                        v2::Input::Vector(v)
                    }
                }
                animation::Sample::Bool(v) => v2::Input::Bool(*v),
                animation::Sample::Style(_) => unreachable!("CSS/SVG channels do not bind material inputs"),
                animation::Sample::String(v) => serde_json::from_value(json!(v.as_ref()))?,
            };
            let inputs = &mut materials.get_mut(name).unwrap().inputs;
            if let Some(v2::Input::Attribute(binding)) = inputs.get_mut(input) {
                binding.factor = match value {
                    v2::Input::Number(value) => Some(v2::Factor::Number(value)),
                    v2::Input::Vector(value) => Some(v2::Factor::Vector(value)),
                    _ => anyhow::bail!("ask: an attribute factor must be numeric"),
                };
            } else {
                inputs.insert(input.clone(), value);
            }
        }
        ensure!(
            worlds.iter().all(|m| m.is_finite()),
            "ask: sampled transform is not finite; correct the node's animation"
        );
        Ok(Pose {
            nodes,
            worlds,
            samples,
            materials,
            render,
        })
    }

    pub fn node(&self, name: &str) -> Result<usize> {
        let found: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.node.name == name)
            .map(|(i, _)| i)
            .collect();
        ensure!(
            found.len() == 1,
            "ask: node {name:?} must name exactly one node"
        );
        Ok(found[0])
    }

    pub fn sample_offset(&self, request: &vork_time::Request, offset: f32) -> Result<Pose> {
        let mut samples = self.animation.samples();
        let time = request.numeric()?;
        self.animation.sample_nodes_request(
            request,
            time,
            offset,
            &mut samples,
            &(0..self.nodes.len()).collect::<Vec<_>>(),
            &(0..self.assets.len()).collect::<Vec<_>>(),
        )?;
        let mut values = values::sample_storage(&self.nodes);
        let mut worlds = self.nodes.iter().map(|n| n.world).collect::<Vec<_>>();
        values::apply_pose_samples(
            &self.nodes,
            &self.assets,
            &self.animation,
            &self.targets,
            time + offset,
            &samples,
            &mut values,
            &mut worlds,
        )?;
        let nodes = values
            .into_iter()
            .zip(&self.nodes)
            .map(|(n, original)| n.unwrap_or_else(|| original.clone()))
            .collect();
        Ok(Pose {
            nodes,
            worlds,
            samples,
            materials: self.scene.materials.clone(),
            render: self.scene.render.clone(),
        })
    }

    pub fn under(&self, mut node: usize, root: usize) -> bool {
        loop {
            if node == root {
                return true;
            }
            match self.nodes[node].parent {
                Some(parent) => node = parent,
                None => return false,
            }
        }
    }
}
