//! Immutable glTF tracks; only poses and morph weights are sampled per frame.
use super::*;
#[cfg(test)]
use glam::Quat;
use std::sync::Arc;
mod tracks;
pub(crate) mod style;
use tracks::Combine;
use tracks::{Curve, Interpolation};
pub(crate) use tracks::{blend, fold_material_inputs,ClockNode,FootageClock,MaterialBinding, Plan, Sample, Samples, SoundPlan};
pub(crate) use tracks::Combine as BlendKind;

type BindNodes = Vec<(Option<usize>, vork_scene::Transform, Vec<f32>)>;
type SkinBindings = Vec<Vec<(usize, Mat4)>>;
type ClipFile = (gltf::Document, Vec<gltf::buffer::Data>);
#[derive(Default)]
pub(super) struct SourceCache {
    binds: std::collections::HashMap<String, (Arc<BindNodes>, Arc<SkinBindings>)>,
    clips: std::collections::HashMap<(String, Option<PathBuf>, usize), Arc<Vec<Track>>>,
    files: std::collections::HashMap<PathBuf, Arc<ClipFile>>,
}

#[derive(Clone)]
pub(super) struct Asset {
    layers: Arc<Vec<Layer>>,
    native: Option<(Arc<Plan>, usize)>,
    native_asset: Option<usize>,
    pub active: bool,
    /// Immutable install-time pose for assets without animation.
    bind_pose: Option<Arc<Pose>>,
    nodes: Arc<BindNodes>,
    visibility: Vec<vork_scene::Visibility>,
    skins: Arc<SkinBindings>,
}
/// One clip layer of the scene plan, by plan layer index, and its decoded channels.
struct Layer {
    layer: usize,
    tracks: Arc<Vec<Track>>,
}
struct Track {
    node: usize,
    property: gltf::animation::Property,
    curve: Arc<Curve>,
}
pub(super) struct Pose {
    pub collapsed: Vec<bool>,
    pub worlds: Vec<Mat4>,
    pub weights: Vec<Vec<f32>>,
    pub visibility: Vec<u32>,
}
impl Asset {
    /// Bind and clip data for one asset node; `clips` are the plan's clip layers
    /// that target it, by layer index. Templates share decoded bindings and tracks.
    pub fn cached(
        doc: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        clips: &[(usize, &vork_scene::Clip)],
        base: &Path,
        source_key: &str,
        cache: &mut SourceCache,
    ) -> Result<Self> {
        let nodes: Arc<BindNodes> = if let Some((nodes, _)) = cache.binds.get(source_key) {
            Arc::clone(nodes)
        } else {
            let mut nodes: Vec<_> = doc
                .nodes()
                .map(|n| {
                    let (translation, rotation, scale) = n.transform().decomposed();
                    (
                        None,
                        vork_scene::Transform {
                            translation,
                            rotation: Some(rotation),
                            scale,
                        },
                        n.weights()
                            .or_else(|| n.mesh().and_then(|m| m.weights()))
                            .unwrap_or(&[])
                            .to_vec(),
                    )
                })
                .collect();
            for n in doc.nodes() {
                for c in n.children() {
                    nodes[c.index()].0 = Some(n.index());
                }
            }
            Arc::new(nodes)
        };
        let mut layers = Vec::new();
        for &(index, clip) in clips {
            let external_path = clip.file.as_ref().map(|p| base.join(p));
            let external = external_path
                .as_ref()
                .map(|path| {
                    if let Some(file) = cache.files.get(path) {
                        return Ok(Arc::clone(file));
                    }
                    // Parse an external clip file once, even when its clips/playbacks differ.
                    let gltf = crate::import::open(path)
                        .with_context(|| format!("load clip {}", path.display()))?;
                    let buffers = crate::import::buffers(&gltf.document, path.parent(), gltf.blob)?;
                    let file = Arc::new((gltf.document, buffers));
                    cache.files.insert(path.clone(), Arc::clone(&file));
                    Ok::<_, anyhow::Error>(file)
                })
                .transpose()?;
            let (source_doc, buffers) = external
                .as_deref()
                .map_or((doc, buffers), |(d, b)| (d, b.as_slice()));
            let count = source_doc.animations().len();
            let selected = if let Some(name) = &clip.name {
                ensure!(count > 0, "clip.name {name:?}: asset has no clips");
                let matches = source_doc
                    .animations()
                    .filter(|a| a.name() == Some(name))
                    .count();
                ensure!(
                    matches > 0,
                    "{}",
                    ::vork_scene::v2::diagnostics::unknown(
                        "animation clip",
                        name,
                        source_doc.animations().filter_map(|a| a.name())
                    )
                );
                ensure!(
                    matches == 1,
                    "animation clip name {name:?} is ambiguous ({matches} matches); select clip.index"
                );
                source_doc.animations().find(|a| a.name() == Some(name))
            } else {
                let index = clip.index.unwrap_or(0) as usize;
                ensure!(count > 0, "clip.index {index}: asset has no clips");
                ensure!(
                    index < count,
                    "clip.index {index} does not exist; asset has {count} clips (indices 0..{count})"
                );
                source_doc.animations().nth(index)
            };
            let clip_source = clip;
            let clip = selected.expect("clip selection was checked");
            let clip_key = (source_key.to_owned(), external_path, clip.index());
            if let Some(tracks) = cache.clips.get(&clip_key) {
                layers.push(Layer {
                    layer: index,
                    tracks: Arc::clone(tracks),
                });
                continue;
            }
            let mut tracks = Vec::new();
            for c in clip.channels() {
                let target = if clip_source.file.is_some() {
                    let source = c.target().node();
                    let name = source
                        .name()
                        .context("external animation target needs a node name")?;
                    ensure!(
                        source_doc
                            .nodes()
                            .filter(|n| n.name() == Some(name))
                            .count()
                            == 1,
                        "ambiguous external animation node {name}"
                    );
                    let matches: Vec<_> = doc.nodes()
                        .filter(|node| node.name() == Some(name))
                        .map(|node| node.index())
                        .collect();
                    ensure!(
                        matches.len() == 1,
                        "external animation target {name} must match exactly one named node"
                    );
                    matches[0]
                } else {
                    c.target().node().index()
                };
                let reader = c.reader(|b| Some(&buffers[b.index()]));
                let times: Vec<_> = reader.read_inputs().context("animation inputs")?.collect();
                let property = c.target().property();
                let values: Vec<f32> = match reader.read_outputs().context("animation outputs")? {
                    gltf::animation::util::ReadOutputs::Translations(v) => v.flatten().collect(),
                    gltf::animation::util::ReadOutputs::Rotations(v) => {
                        v.into_f32().flatten().collect()
                    }
                    gltf::animation::util::ReadOutputs::Scales(v) => v.flatten().collect(),
                    gltf::animation::util::ReadOutputs::MorphTargetWeights(v) => {
                        v.into_f32().collect()
                    }
                };
                let interpolation = c.sampler().interpolation();
                ensure!(!times.is_empty(), "empty animation track");
                let width = values.len()
                    / times.len()
                    / if interpolation == gltf::animation::Interpolation::CubicSpline {
                        3
                    } else {
                        1
                    };
                let expected = match property {
                    gltf::animation::Property::Translation | gltf::animation::Property::Scale => 3,
                    gltf::animation::Property::Rotation => 4,
                    gltf::animation::Property::MorphTargetWeights => {
                        let mesh = doc.nodes().nth(target).unwrap().mesh()
                            .context("weights animation requires mesh")?;
                        let expected = mesh.primitives().next()
                            .context("weights animation requires primitive")?.morph_targets().len();
                        ensure!(mesh.primitives().all(|p| p.morph_targets().len() == expected),
                            "animation target mesh primitives have inconsistent morph counts");
                        expected
                    },
                };
                ensure!(
                    width == expected
                        && width > 0
                        && values.len()
                            == width
                                * times.len()
                                * if interpolation == gltf::animation::Interpolation::CubicSpline {
                                    3
                                } else {
                                    1
                                },
                    "animation output shape mismatch"
                );
                ensure!(
                    times.iter().all(|t| t.is_finite() && *t >= 0.)
                        && times.windows(2).all(|t| t[0] < t[1])
                        && values.iter().all(|v| v.is_finite()),
                    "invalid animation keys"
                );
                tracks.push(Track {
                    node: target,
                    property,
                    curve: Arc::new(Curve {
                        times,
                        values,
                        width,
                        rotation: property == gltf::animation::Property::Rotation,
                        euler: false,
                        interpolation: match interpolation {
                            gltf::animation::Interpolation::Step => Interpolation::Step,
                            gltf::animation::Interpolation::Linear => Interpolation::Linear,
                            gltf::animation::Interpolation::CubicSpline => Interpolation::Cubic,
                        },
                    }),
                });
            }
            let tracks = Arc::new(tracks);
            cache.clips.insert(clip_key, Arc::clone(&tracks));
            layers.push(Layer {
                layer: index,
                tracks,
            });
        }
        let skins: Arc<SkinBindings> = if let Some((_, skins)) = cache.binds.get(source_key) {
            Arc::clone(skins)
        } else {
            Arc::new(
                doc.skins()
                    .map(|s| {
                        let inverse: Vec<_> = s
                            .reader(|b| Some(&buffers[b.index()]))
                            .read_inverse_bind_matrices()
                            .map(|v| v.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                            .unwrap_or_else(|| vec![Mat4::IDENTITY; s.joints().len()]);
                        s.joints()
                            .zip(inverse)
                            .map(|(j, m)| (j.index(), m))
                            .collect()
                    })
                    .collect(),
            )
        };
        cache
            .binds
            .entry(source_key.to_owned())
            .or_insert_with(|| (Arc::clone(&nodes), Arc::clone(&skins)));
        let mut asset = Self {
            layers: Arc::new(layers),
            native: None,
            native_asset: None,
            active: false,
            bind_pose: None,
            visibility: vec![Default::default(); nodes.len()],
            nodes,
            skins,
        };
        asset.bind_pose = Some(Arc::new(asset.evaluate_sampled(None, None)));
        Ok(asset)
    }
    /// Registers this asset in the one scene plan; freeze/attach it after all imports.
    pub fn register_native(
        mut self,
        plan: &mut Plan,
        doc: &gltf::Document,
        root: &str,
        asset: usize,
        overrides: &vork_scene::Table<vork_scene::AssetNodeOverrides>,
    ) -> Result<Self> {
        self.visibility = doc.nodes().map(|node| crate::buffers::imported_override(overrides, node.index(), node.name().unwrap_or(""))
            .and_then(|value| value.visibility.clone()).unwrap_or_default()).collect();
        self.bind_pose = Some(Arc::new(self.evaluate_sampled(None, None)));
        let mut active = plan.bind_asset(doc, root, asset)?;
        for imported in self.layers.iter() {
            for track in imported.tracks.iter() {
                plan.add_clip_curve(
                    imported.layer,
                    Arc::clone(&track.curve),
                    tracks::AssetTarget {
                        asset,
                        node: track.node,
                        property: track.property,
                        component: None,
                    },
                );
                active = true;
            }
        }
        if active {
            self.native_asset = Some(asset);
        }
        Ok(self)
    }
    pub fn attach_plan(&mut self, plan: Arc<Plan>) {
        if let Some(asset) = self.native_asset {
            self.native = Some((plan, asset));
        }
    }
    pub fn animated(&self) -> bool {
        self.native_asset.is_some() || !self.layers.is_empty()
    }
    #[cfg(test)]
    pub fn pose(&self, time: f32) -> Arc<Pose> {
        if let Some((plan, _)) = &self.native {
            let mut samples = plan.samples();
            plan.sample_into(time, &mut samples);
            self.pose_sampled(&samples)
        } else {
            Arc::new(self.evaluate_sampled(None, None))
        }
    }
    pub fn pose_sampled(&self, samples: &Samples) -> Arc<Pose> {
        if !self.animated() {
            return Arc::clone(self.bind_pose.as_ref().expect("installed bind pose"));
        }
        Arc::new(self.evaluate_sampled(Some(samples), self.native.as_ref().map(|(plan, _)| plan.as_ref())))
    }
    pub(crate) fn pose_with_plan(&self, samples: &Samples, plan: &Plan) -> Arc<Pose> {
        if !self.animated() { return Arc::clone(self.bind_pose.as_ref().expect("installed bind pose")); }
        Arc::new(self.evaluate_sampled(Some(samples), Some(plan)))
    }
    pub(crate) fn local_transforms_with_plan(&self, samples: &Samples, plan: &Plan) -> Vec<vork_scene::Transform> {
        self.sample_local_inputs(Some(samples), Some(plan)).0
    }
    fn sample_local_inputs(&self, sampled: Option<&Samples>, plan: Option<&Plan>)
        -> (Vec<vork_scene::Transform>, Vec<Vec<f32>>, Vec<vork_scene::Visibility>) {
        let mut transforms: Vec<_> = self.nodes.iter().map(|n| n.1.clone()).collect();
        let mut weights: Vec<_> = self.nodes.iter().map(|n| n.2.clone()).collect();
        let mut visibility = self.visibility.clone();
        if let Some((plan, asset)) = plan.zip(self.native_asset) {
            let mut apply =
                |target: tracks::AssetTarget, sample: &tracks::Sample, weight, additive| {
                    let tracks::Sample::Numeric(sample) = sample else {
                        return;
                    };
                    use gltf::animation::Property;
                    let node = target.node;
                    match target.property {
                        Property::Translation => tracks::blend(
                            &mut transforms[node].translation,
                            sample,
                            weight,
                            additive,
                            Combine::Add,
                        ),
                        Property::Scale => tracks::blend(
                            &mut transforms[node].scale,
                            sample,
                            weight,
                            additive,
                            Combine::Multiply,
                        ),
                        Property::Rotation => tracks::blend(
                            transforms[node].rotation_mut(),
                            sample,
                            weight,
                            additive,
                            Combine::Rotation,
                        ),
                        Property::MorphTargetWeights => {
                            if let Some(component) = target.component {
                                let length = weights[node].len().max(component + 1);
                                weights[node].resize(length, 0.);
                                tracks::blend(
                                    &mut weights[node][component..component + 1],
                                    sample,
                                    weight,
                                    additive,
                                    Combine::Add,
                                );
                            } else {
                                weights[node].resize(sample.len(), 0.);
                                tracks::blend(
                                    &mut weights[node],
                                    sample,
                                    weight,
                                    additive,
                                    Combine::Add,
                                );
                            }
                        }
                    }
                };
            if let Some(samples) = sampled {
                plan.apply_asset(asset, samples, &mut apply);
                plan.apply_asset_visibility(asset, samples, &mut visibility);
            }
        }
        (transforms, weights, visibility)
    }
    fn evaluate_sampled(&self, sampled: Option<&Samples>, plan: Option<&Plan>) -> Pose {
        let (transforms, weights, visibility) = self.sample_local_inputs(sampled, plan);
        fn world(
            i: usize,
            nodes: &[(Option<usize>, vork_scene::Transform, Vec<f32>)],
            local: &[vork_scene::Transform],
            out: &mut [Option<Mat4>],
        ) -> Mat4 {
            if let Some(m) = out[i] {
                return m;
            }
            let m = nodes[i]
                .0
                .map_or(Mat4::IDENTITY, |p| world(p, nodes, local, out))
                * local[i].matrix();
            out[i] = Some(m);
            m
        }
        let mut worlds = vec![None; self.nodes.len()];
        for i in 0..worlds.len() {
            world(i, &self.nodes, &transforms, &mut worlds);
        }
        fn mask(i: usize, nodes: &BindNodes, local: &[vork_scene::Visibility], out: &mut [Option<u32>]) -> u32 {
            if let Some(mask) = out[i] { return mask; }
            let value = crate::lights::object_visibility_mask(&local[i])
                & nodes[i].0.map_or(constants::RAY_VISIBILITY_ALL, |p| mask(p,nodes,local,out));
            out[i] = Some(value);
            value
        }
        let mut masks = vec![None; self.nodes.len()];
        for i in 0..masks.len() { mask(i,&self.nodes,&visibility,&mut masks); }
        fn collapsed(i: usize, nodes: &BindNodes, local: &[vork_scene::Transform], out: &mut [Option<bool>]) -> bool {
            if let Some(value) = out[i] {return value;}
            let value = local[i].scale.contains(&0.) || nodes[i].0.is_some_and(|p| collapsed(p,nodes,local,out));
            out[i] = Some(value);
            value
        }
        let mut collapsed_nodes = vec![None; self.nodes.len()];
        for i in 0..collapsed_nodes.len() {collapsed(i,&self.nodes,&transforms,&mut collapsed_nodes);}
        Pose {
            collapsed: collapsed_nodes.into_iter().map(Option::unwrap).collect(),
            visibility: masks.into_iter().map(Option::unwrap).collect(),
            worlds: worlds.into_iter().map(Option::unwrap).collect(),
            weights,
        }
    }
    pub fn matrices(&self, pose: &Pose, node: usize, skin: Option<usize>) -> Vec<Mat4> {
        skin.map_or_else(
            || vec![pose.worlds[node]],
            |s| {
                self.skins[s]
                    .iter()
                    .map(|(j, b)| pose.worlds[*j] * *b)
                    .collect()
            },
        )
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(non_snake_case)]
pub(super) struct Influence {
    pub JOINTS_0: [u32; 4],
    pub WEIGHTS_0: [f32; 4],
}
#[derive(Clone)]
pub(super) struct Deform {
    pub cpu: Option<Arc<crate::subdivision::CpuDeform>>,
    pub native: Option<Arc<crate::geometry::NativeDeform>>,
    pub job: usize,
    pub leader: usize,
    pub asset: Option<usize>,
    pub node: usize,
    pub skin: Option<usize>,
    // Per joint: rest-position interval followed by each morph-position interval.
    pub bounds: Arc<Vec<Vec<(Vec3, Vec3)>>>,
    pub source: wgpu::Buffer,
    pub influences: wgpu::Buffer,
    pub morphs: wgpu::Buffer,
    pub count: u32,
    pub targets: usize,
    pub joints: usize,
}
#[derive(Clone)]
pub(super) struct FrameDeform {
    pub leader: usize,
    pub chained: Option<crate::passes::deform::Chained>,
    /// Current pose, in the pool ray hits read (hit.slang).
    pub output: crate::geometry::VertexStream,
    pub previous_output: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub previous_params: wgpu::Buffer,
    pub bind: wgpu::BindGroup,
}

/// An installed asset named "actor" with the scene's clip layers and tracks.
#[cfg(test)]
fn actor(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    scene: serde_json::Value,
    cache: &mut SourceCache,
) -> Result<(Asset, Arc<Plan>)> {
    let mut scene = scene;
    scene["version"] = 2.into();
    scene["nodes"] = serde_json::json!([{"type":"asset","name":"actor","file":"fixture.gltf"}]);
    let input: ::vork_scene::v2::Scene = serde_json::from_value(scene)?;
    let mut plan = Plan::compile(&input, Path::new("."))?;
    let clips: Vec<_> = input
        .animation
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| layer.clip.as_ref().map(|clip| (index, clip)))
        .collect();
    let mut asset = Asset::cached(doc, buffers, &clips, Path::new("."), "fixture", cache)?
        .register_native(&mut plan, doc, "actor", 0, &Default::default())?;
    plan.finish_asset_bindings()?;
    let plan = Arc::new(plan);
    asset.attach_plan(Arc::clone(&plan));
    Ok((asset, plan))
}

#[cfg(test)]
mod sharing_tests {
    use super::*;
    #[test]
    fn external_clips_bind_all_named_nodes_and_validate_morph_widths() -> Result<()> {
        let original: serde_json::Value = serde_json::from_str(include_str!("../tests/goldens/skinned-box.gltf"))?;
        for morph in [false, true] {
            let mut source = original.clone();
            source["nodes"][1]["name"] = "RootMotion".into();
            if morph {
                source["nodes"][1]["mesh"] = 0.into();
                let position = source["meshes"][0]["primitives"][0]["attributes"]["POSITION"].clone();
                source["meshes"][0]["primitives"][0]["targets"] = serde_json::json!([
                    {"POSITION":position},{"POSITION":position},{"POSITION":position}]);
                source["animations"][0]["channels"][0]["target"]["path"] = "weights".into();
                source["accessors"][6]["type"] = "SCALAR".into();
                source["accessors"][6]["count"] = 6.into();
            }
            let (external_doc, external_buffers, _) = gltf::import_slice(serde_json::to_vec(&source)?)?;
            let mut target = source.clone();
            target.as_object_mut().unwrap().remove("skins");
            target["nodes"][0].as_object_mut().unwrap().remove("skin");
            target["nodes"].as_array_mut().unwrap().insert(0, serde_json::json!({"name":"padding"}));
            target["animations"][0]["channels"][0]["target"]["node"] = 2.into();
            let (doc, buffers, _) = gltf::import_slice(serde_json::to_vec(&target)?)?;
            let mut cache = SourceCache::default();
            cache.files.insert(Path::new(".").join("external.gltf"), Arc::new((external_doc, external_buffers)));
            let (embedded, _) = actor(&doc, &buffers, serde_json::json!({"animation":[{"clip":{"target":"actor"}}]}), &mut cache)?;
            let external_scene = serde_json::json!({"animation":[{"clip":{"target":"actor","file":"external.gltf"}}]});
            let (external, _) = actor(&doc, &buffers, external_scene.clone(), &mut cache)?;
            for time in [0., 0.5, 1., 0.] {
                assert_eq!(embedded.pose(time).worlds, external.pose(time).worlds);
                assert_eq!(embedded.pose(time).weights, external.pose(time).weights);
            }
            // A separate target document must not reuse bindings for another asset.
            for missing in [false, true] {
                let mut invalid = target.clone();
                invalid["nodes"][if missing {2} else {0}]["name"] = if missing {"other"} else {"RootMotion"}.into();
                let (invalid, buffers, _) = gltf::import_slice(serde_json::to_vec(&invalid)?)?;
                cache.clips.clear(); cache.binds.clear();
                let error = actor(&invalid, &buffers, external_scene.clone(), &mut cache).err().context("ambiguous or missing target accepted")?;
                assert!(format!("{error:#}").contains("exactly one named node"));
            }
            if morph {
                target["meshes"][0]["primitives"][0]["targets"].as_array_mut().unwrap().pop();
                target["animations"] = serde_json::json!([]);
                let (invalid, buffers, _) = gltf::import_slice(serde_json::to_vec(&target)?)?;
                cache.clips.clear(); cache.binds.clear();
                let error = actor(&invalid, &buffers, external_scene, &mut cache).err().context("mismatched morph width accepted")?;
                assert!(format!("{error:#}").contains("animation output shape mismatch"));
            }
        }
        Ok(())
    }
    #[test]
    fn imported_visible_keys_gate_subtrees_and_restore_static_refinements() -> Result<()> {
        let doc = gltf::Gltf::from_slice(br#"{"asset":{"version":"2.0"},"nodes":[{"name":"parent","children":[1]},{"name":"child"},{"name":"sibling"}]}"#)?;
        let mut scene: vork_scene::Scene = serde_json::from_value(serde_json::json!({"version":2,
            "nodes":[{"type":"asset","name":"actor","file":"fixture.gltf","nodes":{"parent":{"visible":false},"child":{"visible_glossy":false}}}],
            "animation":[{"tracks":[{"path":["nodes","actor","nodes","parent","visible"],"keys":[[0,false],[1,true]]}]}]
        }))?;
        let mut plan = Plan::compile(&scene, Path::new("."))?;
        let vork_scene::NodeKind::Asset {nodes:overrides,..} = &scene.nodes[0].kind else {unreachable!()};
        let mut asset = Asset::cached(&doc.document,&[],&[],Path::new("."),"visible",&mut SourceCache::default())?
            .register_native(&mut plan,&doc.document,"actor",0,overrides)?;
        plan.finish_asset_bindings()?;
        // Installation samples imported poses before final program-word binding
        // freezes and attaches the plan.
        let mut samples = plan.samples();
        plan.sample_into(1., &mut samples);
        assert_eq!(asset.pose_with_plan(&samples, &plan).visibility[0], constants::RAY_VISIBILITY_ALL);
        plan.sample_into(0., &mut samples);
        assert_eq!(asset.pose_with_plan(&samples, &plan).visibility[0], 0);
        asset.attach_plan(Arc::new(plan));
        for (time,shown) in [(0.,false),(0.99,false),(1.,true),(0.5,false)] {
            let pose = asset.pose(time);
            assert_eq!(pose.visibility[0], if shown {constants::RAY_VISIBILITY_ALL} else {0});
            assert_eq!(pose.visibility[1], if shown {constants::RAY_VISIBILITY_ALL & !constants::RAY_SPECULAR} else {0});
            assert_eq!(pose.visibility[2],constants::RAY_VISIBILITY_ALL);
        }
        // Index selectors use the same binding as names.
        scene.animation[0].tracks.as_mut().unwrap()[0].path[3] = "#0".into();
        let mut plan = Plan::compile(&scene,Path::new("."))?;
        assert!(plan.bind_asset(&doc.document,"actor",0)?);
        Ok(())
    }
    #[test]
    fn static_pose_is_shared_across_sample_times_and_instance_offsets() -> Result<()> {
        let doc = gltf::Gltf::from_slice(
            br#"{"asset":{"version":"2.0"},"nodes":[{"translation":[1,2,3]}]}"#,
        )?;
        let (asset, plan) = actor(
            &doc.document,
            &[],
            serde_json::json!({}),
            &mut SourceCache::default(),
        )?;
        let samples = plan.samples();
        assert!(!asset.animated());
        let a = asset.pose_sampled(&samples);
        let b = asset.pose_sampled(&samples);
        let c = asset.pose_sampled(&samples);
        assert!(Arc::ptr_eq(&a, &b) && Arc::ptr_eq(&a, &c));
        assert_eq!(a.worlds[0].w_axis.truncate(), Vec3::new(1., 2., 3.));
        Ok(())
    }
    #[test]
    fn playback_variants_share_bindings_and_tracks_but_not_poses() -> Result<()> {
        let (doc, buffers, _) =
            gltf::import_slice(include_bytes!("../tests/goldens/skinned-box.gltf"))?;
        let mut cache = SourceCache::default();
        let (a, _) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[{"clip":{"target":"actor"}}]}),
            &mut cache,
        )?;
        let (b, _) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[{"clip":{"target":"actor"},"speed":2}]}),
            &mut cache,
        )?;
        assert!(Arc::ptr_eq(&a.nodes, &b.nodes));
        assert!(Arc::ptr_eq(&a.skins, &b.skins));
        assert!(Arc::ptr_eq(&a.layers[0].tracks, &b.layers[0].tracks));
        assert_ne!(a.pose(0.25).worlds, b.pose(0.25).worlds);
        Ok(())
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;
    #[test]
    fn authored_joint_track_uses_weighted_layers_without_joint_target_nodes() -> Result<()> {
        let (doc, buffers, _) =
            gltf::import_slice(include_bytes!("../tests/goldens/skinned-box.gltf"))?;
        let joint = doc.skins().next().unwrap().joints().next().unwrap();
        let name = joint.name().context("fixture joint needs name")?;
        let (actor, _) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[
                {"tracks":[{"path":["nodes","actor","nodes",name,"translation"],"keys":[[0,[0,0,0]],[2,[0,4,0]]]}],"weight":0.5},
                {"tracks":[{"path":["nodes","actor","nodes",name,"translation"],"keys":[[0,[0,0,0]],[2,[0,2,0]]]}],"blend":"combine"}
            ]}),
            &mut SourceCache::default(),
        )?;
        assert!(actor.animated());
        let local = actor.pose(1.).worlds[joint.index()].w_axis.y;
        assert!(
            (local - 2.).abs() < 1e-6,
            "joint should replace half then add: {local}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod unified_pose_tests {
    use super::*;
    #[test]
    fn named_morph_at_nonzero_index_without_default_weights() -> Result<()> {
        let mut source: serde_json::Value =
            serde_json::from_str(include_str!("../tests/goldens/skinned-box.gltf"))?;
        source["nodes"][0]["name"] = "Face".into();
        source["meshes"][0]["extras"] = serde_json::json!({"targetNames":["blink","smile"]});
        let position = source["meshes"][0]["primitives"][0]["attributes"]["POSITION"].clone();
        source["meshes"][0]["primitives"][0]["targets"] =
            serde_json::json!([{"POSITION":position},{"POSITION":position}]);
        let (doc, buffers, _) = gltf::import_slice(serde_json::to_vec(&source)?)?;
        let (actor, plan) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[{"tracks":[{"path":["nodes","actor","nodes","Face","morph_weights","smile"],"keys":[[0,0],[1,1]]}]}]}),
            &mut SourceCache::default(),
        )?;
        let mut sampled = plan.samples();
        plan.sample_into(0.75, &mut sampled);
        assert_eq!(actor.pose_sampled(&sampled).weights[0], [0., 0.75]);
        Ok(())
    }
    #[test]
    fn clip_weight_and_additive_authored_pose_use_the_same_layer_order() -> Result<()> {
        let (doc, buffers, _) =
            gltf::import_slice(include_bytes!("../tests/goldens/skinned-box.gltf"))?;
        let (clip_only, _) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[{"clip":{"target":"actor"}}]}),
            &mut SourceCache::default(),
        )?;
        let reference = clip_only.pose(0.5).worlds[1].w_axis.truncate() * 0.5 + Vec3::Y * 0.5;
        let (actor, plan) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[
                {"tracks":[{"path":["animation","walk","weight"],"keys":[[0,0],[1,1]]}]},
                {"name":"walk","clip":{"target":"actor","index":0}},
                {"blend":"combine","weight":0.5,"tracks":[{"path":["nodes","actor","nodes","joint","translation"],"keys":[[0,[0,1,0]]]},{"path":["nodes","actor","nodes","joint","rotation"],"keys":[[0,Quat::from_rotation_y(0.8).to_array()]]}]}
            ]}),
            &mut SourceCache::default(),
        )?;
        let mut sampled = plan.samples();
        plan.sample_into(0.5, &mut sampled);
        let pose = actor.pose_sampled(&sampled);
        assert!((pose.worlds[1].w_axis.truncate() - reference).length() < 1e-6);
        let (_, rotation, _) = pose.worlds[1].to_scale_rotation_translation();
        assert!(rotation.dot(Quat::from_rotation_y(0.4)).abs() > 0.99999);
        Ok(())
    }
}

#[cfg(test)]
mod clip_layer_binding_tests {
    use super::*;
    #[test]
    fn two_clips_keep_their_layer_indices_and_reject_later_bad_joint_shape() -> Result<()> {
        let mut source: serde_json::Value =
            serde_json::from_str(include_str!("../tests/goldens/skinned-box.gltf"))?;
        source["animations"][0]["name"] = "Walk".into();
        let mut second = source["animations"][0].clone();
        second["name"] = "Idle".into();
        source["animations"].as_array_mut().unwrap().push(second);
        let (doc, buffers, _) = gltf::import_slice(serde_json::to_vec(&source)?)?;
        let (actor_pose, _) = actor(
            &doc,
            &buffers,
            serde_json::json!({"animation":[
                {"tracks":[{"path":["animation","idle","weight"],"keys":[[0,0],[1,1]]}]},
                {"name":"walk","clip":{"target":"actor","name":"Walk"}},
                {"name":"idle","clip":{"target":"actor","name":"Idle"},"speed":0,"weight":0}
            ]}),
            &mut SourceCache::default(),
        )?;
        let start = actor_pose.pose(0.);
        let midpoint = actor_pose.pose(0.5);
        let end = actor_pose.pose(1.);
        assert_eq!(
            end.worlds, start.worlds,
            "the second clip's lower-layer keyed weight must reach one"
        );
        assert_ne!(
            midpoint.worlds, start.worlds,
            "clip layers must not collapse to the rest pose"
        );
        let joint = doc.skins().next().unwrap().joints().next().unwrap();
        let name = joint.name().unwrap();
        assert!(
            actor(
                &doc,
                &buffers,
                serde_json::json!({"animation":[
                    {"tracks":[{"path":["nodes","actor","nodes",name,"translation"],"keys":[[0,[0,0,0]]]}]},
                    {"tracks":[{"path":["nodes","actor","nodes",name,"translation"],"keys":[[0,[0,0]]]}]}
                ]}),
                &mut SourceCache::default(),
            )
            .is_err()
        );
        Ok(())
    }
}
