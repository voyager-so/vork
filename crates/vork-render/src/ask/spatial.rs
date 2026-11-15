use super::evaluation::{CpuScene, Pose};
use crate::{buffers, programs};
use anyhow::{Context, Result, ensure};
use glam::{Mat4, Vec3};
use vork_scene::{NodeKind, v2};

pub(super) type Bounds = (Vec3, Vec3);

fn include(bounds: &mut Option<Bounds>, p: Vec3) -> Result<()> {
    ensure!(
        p.is_finite(),
        "ask: evaluated geometry is not finite; correct its positions or animation"
    );
    *bounds = Some(bounds.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
    Ok(())
}

/// Decode influence columns into the renderer's shared vertex-major evaluator.
fn deform(
    positions: &mut Vec<[f32; 3]>,
    sets: &[(Vec<[u32; 4]>, Vec<[f32; 4]>)],
    matrices: &[Mat4],
    morphs: &[[f32; 3]],
    weights: &[f32],
) -> Result<()> {
    ensure!(
        sets.iter()
            .all(|(j, w)| j.len() == positions.len() && w.len() == positions.len()),
        "skin attribute count differs from positions"
    );
    ensure!(
        morphs.len() == positions.len() * weights.len(),
        "morph count differs from positions"
    );
    let count = sets.len().max(1);
    let mut influences = Vec::with_capacity(positions.len() * count);
    for i in 0..positions.len() {
        if sets.is_empty() {
            influences.push(crate::animation::Influence {
                JOINTS_0: [0; 4],
                WEIGHTS_0: [1., 0., 0., 0.],
            });
        } else {
            influences.extend(sets.iter().map(|(j, w)| crate::animation::Influence {
                JOINTS_0: j[i],
                WEIGHTS_0: w[i],
            }));
        }
    }
    crate::deformation::normalize(&mut influences, positions.len(), count, matrices.len())?;
    *positions =
        crate::deformation::positions(positions, &influences, count, morphs, matrices, weights);
    Ok(())
}

impl CpuScene {
    fn authored_buffer(&self, name: &str) -> Result<&buffers::AttributeData> {
        ensure!(
            !crate::compute::writes_buffer(&self.scene, name),
            "not built yet: this answer needs the GPU program writing buffer {name:?}"
        );
        self.buffers
            .data
            .get(name)
            .with_context(|| format!("ask: unknown numeric buffer {name:?}"))
    }

    fn displacement(&self, material: &str) -> Result<()> {
        let material = self
            .scene
            .materials
            .get(material)
            .context("unknown mesh material")?;
        if material.program == "std/openpbr" {
            return Ok(());
        }
        let (source, origin) =
            programs::material_source(&self.scene, &self.base, &material.program)?;
        let (stages, _) =
            vork_slang::Compiler::discover().stages(&programs::abi(), &source, &origin)?;
        ensure!(
            !stages.contains("displace"),
            "not built yet: this answer needs the GPU program {:?}",
            material.program
        );
        Ok(())
    }

    pub fn placements(
        &self,
        node: usize,
        pose: &Pose,
        instance: Option<usize>,
    ) -> Result<Vec<(Mat4, f32)>> {
        let NodeKind::Mesh {
            instances: Some(definition),
            ..
        } = &pose.nodes[node].node.kind
        else {
            ensure!(
                instance.is_none(),
                "ask: instance requires an instanced node"
            );
            return Ok(vec![(Mat4::IDENTITY, 0.)]);
        };
        for name in definition
            .attributes
            .values()
            .chain(definition.count.iter())
        {
            self.authored_buffer(name)?;
        }
        let transforms = buffers::instance_transforms(definition, &self.buffers)?;
        let capacity = transforms.len();
        let count = definition
            .count
            .as_ref()
            .map(|name| self.authored_buffer(name).map(|b| b.words[0] as usize))
            .transpose()?
            .unwrap_or(capacity);
        ensure!(
            count <= capacity,
            "ask: instance count exceeds capacity; reduce the authored count"
        );
        if let Some(i) = instance {
            ensure!(i < count, "ask: instance is outside the live count");
        }
        let offsets = definition
            .attributes
            .get("time_offset")
            .map(|name| self.authored_buffer(name)?.floats::<1>())
            .transpose()?;
        Ok(transforms
            .into_iter()
            .take(count)
            .enumerate()
            .filter(|(i, _)| instance.is_none_or(|wanted| wanted == *i))
            .map(|(i, m)| (m, offsets.as_ref().map_or(0., |v| v[i][0])))
            .collect())
    }

    fn native_points(&self, node: usize, pose: &Pose) -> Result<Vec<[f32; 3]>> {
        let NodeKind::Mesh {
            geometry: name,
            skin: binding,
            morph_weights,
            materials,
            ..
        } = &pose.nodes[node].node.kind
        else {
            return Ok(Vec::new());
        };
        for material in materials {
            self.displacement(material)?;
        }
        let source = &self.scene.geometry[name];
        if matches!(
            source.topology,
            v2::Topology::Implicit | v2::Topology::Volume
        ) {
            // These are the renderer's installed field extents, not an isosurface
            // extraction or a claim about occupied voxels.
            for buffer in source.attributes.values() {
                ensure!(
                    !crate::compute::writes_buffer(&self.scene, buffer),
                    "not built yet: this answer needs the GPU program writing buffer {buffer:?}"
                );
            }
            let geometry = buffers::geometry_v2(&self.buffers, source, &[], 0, None)?;
            return Ok(corners(geometry.bounds).map(|p| p.to_array()).to_vec());
        }
        for name in source
            .attributes
            .values()
            .chain(source.indices.iter())
            .chain(source.widths.iter())
            .chain(source.count.iter())
        {
            self.authored_buffer(name)?;
        }
        if !matches!(source.topology, v2::Topology::Triangles) {
            // Widths describe diameter. Their enclosing cubes conservatively contain the
            // renderer's camera-dependent point/curve tessellation without constructing it.
            let positions = self
                .authored_buffer(&source.attributes["POSITION"])?
                .floats::<3>()?;
            let widths = self
                .authored_buffer(source.widths.as_deref().context("missing widths")?)?
                .floats::<1>()?;
            ensure!(
                positions.len() == widths.len(),
                "width count differs from positions"
            );
            let count = source
                .count
                .as_ref()
                .map(|n| self.authored_buffer(n).map(|b| b.words[0] as usize))
                .transpose()?
                .unwrap_or(positions.len());
            ensure!(count <= positions.len(), "point count exceeds capacity");
            return Ok(positions
                .into_iter()
                .zip(widths)
                .take(count)
                .flat_map(|(p, w)| {
                    let p = Vec3::from_array(p);
                    corners((p - Vec3::splat(w[0] * 0.5), p + Vec3::splat(w[0] * 0.5)))
                        .map(|v| v.to_array())
                })
                .collect());
        }
        let mut positions = self
            .authored_buffer(&source.attributes["POSITION"])?
            .floats::<3>()?;
        let mut indices = source
            .indices
            .as_ref()
            .map(|n| self.authored_buffer(n).map(|b| b.words.clone()))
            .transpose()?
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        // Renderer installation triangulates the rest cage before deformation.
        if !source
            .subdivision
            .as_ref()
            .is_some_and(|s| s.render_levels > 0)
        {
            if let Some(faces) = &source.face_vertex_counts {
                indices = crate::polygon::triangulate(&positions, &indices, faces)?;
            }
        }
        let mut morphs = Vec::new();
        let mut weights = Vec::new();
        for (name, morph) in &source.morphs {
            let delta = self.authored_buffer(&morph.POSITION)?.floats::<3>()?;
            ensure!(
                delta.len() == positions.len(),
                "morph count differs from positions"
            );
            morphs.extend(delta);
            weights.push(morph_weights.get(name).copied().unwrap_or(0.));
        }
        if binding.is_some() {
            self.authored_buffer(&binding.as_ref().unwrap().inverse_bind_matrices)?;
            let names = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| !n.node.name.is_empty())
                .map(|(i, n)| (n.node.name.clone(), i))
                .collect();
            let binding =
                buffers::native_binding(&self.scene, &self.nodes, &names, node, &self.buffers)?;
            let (matrices, _) = binding.palette(&pose.worlds, morph_weights);
            let joints = self.authored_buffer(
                source
                    .attributes
                    .get("JOINTS_0")
                    .context("skin requires JOINTS_0")?,
            )?;
            let skin_weights = self
                .authored_buffer(
                    source
                        .attributes
                        .get("WEIGHTS_0")
                        .context("skin requires WEIGHTS_0")?,
                )?
                .floats::<4>()?;
            let joints = joints
                .words
                .chunks_exact(4)
                .map(|j| j.try_into().unwrap())
                .collect();
            deform(
                &mut positions,
                &[(joints, skin_weights)],
                &matrices,
                &morphs,
                &weights,
            )?;
        } else if !morphs.is_empty() {
            deform(&mut positions, &[], &[Mat4::IDENTITY], &morphs, &weights)?;
        }
        if let Some(subdivision) = source.subdivision.as_ref().filter(|s| s.render_levels > 0) {
            let plan = vork_mesh::Plan::new(
                positions.len(),
                &indices,
                source.face_vertex_counts.as_deref(),
                &source
                    .crease_edge
                    .iter()
                    .map(|e| (e.vertices, e.crease))
                    .collect::<Vec<_>>(),
                subdivision.render_levels,
                matches!(subdivision.subdivision_type, v2::SubdivisionType::Simple),
                subdivision.use_limit_surface,
            )?;
            positions = plan.evaluate(&positions);
            indices = plan.indices;
        }
        ensure!(
            indices.len() % 3 == 0 && indices.iter().all(|&i| (i as usize) < positions.len()),
            "invalid triangle indices"
        );
        Ok(indices.into_iter().map(|i| positions[i as usize]).collect())
    }

    fn imported_points(&self, asset: usize, local: usize, pose: &Pose) -> Result<Vec<[f32; 3]>> {
        let imported = &self.imports[asset];
        let source = imported
            .document
            .nodes()
            .nth(local)
            .context("unknown imported node")?;
        let Some(mesh) = source.mesh() else {
            return Ok(Vec::new());
        };
        if let NodeKind::Asset { materials, .. } = &self.nodes[imported.root].node.kind {
            for material in materials.values() {
                self.displacement(material)?;
            }
        }
        let sampled = self.assets[asset].pose_with_plan(&pose.samples, &self.animation);
        let mut result = Vec::new();
        for primitive in mesh.primitives() {
            ensure!(
                primitive.mode() == gltf::mesh::Mode::Triangles,
                "only triangle primitives are supported"
            );
            let reader = primitive.reader(|b| Some(&imported.buffers[b.index()]));
            let mut positions: Vec<_> = reader
                .read_positions()
                .context("mesh missing POSITION")?
                .collect();
            let mut morphs = Vec::new();
            let mut weights = Vec::new();
            for (target, (delta, _, _)) in reader.read_morph_targets().enumerate() {
                let delta: Vec<_> = delta
                    .map(|d| d.collect())
                    .unwrap_or_else(|| vec![[0.; 3]; positions.len()]);
                ensure!(
                    delta.len() == positions.len(),
                    "morph count differs from positions"
                );
                morphs.extend(delta);
                // glTF may omit defaults, and a named track may sample only a
                // prefix. Renderer uploads fill every remaining target with zero.
                weights.push(sampled.weights[local].get(target).copied().unwrap_or(0.));
            }
            let matrices =
                self.assets[asset].matrices(&sampled, local, source.skin().map(|s| s.index()));
            if source.skin().is_some() {
                let count = primitive
                    .attributes()
                    .filter_map(|(semantic, _)| match semantic {
                        gltf::Semantic::Joints(i) => Some(i + 1),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                ensure!(count > 0, "skinned mesh requires joint influences");
                let sets = (0..count)
                    .map(|i| {
                        Ok((
                            reader
                                .read_joints(i)
                                .context("missing JOINTS set")?
                                .into_u16()
                                .map(|v| v.map(u32::from))
                                .collect(),
                            reader
                                .read_weights(i)
                                .context("missing WEIGHTS set")?
                                .into_f32()
                                .collect(),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                deform(&mut positions, &sets, &matrices, &morphs, &weights)?;
            } else {
                deform(&mut positions, &[], &matrices, &morphs, &weights)?;
            }
            let indices: Vec<_> = reader
                .read_indices()
                .map(|v| v.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            ensure!(
                indices.len() % 3 == 0 && indices.iter().all(|&i| (i as usize) < positions.len()),
                "invalid triangle indices"
            );
            result.extend(indices.into_iter().map(|i| positions[i as usize]));
        }
        Ok(result)
    }

    pub fn bounds(
        &self,
        root: usize,
        pose: &Pose,
        request: &vork_time::Request,
        instance: Option<usize>,
    ) -> Result<Option<Bounds>> {
        if instance.is_some() {
            self.placements(root, pose, instance)?;
        }
        let mut bounds = None;
        for node in 0..self.nodes.len() {
            if !self.under(node, root) {
                continue;
            }
            if let Some((asset, local, owner, _)) = self.nodes[node].embedded {
                for p in self.imported_points(asset, local, pose)? {
                    include(
                        &mut bounds,
                        pose.worlds[owner].transform_point3(Vec3::from_array(p)),
                    )?;
                }
            } else if let NodeKind::Asset {file, materials, ..} = &self.nodes[node].node.kind
                && crate::splats::is_file(file)
            {
                for material in materials.values() { self.displacement(material)?; }
                let cloud=crate::splats::read(&self.base.join(file),u64::MAX)?;
                let mut local=None;
                for point in 0..cloud.metadata.count as usize {
                    let (lo,hi)=cloud.shape(point).bounds(3.)?;
                    include(&mut local,lo)?;include(&mut local,hi)?;
                }
                // The installed capture's support bounds, including its tails.
                // Object transforms act on the same domain used by rendering.
                if let Some(local)=local {
                    for point in corners(local) {
                        include(&mut bounds,pose.worlds[node].transform_point3(point))?;
                    }
                }
            } else if matches!(self.nodes[node].node.kind, NodeKind::Mesh { .. }) {
                for (local, offset) in
                    self.placements(node, pose, if node == root { instance } else { None })?
                {
                    let shifted;
                    let pose = if offset == 0. {
                        pose
                    } else {
                        shifted = self.sample_offset(request, offset)?;
                        &shifted
                    };
                    let matrix = pose.worlds[node] * local;
                    for p in self.native_points(node, pose)? {
                        include(&mut bounds, matrix.transform_point3(Vec3::from_array(p)))?;
                    }
                }
            }
        }
        Ok(bounds)
    }

    pub fn camera(
        &self,
        pose: &Pose,
        name: Option<&str>,
        time: f32,
    ) -> Result<(Mat4, Mat4, [[f32; 4]; 4])> {
        let camera = if let Some(name) = name.or(pose.render.camera.as_deref()) {
            self.node(name)?
        } else {
            let authored: Vec<_> = pose
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    n.embedded.is_none() && matches!(n.node.kind, NodeKind::Camera { .. })
                })
                .map(|(i, _)| i)
                .collect();
            ensure!(
                authored.len() <= 1,
                "ask: select a camera with render.camera or camera"
            );
            authored
                .first()
                .copied()
                .or_else(|| {
                    pose.nodes
                        .iter()
                        .position(|n| matches!(n.node.kind, NodeKind::Camera { .. }))
                })
                .context("ask: screen requires a camera")?
        };
        let NodeKind::Camera {
            material,
            focus_object,
        } = &pose.nodes[camera].node.kind
        else {
            anyhow::bail!("ask: camera must name a camera node")
        };
        let default: v2::Material =
            serde_json::from_value(serde_json::json!({"program":"std/camera"}))?;
        let material = material
            .as_ref()
            .map_or(&default, |name| &pose.materials[name]);
        ensure!(
            material.program == "std/camera",
            "not built yet: this answer needs the GPU program {:?}",
            material.program
        );
        let mut programs = self.cameras.borrow_mut();
        if !programs.contains_key(&material.program) {
            let (source, origin) =
                programs::material_source(&self.scene, &self.base, &material.program)?;
            let program = vork_slang::Compiler::discover().compile_camera(
                &programs::abi(),
                &source,
                &origin,
                "askcamera",
            )?;
            programs.insert(material.program.clone(), std::sync::Arc::new(program));
        }
        let program = programs[&material.program].clone();
        drop(programs);
        let values = material
            .inputs
            .iter()
            .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
            .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        let words = program.inputs.pack(&values, |name, _| {
            anyhow::bail!("ask: camera input {name} must be numeric")
        })?;
        let focus = focus_object
            .as_ref()
            .map(|name| {
                Ok::<_, anyhow::Error>(crate::targets::focus_distance(
                    pose.worlds[camera],
                    pose.worlds[self.node(name)?],
                ))
            })
            .transpose()?
            .unwrap_or(-1.);
        let state = crate::camera_program::State {
            program,
            words,
            offset: 0,
            focus,
            time,
            evaluated: Default::default(),
        };
        let [width, height] = pose.render.resolution;
        ensure!(
            width > 0 && height > 0,
            "ask: screen requires a positive output resolution"
        );
        let aspect = width as f32 / height as f32;
        let calibration = state
            .calibration(aspect)?
            .context("ask: screen for an uncalibrated camera is not built yet")?;
        let mut node = pose.nodes[camera].clone();
        node.camera = Some(state);
        Ok((
            crate::values::node_view(camera, &pose.nodes, &pose.worlds)?,
            crate::values::camera_projection(&node, aspect),
            calibration,
        ))
    }
}

pub(super) fn corners((lo, hi): Bounds) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    })
}
