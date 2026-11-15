//! Install-time subdivision stencils and CPU evaluation after live deformation.
//! Static meshes finish here. Animated output uses the existing two-pose morph
//! stream, dirty uploads and deformation dispatch; no new pass or shader.
use crate::*;
use ::vork_scene::v2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vork_mesh::Plan;
use crate::geometry_topology::{AttributeDomain, Connectivity, RowMap, Topology, Transition};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Refinement {
    pub(crate) plan: Plan,
    pub(crate) positions: Vec<[f32; 3]>,
    input_indices: Vec<u32>,
    input_faces: Option<Vec<u32>>,
    /// Stable source identities stay separate from weighted output rows.
    input_vertex_ids: Vec<u64>,
    uv: Vec<[f32; 2]>,
}
impl Refinement {
    pub(crate) fn topology(&self) -> Transition<'_> {
        Transition {
            input: Topology { vertices: self.positions.len(), connectivity: self.input_faces.as_ref().map_or(
                Connectivity::Triangles(&self.input_indices), |counts| Connectivity::Polygons { indices: &self.input_indices, counts }) },
            output: Topology { vertices: self.plan.stencils.len(), connectivity: Connectivity::Triangles(&self.plan.indices) },
            vertices: RowMap::Stencils(&self.plan), primitives: RowMap::Indexed(&self.plan.primitive_ids), corners: None,
            source_vertex_ids: &self.input_vertex_ids,
        }
    }
    fn validate_topology(&self) -> Result<()> {
        self.topology().validate()
    }
    fn vertex_attribute<const N: usize>(&self, values: &[[f32;N]]) -> Vec<[f32;N]> {
        self.topology().remap(AttributeDomain::Vertex,values).expect("validated subdivision vertex domain").into_owned()
    }
    pub(crate) fn evaluate(&self, cage: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let positions = self.vertex_attribute(cage);
        let normals = self.plan.normals(&positions);
        (positions, normals)
    }
}
#[derive(Serialize, Deserialize)]
struct Baked {
    refinement: Refinement,
    geometry: crate::geometry::Geometry,
}

pub(crate) fn prepare(
    r: &Renderer,
    buffers: &buffers::SceneBuffers,
    source: &v2::Geometry,
    geometry: crate::geometry::Geometry,
) -> Result<crate::geometry::Geometry> {
    if source
        .subdivision
        .as_ref()
        .is_none_or(|s| s.render_levels == 0)
    {
        return Ok(geometry);
    }
    ensure!(
        source.material_ranges.is_empty(),
        "subdivision requires a single-material cage (no material_ranges)"
    );
    let mut key = blake3::Hasher::new();
    key.update(&serde_json::to_vec(source)?);
    for name in source
        .attributes
        .values()
        .chain(source.indices.iter())
        .chain(
            source
                .morphs
                .values()
                .flat_map(|m| std::iter::once(&m.POSITION).chain(m.NORMAL.iter())),
        )
    {
        for word in &buffers.data[name].words {
            key.update(&word.to_le_bytes());
        }
    }
    let indices = source
        .indices
        .as_ref()
        .map(|n| buffers.data[n].words.clone())
        .unwrap_or_else(|| (0..geometry.vertices.len() as u32).collect());
    let (levels, simple, limit) = source.subdivision.as_ref().map_or((0, false, false), |s| {
        (
            s.render_levels,
            matches!(s.subdivision_type, v2::SubdivisionType::Simple),
            s.use_limit_surface,
        )
    });
    let (vertices, indices_count) = Plan::refined_counts(geometry.vertices.len(), &indices, source.face_vertex_counts.as_deref(), levels)?;
    let limits = r.device.limits();
    // Every output vertex has a float3 position. Other retained streams check
    // their exact byte demands when packing; do not impose a full-Vertex stride
    // on a split position/attribute buffer.
    crate::gpu_memory::storage_binding(&limits, "subdivision positions", vertices * 12, "reduce subdivision level or source mesh size")?;
    ensure!(indices_count*4 <= limits.max_buffer_size,
        crate::gpu_memory::device_limit("subdivision triangle indices", format_args!("{} bytes",indices_count*4), format_args!("max_buffer_size={} bytes",limits.max_buffer_size), "reduce subdivision level or source mesh size"));
    let path = r.disk_cache.path("subdivision", 2, key.finalize());
    if let Some(bytes) = disk_cache::read(&path) {
        if let Ok(mut baked) = serde_json::from_slice::<Baked>(&bytes) {
            baked.refinement.validate_topology()?;
            let count=baked.refinement.plan.stencils.len();
            ensure!(baked.geometry.vertices.len()==count && baked.refinement.uv.len()==count
                && baked.geometry.attributes.values().all(|rows|rows.len()==count)
                && baked.geometry.indices==baked.refinement.plan.indices,
                "cached subdivision output violates its topology/domain contract");
            baked.geometry.refinement = Some(Arc::new(baked.refinement));
            return Ok(baked.geometry);
        }
    }
    let plan = Plan::new(
        geometry.vertices.len(),
        &indices,
        source.face_vertex_counts.as_deref(),
        &source.crease_edge.iter().map(|e| (e.vertices, e.crease)).collect::<Vec<_>>(),
        levels,
        simple,
        limit,
    )?;
    let positions: Vec<_> = geometry.vertices.iter().map(|v| v.POSITION).collect();
    let mut refinement = Refinement {
        plan,
        positions,
        input_indices: indices,
        input_faces: source.face_vertex_counts.clone(),
        input_vertex_ids: geometry.vertex_ids.clone(),
        uv: Vec::new(),
    };
    refinement.validate_topology()?;
    refinement.uv = refinement.vertex_attribute(&geometry.vertices.iter().map(|vertex|vertex.TEXCOORD_0).collect::<Vec<_>>());
    let (positions, normals) = refinement.evaluate(&refinement.positions);
    ensure!(
        positions.iter().flatten().all(|v| v.is_finite()),
        "subdivision produced non-finite positions"
    );
    let uv1 = refinement.vertex_attribute(
        &geometry
            .vertices
            .iter()
            .map(|v| v.TEXCOORD_1)
            .collect::<Vec<_>>(),
    );
    let colors = refinement.vertex_attribute(
        &geometry
            .vertices
            .iter()
            .map(|v| v.COLOR_0)
            .collect::<Vec<_>>(),
    );
    let uvs = refinement
        .uv
        .iter()
        .zip(uv1)
        .map(|(a, b)| [*a, b])
        .collect();
    let mut result = crate::geometry::Geometry::channels(
        positions,
        normals,
        uvs,
        refinement.plan.indices.clone(),
        geometry.uv_sets,
    );
    result.primitive_ids=if levels==0 {geometry.primitive_ids.clone()} else {refinement.plan.primitive_ids.clone()};
    for (v, color) in result.vertices.iter_mut().zip(colors) {
        v.COLOR_0 = color;
    }
    for (name, values) in &geometry.attributes {
        result
            .attributes
            .insert(name.clone(), refinement.vertex_attribute(values));
    }
    // Reconstruct tangent space for the new topology and normals.
    tangents(&mut result);
    let baked = Baked {
        refinement: refinement.clone(),
        geometry: result,
    };
    if let Ok(bytes) = serde_json::to_vec(&baked) {
        let _ = disk_cache::write(&path, &bytes);
    }
    let mut result = baked.geometry;
    result.refinement = Some(Arc::new(refinement));
    Ok(result)
}
fn tangents(geometry: &mut crate::geometry::Geometry) {
    let mut t = vec![Vec3::ZERO; geometry.vertices.len()];
    let mut b = t.clone();
    for face in geometry.indices.chunks_exact(3) {
        let a = &geometry.vertices[face[0] as usize];
        let x = &geometry.vertices[face[1] as usize];
        let y = &geometry.vertices[face[2] as usize];
        let e1 = Vec3::from(x.POSITION) - Vec3::from(a.POSITION);
        let e2 = Vec3::from(y.POSITION) - Vec3::from(a.POSITION);
        let u1 = glam::Vec2::from(x.TEXCOORD_0) - glam::Vec2::from(a.TEXCOORD_0);
        let u2 = glam::Vec2::from(y.TEXCOORD_0) - glam::Vec2::from(a.TEXCOORD_0);
        let det = u1.perp_dot(u2);
        if det.abs() > f32::EPSILON {
            for &i in face {
                t[i as usize] += (e1 * u2.y - e2 * u1.y) / det;
                b[i as usize] += (e2 * u1.x - e1 * u2.x) / det;
            }
        }
    }
    for (i, v) in geometry.vertices.iter_mut().enumerate() {
        let n = Vec3::from(v.NORMAL);
        let tangent = (t[i] - n * n.dot(t[i])).normalize_or_zero();
        v.TANGENT = tangent
            .extend(if n.cross(tangent).dot(b[i]) < 0. {
                -1.
            } else {
                1.
            })
            .to_array();
    }
}

pub(crate) struct CpuDeform {
    pub(crate) base: Vec<Vertex>,
    refinement: Arc<Refinement>,
    influences: Vec<Influence>,
    morphs: Vec<[f32; 3]>,
}
impl CpuDeform {
    pub(crate) fn new(
        refinement: Arc<Refinement>,
        mut influences: Vec<Influence>,
        morphs: Vec<[f32; 3]>,
        joints: usize,
    ) -> Result<Self> {
        crate::deformation::normalize(&mut influences, refinement.positions.len(), 1, joints)
            .map_err(|_| anyhow::anyhow!("invalid subdivision skin weights"))?;
        Ok(Self {
            base: Vec::new(),
            refinement,
            influences,
            morphs,
        })
    }
    pub(crate) fn vertices(&self, matrices: &[Mat4], weights: &[f32]) -> Vec<Vertex> {
        let (positions, normals) = self.evaluate(matrices, weights);
        let mut geometry = crate::geometry::Geometry::new(self.base.clone(), self.refinement.plan.indices.clone(), 1);
        for ((v,p),n) in geometry.vertices.iter_mut().zip(positions).zip(normals) {v.POSITION=p;v.NORMAL=n;}
        tangents(&mut geometry);
        geometry.vertices
    }
    pub(crate) fn evaluate(
        &self,
        matrices: &[Mat4],
        weights: &[f32],
    ) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let cage = crate::deformation::positions(&self.refinement.positions,
            &self.influences, 1, &self.morphs, matrices, weights);
        self.refinement.evaluate(&cage)
    }
}

/// Each frame slot owns its two-pose CPU results. Alias draws share the leader.
pub(crate) fn frame_resources(
    s: &InstalledScene,
    r: &Renderer,
    frames: &mut [Option<FrameDeform>],
) -> Vec<Option<wgpu::Buffer>> {
    let mut buffers: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(s.draws.len());
    for (i, draw) in s.draws.iter().enumerate() {
        let Some(deform) = draw.deform.as_ref().filter(|d| d.cpu.is_some()) else {
            buffers.push(None);
            continue;
        };
        if deform.leader < i {
            buffers.push(buffers[deform.leader].clone());
            continue;
        }
        let morphs = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some("frame subdivision poses"),
            size: deform.count as u64 * 3 * constants::MORPH_FLOATS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame = frames[i].as_mut().unwrap();
        let pipeline = s.pipelines.deform_pipeline.get().unwrap();
        frame.bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CPU subdivision through deformation"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, deform.source.as_entire_binding()),
                entry(1, deform.influences.as_entire_binding()),
                entry(2, morphs.as_entire_binding()),
                entry(3, frame.params.as_entire_binding()),
                entry(4, frame.output.binding()),
                entry(5, frame.previous_params.as_entire_binding()),
                entry(6, frame.previous_output.as_entire_binding()),
            ],
        });
        buffers.push(Some(morphs));
    }
    buffers
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    #[cfg(vork_private_tests)] // Uses private subdivision scenes.
    fn subdivision_fixture_frames_preserve_repeated_requests() -> Result<()> {
        let r = crate::test_renderer()?;
        for (name,source) in [
            ("skin",include_str!("../../../examples/subdivision-skin.json")), // export-example-private: cfg(vork_private_tests) fixture
            ("morph",include_str!("../../../examples/subdivision-deform.json")), // export-example-private: cfg(vork_private_tests) fixture
            ("hero",include_str!("../../../scenes/subdivision-hero.json")),
        ] {
            let mut scene:serde_json::Value=vork_scene::v2::from_test_str(source)?;
            let resolution=scene["render"]["resolution"].as_array().unwrap();
            let width=(resolution[0].as_u64().unwrap()/4).max(32) as u32;
            let height=(resolution[1].as_u64().unwrap()/4).max(32) as u32;
            scene["render"]["resolution"]=serde_json::json!([width,height]);
            v2::replace_integrator(&mut scene,serde_json::json!({"integrator":"raster","samples":4,"fast_gi":null,"reflections":"environment"}));
            let installed=r.install(v2::from_test_value(scene)?,std::path::Path::new("."))?;
            let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            let mut images:std::collections::BTreeMap<u32,Vec<[f32;4]>>=std::collections::BTreeMap::new();
            for (index,time) in [0f32,0.25,0.,0.25].into_iter().enumerate() {
                slot.submit(&installed,&r,time)?;
                let output=slot.collect(&r)?;
                ensure!(output.rgba.chunks_exact(4).any(|pixel|pixel[..3].iter().any(|&value|value>100)),"{name}: fixture has no visible lit geometry");
                if let Some(expected)=images.get(&time.to_bits()) {
                    crate::test_linear::assert_linear_eq(output.linear_rgba.as_ref().unwrap(),expected);
                } else {images.insert(time.to_bits(),output.linear_rgba.clone().unwrap());}
                if let Some(directory)=std::env::var_os("GEOCHAIN_OUT") {
                    image::save_buffer(std::path::PathBuf::from(directory).join(format!("subdivision-{name}-{index}.png")),&output.rgba,width,height,image::ColorType::Rgba8)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn subdivision_follows_skinning_and_morphs() -> Result<()> {
        let p = vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        let refinement = Arc::new(Refinement {
            plan: Plan::new(4, &[0, 1, 2, 3], Some(&[4]), &[], 1, true, false)?,
            positions: p,
            input_indices: vec![0,1,2,3],
            input_faces: Some(vec![4]),
            input_vertex_ids: Vec::new(),
            uv: vec![[0.; 2]; 9],
        });
        let influences = (0..4)
            .map(|i| Influence {
                JOINTS_0: [if i == 1 || i == 2 { 1 } else { 0 }; 4],
                WEIGHTS_0: [1., 0., 0., 0.],
            })
            .collect();
        let cpu = CpuDeform::new(
            refinement,
            influences,
            vec![[0., 0., 0.], [0., 0., 0.], [0., 0.4, 0.], [0., 0., 0.]],
            2,
        )?;
        let (positions, _) = cpu.evaluate(
            &[
                Mat4::IDENTITY,
                Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ],
            &[1.],
        );
        assert!(
            positions
                .iter()
                .any(|p| Vec3::from(*p).abs_diff_eq(Vec3::new(0., 0.6, -0.5), 1e-6)),
            "subdivide the deformed cage, rather than skinning interpolated bind vertices"
        );
        Ok(())
    }
}
