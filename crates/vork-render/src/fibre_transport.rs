//! Geometry-derived fibre transport, independent of the shadow technique.
//! Four count prefixes bound local returns; forward visibility uses ray crossings.
use crate::*;
use glam::Vec4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Stage {
    Near,
    Far,
    Accumulate,
}
impl Stage {
    pub(crate) const ALL: [Self; 3] = [Self::Near, Self::Far, Self::Accumulate];
    pub(crate) fn additive(self) -> bool { matches!(self, Self::Accumulate) }
    pub(crate) fn index(self) -> usize { Self::ALL.iter().position(|s| *s == self).unwrap() }
    pub(crate) fn entry(self) -> &'static str {
        match self {
            Self::Near => "fibre_near_fs",
            Self::Far => "fibre_far_fs",
            Self::Accumulate => "fibre_accumulate_fs",
        }
    }
}

/// `FibreLight` in fibre_transport.slang. Matrices use the ordinary GPU convention.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LightRecord {
    matrices: [Mat4; 6],
    tiles: [[f32; 4]; 6],
    /// Per-face linear depth endpoints, projection kind (0=orthographic), face count.
    clip: [[f32; 4]; 6],
}
#[derive(Clone)]
pub(crate) struct Face {
    pub light: usize,
    pub face: usize,
    pub origin: [u32; 2],
    pub resolution: u32,
}
#[derive(Clone)]
pub(crate) struct Plan {
    pub side: u32,
    pub faces: Vec<Face>,
}
impl Plan {
    /// Reserve each local light's six faces so a moving source can enter the
    /// fibre bounds without changing the installed structure. Suns need one.
    pub(crate) fn new(
        suns: &[bool],
        resolution: u32,
        max_allocation: u64,
        limits: &wgpu::Limits,
    ) -> Result<Self> {
        let count: usize = suns.iter().map(|sun| if *sun { 1 } else { 6 }).sum();
        let columns = (count.max(1) as f64).sqrt().ceil() as u32;
        let side = resolution
            .checked_mul(columns)
            .context("fibre atlas dimensions overflow")?;
        // Counts (RGBA16F) and identity (RG32UI) each require 8 bytes per texel.
        let allocation = u64::from(side).pow(2) * 8;
        ensure!(
            side > 0 && side <= limits.max_texture_dimension_2d && allocation <= max_allocation,
            crate::gpu_memory::device_limit("fibre transport atlas", format_args!("{side}x{side} pixels, {allocation} bytes per count/identity target"), format_args!("{} pixels per axis and {max_allocation} bytes per allocation",limits.max_texture_dimension_2d), "reduce render resolution or shadow light count")
        );
        let mut faces = Vec::with_capacity(count);
        for (light, sun) in suns.iter().enumerate() {
            for face in 0..if *sun { 1 } else { 6 } {
                let index = faces.len() as u32;
                faces.push(Face {
                    light,
                    face,
                    origin: [index % columns * resolution, index / columns * resolution],
                    resolution,
                });
            }
        }
        Ok(Self { side, faces })
    }
    pub(crate) fn records(&self, lights: &[Light], bounds: (Vec3, Vec3)) -> Vec<LightRecord> {
        let centre = (bounds.0 + bounds.1) * 0.5;
        let radius = ((bounds.1 - bounds.0).length() * 0.5).max(f32::MIN_POSITIVE);
        let cube = [
            (Vec3::X, Vec3::NEG_Y),
            (Vec3::NEG_X, Vec3::NEG_Y),
            (Vec3::Y, Vec3::Z),
            (Vec3::NEG_Y, Vec3::NEG_Z),
            (Vec3::Z, Vec3::NEG_Y),
            (Vec3::NEG_Z, Vec3::NEG_Y),
        ];
        let mut records = vec![LightRecord::zeroed(); lights.len()];
        for (index, l) in lights.iter().enumerate() {
            let record = &mut records[index];
            if l.position[3] == 0. {
                let direction = Vec3::from_slice(&l.direction[..3]).normalize();
                let view = Mat4::look_to_rh(
                    centre - direction * (2. * radius),
                    direction,
                    direction.any_orthonormal_vector(),
                );
                let near = radius;
                let far = 3. * radius;
                record.matrices[0] = Mat4::orthographic_rh(-radius, radius, -radius, radius, near, far) * view;
                record.clip[0] = [near, far, 0., 1.];
            } else {
                let position = Vec3::from_slice(&l.position[..3]);
                // Perspective XY with constant clip Z: the fragment writes linear
                // face depth. This includes geometry through the source plane
                // without a world-space near cutoff or a singular depth inverse.
                let projection = Mat4::from_cols(
                    Vec4::X, Vec4::Y, Vec4::new(0., 0., -0.5, -1.), Vec4::ZERO,
                );
                let support = bounds;
                for (face, (direction, up)) in cube.into_iter().enumerate() {
                    let extent = (support.0-position).dot(direction).max((support.1-position).dot(direction));
                    // Cover subtraction/interpolation roundoff at the support edge;
                    // the bound scales with coordinates, not a world-space epsilon.
                    let magnitude = support.0.abs().max(support.1.abs()) + position.abs();
                    let error = vork_slang::registry::FP32_GAMMA7 * magnitude.dot(direction.abs());
                    let far = (extent + error).max(f32::MIN_POSITIVE).next_up();
                    record.matrices[face] = projection * Mat4::look_to_rh(position, direction, up);
                    record.clip[face] = [0., far, 1., 6.];
                }
            }
        }
        for face in &self.faces {
            records[face.light].tiles[face.face] = [
                face.origin[0] as f32,
                face.origin[1] as f32,
                face.resolution as f32,
                self.side as f32,
            ];
        }
        records
    }
}

pub(crate) fn capture_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        pipelines::texture_entry(0, wgpu::TextureSampleType::Depth, false),
        pipelines::texture_entry(1, wgpu::TextureSampleType::Depth, false),
    ]
}
pub(crate) fn read_entries(first: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![pipelines::buffer_entry(
        first,
        wgpu::ShaderStages::FRAGMENT,
        wgpu::BufferBindingType::Storage { read_only: true },
    )];
    entries.extend(
        (1..=2).map(|i| pipelines::texture_entry(first + i, wgpu::TextureSampleType::Depth, false)),
    );
    entries.push(pipelines::texture_entry(
        first + 3,
        wgpu::TextureSampleType::Float { filterable: false },
        false,
    ));
    entries.push(pipelines::texture_entry(
        first + 4,
        wgpu::TextureSampleType::Uint,
        false,
    ));
    entries
}

pub(crate) struct Persistent {
    device: wgpu::Device,
    plan: Plan,
    lights: usize,
    layout: wgpu::BindGroupLayout,
    pub(crate) dependencies: vork_slang::ProgramDependencies,
    deformed: bool,
    cache: crate::residency::Cache<Storage>,
}
impl Persistent {
    pub(crate) fn new(r: &Renderer, plan: Plan, lights: usize, layout: &wgpu::BindGroupLayout,
        dependencies: vork_slang::ProgramDependencies, deformed: bool) -> Self {
        Self {device:r.device.clone(), plan, lights, layout:layout.clone(), dependencies, deformed,
            cache:crate::residency::Cache::new(&r.residency)}
    }
    pub(crate) fn storage(&self) -> std::sync::Arc<Storage> {
        self.cache.get_or_try_init(|| {
            let storage=Storage::new(&self.device,self.plan.clone(),self.lights,&self.layout,self.dependencies,self.deformed);
            let bytes=u64::from(self.plan.side).pow(2)*(4+4+8+8)+storage.records.size();
            Ok((storage,bytes))
        }).expect("fibre storage recipe")
    }
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {self.storage().clear(encoder);}
}
pub(crate) struct Storage {
    pub plan: Plan,
    pub near: wgpu::TextureView,
    pub far: wgpu::TextureView,
    pub near_ids: wgpu::TextureView,
    pub counts: wgpu::TextureView,
    pub records: wgpu::Buffer,
    pub capture: wgpu::BindGroup,
    pub(crate) dependencies: vork_slang::ProgramDependencies,
    deformed: bool,
}
impl Storage {
    pub(crate) fn new(
        d: &wgpu::Device,
        plan: Plan,
        lights: usize,
        layout: &wgpu::BindGroupLayout,
        dependencies: vork_slang::ProgramDependencies,
        deformed: bool,
    ) -> Self {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let target = |name, format| {
            texture(d, name, plan.side, plan.side, 1, format, usage)
                .create_view(&Default::default())
        };
        let near = target("fibre first crossing", wgpu::TextureFormat::Depth32Float);
        let far = target("fibre last crossing", wgpu::TextureFormat::Depth32Float);
        let near_ids = target("fibre first identity", wgpu::TextureFormat::Rg32Uint);
        let counts = target(
            "fibre cumulative crossings",
            wgpu::TextureFormat::Rgba16Float,
        );
        let records = init(
            d,
            "fibre light coordinates",
            bytemuck::cast_slice(&vec![LightRecord::zeroed(); lights.max(1)]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let capture = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fibre capture depth bounds"),
            layout,
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(&near)),
                entry(1, wgpu::BindingResource::TextureView(&far)),
            ],
        });
        Self {
            plan,
            near,
            far,
            near_ids,
            counts,
            records,
            capture,
            dependencies,
            deformed,
        }
    }
    /// Empty transport for seed captures. The subsequent fibre pass rebuilds it.
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        for view in [&self.near] {
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("seed capture empty fibre transport"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.), store: wgpu::StoreOp::Store }), stencil_ops: None,
                }), timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            }));
        }
    }
    pub(crate) fn read_bindings(&self, first: u32) -> Vec<wgpu::BindGroupEntry<'_>> {
        let mut entries = vec![
            entry(first, self.records.as_entire_binding()),
            entry(first + 1, wgpu::BindingResource::TextureView(&self.near)),
            entry(first + 2, wgpu::BindingResource::TextureView(&self.far)),
        ];
        entries.push(entry(
            first + 3,
            wgpu::BindingResource::TextureView(&self.counts),
        ));
        entries.push(entry(
            first + 4,
            wgpu::BindingResource::TextureView(&self.near_ids),
        ));
        entries
    }
}

pub(crate) struct Slot {
    pub uniforms: Vec<wgpu::Buffer>,
    bundles: Vec<[wgpu::RenderBundle; Stage::ALL.len() * 2]>,
}
impl Slot {
    pub(crate) fn new(
        p: &Storage,
        recorder: &crate::bundles::Recorder<'_>,
        global: impl Fn(&wgpu::Buffer) -> wgpu::BindGroup,
    ) -> anyhow::Result<Self> {
        let capture = crate::bundles::Recorder {
            fibre_capture: Some(&p.capture),
            ..*recorder
        };
        let mut uniforms = Vec::new();
        let mut bundles = Vec::new();
        let d = &recorder.r.device;
        for _ in &p.plan.faces {
            let uniform = init(
                d,
                "fibre light view",
                bytemuck::bytes_of(&Uniform::zeroed()),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let bind = global(&uniform);
            let recorded = (0..Stage::ALL.len() * 2).map(|i| capture.record(
                "fibre geometry",
                crate::bundles::Kind::Fibre(Stage::ALL[i / 2], i % 2 != 0),
                &bind,
                None,
                None,
            )).collect::<anyhow::Result<Vec<_>>>()?;
            bundles.push(recorded.try_into().ok().expect("both variants of every fibre stage"));
            uniforms.push(uniform);
        }
        Ok(Self { uniforms, bundles })
    }
    pub(crate) fn prepare(
        &self,
        p: &Storage,
        s: &InstalledScene,
        objects: &[Object],
        lights: &[Light],
        uniform: Uniform,
        material_records: Option<&wgpu::Buffer>,
        changes: &mut values::ChangeTracker,
        history: &mut values::LightingTracker,
    ) {
        let mut bounds: Option<(Vec3, Vec3)> = None;
        let mut key = Vec::new();
        for draw in &s.draws {
            if !s.materials[draw.material].hair {continue;}
            for object in draw.instances.clone() {
                let o = &objects[object as usize];
                key.extend_from_slice(bytemuck::bytes_of(&o.gpu()));
                let b = shadow_envelope::transformed(o.model, draw.bounds);
                bounds = Some(bounds.map_or(b, |a| (a.0.min(b.0), a.1.max(b.1))));
            }
        }
        let Some(bounds) = bounds else {
            return;
        };
        let records = p.plan.records(lights, bounds);
        key.extend_from_slice(bytemuck::cast_slice(&records));
        if p.dependencies.time || p.deformed {
            key.extend_from_slice(&uniform.scene[0].to_le_bytes());
        }
        if p.dependencies.camera {
            key.extend_from_slice(bytemuck::bytes_of(&uniform.view));
            key.extend_from_slice(bytemuck::bytes_of(&uniform.vp));
        }
        if p.dependencies.object_transforms {
            key.extend_from_slice(bytemuck::cast_slice(&gpu_objects(objects)));
        }
        if let Some(records) = material_records {
            changes.append_buffer_inputs(records, &mut key);
        }
        // GPU-authored geometry/images have no CPU value snapshot here.
        if s.compute.is_some() || p.dependencies.images {
            history.changed(values::LightingKey::Fibre, b"live GPU fibre inputs");
        }
        history.changed(values::LightingKey::Fibre, &key);
        changes.upload(&p.records, 0, bytemuck::cast_slice(&records));
        for (i, face) in p.plan.faces.iter().enumerate() {
            let l = &lights[face.light];
            let record = &records[face.light];
            let mut u = uniform;
            u.light_vp = record.matrices[face.face];
            u.params[0] = record.clip[face.face][0];
            u.params[1] = record.clip[face.face][1];
            u.eye = if l.position[3] == 0. {
                [l.direction[0], l.direction[1], l.direction[2], 0.]
            } else {
                [l.position[0], l.position[1], l.position[2], 1.]
            };
            let selection = s.links.shadowed_by(s.light_plan.nodes[face.light]);
            u.shadow_links = [selection.mask, u32::from(selection.all), 0, 0];
            changes.upload(&self.uniforms[i], 0, bytemuck::bytes_of(&u));
        }
    }
    pub(crate) fn encode(
        &self,
        p: &Storage,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
    ) {
        for (stage_index, stage) in Stage::ALL.into_iter().enumerate() {
            let colors: Vec<_> = if stage.additive() {
                let views = [&p.counts];
                views.into_iter()
                    .map(|view| {
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })
                    })
                    .collect()
            } else if stage == Stage::Near {
                vec![Some(wgpu::RenderPassColorAttachment {
                    view: &p.near_ids,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })]
            } else {
                Vec::new()
            };
            let view = match stage {Stage::Far => &p.far, _ => &p.near};
            let depth = wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: if stage.additive() {
                    None
                } else {
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if stage == Stage::Far { 0. } else { 1. }),
                        store: wgpu::StoreOp::Store,
                    })
                },
                stencil_ops: None,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fibre transport"),
                color_attachments: &colors,
                depth_stencil_attachment: Some(depth),
                timestamp_writes: (stage == Stage::Near || stage == Stage::Accumulate).then_some(
                    wgpu::RenderPassTimestampWrites {
                        query_set: queries,
                        beginning_of_pass_write_index: (stage == Stage::Near)
                            .then_some(timing::query("fibre")),
                        end_of_pass_write_index: (stage == Stage::Accumulate)
                            .then_some(timing::query("fibre") + 1),
                    },
                ),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (i, face) in p.plan.faces.iter().enumerate() {
                let [x, y] = face.origin;
                let side = face.resolution;
                pass.set_viewport(x as f32, y as f32, side as f32, side as f32, 0., 1.);
                pass.set_scissor_rect(x, y, side, side);
                pass.execute_bundles([
                    &self.bundles[i][stage_index * 2],
                    &self.bundles[i][stage_index * 2 + 1],
                ]);
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod hair_fixture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn diffuse_hair_reserves_no_physical_capture_or_exit_tree() -> Result<()> {
        let r = crate::test_renderer()?;
        let fixture = hair_fixture::with_test_hair(vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/fibre-receiver.json"), include_str!("../tests/presets/fibre-receiver-2172aaaf.json"))?);
        for model in ["diffuse", "chiang"] {
            let mut scene = fixture.clone();
            let inputs = scene["materials"]["hair"]["inputs"].as_object_mut().unwrap();
            inputs.insert("model".into(), model.into());
            let installed = r.install(vork_scene::v2::from_test_value(scene)?, Path::new("."))?;
            let physical = model == "chiang";
            // Physical scattering promotes the complete camera view to path
            // transport, which reserves no raster fibre capture for either model.
            assert_eq!(installed.render().settings.integrator.is_path(), physical, "{model:?}");
            assert!(installed.fibre.is_none(), "{model:?}");
            assert_eq!(installed.objects.iter().any(|o| o.material_index[2] != 0), physical, "{model:?}");
        }
        let mut mixed = fixture.clone();
        mixed["materials"]["diffuse"] = mixed["materials"]["hair"].clone();
        mixed["materials"]["diffuse"]["inputs"]["model"] = "diffuse".into();
        let mut diffuse = mixed["nodes"][2].clone();
        diffuse["materials"] = serde_json::json!(["diffuse"]);
        mixed["nodes"].as_array_mut().unwrap().push(diffuse);
        let installed = r.install(vork_scene::v2::from_test_value(mixed)?, Path::new("."))?;
        assert!(installed.render().settings.integrator.is_path(), "one physical strand selects the path owner");
        assert!(installed.fibre.is_none());
        assert_eq!(installed.plan.draws.iter().filter(|d| d.hair).count(), 1);
        assert_eq!(installed.objects.iter().filter(|o| o.material_index[2] != 0).count(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn eviction_respects_shadow_leases_and_rebuilds_fibre() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut fixture = hair_fixture::with_test_hair(vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/fibre-receiver.json"), include_str!("../tests/presets/fibre-receiver-2172aaaf.json"))?);
        fixture["materials"]["hair"]["inputs"]["model"] = "chiang".into();
        fixture["render"]["resolution"] = serde_json::json!([32,32]);
        let scene = r.install(vork_scene::v2::from_test_value(fixture)?, Path::new("."))?;
        // Physical Hair is owned by complete-camera path transport: there is no
        // raster fibre capture for frame bindings or shadow backing to lease.
        assert!(scene.render().settings.integrator.is_path(), "physical Hair uses the complete-camera path owner");
        assert!(scene.fibre.is_none(), "path-owned physical Hair reserves no raster fibre capture");
        let mut frame = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        frame.submit(&scene, &r, 0.)?;
        let before = frame.collect(&r)?.linear_rgba.unwrap();
        assert!(before.iter().flatten().all(|v| v.is_finite()));
        let shadows = scene.shadows();
        // Eviction with live frame bindings, with only the shadow backing, and
        // with neither must leave a scene that rebuilds the same frame.
        r.residency.evict(u64::MAX);
        drop(frame);
        r.residency.evict(u64::MAX);
        drop(shadows);
        r.residency.evict(u64::MAX);
        let mut frame = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        frame.submit(&scene, &r, 0.)?;
        crate::test_linear::assert_linear_eq(&before,&frame.collect(&r)?.linear_rgba.unwrap());
        r.errors.check()
    }

    #[test]
    fn capture_plan_preserves_light_faces_and_checks_allocations() -> Result<()> {
        let limits = wgpu::Limits::default();
        let plan = Plan::new(&[true, false], 64, u64::MAX, &limits)?;
        assert_eq!(plan.faces.len(), 7);
        assert_eq!(plan.side, 192);
        assert_eq!(plan.faces.iter().filter(|f| f.light == 1).count(), 6);
        assert!(Plan::new(&[false], 64, 8, &limits).is_err());
        // A device that admits the live RGBA16F/RG32UI targets must not be
        // refused for the retired four-layer optical-depth texture.
        assert!(Plan::new(&[true], 64, 64*64*8, &limits).is_ok());
        assert!(Plan::new(&[true], 64, 64*64*8-1, &limits).is_err());
        assert!(
            Plan::new(
                &[true],
                limits.max_texture_dimension_2d + 1,
                u64::MAX,
                &limits
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn compiled_fibre_transport_obeys_path_sums_and_absorption_limits() -> Result<()> {
        let r = crate::test_renderer()?;
        let native = medium::Native {
            includes: vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../libraries")],
            ..Default::default()
        };
        let source = r#"
import std.hair_bsdf;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> output;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    if (id.x>=8u) {
        var h=principled_hair(float3(.5),.3,.3,0.,1.55,0.);
        h.absorption_coefficient=float3(id.x==8u ? 1e6 : 0.);
        output[id.x]=float4(hair_forward_at(h,float3(0.,1.,0.),float3(.5,0.,sqrt(.75)),float3(0.,0.,1.)),hair_transport_moments(h,0.).forward.x);
        return;
    }
    HairTransportMoments m={};m.forward=float3(.2,.6,.9);m.backward=float3(.1,.2,.05);
    m.forward_variance=float3(.03);m.backward_variance=float3(.02);
    output[id.x]=float4(hair_bundle_backscatter(m,float(id.x),0.,0.,0.,float3(0.),float3(0.),false),0.);
}"#;
        let entries = [pipelines::buffer_entry(
            0,
            wgpu::ShaderStages::COMPUTE,
            wgpu::BufferBindingType::Storage { read_only: false },
        )];
        let pipeline = medium::test_pipeline(&r, source, &native, &entries, (1, 1, 1))?;
        let output = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("finite fibre returns"),
            size: 10 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("finite fibre returns readback"),
            size: 10 * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[entry(0, output.as_entire_binding())],
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(10, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 10 * 16);
        r.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |v| {
            let _ = tx.send(v);
        });
        r.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        rx.recv()??;
        let mapped = readback.get_mapped_range(..);
        let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        for (count, actual) in values[..8].iter().enumerate() {
            for (channel, (af, ab)) in [(0.2_f64, 0.1_f64), (0.6, 0.2), (0.9, 0.05)]
                .into_iter()
                .enumerate()
            {
                // Independently enumerate one- and three-return paths. Their
                // longitudinal variance grows with each forward/backward event.
                let r = af * af;
                let mut first = 0.;
                let mut first_moment = 0.;
                let mut third = 0.;
                let mut third_moment = 0.;
                for i in 1..=count as i32 {
                    let weight = ab * r.powi(i);
                    first += weight;
                    first_moment += weight * f64::from(i);
                    for j in 0..i {
                        for k in j + 1..=count as i32 {
                            let power = i - j - 1 + k;
                            let weight = ab.powi(3) * r.powi(power);
                            third += weight;
                            third_moment += weight * f64::from(power);
                        }
                    }
                }
                let energy = first + third;
                let variance = if energy == 0. {
                    1.
                } else {
                    (2. * first_moment * 0.03
                        + first * 0.02
                        + 2. * third_moment * 0.03
                        + third * 3. * 0.02)
                        / energy
                };
                let expected = 2. * energy
                    / (std::f64::consts::PI * (2. * std::f64::consts::PI * variance).sqrt());
                let actual = f64::from(actual[channel]);
                assert!(
                    actual.is_finite() && (actual - expected).abs() < 2e-4 * expected.max(0.01),
                    "{count} neighbours channel {channel}: {actual} != {expected}"
                );
            }
        }
        assert!(values[8][..3].iter().all(|v|*v==0.), "an opaque interior cannot transmit the R reflection lobe: {:?}",values[8]);
        assert!(values[8][3]>0., "opaque fibres still scatter reflected light");
        assert!(values[9][..3].iter().all(|v|*v>0. && *v<=1.), "a clear fibre must transmit bounded energy: {:?}",values[9]);
        Ok(())
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    #[test]
    fn local_capture_support_is_finite_through_the_source_plane() {
        let plan=Plan::new(&[false],16,u64::MAX,&wgpu::Limits::default()).unwrap();
        let directions=[Vec3::X,Vec3::NEG_X,Vec3::Y,Vec3::NEG_Y,Vec3::Z,Vec3::NEG_Z];
        for kind in [1.,2.,3.] {
            for scale in [1e-4,1.,1e4] {
                for offset in [Vec3::ZERO,Vec3::new(3.,-2.,4.)*scale] {
                    let light=Light {position:[offset.x,offset.y,offset.z,kind],..Light::zeroed()};
                    let bounds=(offset-Vec3::splat(scale),offset+Vec3::splat(scale));
                    let record=plan.records(&[light],bounds)[0];
                    for (face,axis) in directions.into_iter().enumerate() {
                        assert!(record.matrices[face].is_finite());
                        assert!(record.clip[face][1]>scale);
                        // Forward support reaches arbitrarily close to the source;
                        // behind-source geometry is clipped by homogeneous W.
                        for distance in [0.001,0.5,1.] {
                            let point=offset+axis*(scale*distance);
                            let clip=record.matrices[face]*point.extend(1.);
                            assert!(clip.is_finite() && clip.w>0.);
                            assert!((clip.z/clip.w-0.5).abs()<0.001);
                            let depth=(point-offset).dot(axis)/record.clip[face][1];
                            assert!(depth>0. && depth<1.);
                            let recovered=depth*record.clip[face][1];
                            assert!((recovered-scale*distance).abs()<scale*1e-6);
                        }
                        let behind=record.matrices[face]*(offset-axis*scale).extend(1.);
                        assert!(behind.w<0.);
                    }
                }
            }
        }
    }
}
