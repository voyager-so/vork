//! Hardware ray queries over the installed geometry. Install builds one BLAS per
//! unique static geometry set straight from the shared vertex/index pages;
//! each frame slot owns its TLAS and the BLAS of its deformed streams. The
//! change tracker's object and skin uploads decide when they rebuild. Visibility
//! queries share opaque static hierarchies; identity consumers retain primitives.
//! Fixed-topology evaluated BLASes prefer validated native refits after their first build.
use crate::*;

const BUILD_FLAGS: wgpu::AccelerationStructureFlags =
    wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;

/// The scene TLAS at `first` and the alpha table at `first + 1`.
pub(crate) fn layout_entries(first: u32, stages: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: first,
            visibility: stages,
            ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
            count: None,
        },
        pipelines::buffer_entry(
            first + 1,
            stages,
            wgpu::BufferBindingType::Storage { read_only: true },
        ),
    ]
}

/// AO/GI query outputs: visibility/radiance and the selected world position.
fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
            format, view_dimension: wgpu::TextureViewDimension::D2 }, count: None }
}
/// Ray GI group 0: the shared scene globals (frame.slang), then depth, normals,
/// the AO/visibility output, camera colour, incident output, TLAS and hit table.
/// Group 1 is the scene material table, group 2 the hit buffer array.
pub(crate) fn gi_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let compute = wgpu::ShaderStages::COMPUTE;
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        visibility: compute,
        ..pipelines::texture_entry(binding, wgpu::TextureSampleType::Float { filterable: false }, false)
    };
    let mut entries = pipelines::global_entries();
    entries.extend([texture(10), texture(11), storage_entry(12,wgpu::TextureFormat::Rgba32Float), storage_entry(13,wgpu::TextureFormat::Rgba32Float)]);
    entries.extend(layout_entries(14, compute));
    entries.push(storage_entry(16,wgpu::TextureFormat::Rgba32Float));
    entries.push(pipelines::buffer_entry(17,compute,wgpu::BufferBindingType::Uniform));
    entries.push(pipelines::buffer_entry(18,compute,wgpu::BufferBindingType::Storage {read_only:true}));
    entries.push(texture(19));
    entries.push(wgpu::BindGroupLayoutEntry {
        visibility: compute,
        ..pipelines::texture_entry(21, wgpu::TextureSampleType::Uint, false)
    });
    entries.push(pipelines::buffer_entry(20,compute,wgpu::BufferBindingType::Storage {read_only:true}));
    for e in &mut entries {
        e.visibility = compute;
    }
    entries
}

/// With probes, group 2 reads the irradiance volume at hits (probes.rs).
pub(crate) fn gi_layout(
    d: &wgpu::Device,
    materials: &wgpu::BindGroupLayout,
    probes: bool,
) -> wgpu::PipelineLayout {
    let entries = gi_entries();
    let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("local GI"),
        entries: &entries,
    });
    d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ray GI layout"),
        bind_group_layouts: &[Some(&group), Some(materials), probes.then(|| crate::probes::read_layout(d)).as_ref(), Some(&gi_read_layout(d))],
        immediate_size: 0,
    })
}
pub(crate) fn gi_read_entries()->Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries=vec![hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE)];
    entries.push(pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}));
    entries.push(pipelines::buffer_entry(2,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}));
    entries.push(pipelines::buffer_entry(3,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}));
    entries
}
pub(crate) fn gi_read_layout(d:&wgpu::Device)->wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("GI hit and accumulation buffers"),entries:&gi_read_entries()})
}
/// Group 3 of hit-shading compute kernels: only the hit buffer array (wgpu
/// keeps binding arrays out of groups with uniform buffers; group 2 is the
/// shadow modules' own).
pub(crate) fn hits_layout(d: &wgpu::Device) -> wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hit buffers"),
        entries: &[hit_buffers_entry(0, wgpu::ShaderStages::COMPUTE)],
    })
}

/// Ray reflections that shade unseen hits: the scene globals (frame.slang), then
/// colour, depth, normals, ambient, the output, TLAS and hit table. Group 1 is
/// the scene material table, group 3 the hit geometry and reflection field arrays.
pub(crate) fn reflections_hits_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let compute = wgpu::ShaderStages::COMPUTE;
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        visibility: compute,
        ..pipelines::texture_entry(binding, wgpu::TextureSampleType::Float { filterable: false }, false)
    };
    let mut entries = pipelines::global_entries();
    entries.extend([texture(10), texture(11), texture(12), texture(13)]);
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 14,
        visibility: compute,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2Array,
        },
        count: None,
    });
    entries.extend(layout_entries(15, compute));
    entries.push(pipelines::buffer_entry(17, compute, wgpu::BufferBindingType::Storage {read_only:true}));
    entries.push(pipelines::buffer_entry(18, compute, wgpu::BufferBindingType::Storage {read_only:true}));
    entries.push(pipelines::buffer_entry(19, compute, wgpu::BufferBindingType::Storage {read_only:true}));
    entries.push(pipelines::buffer_entry(20,compute,wgpu::BufferBindingType::Uniform));
    entries.push(pipelines::buffer_entry(21,compute,wgpu::BufferBindingType::Storage {read_only:false}));
    entries.push(pipelines::buffer_entry(22,compute,wgpu::BufferBindingType::Storage {read_only:true}));
    for e in &mut entries {
        e.visibility = compute;
    }
    entries
}
pub(crate) fn reflections_hits_layout(d: &wgpu::Device, materials: &wgpu::BindGroupLayout, probes: bool) -> wgpu::PipelineLayout {
    let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ray reflection hits"),
        entries: &reflections_hits_entries(),
    });
    d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ray reflection hits layout"),
        bind_group_layouts: &[Some(&group), Some(materials), probes.then(|| crate::probes::read_layout(d)).as_ref(), Some(&reflection_read_layout(d))],
        immediate_size: 0,
    })
}

/// Geometry and reflection records share the hit-table group.
pub(crate) fn reflection_read_entries()->Vec<wgpu::BindGroupLayoutEntry> {
    vec![hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE),crate::passes::reflections::record_fields_entry(1,true)]
}
pub(crate) fn reflection_read_layout(d:&wgpu::Device)->wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("reflection hit buffers"),entries:&reflection_read_entries()})
}

/// Raster geometry buffers double as BLAS inputs only in scenes that trace rays.
pub(crate) fn geometry_usage(rays: bool) -> wgpu::BufferUsages {
    if rays {
        wgpu::BufferUsages::BLAS_INPUT
    } else {
        wgpu::BufferUsages::empty()
    }
}

fn intern_buffer(buffers: &mut Vec<HitBuffer>, buffer: HitBuffer) -> u32 {
    buffers.iter().position(|b| *b == buffer).unwrap_or_else(|| {
        buffers.push(buffer);
        buffers.len() - 1
    }) as u32
}

/// Where one draw's ray-hit corners live in the raster buffers (hit.slang):
/// positions, indices and the packed attribute stream of its VertexLayout,
/// addressed through the hit buffer array. No second copy of geometry.
fn hit_record(
    draw: &install::Draw,
    buffers: &mut Vec<HitBuffer>,
    leader: Option<usize>,
    displaced: Option<(usize, u64)>,
) -> [u32; constants::RAY_HIT_RECORD_WORDS] {
    let mut slot = |buffer| intern_buffer(buffers, buffer);
    let layout = draw.layout;
    let deformed = layout.deform;
    let (positions, base, stride) = if let Some((page, offset)) = displaced {
        (slot(HitBuffer::DisplacedPage(page)), (offset / 4) as u32, 10)
    } else {
        match leader {
        Some(leader) => (slot(HitBuffer::Deformed(leader)), 0, (std::mem::size_of::<DeformedVertex>() / 4) as u32),
        None => (slot(HitBuffer::Static(draw.vertices.clone())), (draw.vertex_offset / 4) as u32, (layout.vertex_stride()/4) as u32),
        }
    };
    let indices = slot(HitBuffer::Static(draw.indices.clone()));
    let attributes = slot(if draw.computed || layout.points { HitBuffer::Attributes(leader.expect("evaluated attribute leader")) } else { HitBuffer::Static(draw.attributes.clone()) });
    const NONE: u32 = 0xff;
    // Float offsets within a vertex: deformed normals/tangents follow the pose.
    let (normal, tangent, rest, surface) = if deformed {
        (3, 6, 0, 3)
    } else {
        (0, if layout.tangent { 3 } else { NONE }, NONE, 3 + if layout.tangent { 4 } else { 0 })
    };
    let uv = |set: usize| {
        let packed = layout.uv[set];
        if packed == constants::VERTEX_DEFAULT_UV { NONE } else { surface + 2 * u32::from(packed) }
    };
    let uv_count = layout.uv.iter().filter(|&&p| p != constants::VERTEX_DEFAULT_UV).map(|&p| u32::from(p) + 1).max().unwrap_or(0);
    let color = surface + 2 * uv_count;
    let custom = color + if layout.color { 4 } else { 0 };
    [
        positions,
        base,
        stride,
        indices,
        (draw.index_offset / 4) as u32,
        attributes,
        if layout.points {0} else {(draw.attribute_offset() / 4) as u32},
        (layout.attribute_stride() / 4) as u32,
        normal | tangent << 8 | rest << 16 | u32::from(deformed) << 24 | u32::from(layout.points) << 26,
        uv(0) | uv(1) << 8 | (if layout.color { color } else { NONE }) << 16 | (if layout.custom > 0 { custom } else { NONE }) << 24,
        0, // Primitive offset within the authored draw.
    ]
}
/// One element of the hit buffer array: an installed raster buffer, or a
/// deformation leader's range of the frame slot's deform pool.
#[derive(Clone, PartialEq)]
pub(crate) enum HitBuffer {
    Static(wgpu::Buffer),
    Deformed(usize),
    DisplacedPage(usize),
    Attributes(usize),
}
/// Installed buffers are bound whole by both camera and ray-hit readers.
/// Evaluated ranges have already been admitted by the geometry page planner.
fn admit_static_hit_buffers(limits: &wgpu::Limits, buffers: &[HitBuffer]) -> Result<()> {
    for (index, buffer) in buffers.iter().enumerate() {
        if let HitBuffer::Static(buffer) = buffer {
            crate::gpu_memory::storage_binding(limits, &format!("hit geometry buffer {index}"), buffer.size(), "reduce geometry or instance count")?;
        }
    }
    Ok(())
}
/// Resolve installed hit records to the frame slot's evaluated geometry ranges.
pub(crate) fn resolve_hit_buffers(buffers: &[HitBuffer], deforms: &[Option<EvaluatedGeometry>]) -> Vec<(wgpu::Buffer, u64, u64)> {
    buffers.iter().map(|b| match b {
        HitBuffer::Attributes(i) => { let b=deforms[*i].as_ref().unwrap().attributes.as_ref().unwrap(); (b.clone(),0,b.size()) },
        HitBuffer::Static(buffer) => (buffer.clone(), 0, buffer.size()),
        HitBuffer::DisplacedPage(leader) => {
            let output = &deforms[*leader].as_ref().expect("evaluated page").output;
            (output.buffer.clone(), 0, output.buffer.size())
        }
        HitBuffer::Deformed(leader) => {
            let output = &deforms[*leader].as_ref().expect("deformation leader output").output;
            (output.buffer.clone(), output.offset, output.size)
        }
    }).collect()
}
/// The hit buffer array binding; unused slots stay unbound (partially bound).
pub(crate) fn hit_bindings(buffers: &[(wgpu::Buffer, u64, u64)]) -> Vec<wgpu::BufferBinding<'_>> {
    buffers.iter().map(|(buffer, offset, size)| wgpu::BufferBinding {
        buffer, offset: *offset, size: std::num::NonZeroU64::new(*size),
    }).collect()
}
/// The hit-buffer group shared by compute surface kernels.
pub(crate) fn hits_bind(d: &wgpu::Device, buffers: &[(wgpu::Buffer, u64, u64)]) -> wgpu::BindGroup {
    let bindings = hit_bindings(buffers);
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hit buffers"),
        layout: &hits_layout(d),
        entries: &[entry(0, wgpu::BindingResource::BufferArray(&bindings))],
    })
}
/// Device features the hit buffer array needs (rt::require).
pub(crate) const HIT_BUFFER_FEATURES: wgpu::Features = wgpu::Features::BUFFER_BINDING_ARRAY
    .union(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY)
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    .union(wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY);
/// Path, ray GI, ray refraction and probes shade hit surfaces (hit.slang).
pub(crate) fn traces_hit_surfaces(scene: &Scene) -> bool {
    ray_refraction(scene) || other_hit_surfaces(scene)
}
/// Hit shading still needed after installed materials prove there is no solid glass.
pub(crate) fn other_hit_surfaces(scene: &Scene) -> bool {
    ray_indirect(scene) || scene.render.settings.integrator.reflections() == Some(vork_scene::Reflections::Ray) || scene.render.settings.integrator.is_path()
        || scene.render.settings.integrator.probe_volume().is_some()
}
/// The hit buffer array layout entry (`count` elements, partially bound).
pub(crate) fn hit_buffers_entry(binding: u32, stages: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        count: std::num::NonZeroU32::new(constants::RAY_HIT_BUFFERS),
        ..pipelines::buffer_entry(binding, stages, wgpu::BufferBindingType::Storage { read_only: true })
    }
}

/// Installed triangle-shadow permission, including automatic local lights.
pub(crate) fn traces_shadows(kind: &NodeKind) -> bool {
    kind.casts_shadows()
        && !matches!(kind, NodeKind::Light { shape: vork_scene::LightShape::Environment, .. })
}

// Reserve unknown environment exports and imported lights before compilation.
// Installed consumers refine this reservation against actual evaluated lights.
fn automatic_light_shadows(scene: &Scene) -> bool {
    fn possible(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| (matches!(node.kind,
            NodeKind::Light { .. }) && node.kind.casts_shadows())
            || matches!(node.kind, NodeKind::Light { shape: vork_scene::LightShape::Environment, .. } | NodeKind::Asset { .. })
            || possible(&node.children))
    }
    scene.render.settings.integrator.shadow_method() == vork_scene::ShadowMethod::Auto
        && surface_transport_required(scene) && possible(&scene.nodes)
}
pub(crate) fn installed_shadow_rays(scene: &Scene, nodes: &[EvalNode]) -> bool {
    if scene.render.settings.integrator.shadow_method()!=vork_scene::ShadowMethod::Auto {return ray_shadows(scene);}
    scene.render.settings.integrator.subsurface_method().is_some()
        || (surface_transport_required(scene)
            && nodes.iter().any(|node| traces_shadows(&node.node.kind)))
}

/// Scene options that trace rays, as their scene paths.
fn options(scene: &Scene) -> Vec<String> {
    ray_options(scene, ray_refraction(scene))
}
pub(crate) fn ray_options(scene: &Scene, refraction: bool) -> Vec<String> {
    let mut out = Vec::new();
    if crate::catcher::has_mattes(scene) {out.push("node matte transport (is_holdout/is_shadow_catcher)".into());}
    if !scene.render.settings.integrator.is_path()
        && scene.render.settings.integrator.shadow_method() == vork_scene::ShadowMethod::Ray {
        out.push("ray-traced shadows".into());
    } else if automatic_light_shadows(scene) {
        out.push("automatic light shadows".into());
    }
    if scene.render.settings.integrator.is_path() {
        out.push("path transport".into());
    }
    if scene.render.settings.integrator.subsurface_method().is_some() {
        out.push("spatial subsurface scattering".into());
    }
    if scene.render.settings.integrator.reflections() == Some(vork_scene::Reflections::Ray) {
        out.push("glossy reflections".into());
    }
    if refraction {out.push("glass refraction".into());}
    if ray_indirect(scene) {
        out.push("diffuse indirect lighting".into());
    }
    if scene.render.settings.integrator.probe_volume().is_some() && surface_transport_required(scene) {
        out.push("retained indirect lighting".into());
    }
    out
}
/// Ray-traced diffuse indirect: install keeps hit attributes in the ray table.
pub(crate) fn ray_indirect(scene: &Scene) -> bool {
    scene.render.settings.integrator.indirect().is_some_and(|gi| gi.method == vork_scene::FastGiMethod::Ray)
}

// Groups, cameras and lights cannot receive surface transport. Imported and
// procedural geometry remain required/unknown; never infer absence from views
// or visibility, since captures and secondary rays can see other receivers.
fn surface_transport_required(scene:&Scene)->bool {
    fn any(nodes:&[vork_scene::v2::Node])->bool {
        nodes.iter().any(|node| !matches!(&node.kind,NodeKind::Node{} | NodeKind::Camera{..} | NodeKind::Light{..}) || any(&node.children))
    }
    any(&scene.nodes)
}

/// Automatic diffuse transport is independent of reflection and refraction.
/// Receiver queries integrate diffuse lighting, with retained irradiance supplying
/// further bounces when installed. Screen GI cannot supply offscreen transport.
pub(crate) fn resolve_indirect(scene:&mut Scene,features:wgpu::Features)->Result<()> {
    let required=surface_transport_required(scene);
    // Retained probe support is planned from CPU geometry. Arbitrary computed
    // streams can leave that support; the complete provider evaluates each
    // requested snapshot instead of reusing rest-space masks.
    fn computed_mesh(scene: &Scene, nodes: &[vork_scene::v2::Node]) -> bool {
        nodes.iter().any(|node| matches!(&node.kind, vork_scene::v2::NodeKind::Mesh { geometry, .. }
            if scene.geometry.get(geometry).is_some_and(|g| !matches!(g.topology,vork_scene::v2::Topology::Implicit) && crate::computed_geometry::dynamic(scene, g)))
            || computed_mesh(scene, &node.children))
    }
    if automatic_indirect_requested(scene) && scene.render.settings.integrator.probe_volume().is_some()
        && computed_mesh(scene, &scene.nodes) {
        promote_to_path(scene);
        return Ok(());
    }
    // An aggregate field cannot separate World receiver classes. Preserve the
    // scene's links with the complete per-receiver query provider instead.
    fn linked_world(nodes:&[vork_scene::v2::Node])->bool {
        nodes.iter().any(|n| matches!(&n.kind,NodeKind::Light {shape:vork_scene::LightShape::Environment,illuminates:Some(_),..}) || linked_world(&n.children))
    }
    let linked_world=linked_world(&scene.nodes) || scene.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit));
    let Some(raster)=scene.render.settings.integrator.raster_mut() else {return Ok(())};
    let Some(gi)=raster.fast_gi.as_mut() else {return Ok(())};
    if gi.method!=vork_scene::FastGiMethod::Auto {return Ok(())}
    if !required {raster.fast_gi=None;raster.probes=None;return Ok(())}
    let queries=features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | HIT_BUFFER_FEATURES);
        ensure!(queries,"effect unavailable: diffuse indirect lighting needs complete scene queries; use a ray-query capable adapter, simplify the scene");
        if linked_world { raster.probes=None; }
        // Integrate the receiver hemisphere before displaying retained lighting.
        // Exact receiver-to-probe connections remain cache leak guards, not
        // point-light silhouettes projected directly onto the primary surface.
        gi.method=vork_scene::FastGiMethod::Ray;
    Ok(())
}

/// Keep the call's sample budget and denoise intent when retained transport
/// cannot represent the scene. Never substitute an unoccluded World fill.
pub(crate) fn promote_to_path(scene:&mut Scene) {
    let denoiser=scene.render.settings.integrator.raster().and_then(|r|r.denoiser.clone());
    scene.render.settings.reconstruct=false;
    scene.render.settings.integrator=vork_scene::Integrator::Path(vork_scene::PathIntegrator {denoiser,..Default::default()});
}

/// Only the newly owned effects change the default denoise policy. Existing
/// dry camera/catcher promotions keep the resolved profile's denoiser unchanged.
fn promote_transport_owner(scene:&mut Scene) {
    let denoiser=scene.render.settings.promotion_denoise.unwrap_or(true).then(vork_scene::Denoiser::default);
    if let Some(raster)=scene.render.settings.integrator.raster_mut() {raster.denoiser=denoiser;}
    promote_to_path(scene);
}

pub(crate) fn automatic_indirect_requested(scene:&Scene)->bool {
    scene.render.settings.integrator.raster().is_some_and(|r|r.fast_gi.as_ref().is_some_and(|gi|gi.method==vork_scene::FastGiMethod::Auto))
        && surface_transport_required(scene)
}

/// Conservative preflight before assets/programs are installed. Imported materials
/// and program outputs are unknown here; native OpenPBR uses the normal compiler's
/// transmission capability, including animated and textured weights.
pub(crate) fn may_refract(scene: &Scene) -> bool {
    fn assets(nodes: &[Node]) -> bool {
        nodes.iter().any(|n| matches!(n.kind, NodeKind::Asset { .. }) || assets(&n.children))
    }
    if assets(&scene.nodes) {return true;}
    let Ok(animated) = materials::animated_inputs(scene) else {return true;};
    scene.materials.iter().any(|(name, material)| {
        if material.program != "std/openpbr" {return true;}
        let inputs = animated.get(name).cloned().unwrap_or_default();
        materials::openpbr_values(scene, Some(name), &inputs, |_| Ok(0))
            .map_or(true, |m|m.transmissive)
    })
}

/// Device creation precedes compiled program capabilities. Reserve queries for
/// possible solid SSS or RGB connections; installation refines unknown programs.
pub(crate) fn possible_transport_queries(scene: &Scene) -> bool {
    let Ok(animated)=materials::animated_inputs(scene) else {return true;};
    for (name,material) in &scene.materials {
        if material.program!="std/openpbr" {return true;}
        let inputs=animated.get(name).cloned().unwrap_or_default();
        let Ok(record)=materials::openpbr_values(scene,Some(name),&inputs,|_|Ok(0)) else {return true;};
        if record.spatial_subsurface() {return true;}
    }
    may_refract(scene)
}

/// Thin and solid camera interfaces both need scene boundary transport.
pub(crate) fn automatic_refraction(scene: &Scene) -> bool {
    scene.render.settings.integrator.raster().is_some_and(|r|r.refraction == vork_scene::Refraction::Auto) && may_refract(scene)
}
pub(crate) fn automatic_refraction_supported(_device_type: wgpu::DeviceType, features: wgpu::Features) -> bool {
    // Complete interface transport depends on capabilities, including software
    // queries; a CPU classification cannot strand it after public overrides retire.
    features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | HIT_BUFFER_FEATURES)
}

/// Resolve the automatic technique before requirements, prewarm and installation.
/// Explicit ray remains an error on unsupported devices; screen is always retained.
pub(crate) fn resolve_refraction(scene: &mut Scene, device_type: wgpu::DeviceType, features: wgpu::Features) {
    if !scene.render.settings.integrator.raster().is_some_and(|r| r.refraction == vork_scene::Refraction::Auto) {return;}
    let transmission = may_refract(scene);
    let ray = automatic_refraction(scene) && automatic_refraction_supported(device_type, features);
    if let Some(raster) = scene.render.settings.integrator.raster_mut() {
        if raster.refraction == vork_scene::Refraction::Auto {
            raster.refraction = if !transmission {vork_scene::Refraction::Environment}
                else if ray {vork_scene::Refraction::Ray} else {vork_scene::Refraction::Screen};
        }
    }
}

pub(crate) fn ray_refraction(scene: &Scene) -> bool {
    scene.render.settings.integrator.raster().is_some_and(|r|r.refraction == vork_scene::Refraction::Ray)
}

/// Resolve the authored automatic reflection technique once, before planning.
/// Adapter admission already guarantees ray queries; scene absence disables work.
pub(crate) fn resolve_reflections(scene: &mut Scene) {
    let required=surface_transport_required(scene);
    if let Some(raster) = scene.render.settings.integrator.raster_mut()
        && raster.reflections == vork_scene::Reflections::Auto
    {
        if !required {
            raster.reflections=vork_scene::Reflections::Environment;
            return;
        }
        // Neither World nor the directional capture currently certifies the
        // filtered lobe's local visibility/radiance residual. Unknown error
        // cannot qualify an approximation at any roughness, including one.
        // Keep sampling instead of turning roughness into a correctness limit.
        raster.reflections = vork_scene::Reflections::Ray;
    }
}

/// Automatic transport requests the optional query capability during startup;
/// only the resolved scene determines whether installation needs an AS.
pub(crate) fn automatic_reflections_requested(scene: &Scene) -> bool {
    scene.render.settings.integrator.reflections() == Some(vork_scene::Reflections::Auto)
        && surface_transport_required(scene)
}

/// Any ray technique: the install builds acceleration structures.
pub(crate) fn uses_rays(scene: &Scene) -> bool {
    installed_rays(scene, ray_refraction(scene))
}
/// Refine conservative preflight after imported/program material capabilities exist.
pub(crate) fn installed_rays(scene: &Scene, refraction: bool) -> bool {
    !ray_options(scene, refraction).is_empty()
}
/// Ray AO is required when authored or keyed strength can activate it.

/// Ray-traced light shadows: the forward shader variant reads the TLAS.
pub(crate) fn ray_shadows(scene: &Scene) -> bool {
    automatic_light_shadows(scene)
        || scene.render.settings.integrator.shadow_method() == vork_scene::ShadowMethod::Ray
        || scene.render.settings.integrator.subsurface_method().is_some()
}

/// CPU rendering is explicit; never fall back from a GPU to software.
pub(crate) fn cpu_requested() -> Result<bool> {
    Ok(crate::selected_gpu()? == Some(crate::GpuSelection::Cpu))
}

pub(crate) fn adapter_help() -> &'static str {
    if cfg!(target_os = "macos") {
        "No supported ray-tracing GPU; Apple Silicon is required on macOS."
    } else if cfg!(target_os = "windows") {
        "No supported ray-tracing GPU. For slow CPU rendering, install a Vulkan loader and a recent Mesa lavapipe Vulkan driver (with its ICD manifest), then set VORK_GPU=cpu (or --gpu cpu). Windows CPU rendering has not been qualified."
    } else {
        "No supported ray-tracing GPU (NVIDIA, or AMD on a Vulkan driver with ray queries such as Mesa RADV). For slow CPU rendering, install recent Mesa Vulkan drivers (mesa-vulkan-drivers on Debian/Ubuntu; mesa-vulkan-drivers or vulkan-swrast elsewhere) and set VORK_GPU=cpu (or --gpu cpu)."
    }
}

pub(crate) fn require_adapter(info: &wgpu::AdapterInfo, features: wgpu::Features) -> Result<()> {
    let software = cfg!(any(test, feature = "test-support")) || cpu_requested()?;
    ensure!(features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | HIT_BUFFER_FEATURES)
        && (software || info.device_type != wgpu::DeviceType::Cpu),
        "{} Adapter {} ({:?}) requires ray queries and storage buffer arrays; missing {:?}.",
        adapter_help(), info.name, info.backend,
        (wgpu::Features::EXPERIMENTAL_RAY_QUERY | HIT_BUFFER_FEATURES) - features);
    Ok(())
}

/// Ray techniques are explicit scene choices; a device without ray queries is an
/// error naming the option, never a silent fallback to another technique.
pub(crate) fn require(r: &Renderer, scene: &Scene) -> Result<()> {
    let supported = r
        .device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    let adapter = format!("{} ({:?})", r.adapter_info.name, r.adapter_info.backend);
    if let Some(message) = missing(scene, supported, &adapter) {
        bail!(message);
    }
    ensure!(
        !traces_hit_surfaces(scene) || r.device.features().contains(HIT_BUFFER_FEATURES),
        "{} reads hit geometry through storage buffer arrays ({HIT_BUFFER_FEATURES:?}); adapter {adapter} lacks them",
        options(scene).join(", ")
    );
    Ok(())
}
fn missing(scene: &Scene, supported: bool, adapter: &str) -> Option<String> {
    let chosen = options(scene);
    (!supported && !chosen.is_empty()).then(|| {
        let effect=if ray_indirect(scene) {"diffuse indirect lighting; "} else {""};
        format!(
            "effect unavailable: {effect}{} needs hardware ray queries (wgpu EXPERIMENTAL_RAY_QUERY); adapter {adapter} has none; use a ray-query capable adapter",
            chosen.join(", ")
        )
    })
}

fn triangles(
    alpha_tested: bool,
    vertex_count: u32,
    index_count: u32,
) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(index_count),
        // Alpha-tested draws yield candidates the shaders test against the
        // material cutoff; every other caster is opaque.
        flags: if alpha_tested {
            wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION
        } else {
            wgpu::AccelerationStructureGeometryFlags::OPAQUE
        },
    }
}
fn blas(
    d: &wgpu::Device,
    label: &str,
    size: &wgpu::BlasTriangleGeometrySizeDescriptor,
) -> wgpu::Blas {
    blas_geometries(d, label, std::slice::from_ref(size))
}
fn blas_geometries(d: &wgpu::Device, label: &str, sizes: &[wgpu::BlasTriangleGeometrySizeDescriptor]) -> wgpu::Blas {
    d.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some(label),
            flags: BUILD_FLAGS,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: sizes.to_vec(),
        },
    )
}
fn geometry<'a>(
    size: &'a wgpu::BlasTriangleGeometrySizeDescriptor,
    vertices: &'a wgpu::Buffer,
    first_vertex: u32,
    stride: u64,
    indices: &'a wgpu::Buffer,
    first_index: u32,
) -> wgpu::BlasGeometries<'a> {
    wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
        size,
        vertex_buffer: vertices,
        first_vertex,
        vertex_stride: stride,
        index_buffer: Some(indices),
        first_index: Some(first_index),
        transform_buffer: None,
        transform_buffer_offset: None,
    }])
}

/// Install-owned static BLAS, shared by every frame slot, and the ray table:
/// RAY_SLOT_WORDS per TLAS instance (first corner word, or NO_ALPHA when opaque;
/// the material record; first hit-triangle word, or NO_ALPHA; the first slot
/// points to the optional shadow-containment table), then TEXCOORD_0,
/// TEXCOORD_1 and vertex-colour alpha of each index of the alpha-tested draws
/// (five words per corner), then the hit triangles of `indirect.method: ray` scenes.
pub(crate) struct Rays {
    /// All installed BLAS geometry accepts opaque intersections directly.
    pub(crate) opaque: bool,
    /// Offset of the path emitter topology in the shared ray table.
    pub(crate) emitters: u32,
    pub(crate) probe_emitters: bool,
    pub(crate) emitter_weights: Option<crate::emitter_weights::Plan>,
    /// Offset of per-instance solid-boundary capability descriptors.
    pub(crate) boundaries: u32,
    /// Per-object solid identity for raster camera interfaces.
    pub(crate) object_boundaries: u32,
    /// Offset of per-instance authored mesh identities for spatial BSSRDFs.
    pub(crate) subsurface: u32,
    /// Offset of per-solid records (constants::PATH_SOLID_WORDS each).
    pub(crate) occupancy_solids: u32,
    /// Offset of the path camera film inverse CDF in the shared ray table.
    pub(crate) film_filter: u32,
    /// Per draw; None for deformed draws, whose BLAS are frame-private.
    blas: Vec<Option<wgpu::Blas>>,
    visibility_groups: Vec<VisibilityGroup>,
    points: Vec<Option<PointShape>>,
    point_rows: Vec<Option<u32>>,
    /// Dense TLAS slots represented by shared opaque visibility hierarchies.
    #[cfg(test)]
    grouped_instances: Vec<bool>,
    /// Ray topology; opacity headers use this same CandidateInstanceIndex order.
    pub(crate) instances: Vec<(usize,u32)>,
    /// Every frame slot uses the installed topology and descriptor ABI.
    placement_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    /// Per draw: candidates are alpha-tested (non-opaque BLAS geometry).
    alpha_tested: Vec<bool>,
    shadow_solids: Vec<bool>,
    shadow_bounds: Option<crate::shadow_containment::Plan>,
    pub(crate) alpha: wgpu::Buffer,
    /// The hit buffer array's elements (hit records index them).
    pub(crate) hit_buffers: Vec<HitBuffer>,
    pub(crate) implicit: [u32;4],
}
/// An immutable query snapshot for compute. Every potentially selected target must
/// be static; simulation writers cannot create an implicit read/write AS cycle.
pub(crate) struct ComputeSnapshot {
    pub objects: wgpu::Buffer,
    pub rays: Rays,
    pub tlas: wgpu::Tlas,
}
impl ComputeSnapshot {
    pub(crate) fn new(r:&Renderer,filters:&[vork_slang::QueryFilter],source:&vork_scene::v2::Scene,nodes:&[install::EvalNode],roots:&[install::MeshRoot],draws:&[install::Draw],objects:&[Object],materials:&[Material],integrator:&vork_scene::Integrator)->Result<Self> {
        let mut selected=Vec::new();
        let solids=crate::passes::path::solids(draws,roots,materials)?;
        let absorbing:std::collections::BTreeSet<_>=draws.iter().filter(|d|d.implicit.is_none() && materials[d.material].solid_absorption)
            .flat_map(|d|d.instances.clone()).filter_map(|object|solids.owners.get(object as usize).copied().filter(|&id|id!=u32::MAX)).collect();
        for (draw_id,draw) in draws.iter().enumerate() {
            for object in draw.instances.clone() {
                let record=&objects[object as usize];
                let public=record.opacity[1] as u32;
                // An animated mask may enter any category; only exact identity
                // proves that a changing object is outside the read set.
                let reads=filters.iter().any(|f|f.mask!=0 && f.object.is_none_or(|id|id==public));
                if !reads {continue;}
                let node=&nodes[roots[object as usize].node];
                ensure!(!node.dynamic && !draw.evaluated() && !crate::compute::places_instances(source,&node.node),
                    "compute trace requires immutable target geometry; object {public} can change or depends on compute; select a static collider with object_id");
                ensure!(draw.implicit.is_none(),"compute trace does not yet support implicit target geometry");
                // Match transport's installed mask, including its auxiliary
                // opaque/solid categories, without evaluating a hit material.
                let determinant=record.model.determinant();
                let mut mask=if determinant==0. || !determinant.is_finite() || !record.model.is_finite() {0} else {record.metadata[1]};
                if !materials[draw.material].transparent && mask&constants::RAY_DIFFUSE!=0 {mask|=constants::RAY_GI_OCCLUDER;}
                if solids.owners.get(object as usize).is_some_and(|id|absorbing.contains(id)) && mask&constants::RAY_SHADOW!=0 {mask|=constants::RAY_SHADOW_SOLID;}
                if filters.iter().any(|f|f.mask as u32&mask!=0 && f.object.is_none_or(|id|id==public)) {
                    selected.push((draw_id,object));
                }
            }
        }
        let rays=Rays::new(r,draws,&selected,objects,&[],true,materials,integrator,&[],&[],&[],&[],false)?;
        let packed=gpu_objects(objects);
        let object_buffer=init(&r.device,"compute query objects",bytemuck::cast_slice(&packed),wgpu::BufferUsages::STORAGE);
        let tlas=r.device.create_tlas(&wgpu::CreateTlasDescriptor {label:Some("compute query snapshot"),max_instances:rays.instances.len().max(1) as u32,flags:BUILD_FLAGS,update_mode:wgpu::AccelerationStructureUpdateMode::Build});
        let mut encoder=r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label:Some("compute query snapshot build")});
        if rays.instances.is_empty() {
            encoder.build_acceleration_structures(std::iter::empty(),std::iter::once(&tlas));
        } else {
            let instances:Vec<_>=rays.instances.iter().enumerate().map(|(slot,&(draw,object))|(object,!materials[draws[draw].material].transparent,true,rays.shadow_solids[slot],rays.blas[draw].as_ref().expect("static query BLAS").clone())).collect();
            let poses:Vec<_>=rays.instances.iter().zip(&rays.point_rows).map(|(&(draw,_),row)|row.map(|row|(rays.points[draw].as_ref().unwrap().rest_poses.clone(),row*8+4))).collect();
            let placement=crate::gpu_tlas::Placement::with_points(r,&object_buffer,&instances,&rays.placement_pipeline,&poses)?;
            placement.encode(&mut encoder,&tlas, false);
        }
        crate::command_finish::Segments::submit(&r.queue,encoder);
        Ok(Self {objects:object_buffer,rays,tlas})
    }
}

/// Visibility-only opaque geometry needs neither primitive material IDs nor hit
/// attributes. Static instances with identical transforms, masks, groups and material sidedness can
/// share one triangle hierarchy without changing the rays' intersection space.
struct VisibilityGroup { slot: usize, blas: wgpu::Blas }

/// Shared immutable shape for compact computed point placements. The ray hit
/// record retains the original draw/point identity independently of TLAS slots.
struct PointShape {
    blas: wgpu::Blas,
    count: u32,
    indices: u32,
    rest_poses: wgpu::Buffer,
}
fn point_shapes(r: &Renderer, draws: &[install::Draw]) -> Result<Vec<Option<PointShape>>> {
    let mut shared=std::collections::HashMap::new();
    draws.iter().map(|draw| {
        let Some(mesh)=draw.exit_geometry.as_ref().filter(|mesh|!mesh.point_centers.is_empty()) else {return Ok(None)};
        let count=mesh.point_centers.len();
        ensure!(mesh.vertices.len()%count==0 && mesh.indices.len()%count==0,"invalid repeated point topology");
        let vertices=mesh.vertices.len()/count;
        let indices=mesh.indices.len()/count;
        let positions:Vec<_>=mesh.vertices[..vertices].iter().map(|v|v.NORMAL).collect();
        let connectivity=&mesh.indices[..indices];
        let mut hash=blake3::Hasher::new();hash.update(bytemuck::cast_slice(&positions));hash.update(bytemuck::cast_slice(connectivity));hash.update(&[u8::from(draw.ray_corners.is_some())]);
        let key=hash.finalize();
        let built=if let Some(built)=shared.get(&key) {wgpu::Blas::clone(built)} else {
            let size=triangles(draw.ray_corners.is_some(),vertices.try_into()?,indices.try_into()?);
            let built=blas(&r.device,"shared point shape BLAS",&size);
            let usage=wgpu::BufferUsages::BLAS_INPUT;
            let positions=init(&r.device,"shared point shape positions",bytemuck::cast_slice(&positions),usage);
            let connectivity=init(&r.device,"shared point shape indices",bytemuck::cast_slice(connectivity),usage);
            let mut encoder=r.device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures([&wgpu::BlasBuildEntry {blas:&built,geometry:geometry(&size,&positions,0,12,&connectivity,0)}],std::iter::empty());
            crate::command_finish::Segments::submit(&r.queue,encoder);
            shared.insert(key,built.clone());built
        };
        ensure!(mesh.point_widths.len()==count,"point widths lost during shape installation");
        let poses:Vec<_>=mesh.point_centers.iter().zip(&mesh.point_widths).flat_map(|(center,width)|[[center[0],center[1],center[2],*width];2]).collect();
        let rest_poses=init(&r.device,"authored point endpoints",bytemuck::cast_slice(&poses),wgpu::BufferUsages::STORAGE);
        Ok(Some(PointShape {blas:built,count:count.try_into()?,indices:indices.try_into()?,rest_poses}))
    }).collect()
}


fn checked_placement_count(counts:impl IntoIterator<Item=u32>,limit:u32)->Result<u32> {
    let mut total=0u32;
    for count in counts {total=total.checked_add(count).filter(|&n|n<=limit).with_context(|| crate::gpu_memory::limit_message("point placements in TLAS", format_args!("{} instances",u64::from(total)+u64::from(count)), format_args!("{limit} instances"), "reduce point pool size or instance count"))?;}
    Ok(total)
}

/// Material bindings are final only after geometry reuse and primitive groups.
/// Retain UV/alpha corners here so an opaque first binding cannot make later
/// transparent bindings opaque to ray queries. Geometry buffers remain shared.
pub(crate) fn prepare_coverage(draws: &mut [install::Draw], materials: &[Material]) -> Result<()> {
    let mut shared = std::collections::HashMap::new();
    for draw in draws {
        let material = &materials[draw.material];
        if !material.surface_coverage && !material.transmissive {
            draw.ray_corners = None;
            continue;
        }
        let geometry = draw.exit_geometry.as_ref().context("ray opacity needs retained geometry")?;
        let key = (std::sync::Arc::as_ptr(geometry), draw.primitive_start, draw.count);
        draw.ray_corners = Some(shared.entry(key).or_insert_with(|| {
            let start = draw.primitive_start as usize * 3;
            std::sync::Arc::new(geometry.indices[start..start + draw.count as usize].iter().map(|&i| {
                let v = &geometry.vertices[i as usize];
                [v.TEXCOORD_0[0], v.TEXCOORD_0[1], v.TEXCOORD_1[0], v.TEXCOORD_1[1], v.COLOR_0[3]]
            }).collect::<Vec<_>>())
        }).clone());
    }
    Ok(())
}
const NO_ALPHA: u32 = u32::MAX;
/// The decoupled camera's draw table (decoupled.slang CameraGeometry): per draw,
/// RAY_SLOT_WORDS words whose third addresses its hit record, then the records.
/// Camera slots read their triangles through the records ray hits use, with
/// the draw index in place of a TLAS instance.
pub(crate) fn draw_table(r: &Renderer, draws: &[install::Draw]) -> Result<(wgpu::Buffer, Vec<HitBuffer>)> {
    let (displaced, _) = crate::displacement::page_layout(draws, &r.device.limits())?;
    let slot_words = constants::RAY_SLOT_WORDS;
    // Exit-tree buffer index, reserved, hit record; one slot even without draws.
    let mut words = vec![NO_ALPHA; draws.len().max(1) * slot_words];
    let mut buffers = Vec::new();
    for (i, draw) in draws.iter().enumerate() {
        words[slot_words * i + 2] = words.len() as u32;
        words.extend(hit_record(draw, &mut buffers, draw.evaluation_leader().or_else(||draw.computed.then_some(i)), displaced[i]));
        if let Some(field) = &draw.implicit {
            words[slot_words * i + 1] = words.len() as u32;
            words.extend(field.dims);
            words.push(field.spans.len() as u32);
            for (buffer, span) in &field.spans {
                let slot = intern_buffer(&mut buffers, HitBuffer::Static(buffer.clone()));
                words.extend([slot, span.offset, span.first, span.count]);
            }
            words.push(intern_buffer(&mut buffers,HitBuffer::Static(field.acceleration.buffer.clone())));
        }
        if let Some(tree) = &draw.exit_bvh {
            words[slot_words * i] = intern_buffer(&mut buffers, HitBuffer::Static(tree.clone()));
        }
    }
    ensure!(
        buffers.len() <= constants::RAY_HIT_BUFFERS as usize,
        "camera hits address {} geometry buffers; the hit buffer array holds {}",
        buffers.len(),
        constants::RAY_HIT_BUFFERS
    );
    admit_static_hit_buffers(&r.device.limits(), &buffers)?;
    crate::gpu_memory::storage_binding(&r.device.limits(), "camera draw table", words.len() as u64 * 4, "reduce geometry or instance count")?;
    let table = init(&r.device, "decoupled draw table", bytemuck::cast_slice(&words), wgpu::BufferUsages::STORAGE);
    // A binding array needs one element even without draws.
    if buffers.is_empty() {
        buffers.push(HitBuffer::Static(table.clone()));
    }
    Ok((table, buffers))
}
fn alpha_table(
    r: &Renderer,
    draws: &[install::Draw],
    instances: &[(usize, u32)],
    hits: bool,
    point_rows: &[Option<u32>],
    points: &[Option<PointShape>],
    path: bool,
    buffers: &mut Vec<HitBuffer>,
    emitters: &[u32],
    implicit_sources: &[u32],
    boundary_ids: &[u32],
    subsurface_ids: &[u32],
    solids: &[crate::passes::path::PathSolid],
    shadow_solids: &[bool],
    solid_owners: &[u32],
    shadow_bounds: Option<&mut crate::shadow_containment::Plan>,
) -> Result<(wgpu::Buffer, u32, u32, u32, u32, u32, u32, [u32;4])> {
    let (displaced, _) = crate::displacement::page_layout(draws, &r.device.limits())?;
    let slot_words = constants::RAY_SLOT_WORDS;
    let mut words: Vec<u32> = instances
        .iter()
        .flat_map(|&(draw, _)| [NO_ALPHA, draws[draw].material as u32, NO_ALPHA, NO_ALPHA])
        .collect();
    words.resize(words.len().max(slot_words), NO_ALPHA);
    let mut first = std::collections::HashMap::new();
    for (slot, &(draw, _)) in instances.iter().enumerate() {
        let Some(corners) = &draws[draw].ray_corners else {
            continue;
        };
        let start = *first
            .entry(std::sync::Arc::as_ptr(corners))
            .or_insert_with(|| words.len() as u32);
        if start == words.len() as u32 {
            words.extend(corners.iter().flatten().map(|v| v.to_bits()));
        }
        words[slot_words * slot] = start + point_rows[slot].map_or(0,|row|row*points[draw].as_ref().unwrap().indices*5);
    }
    // Hit records, one per draw: aliases of a deformation share its leader's pose.
    let mut records = std::collections::HashMap::new();
    for (slot, &(draw, _)) in instances.iter().enumerate().filter(|_| hits) {
        let start = *records.entry((draw,point_rows[slot])).or_insert_with(|| {
            let start = words.len() as u32;
            let leader = draws[draw].evaluation_leader().or_else(||draws[draw].computed.then_some(draw));
            let mut record=hit_record(&draws[draw], buffers, leader, displaced[draw]);
            if let Some(row)=point_rows[slot] {
                let first=row*points[draw].as_ref().unwrap().indices;
                record[4]+=first;
                record[8]|=1<<25;
                record[10]=first/3;
            }
            words.extend(record);
            start
        });
        words[slot_words * slot + 2] = start;
    }
    ensure!(
        buffers.len() <= constants::RAY_HIT_BUFFERS as usize,
        "ray hits address {} geometry buffers; the hit buffer array holds {}",
        buffers.len(),
        constants::RAY_HIT_BUFFERS
    );
    let implicit_offset=words.len() as u32;
    let fields:Vec<_>=instances.iter().enumerate().filter(|(_, (d,_))|draws[*d].implicit.is_some()).collect();
    words.resize(words.len()+fields.len()*3,0);
    for (row,&(instance,&(draw,object))) in fields.iter().enumerate() {
        let field=draws[draw].implicit.as_ref().unwrap();
        let descriptor=words.len() as u32;
        words[implicit_offset as usize+row*3..implicit_offset as usize+row*3+3].copy_from_slice(&[instance as u32,object,descriptor]);
        words.extend(field.dims);
        words.push(field.spans.len() as u32);
        for (buffer,span) in &field.spans {
            let slot=intern_buffer(buffers,HitBuffer::Static(buffer.clone()));
            words.extend([slot,span.offset,span.first,span.count]);
        }
        words.push(intern_buffer(buffers,HitBuffer::Static(field.acceleration.buffer.clone())));
    }
    admit_static_hit_buffers(&r.device.limits(), buffers)?;
    ensure!(buffers.len()<=constants::RAY_HIT_BUFFERS as usize,"implicit grids exceed hit-buffer capacity");
    let implicit=[implicit_offset,fields.len() as u32,words.len() as u32,implicit_sources.len() as u32];
    words.extend_from_slice(implicit_sources);
    let emitter_offset = words.len() as u32;
    words.extend_from_slice(emitters);
    let boundary_offset = words.len() as u32;
    if !boundary_ids.is_empty() {
        words.extend(instances.iter().map(|&(_, object)| boundary_ids[object as usize]));
    }
    let object_boundary_offset=words.len() as u32;
    words.extend_from_slice(boundary_ids);
    let subsurface_offset=words.len() as u32;
    if !subsurface_ids.is_empty() {
        words.extend(instances.iter().map(|&(_,object)|subsurface_ids[object as usize]));
    }
    if !solids.is_empty() {words.push(solids.len() as u32);}
    let bounds_offset = words.len() as u32;
    words.extend(solids.iter().flat_map(|s| -> [u32; constants::PATH_SOLID_WORDS] {
        [s.lo.x.to_bits(), s.lo.y.to_bits(), s.lo.z.to_bits(), s.hi.x.to_bits(), s.hi.y.to_bits(), s.hi.z.to_bits(), s.row]
    }));
    if solids.len()<=8 {
        words.extend(solids.iter().flat_map(|s|s.contact_bounds.0.to_array().into_iter().chain(s.contact_bounds.1.to_array())).map(f32::to_bits));
    }
    // The first slot publishes only potentially absorbing solid identities.
    // Clear/depth-zero interfaces need no occupancy query beyond an emitter.
    let eligible=instances.iter().zip(shadow_solids).filter_map(|(&(_,object),&absorbing)| {
        let solid=solid_owners.get(object as usize).copied().unwrap_or(NO_ALPHA);
        (absorbing && solid!=NO_ALPHA).then_some(solid)
    }).collect::<std::collections::BTreeSet<_>>();
    if !eligible.is_empty() {
        let identities=words.len() as u32;
        words.extend(instances.iter().map(|&(_,object)|solid_owners.get(object as usize).copied().unwrap_or(NO_ALPHA)));
        let unbounded:Vec<_>=eligible.into_iter().filter(|&id|solids[id as usize].shadow_bounds.is_none()).collect();
        let (buffer,count)=if let Some(plan)=shadow_bounds {
            plan.slot=intern_buffer(buffers,HitBuffer::Static(plan.template.clone()));
            (plan.slot,plan.count)
        } else {(NO_ALPHA,0)};
        ensure!(buffers.len()<=constants::RAY_HIT_BUFFERS as usize,"shadow bounds exceed ray hit-buffer capacity");
        let table=words.len() as u32;
        words.extend([unbounded.len() as u32,bounds_offset,identities,buffer,count]);
        words.extend(unbounded);
        words[constants::RAY_SHADOW_TABLE_WORD]=table;
    }
    let film_filter = words.len() as u32;
    if path { words.extend(crate::sampling::path_filter_table().iter().map(|v| v.to_bits())); }
    let bytes = words.len() as u64 * 4;
    let limits = r.device.limits();
    ensure!(
        bytes <= u64::from(limits.max_storage_buffer_binding_size)
            && bytes <= limits.max_buffer_size,
        "{}", crate::gpu_memory::limit_message("ray table", format_args!("{bytes} bytes"), format_args!("storage binding={} bytes, buffer={} bytes",limits.max_storage_buffer_binding_size,limits.max_buffer_size), "reduce geometry and light count or choose --profile std/draft for a lower sample budget")
    );
    let table = init(&r.device, "ray table", bytemuck::cast_slice(&words), wgpu::BufferUsages::STORAGE);
    if buffers.is_empty() {
        buffers.push(HitBuffer::Static(table.clone()));
    }
    Ok((table, emitter_offset, boundary_offset, subsurface_offset, bounds_offset, film_filter, object_boundary_offset, implicit))
}
// Intern geometry sets independently of their instance transforms. Repeated
// opaque assemblies retain shared BLAS instead of duplicating geometry per instance.
fn geometry_key(draw: &install::Draw) -> (wgpu::Buffer,wgpu::Buffer,u64,u64,u32,u32,u64,bool) {
    (draw.vertices.clone(),draw.indices.clone(),draw.vertex_offset,draw.index_offset,
        draw.count,draw.vertex_count,draw.layout.vertex_stride(),draw.ray_corners.is_some())
}
/// A shared position stream still needs separate BLAS for distinct topology or
/// coverage flags. Static geometry retains its installed ownership separately.
fn evaluated_owners(draws: &[install::Draw]) -> Vec<usize> {
    let mut geometries = std::collections::HashMap::new();
    draws.iter().enumerate().map(|(i, draw)| {
        if !draw.evaluated() { return i; }
        *geometries.entry((draw.evaluation_leader().unwrap_or(i), geometry_key(draw))).or_insert(i)
    }).collect()
}
fn triangles_size(draw: &install::Draw) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    triangles(draw.ray_corners.is_some(),draw.vertex_count,draw.count)
}
impl Rays {
    pub(crate) fn new(
        r: &Renderer,
        draws: &[install::Draw],
        instances: &[(usize, u32)],
        objects: &[Object],
        static_casters: &[(usize, u32)],
        hits: bool,
        materials: &[Material],
        integrator: &vork_scene::Integrator,
        boundary_ids: &[u32],
        solid_owners: &[u32],
        subsurface_ids: &[u32],
        solids: &[crate::passes::path::PathSolid],
        medium_sources: bool,
    ) -> Result<Self> {
        let path=integrator.is_path();
        ensure!(
            objects.len() as u64 <= u64::from(constants::RAY_MAX_INSTANCE_INDEX) + 1,
            "{} object instances exceed the 24-bit ray instance index", objects.len()
        );
        let d = &r.device;
        let points=point_shapes(r,draws)?;
        let mut pending: Vec<(Vec<(usize,wgpu::BlasTriangleGeometrySizeDescriptor)>,wgpu::Blas)> = Vec::new();
        let mut unique = std::collections::HashMap::new();
        let mut visibility_groups = Vec::new();
        let mut grouped_instances = vec![false;instances.len()];
        if !hits && integrator.subsurface_method().is_none() {
            let eligible: std::collections::HashSet<_> = static_casters.iter().copied().collect();
            let mut groups = std::collections::BTreeMap::new();
            for (slot,&(draw,object)) in instances.iter().enumerate() {
                let source=&draws[draw];
                if !eligible.contains(&(draw,object)) || source.evaluated() || source.ray_corners.is_some() || points[draw].is_some() {continue;}
                let value=&objects[object as usize];
                let mask=if collapsed(value.model) {0} else {value.metadata[1]};
                let key=(value.model.to_cols_array().map(f32::to_bits),value.metadata[0],mask,!materials[source.material].transparent,materials[source.material].double_sided);
                groups.entry(key).or_insert_with(Vec::new).push((slot,draw,object));
            }
            let limits=d.limits();
            let primitive_limit=u64::from(limits.max_blas_primitive_count).min(constants::RAY_BLAS_BATCH_TRIANGLES);
            for members in groups.values() {
                let mut start=0;
                while start<members.len() {
                    let mut end=start;let mut triangles=0u64;
                    while end<members.len() {
                        let count=u64::from(draws[members[end].1].count/3);
                        if end>start && ((end-start) as u32==limits.max_blas_geometry_count || triangles+count>primitive_limit) {break;}
                        triangles+=count;end+=1;
                    }
                    let records:Vec<_>=members[start..end].iter().map(|&(slot,draw,_)| {
                        grouped_instances[slot]=true;
                        (draw,triangles_size(&draws[draw]))
                    }).collect();
                    let key:Vec<_>=records.iter().map(|(draw,_)|geometry_key(&draws[*draw])).collect();
                    let built=unique.entry(key).or_insert_with(|| {
                        let sizes:Vec<_>=records.iter().map(|(_,size)|size.clone()).collect();
                        let built=blas_geometries(d,"static opaque visibility BLAS",&sizes);
                        pending.push((records,built.clone()));built
                    }).clone();
                    visibility_groups.push(VisibilityGroup {slot:members[start].0,blas:built});
                    start=end;
                }
            }
        }
        let mut primitive_needed=vec![false;draws.len()];
        for (slot,&(draw,_)) in instances.iter().enumerate() {primitive_needed[draw]|=!grouped_instances[slot];}
        let blas = draws.iter().enumerate().map(|(index,draw)| {
            if let Some(shape)=&points[index] {return Some(shape.blas.clone());}
            if draw.evaluated() || !primitive_needed[index] {return None;}
            let key=vec![geometry_key(draw)];
            Some(unique.entry(key).or_insert_with(|| {
                let size=triangles_size(draw);
                let built=blas(d,"static primitive BLAS",&size);
                pending.push((vec![(index,size)],built.clone()));built
            }).clone())
        }).collect();
        // Bounded submissions keep build scratch small (the M5 forest spike OOM):
        // each batch's scratch is released before the next batch allocates its own.
        let mut start = 0;
        let mut previous = None;
        while start < pending.len() {
            if let Some(submission) = previous.take() {
                r.wait(Some(submission))?;
            }
            let mut end = start;
            let mut batch = 0u64;
            while end < pending.len()
                && (end == start || batch < constants::RAY_BLAS_BATCH_TRIANGLES)
            {
                batch += pending[end].0.iter().map(|(_,size)|u64::from(size.index_count.unwrap_or(0)/3)).sum::<u64>();
                end += 1;
            }
            let entries: Vec<_> = pending[start..end]
                .iter()
                .map(|(records,blas)| wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(records.iter().map(|(index,size)| {
                        let draw=&draws[*index];let stride=draw.layout.vertex_stride();
                        wgpu::BlasTriangleGeometry {
                            size,vertex_buffer:&draw.vertices,first_vertex:(draw.vertex_offset/stride) as u32,vertex_stride:stride,
                            index_buffer:Some(&draw.indices),first_index:Some((draw.index_offset/4) as u32),
                            transform_buffer:None,transform_buffer_offset:None,
                        }
                    }).collect()),
                })
                .collect();
            let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("static BLAS build"),
            });
            encoder.build_acceleration_structures(entries.iter(), std::iter::empty());
            previous = Some(crate::command_finish::Segments::submit(&r.queue, encoder));
            start = end;
        }
        // CandidateInstanceIndex addresses this dense TLAS order, not the
        // original raster instance list. Keep alpha headers in the same order.
        let mut leaders:std::collections::BTreeMap<_,_>=visibility_groups.iter().enumerate().map(|(i,g)|(g.slot,i)).collect();
        let placement_count=checked_placement_count(instances.iter().enumerate().filter(|(slot,_)|!grouped_instances[*slot] || leaders.contains_key(slot))
            .map(|(_, &(draw,_))|points[draw].as_ref().map_or(1,|p|p.count)),d.limits().max_tlas_instance_count)? as usize;
        let mut ray_instances=Vec::with_capacity(placement_count);let mut ray_grouped=Vec::with_capacity(placement_count);let mut point_rows=Vec::with_capacity(placement_count);
        for (slot,&instance) in instances.iter().enumerate() {
            if let Some(group)=leaders.remove(&slot) {
                visibility_groups[group].slot=ray_instances.len();
            } else if grouped_instances[slot] {continue;}
            if let Some(shape)=&points[instance.0] {
                for row in 0..shape.count {ray_instances.push(instance);ray_grouped.push(false);point_rows.push(Some(row));}
            } else {ray_instances.push(instance);ray_grouped.push(grouped_instances[slot]);point_rows.push(None);}

        }
        debug_assert_eq!(ray_instances.len(),placement_count);
        let mut hit_buffers = Vec::new();
        let probe_emitters = integrator.probe_volume().is_some()
            && crate::emitters::has_sources(draws,instances,materials,true);
        let gi_emitters=integrator.raster().is_some_and(|r|r.fast_gi.as_ref().is_some_and(|g|g.method==vork_scene::FastGiMethod::Ray))
            && crate::emitters::has_sources(draws,instances,materials,true);
        let topology = if path || probe_emitters || gi_emitters || medium_sources || integrator.reflections()==Some(vork_scene::Reflections::Ray) {
            crate::emitters::emitter_topology_counts(draws, &ray_instances, materials, !path, |slot,draw|points[draw].as_ref().filter(|_|point_rows[slot].is_some()).map_or(draws[draw].count,|p|p.indices)/3)?
        } else { vec![0, 0, 0] };
        let absorbing=ray_instances.iter().filter_map(|&(draw,object)| {
            let solid=solid_owners.get(object as usize).copied().unwrap_or(NO_ALPHA);
            (draws[draw].implicit.is_none() && materials[draws[draw].material].solid_absorption && solid!=NO_ALPHA).then_some(solid)
        }).collect::<std::collections::BTreeSet<_>>();
        // All faces of a potentially absorbing solid participate in nearest-boundary
        // discovery, including clear and opaque material groups of that solid.
        let shadow_solids=ray_instances.iter().map(|&(_,object)|absorbing.contains(&solid_owners.get(object as usize).copied().unwrap_or(NO_ALPHA))).collect::<Vec<_>>();
        let mut shadow_bounds=crate::shadow_containment::Plan::new(r,solids,&absorbing,objects,draws,solid_owners)?;
        let (alpha, emitters, boundaries, subsurface, occupancy_solids, film_filter, object_boundaries, implicit) = alpha_table(r, draws, &ray_instances, hits, &point_rows, &points, path, &mut hit_buffers, &topology, &crate::emitters::implicit_sources(draws,&ray_instances,materials), boundary_ids, subsurface_ids, solids, &shadow_solids, solid_owners, shadow_bounds.as_mut())?;
        Ok(Self {
            shadow_solids,
            points,
            point_rows,
            shadow_bounds,
            opaque: draws.iter().all(|draw| draw.ray_corners.is_none()),
            emitters,
            probe_emitters,
            emitter_weights:if path || probe_emitters || gi_emitters || medium_sources {crate::emitter_weights::Plan::new(r,emitters,topology[1])?} else {None},
            boundaries,
            object_boundaries,
            subsurface,
            occupancy_solids,
            film_filter,
            blas,
            visibility_groups,
            #[cfg(test)]
            grouped_instances:ray_grouped,
            instances:ray_instances,
            placement_pipeline: Default::default(),
            alpha_tested: draws.iter().map(|d| d.ray_corners.is_some()).collect(),
            alpha,
            hit_buffers,
            implicit,
        })
    }
}

/// Frame-slot TLAS over every draw instance, plus BLAS of deformed streams.
pub(crate) struct FrameRays {
    shadow_bounds:Option<crate::shadow_containment::Slot>,
    pub(crate) emitter_weights:Option<crate::emitter_weights::Slot>,
    pub(crate) tlas: wgpu::Tlas,
    placement: Option<crate::gpu_tlas::Placement>,
    placement_revision: u64,
    /// Changed object bytes since this slot's last full hierarchy build.
    motion_bytes: u64,
    #[cfg(test)]
    pub(crate) builds: u32,
    /// This slot's hit buffer array (Rays::hit_buffers with deformed ranges
    /// resolved to its deform pool), as (buffer, offset, size).
    pub(crate) hit_buffers: Vec<(wgpu::Buffer, u64, u64)>,
    /// Per draw: the size and BLAS built from this slot's deformed output.
    deformed: Vec<Option<(wgpu::BlasTriangleGeometrySizeDescriptor, wgpu::Blas)>>,
}
impl Rays {
    pub(crate) fn dynamic_shadow_bounds(&self)->bool {self.shadow_bounds.as_ref().is_some_and(|p|p.dynamic())}
}
impl FrameRays {
    pub(crate) fn prepare_shadow_bounds(&self,s:&InstalledScene,local:&[(Vec3,Vec3)],changes:&mut values::ChangeTracker) {
        if let (Some(plan),Some(slot))=(&s.rays.as_ref().unwrap().shadow_bounds,&self.shadow_bounds) {plan.upload(slot,local,changes);}
    }
    /// Resolve the hit buffer array against this slot's deformation outputs.
    pub(crate) fn resolve_hit_buffers(&mut self, s: &InstalledScene, deforms: &[Option<EvaluatedGeometry>]) {
        let Some(installed) = s.rays.as_ref() else { return };
        self.hit_buffers = resolve_hit_buffers(&installed.hit_buffers, deforms);
        if let (Some(plan),Some(slot))=(&installed.shadow_bounds,&self.shadow_bounds) {
            self.hit_buffers[plan.slot as usize]=(slot.buffer.clone(),0,slot.buffer.size());
        }
    }
    /// The hit-buffer group of compute kernels (rt::hits_layout).
    pub(crate) fn hits_bind(&self, d: &wgpu::Device) -> wgpu::BindGroup {
        hits_bind(d, &self.hit_buffers)
    }
    /// The hit buffer array binding; unused slots stay unbound (partially bound).
    pub(crate) fn hit_bindings(&self) -> Vec<wgpu::BufferBinding<'_>> {
        hit_bindings(&self.hit_buffers)
    }

    pub(crate) fn new(s: &InstalledScene, r: &Renderer, objects: &wgpu::Buffer, deforms: &[Option<EvaluatedGeometry>]) -> Result<Option<Self>> {
        let d=&r.device;
        let Some(installed) = s.rays.as_ref() else {return Ok(None);};
        // Evaluated positions can be shared by draws with different index ranges.
        // Own BLAS by the complete geometry identity, not just the pose leader.
        let owners = evaluated_owners(&s.draws);
        let deformed: Vec<Option<(wgpu::BlasTriangleGeometrySizeDescriptor, wgpu::Blas)>> = s
            .draws
            .iter()
            .zip(&installed.alpha_tested)
            .enumerate()
            .map(|(i, (draw, &alpha_tested))| {
                if !draw.evaluated() || owners[i] != i || installed.points[i].is_some() {return None;}
                let size = triangles(alpha_tested, draw.vertex_count, draw.count);
                let built = d.create_blas(&wgpu::CreateBlasDescriptor {
                    label: Some("deformed primitive BLAS"), flags: BUILD_FLAGS,
                    update_mode: if draw.refit_indices.is_some() {wgpu::AccelerationStructureUpdateMode::PreferUpdate} else {wgpu::AccelerationStructureUpdateMode::Build},
                }, wgpu::BlasGeometrySizeDescriptors::Triangles {descriptors: vec![size.clone()]});
                Some((size, built))
            })
            .collect();
        let tlas = d.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene TLAS"),
            max_instances: installed.instances.len().max(1) as u32,
            flags: BUILD_FLAGS,
            update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
        });
        let placement=(!installed.instances.is_empty()).then(|| {
                let visibility_blas:std::collections::HashMap<_,_>=installed.visibility_groups.iter().map(|group|(group.slot,&group.blas)).collect();
                let instances:Vec<_>=installed.instances.iter().enumerate().map(|(slot,&(draw,object))| {
                    let blas=installed.points[draw].as_ref().map(|p|&p.blas).or_else(||visibility_blas.get(&slot).copied())
                        .or(installed.blas[draw].as_ref())
                        .or_else(||deformed[owners[draw]].as_ref().map(|(_,b)|b)).expect("ray instance BLAS");
                    (object,!s.materials[s.draws[draw].material].transparent,s.draws[draw].implicit.is_none(),installed.shadow_solids[slot],blas.clone())
                }).collect();
                let point_poses:Vec<_>=installed.instances.iter().zip(&installed.point_rows).map(|(&(draw,_),row)|row.map(|row| {
                    if let Some(g)=deforms[draw].as_ref().filter(|g|g.point_domain.is_some()) {(g.output.buffer.clone(),(g.output.offset/4) as u32+2+row*10)}
                    else {(deforms[draw].as_ref().and_then(|g|g.point_poses.as_ref()).unwrap_or(&installed.points[draw].as_ref().unwrap().rest_poses).clone(),row*8+4)}
                })).collect();
                crate::gpu_tlas::Placement::with_points(r,objects,&instances,&installed.placement_pipeline,&point_poses)
            }).transpose()?;
        let shadow_bounds=installed.shadow_bounds.as_ref().map(|plan|crate::shadow_containment::Slot::new(r,plan,objects)).transpose()?;
        Ok(Some(Self { shadow_bounds, tlas, placement, placement_revision:0, motion_bytes:0, #[cfg(test)] builds:0, hit_buffers: Vec::new(), deformed, emitter_weights:installed.emitter_weights.as_ref().map(|p|crate::emitter_weights::Slot::new(p,d)) }))
    }
}

/// Rebuild what this sample's uploads changed: deformed BLAS when skin or morph
/// values moved, and the TLAS when any object value (transform or ray mask) moved.
pub(crate) fn encode(encoder: &mut wgpu::CommandEncoder, s: &InstalledScene, f: &mut FrameSlot) {
    let (Some(rays), Some(installed)) = (f.rays.as_mut(), s.rays.as_ref()) else {
        return;
    };
    // The slot's first sample builds everything; later ones follow the uploads.
    let first = f.changes.changed(values::ChangeKey::RayStructures, &[1]);
    let uploaded = |buffer: &wgpu::Buffer| f.changes.buffer_dirty(buffer);
    // CPU subdivision uploads its evaluated poses with fixed skin parameters.
    // Those vertex changes invalidate the same deformed BLAS as GPU skin/morphs.
    let dirty_geometry: Vec<bool> = s.draws.iter().enumerate().map(|(i, draw)| {
        // Static draws and aliases own no evaluated BLAS to invalidate.
        if rays.deformed[i].is_none() {return false;}
        let i = draw.displacement_leader.unwrap_or(i);
        first || f.geometry_chain.spatial_dirty(i,&f.changes)
    }).collect();
    let deform_changed = rays.deformed.iter().zip(&dirty_geometry).any(|(blas, dirty)| blas.is_some() && *dirty)
        || installed.points.iter().enumerate().any(|(i,p)|p.is_some() && (first || f.geometry_chain.spatial_dirty(i,&f.changes)));
    let placement_changed = if rays.placement.is_some() {
        let changed=rays.placement_revision != f.changes.placement_revision;
        rays.placement_revision=f.changes.placement_revision;
        changed
            || f.compute.as_ref().is_some_and(|compute|uploaded(&compute.parents))
    } else {false};
    let objects_changed = first || uploaded(&f.objects) || placement_changed;
    if let Some(bounds)=&rays.shadow_bounds {
        if objects_changed || uploaded(&bounds.buffer) {bounds.encode(encoder);}
    }
    // Emission tracks update proposal mass without rebuilding acceleration
    // structures. Use the existing per-slot upload tracker for both inputs.
    let materials=f.materials.as_ref().map_or_else(||s.material_table.records(),|m|&m.records);
    if deform_changed || objects_changed || uploaded(materials) || f.changes.dirty(values::ChangeKey::MaterialAttributes) {
        if let (Some(slot),Some(plan))=(&rays.emitter_weights,&installed.emitter_weights) {slot.encode(plan,encoder);}
    }
    if !deform_changed && !objects_changed {
        return;
    }
    // Uploads contain only changed records. Accumulate across refits so repeated
    // small changes cannot indefinitely retain the original tree topology.
    // GPU publications have no CPU motion magnitudes: conservatively rebuild,
    // without synchronizing to read back the million-instance placement buffer.
    let changed_bytes = f.changes.uploads.iter()
        .filter(|(buffer, ..)| buffer == &f.objects)
        .map(|(_, _, _, bytes)| *bytes).sum::<u64>();
    let force_rebuild = rebuild_tlas(&mut rays.motion_bytes, changed_bytes,
        f.objects.size(), first || deform_changed || placement_changed);
    let entries: Vec<_> = rays
        .deformed
        .iter()
        .enumerate()
        .filter(|(index, _)| dirty_geometry[*index])
        .filter_map(|(index, deformed)| {
            let (size, blas) = deformed.as_ref()?;
            let draw = &s.draws[index];
            Some(wgpu::BlasBuildEntry {
                blas,
                geometry: geometry(
                    size,
                    &f.deforms[index].as_ref()?.output.buffer,
                    f.deforms[index].as_ref()?.output.first_vertex(),
                    std::mem::size_of::<DeformedVertex>() as u64,
                    draw.refit_indices.as_ref().unwrap_or(&draw.indices),
                    (draw.index_offset / 4) as u32,
                ),
            })
        })
        .collect();
    #[cfg(test)] {rays.builds+=1;}
    if let Some(placement)=&rays.placement {
        encoder.build_acceleration_structures(entries.iter(), std::iter::empty());
        placement.encode(encoder,&rays.tlas, force_rebuild);
    } else {
        encoder.build_acceleration_structures(entries.iter(), std::iter::once(&rays.tlas));
    }
}

fn rebuild_tlas(motion_bytes: &mut u64, changed_bytes: u64, total_bytes: u64, spatial_publication: bool) -> bool {
    *motion_bytes = motion_bytes.saturating_add(changed_bytes);
    let rebuild = spatial_publication || *motion_bytes >= total_bytes.div_ceil(4).max(1);
    if rebuild { *motion_bytes = 0; }
    rebuild
}

#[cfg(all(test, vork_private_tests))]
include!("rt_tests_private.rs");
#[cfg(test)]
#[path = "rt_refit_tests.rs"]
mod refit_tests;
#[cfg(test)]
mod default_shadow_tests {
    use super::*;
    #[test]
    fn repeated_shape_capacity_is_admitted_before_allocation() {
        assert_eq!(checked_placement_count([96,96,1],193).unwrap(),193);
        assert!(checked_placement_count([96,96,1],192).is_err());
        assert!(checked_placement_count([u32::MAX,1],u32::MAX).is_err());
        assert_eq!(checked_placement_count([],0).unwrap(),0);
    }

    #[test]
    fn ray_profile_reserves_queries_for_imported_and_extracted_lights() {
        let scene: Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2, "nodes":[{"type":"light","shape":"environment"}],
            "render":{"settings":{"integrator":"raster","shadow_method":"ray"}}
        })).unwrap();
        assert!(ray_shadows(&scene));
        assert!(ray_options(&scene, false).iter().any(|option| option.contains("ray-traced shadows")));
    }

    #[test]
    fn automatic_lights_admit_the_extracted_provider_after_install() {
        let mut scene: Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2, "nodes":[{"type":"mesh","geometry":"shape_box"},
                {"type":"light","shape":"environment"}]
        })).unwrap();
        assert!(ray_shadows(&scene));
        assert!(!installed_shadow_rays(&scene,&[]));
        let node:Node=vork_scene::v2::from_test_value(serde_json::json!({"type":"light","shape":"sun"})).unwrap();
        let nodes=vec![EvalNode { camera: None,node:node.clone(),parent:None,world:Mat4::IDENTITY,dynamic:false,embedded:None}];
        assert!(installed_shadow_rays(&scene,&nodes));
        scene.nodes.push(node);
        assert!(ray_shadows(&scene));
        assert!(super::missing(&scene, false, "test (Vulkan)").unwrap().contains("automatic light shadows"));
    }

    #[test]
    fn automatic_lights_skip_empty_surface_trees() {
        let scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera"},{"type":"light","shape":"sun"},
                {"type":"light","shape":"environment"}]})).unwrap();
        assert!(!ray_shadows(&scene));
        assert!(!uses_rays(&scene));
    }

    #[test]
    fn environment_lights_cast_no_ray_shadows() {
        for shape in ["environment","sun","point","spot","rectangle","ellipse"] {
            let node:Node=vork_scene::v2::from_test_value(serde_json::json!({"name":"light","type":"light","shape":shape})).unwrap();
            assert_eq!(traces_shadows(&node.kind),shape!="environment");
        }
    }

    #[test]
    fn sampled_alpha_scene_ignores_environment_shadow_defaults() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let mut scene:serde_json::Value=vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"), include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        scene["render"]["settings"]["shadow_method"]="ray".into();
        scene["render"]["settings"]["samples"]=2.into();
        scene["render"]["settings"]["reflections"]="environment".into();
        scene["materials"]=serde_json::json!({"alpha":{"opacity":"transmittance","inputs":{"geometry_opacity":0.5}}});
        for n in scene["nodes"].as_array_mut().unwrap() {
            if n["type"]=="mesh" {n["materials"]=serde_json::json!(["alpha"]);}
        }
        scene["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"light","name":"world","shape":"environment","color":[0.1,0.1,0.1]}));
        let base=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut installed=r.install(vork_scene::v2::from_test_value(scene)?,&base)?;
        let mut frame=FrameSlot::new(&installed,&r,crate::frame::ReadbackPrecision::Packed)?;
        assert!(!installed.render_frame(&r,&mut frame,0.)?.rgba.is_empty());
        Ok(())
    }

}

// Keep default materials and imported material slots conservative. Visibility
// cannot prove absence: captures and secondary rays may consume any surface.
fn surface_stages_required(nodes:&[Node], stages:&std::collections::BTreeMap<String,crate::programs::Stages>)->bool {
    nodes.iter().any(|node| {
        let surface=match &node.kind {
            NodeKind::Mesh {materials,..}=>materials.is_empty() || materials.iter().any(|name|stages.get(name).is_none_or(|s|s.surface)),
            NodeKind::Asset {..}=>true,
            _=>false,
        };
        surface || surface_stages_required(&node.children,stages)
    })
}

fn previs_requires_path<'a>(scene: &Scene, used_materials: impl IntoIterator<Item=&'a str>) -> Result<bool> {
    if !scene.render.settings.previs {return Ok(false);}
    let animated=materials::animated_inputs(scene)?;
    for name in used_materials {
        if scene.materials[name].program!="std/openpbr" {continue;}
        let inputs=animated.get(name).cloned().unwrap_or_default();
        if materials::openpbr_values(scene,Some(name),&inputs,|_|Ok(0))?.previs_requires_path(&inputs) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Resolve unsupported combinations from scene facts before resources or passes
/// are chosen. Quality never controls admission; preserve the call budget.
pub(crate) fn resolve_scene_providers(r:&Renderer, scene:&mut Scene, _base:&Path, facts:crate::install::TransportFacts, stages:&std::collections::BTreeMap<String,crate::programs::Stages>)->Result<()> {
    fn nodes<'a>(roots:&'a [Node], out:&mut Vec<&'a Node>) {
        for node in roots {out.push(node); nodes(&node.children,out);}
    }
    let mut all=Vec::new(); nodes(&scene.nodes,&mut all);
    let ies=all.iter().any(|n|matches!(&n.kind,NodeKind::Light {shape,image:Some(_),..} if !matches!(shape,vork_scene::LightShape::Environment)));
    // Mesh stage reflection is authoritative here, before automatic providers
    // and effect pipelines are selected. A medium boundary is not a receiver.
    if !surface_stages_required(&scene.nodes, &stages) {
        if let Some(raster)=scene.render.settings.integrator.raster_mut() {
            raster.fast_gi=None;
            raster.probes=None;
            if raster.reflections==vork_scene::Reflections::Auto {raster.reflections=vork_scene::Reflections::Environment;}
            if raster.refraction==vork_scene::Refraction::Auto {raster.refraction=vork_scene::Refraction::Environment;}
        }
    }

    // Scalar table alpha uses each material's existing raster coverage slot.
    // Varying coverage and glass retain complete-camera transport until qualified.
    let previs_coverage=previs_requires_path(scene, stages.keys().map(String::as_str))?;
    let requires_path=facts.media || facts.hair || ies || previs_coverage || crate::catcher::has_mattes(scene);
    if requires_path && !scene.render.settings.integrator.is_path() {
        if facts.media || facts.hair {promote_transport_owner(scene);} else {promote_to_path(scene);}
    }
    ensure!(!requires_path || r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | HIT_BUFFER_FEATURES),
        "effect unavailable: participating media, physical hair and complete-camera transport need hardware ray queries");
    Ok(())
}

#[cfg(test)]
mod transport_owner_tests {
    use super::*;

    #[test]
    fn scalar_table_alpha_does_not_promote_other_materials() -> Result<()> {
        let mut scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera"}],
            "materials":{"opaque":{},"alpha":{"inputs":{"geometry_opacity":0.35}}}}))?;
        scene.render.settings=vork_scene::Profile::load("std/previs")?.resolve(&Default::default())?;
        assert!(!previs_requires_path(&scene,["opaque","alpha"])?);
        scene.animation=vork_scene::v2::from_test_value::<Scene>(serde_json::json!({"version":2,
            "nodes":[{"type":"camera"}],
            "animation":[{"tracks":[{"path":["materials","alpha","inputs","geometry_opacity"],"keys":[[0,0.35],[1,0.7]]}]}]}))?.animation;
        assert!(previs_requires_path(&scene,["opaque","alpha"])?);
        assert!(!previs_requires_path(&scene,["opaque"])?);
        Ok(())
    }

    #[test]
    fn previs_scalar_coverage_retains_emission_and_controls() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        for (background, overlapping, cutoff) in [(false,false,None),(true,false,None),(false,true,None),(false,false,Some(0.5)),(false,false,Some(0.2))] {
            let mut document:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/previs-scalar-alpha.json"))?;
            document["render"]["resolution"]=serde_json::json!([32,32]);
            if background {document["nodes"].as_array_mut().unwrap().remove(1);}
            if overlapping {document["materials"]["back"]["inputs"]["geometry_opacity"]=0.6.into();}
            if let Some(cutoff)=cutoff {document["materials"]["front"]["inputs"]["alpha_cutoff"]=cutoff.into();}
            let mut outputs=Vec::new();
            for continuous in [false,true] {
                let mut document=document.clone();
                if continuous {
                    for material in document["materials"].as_object_mut().unwrap().values_mut() {
                        material["opacity"]="transmittance".into();
                    }
                }
                let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
                scene.render.settings=vork_scene::Profile::load("std/previs")?.resolve(&Default::default())?;
                scene.render.settings.samples=512;
                let mut installed=r.install(scene,Path::new("."))?;
                assert!(!installed.render.settings.integrator.is_path());
                let mut frame=installed.create_frame_slot(&r,crate::frame::ReadbackPrecision::Float)?;
                let output=installed.render_frame(&r,&mut frame,0.)?;
                assert!(!output.ids.is_empty());
                assert!(output.depth_f32.as_ref().unwrap().iter().all(|v|v.is_finite()));
                assert!(output.normals_f32.as_ref().unwrap().iter().flatten().all(|v|v.is_finite()));
                outputs.push(output.linear_rgba.unwrap());
            }
            // Coverage and continuous composition converge in scene-linear space.
            // Independent visibility samples retain Bernoulli noise; compare the
            // interior mean and RMSE, rather than requiring matching sample hashes.
            for c in 0..3 {
                let mut bias=0.;let mut squared=0.;let mut maximum=0_f64;
                for y in 8..24 {for x in 8..24 {
                    let delta=f64::from(outputs[0][y*32+x][c]-outputs[1][y*32+x][c]);
                    bias+=delta;squared+=delta*delta;maximum=maximum.max(delta.abs());
                }}
                eprintln!("previs alpha: background={background}, overlap={overlapping}, cutoff={cutoff:?}, channel={c}: bias={}, RMSE={}, max={maximum}",bias/256.,(squared/256.).sqrt());
                assert!((bias/256.).abs()<0.015 && (squared/256.).sqrt()<0.06 && maximum<0.2,
                    "background={background}, overlap={overlapping}, cutoff={cutoff:?}, channel={c}: bias={}, RMSE={}, max={maximum}",bias/256.,(squared/256.).sqrt());
            }
        }
        Ok(())
    }

    #[test]
    fn dry_promotions_preserve_the_resolved_denoiser() -> Result<()> {
        for profile in ["std/previs", "std/draft", "std/final"] {
            let mut scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}]}))?;
            scene.render.settings=vork_scene::Profile::load(profile)?.resolve(&Default::default())?;
            let denoiser=scene.render.settings.integrator.denoiser().cloned();
            promote_to_path(&mut scene);
            assert!(scene.render.settings.integrator.is_path());
            assert_eq!(scene.render.settings.integrator.denoiser(),denoiser.as_ref());
        }
        Ok(())
    }

    #[test]
    fn promoted_views_keep_budget_and_explicit_denoise_opt_out() -> Result<()> {
        for profile in ["std/previs", "std/draft", "std/final"] {
            for intent in [None, Some(false), Some(true)] {
                let mut scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}]}))?;
                let mut explicit=serde_json::Map::new();
                explicit.insert("samples".into(),serde_json::json!(17));
                if let Some(denoise)=intent {explicit.insert("denoise".into(),serde_json::json!(denoise));}
                scene.render.settings=vork_scene::Profile::load(profile)?.resolve(&explicit)?;
                promote_transport_owner(&mut scene);
                assert!(scene.render.settings.integrator.is_path());
                assert!(!scene.render.settings.reconstruct);
                assert_eq!(scene.render.settings.samples,17);
                assert_eq!(scene.render.settings.integrator.denoiser().is_some(),intent.unwrap_or(true));
            }
        }
        Ok(())
    }
}
