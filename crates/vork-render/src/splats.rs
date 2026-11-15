//! Gaussian point assets and GPU packing. Traversal only consumes these packed
//! attributes; parsing, shape mathematics and radiance remain shared modules.
use crate::*;
use vork_splat::{Cloud, Metadata};
use std::{collections::BTreeMap, fs::File, io::BufReader, sync::Arc};

/// The radiance program the importer binds to a capture's SH attributes.
pub(crate) const PROGRAM: &str = "std/gaussian";

pub(crate) fn is_file(path: impl AsRef<Path>) -> bool {
    path.as_ref().extension().is_some_and(|e| e.eq_ignore_ascii_case("ply") || e.eq_ignore_ascii_case("spz"))
}
pub(crate) fn present(scene: &Scene) -> bool {
    fn visit(nodes: &[Node]) -> bool {
        nodes.iter().any(|node|matches!(&node.kind,NodeKind::Asset {file,..} if is_file(file)) || visit(&node.children))
    }
    visit(&scene.nodes)
}
pub(crate) fn inspect(path: &Path) -> Result<Metadata> {
    let file=BufReader::new(File::open(path).with_context(||format!("open Gaussian splat {}",path.display()))?);
    if path.extension().is_some_and(|e|e.eq_ignore_ascii_case("spz")) {vork_splat::inspect_spz(file)} else {vork_splat::inspect_ply(file)}
}
pub(crate) fn read(path: &Path, available: u64) -> Result<Cloud> {
    let file=BufReader::new(File::open(path)?);
    if path.extension().is_some_and(|e|e.eq_ignore_ascii_case("spz")) {vork_splat::read_spz(file,available)} else {vork_splat::read_ply(file,available)}
}

pub(crate) fn material_capabilities(capabilities: &vork_slang::ProgramCapabilities) -> Result<()> {
    ensure!(!capabilities.spatial_subsurface,"Gaussian splats have no solid interior for subsurface scattering; use a triangle mesh or a thin-walled surface material");
    ensure!(!capabilities.hair,"Gaussian splats do not carry fibre geometry; use curves for hair materials");
    ensure!(!capabilities.lit,"Gaussian splats do not support lit callbacks; replace the surface material instead");
    Ok(())
}

fn chunk_size(device: &wgpu::Device, metadata: &Metadata) -> Result<u32> {
    let limits=device.limits();let stride=48+12*metadata.coefficient_count() as u64;
    let count=(limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)/stride)
        .min(u64::from(limits.max_blas_primitive_count)) as u32;
    ensure!(count>0,"Gaussian splat attributes exceed the device storage limit; use a device with larger storage bindings");
    Ok(count)
}
fn native_sizes<A: wgpu::hal::Api>(device: &wgpu::Device,bounds:&wgpu::Buffer,count:u32) -> Option<wgpu::hal::AccelerationStructureBuildSizes> {
    use wgpu::hal::Device as _;
    let native=unsafe {device.as_hal::<A>()}?;
    // A size query reads no buffer contents, offsets or lengths, but Metal's
    // geometry descriptor must still name a buffer (Vulkan ignores it). `bounds`
    // is any live buffer, as in wgpu-core's own size query inside `create_blas`.
    // These are exactly the flags/count used by upload.
    let bounds=unsafe {bounds.as_hal::<A>()}?;
    let entries=wgpu::hal::AccelerationStructureEntries::<A::Buffer>::AABBs(vec![wgpu::hal::AccelerationStructureAABBs {
        buffer:Some(&*bounds),offset:0,count,stride:24,flags:wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION,
    }]);
    Some(unsafe {native.get_acceleration_structure_build_sizes(&wgpu::hal::GetAccelerationStructureBuildSizesDescriptor {
        entries:&entries,flags:wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
    })})
}
fn sizes(device: &wgpu::Device,bounds:&wgpu::Buffer,count:u32) -> Result<wgpu::hal::AccelerationStructureBuildSizes> {
    #[cfg(any(target_os="linux",target_os="windows"))]
    if let Some(sizes)=native_sizes::<wgpu::hal::api::Vulkan>(device,bounds,count) {return Ok(sizes);}
    #[cfg(target_os="macos")]
    if let Some(sizes)=native_sizes::<wgpu::hal::api::Metal>(device,bounds,count) {return Ok(sizes);}
    // The existing CPU validation backend records the same AABB build without
    // opening a GPU driver. Its sizes describe validation storage only.
    if let Some(sizes)=native_sizes::<wgpu::hal::api::Noop>(device,bounds,count) {return Ok(sizes);}
    anyhow::bail!("Gaussian splats require native ray queries; use a Vulkan or Metal ray-tracing device")
}
/// One box of storage, the buffer that admission's size queries name.
fn size_query_bounds(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {label:Some("Gaussian size query"),size:24,
        usage:wgpu::BufferUsages::BLAS_INPUT,mapped_at_creation:false})
}
fn admit(r: &Renderer, metadata: impl Iterator<Item=Metadata>) -> Result<()> {
    let mut attributes=0u64;let mut acceleration=0u64;let mut transient=0u64;
    let bounds=size_query_bounds(&r.device);
    for m in metadata {
        // One packed alignment word and one stable point-identity word per row.
        attributes+=m.attribute_bytes()+u64::from(m.count)*8;
        let chunk=chunk_size(&r.device,&m)?;
        for first in (0..m.count).step_by(chunk as usize) {
            let count=(m.count-first).min(chunk);let sizes=sizes(&r.device,&bounds,count)?;
            acceleration+=sizes.acceleration_structure_size;
            transient=transient.max(sizes.build_scratch_size+u64::from(count)*24);
        }
    }
    let required=attributes+acceleration+transient;
    let available=residency::camera_budget(&r.device,r.device_memory,None).0;
    ensure!(required<=available,"Gaussian splats need {required} device bytes ({attributes} attributes, {acceleration} acceleration, {transient} build storage); only {available} available; use a device with more memory or a smaller capture");
    Ok(())
}

/// Bind the imported radiance as an ordinary program before material planning.
/// The generated names are installation-local, never a second authored surface.
pub(crate) fn prepare(r: &Renderer, scene: &mut Scene, base: &Path) -> Result<()> {
    fn visit(nodes: &mut [Node], materials: &mut BTreeMap<String,vork_scene::v2::Material>, base: &Path,
        seen: &mut BTreeMap<PathBuf,Metadata>) -> Result<()> {
        for node in nodes {
            if let NodeKind::Asset {file,materials: bindings,nodes: overrides}=&mut node.kind {
                if is_file(&*file) {
                    ensure!(overrides.is_empty(),"Gaussian splat assets have no imported nodes; transform the asset node instead");
                    let path=base.join(&*file);
                    if !seen.contains_key(&path) {seen.insert(path.clone(),inspect(&path)?);}
                    let metadata=&seen[&path];
                    if !bindings.contains_key("*") && !bindings.contains_key("#0") {
                        let mut name=String::from("Gaussian radiance");
                        while materials.contains_key(&name) {name.push('_');}
                        let mut material: vork_scene::v2::Material=serde_json::from_value(serde_json::json!({"program":PROGRAM}))?;
                        for degree in 0..=3 {for coefficient in 0..=2*degree {
                            // Higher bands of a lower-degree expansion are zero.
                            // Express that through the ordinary binding factor,
                            // without adding fabricated columns to point storage.
                            let present=degree<=metadata.degree;
                            material.inputs.insert(format!("sh_{degree}_{coefficient}"),vork_scene::v2::Input::Attribute(vork_scene::v2::AttributeBinding {
                                attribute:if present {vork_splat::sh_attribute(degree,coefficient)} else {vork_splat::sh_attribute(0,0)},
                                channel:None,factor:(!present).then_some(vork_scene::v2::Factor::Number(0.)),
                            }));
                        }}
                        materials.insert(name.clone(),material);bindings.insert("*".into(),name);
                    }
                }
            }
            visit(&mut node.children,materials,base,seen)?;
        }
        Ok(())
    }
    let mut seen=BTreeMap::new();
    visit(&mut scene.nodes,&mut scene.materials,base,&mut seen)?;
    if !seen.is_empty() {
        ensure!(r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),"Gaussian splats require native ray queries; use a ray-tracing-capable device");
        // Device admission precedes payload decompression and shader compilation.
        admit(r,seen.into_values())?;
        if !scene.render.settings.integrator.is_path() {rt::promote_to_path(scene);}
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct Geometry {
    pub blas: wgpu::Blas,
    pub stride: u32,
    pub capture_count: u32,
    pub attribute_offsets: wgpu::Buffer,
}

pub(crate) fn attribute(metadata: &Metadata, name: &str) -> Result<u32> {
    let (offset,width)=match name {
        vork_splat::POSITION=>(0,3), vork_splat::SCALE=>(4,3),
        vork_splat::ROTATION=>(8,4), vork_splat::OPACITY=>(3,1),
        _=> {
            let mut found=None;
            for degree in 0..=metadata.degree {for coefficient in 0..=2*degree {
                if name==vork_splat::sh_attribute(degree,coefficient) {found=Some((12+3*(u32::from(degree).pow(2)+u32::from(coefficient)),3));}
            }}
            found.with_context(||format!("Gaussian splat has no attribute {name:?}; bind an attribute present in this capture"))?
        }
    };
    Ok(offset | width<<24)
}

pub(crate) fn upload(import: &mut import::GeometryUpload<'_>, cloud: &Cloud) -> Result<()> {
    let r=import.r;let d=&r.device;
    let first_draw=import.draws.len();
    let stride=12+3*cloud.metadata.coefficient_count() as u32;
    let chunk=chunk_size(d,&cloud.metadata)? as usize;
    // A scene's slots include attributes used by other meshes. Missing slots stay
    // zero; installation checks only the material actually bound to this asset.
    let offsets: Vec<u32>=import.attribute_names.iter().map(|name|attribute(&cloud.metadata,name).unwrap_or(u32::MAX)).collect();
    let offsets=init(d,"Gaussian attribute offsets",bytemuck::cast_slice(if offsets.is_empty() {&[u32::MAX]} else {&offsets}),wgpu::BufferUsages::STORAGE);
    for first in (0..cloud.metadata.count as usize).step_by(chunk) {
        let end=(first+chunk).min(cloud.metadata.count as usize);
        let mut points=Vec::<f32>::with_capacity((end-first)*stride as usize);
        let mut aabbs=Vec::<f32>::with_capacity((end-first)*6);
        let mut lo=Vec3::splat(f32::INFINITY);let mut hi=Vec3::splat(f32::NEG_INFINITY);
        for i in first..end {
            let shape=cloud.shape(i);
            let (a,b)=shape.bounds(3.)?;lo=lo.min(a);hi=hi.max(b);
            aabbs.extend(a.to_array());aabbs.extend(b.to_array());
            points.extend(cloud.positions[i]);points.push(cloud.opacities[i]);
            points.extend(cloud.scales[i]);points.push(0.);points.extend(cloud.rotations[i]);
            for column in &cloud.harmonics {points.extend(column[i]);}
        }
        let vertices=init(d,"Gaussian point attributes",bytemuck::cast_slice(&points),wgpu::BufferUsages::STORAGE);
        let bounds=init(d,"Gaussian bounds",bytemuck::cast_slice(&aabbs),wgpu::BufferUsages::BLAS_INPUT);
        let size=wgpu::BlasAabbGeometrySizeDescriptor {count:(end-first) as u32,flags:wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION};
        let blas=d.create_blas(&wgpu::CreateBlasDescriptor {label:Some("Gaussian points"),flags:wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,update_mode:wgpu::AccelerationStructureUpdateMode::Build},wgpu::BlasGeometrySizeDescriptors::Aabbs {descriptors:vec![size.clone()]});
        let mut encoder=d.create_command_encoder(&wgpu::CommandEncoderDescriptor {label:Some("Gaussian acceleration")});
        encoder.build_acceleration_structures([&wgpu::BlasBuildEntry {blas:&blas,geometry:wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {size:&size,buffer:&bounds,offset:0,stride:24}])}],std::iter::empty());
        r.wait(Some(crate::command_finish::Segments::submit(&r.queue,encoder)))?;
        *import.min=import.min.min(lo);*import.max=import.max.max(hi);
        import.draws.push(install::Draw {
            splats:Some(Arc::new(Geometry {blas,stride,capture_count:cloud.metadata.count,attribute_offsets:offsets.clone()})),
            implicit:None,sort_order:0,vertices:vertices.clone(),attributes:vertices,
            layout:geometry::VertexLayout {points:false,deform:false,tangent:false,uv:[constants::VERTEX_DEFAULT_UV;2],color:false,custom:import.attribute_names.len() as u8},
            uv_sets:0,exit_geometry:None,exit_bvh:None,open_edges:None,indices:import.placeholder.clone(),refit_indices:None,geometry_page:None,
            vertex_offset:0,index_offset:0,count:0,vertex_count:(end-first) as u32,ray_corners:None,
            material:import.default,material_slot:Some(0),object:import.object,instances:import.object..import.object+1,sort_stride:1,
            local:Mat4::IDENTITY,gltf_node:(0,String::new()),gltf_path:Vec::new(),primitive_start:first as u32,center:(lo+hi)*0.5,bounds:(lo,hi),generated_bounds:(lo,hi),
            computed:false,deform:None,displacement:None,displacement_leader:None,
        });
    }
    // Storage chunks share the capture's generated-coordinate domain.
    for draw in &mut import.draws[first_draw..] {draw.generated_bounds=(*import.min,*import.max);}
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Admission sizes a capture before any of its buffers exist. Metal's size
    /// query panicked on a geometry descriptor that named no buffer.
    #[test]
    #[ignore = "requires real Metal GPU; scripts/check-metal-contracts"]
    fn metal_contract_gaussian_admission_sizes_a_capture() -> anyhow::Result<()> {
        let r=crate::metal_contract_renderer()?;
        let bounds=super::size_query_bounds(&r.device);
        let (one,many)=(super::sizes(&r.device,&bounds,1)?,super::sizes(&r.device,&bounds,1<<20)?);
        assert!(one.acceleration_structure_size>0 && one.build_scratch_size>0,"{one:?}");
        assert!(many.acceleration_structure_size>one.acceleration_structure_size,"{one:?} {many:?}");
        Ok(())
    }
}
