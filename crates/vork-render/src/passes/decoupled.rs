//! Decoupled shading (shaders/decoupled.slang, decoupled_merge.slang). Each
//! scene-time stratum collects spatial visibility samples into per-pixel slots,
//! then shades each slot once into resolved HDR. Static exposures collect all
//! spatial samples together; sampled shutters retain one sample per time stratum.
//! Transparent draws list their fragments in front of each pass's opaque depth;
//! the merge folds them into the same slots, and the shade blends them front to
//! back over the opaque slots per visibility sample.
use crate::*;
use crate::pipelines::ShadeStage;
use vork_slang::registry::CAMERA_IMPLICIT_SLOTS;
use constants::{EVALUATION_BAND_RECORDS, EVALUATION_ESTIMATE_HEADROOM, EVALUATION_MEMORY_SHARE, VISIBILITY_DIFFUSION_WORDS, VISIBILITY_KEY_WORDS, VISIBILITY_REFLECTION_WORDS, VISIBILITY_REFLECTION_GUIDE_WORDS, REFLECTION_CLASSES, VISIBILITY_SLOTS};

/// Host allocation and generated shade modules must select the same slot source.
/// Diffusion traverses merged slots to match center and donor objects; reflections
/// use the opaque dense-zero record in either one-sample representation.
pub(crate) fn direct_visibility(samples:u32, transparent:bool, diffusion:bool)->bool {samples==1 && !transparent && !diffusion}

/// What sizes a camera view's whole-frame bindings, besides its extent and samples.
#[derive(Clone, Copy)]
pub(crate) struct ViewShape {
    pub(crate) transparent: bool,
    pub(crate) diffusion: bool,
    pub(crate) reflections: bool,
    pub(crate) screen_refraction: bool,
    /// Every material is a table material (decoupled Install: materials == 2).
    pub(crate) table_only: bool,
    /// Active solid capacity retained for each camera ray in a shading band.
    pub(crate) boundaries: u32,
    /// Installed overflow-pool reservation.
    pub(crate) reserved: u32,
}
/// Speculative overflow is optional; measured reservation demand is mandatory.
/// Bound the film-footprint estimate by every dependent binding.
fn visibility_pool(limits: &wgpu::Limits, shape: ViewShape, extent: [u32; 2], samples: u32) -> u64 {
    let pixels = u64::from(extent[0]) * u64::from(extent[1]);
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let mut radius = [0.5_f32; 2];
    for i in 0..samples {
        let offset = crate::sampling::offset(i as usize, samples as usize);
        for axis in 0..2 { radius[axis] = radius[axis].max(offset[axis].abs()); }
    }
    let estimate = pixels * (4. * radius[0] * radius[1]).ceil() as u64;
    let target = estimate.max(u64::from(shape.reserved).next_power_of_two());
    let (mut lo, mut hi) = (0, target.min(u64::from(u32::MAX)));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let bindings=view_bindings_with_pool(shape, extent, samples, mid);
        if bindings.iter().all(|(_, bytes)| *bytes <= limit) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo.max(u64::from(shape.reserved))
}
fn view_bindings(limits: &wgpu::Limits, shape: ViewShape, extent: [u32; 2], samples: u32) -> [(&'static str, u64); 5] {
    view_bindings_with_pool(shape, extent, samples, visibility_pool(limits, shape, extent, samples))
}
/// Byte sizes of a view's whole-frame storage bindings and minimum containment
/// band. The metadata binding must leave room for at least one tile row.
fn view_bindings_with_pool(shape: ViewShape, [w, h]: [u32; 2], samples: u32, pool: u64) -> [(&'static str, u64); 5] {
    let pixels = u64::from(w) * u64::from(h);
    let direct_keys = direct_visibility(samples, shape.transparent, shape.diffusion);
    // decoupled_slots.slang SlotLayout: dense slots and a list head per pixel, then the pool.
    let stride = u64::from(VISIBILITY_KEY_WORDS + samples.div_ceil(32));
    let slots = if direct_keys {4} else {pixels * (u64::from(VISIBILITY_SLOTS) * stride + 1) * 4 + pool * (1 + stride) * 4};
    let records = pixels * u64::from(VISIBILITY_SLOTS) + pool;
    // Slot records, then (transparent scenes) per-sample transmittance (CameraTransmittance);
    // one table evaluation per pixel needs neither (decoupled.slang direct_evaluations).
    let metadata = if direct_keys && shape.table_only {4} else {
        records * u64::from(vork_slang::registry::CAMERA_EVALUATION_METADATA_WORDS) * 4
            + if shape.transparent {pixels * u64::from(samples) * 4} else {0}
    };
    // Twelve packed words per existing dense/pool slot; no fixed object limit.
    let subsurface = if shape.diffusion {
        let bytes=records * u64::from(VISIBILITY_DIFFUSION_WORDS) * 4;
        // Each half is a storage-buffer slice (WebGPU guarantees <=256-byte alignment).
        if shape.screen_refraction {bytes.div_ceil(256)*256*2} else {bytes}
    } else {4};
    // Each field is a separate binding, with no change to the packed record.
    let reflections = if shape.reflections { ((pixels * u64::from(VISIBILITY_SLOTS) + if direct_keys {0} else {pool}) * 4).div_ceil(256)*256 } else {256};
    let guides = if shape.reflections {pixels * u64::from(REFLECTION_CLASSES * VISIBILITY_REFLECTION_GUIDE_WORDS) * 4} else {4};
    // Leave room for at least one tile row of containment beside the whole-frame
    // metadata, including when the speculative visibility pool fills a binding.
    let metadata=metadata+occupancy_row_bytes(w,samples,shape.boundaries)*u64::from(h.min(TILE));
    [("visibility slots", slots), ("camera evaluation metadata", metadata), ("subsurface records", subsurface),
        ("reflection record field", reflections), ("reflection guides", guides)]
}
/// Fields packed into each reflection-record allocation. The complete array may
/// exceed the device's single-buffer limit, so it is split into several buffers.
fn reflection_fields_per_buffer(field_bytes:u64,max_buffer_size:u64)->u32 {
    (max_buffer_size/field_bytes.max(1)).clamp(1,u64::from(VISIBILITY_REFLECTION_WORDS)) as u32
}

/// Check the camera view's whole-frame bindings against the device, at install
/// before any per-frame resource exists and again when the view is created.
pub(crate) fn check_view(limits: &wgpu::Limits, extent: [u32; 2], samples: u32, shape: ViewShape) -> Result<[u64; 5]> {
    let sizes=gpu_memory::frame_bindings(limits, extent, samples, |extent, samples| view_bindings(limits, shape, extent, samples))?;
    // Record fields are separate bindings packed across allocations
    // (reflection_fields_per_buffer), so the complete array need not fit one buffer.
    Ok(sizes)
}

pub(crate) fn packet_eligible(shape:ViewShape,samples:u32)->bool {
    samples>1 && !shape.transparent && !shape.diffusion
        && !shape.screen_refraction && shape.boundaries==0
}
/// Admission checks the smallest reusable packet and the retained whole-frame
/// guides. View creation subsequently selects and validates its actual extent.
pub(crate) fn check_packet_minimum(limits:&wgpu::Limits,extent:[u32;2],samples:u32,shape:ViewShape)->Result<()> {
    let local=[extent[0],extent[1].min(TILE)];
    let pool=u64::from(local[0])*u64::from(local[1])*u64::from(samples.saturating_sub(VISIBILITY_SLOTS));
    let fields=view_bindings_with_pool(shape,local,samples,pool);
    for (label,bytes) in fields {gpu_memory::storage_binding(limits,label,bytes, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;}
    let (label,bytes)=view_bindings_with_pool(shape,extent,samples,0)[4];
    gpu_memory::storage_binding(limits,label,bytes, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;
    Ok(())
}

/// Reserve at most a quarter of available device memory for camera records,
/// leaving headroom for retained images, geometry and downstream passes. Frames
/// whose existing layout fits use the existing dispatch path without packets.
fn camera_packet_rows(limits:&wgpu::Limits,shape:ViewShape,[w,h]:[u32;2],samples:u32,words:u32,memory:u64,materials:u32,indirect_words:u32)->Result<Option<u32>> {
    let budget=memory/constants::CAMERA_RECORD_MEMORY_SHARE;
    let header=u64::from(vork_slang::registry::CAMERA_INDIRECT_HEADER_WORDS)*4;
    let binding=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let work_limit=if indirect_words==0 {u64::MAX} else {binding.saturating_sub(header)/(u64::from(indirect_words)*4)};
    let full=view_bindings(limits,shape,[w,h],samples);
    let root_bytes=|sizes:[(&str,u64);5],evaluations:u64| {
        sizes[0].1+sizes[1].1+sizes[2].1+sizes[3].1*u64::from(VISIBILITY_REFLECTION_WORDS)
            +u64::from(materials)*24+header
            +evaluations*(u64::from(words)*16+16+u64::from(indirect_words)*4)
    };
    let fits_bindings=|sizes:[(&str,u64);5]|sizes.iter().all(|(_,b)|*b<=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size));
    let records=if shape.table_only {u64::from(w)*u64::from(h)*u64::from(VISIBILITY_SLOTS)+visibility_pool(limits,shape,[w,h],samples)}
        else {u64::from(w)*u64::from(h)*u64::from(samples)};
    let evaluations=records.min(evaluation_budget(limits,memory,words)).min(work_limit);
    if fits_bindings(full) && root_bytes(full,evaluations)<=budget {return Ok(None);}
    let fits=|rows:u32| {
        let pixels=u64::from(w)*u64::from(rows);
        let pool=pixels*u64::from(samples.saturating_sub(VISIBILITY_SLOTS));
        let sizes=view_bindings_with_pool(shape,[w,rows],samples,pool);
        fits_bindings(sizes) && pixels*u64::from(samples)<=evaluation_limit(limits,materials,words).min(work_limit)
            && root_bytes(sizes,pixels*u64::from(samples))<=budget
    };
    let minimum=h.min(TILE);
    // Memory headroom is a scheduling target, never a new scene refusal.
    // Native limits remain hard and are checked again before allocation.
    if !fits(minimum) {return Ok(Some(minimum));}
    let mut lo=1;let mut hi=h.div_ceil(TILE);
    while lo<hi {let mid=(lo+hi+1)/2;if fits((mid*TILE).min(h)) {lo=mid;} else {hi=mid-1;}}
    let maximum=(lo*TILE).min(h);
    // Once the pass count is fixed, balance the packets. Keeping a maximum-size
    // backing for four full packets and a tiny fifth retains memory for no
    // scheduling benefit. Round in tile units so the sample/workgroup mapping
    // remains unchanged and the balanced size cannot exceed the fitted size.
    let packets=h.div_ceil(maximum);
    Ok(Some((h.div_ceil(TILE).div_ceil(packets)*TILE).min(h)))
}

/// One key per pixel: (object + 1, primitive).
pub(crate) const KEY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Merge and shade workgroup side (REG_COMPUTE_TILE).
const TILE: u32 = 8;

/// Keep the fixed field offsets, but do not stride over an unused result tail.
/// These are the same runtime gates as camera_native::evaluation_source.
fn evaluation_words(features:crate::forward_variant::FeatureMask,transparent:bool,diffusion:bool,thickness:bool,aovs:u32)->u32 {
    use crate::forward_variant::{FEATURE_CAMERA_RECEIVERS,FEATURE_REFLECTIONS,FEATURE_INDIRECT_CONTROLS};
    if features&FEATURE_REFLECTIONS!=0 {return 8+u32::from(diffusion)*6+aovs+crate::constants::REFLECTION_BSDF_VECTORS;}
    if diffusion || aovs!=0 {return 8+u32::from(diffusion)*6+aovs;}
    if transparent {return 8;}
    if thickness || features&FEATURE_CAMERA_RECEIVERS!=0 {return 7;}
    if features&FEATURE_INDIRECT_CONTROLS!=0 {return 4;}
    2
}
fn evaluation_page_capacity(limits: &wgpu::Limits,words:u32) -> u32 {
    (limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) / (u64::from(words)*16)) as u32
}
/// The bin binding can retain at most binding/16 records. Size the descriptor
/// table for the largest record this scene can produce, including its AOVs.
pub(crate) fn evaluation_pages(limits:&wgpu::Limits,aovs:usize)->u32 {
    let words=8+6+crate::constants::REFLECTION_BSDF_VECTORS as u32+aovs as u32;
    let binding=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    (binding/16).div_ceil(u64::from(evaluation_page_capacity(limits,words).max(1))).max(1) as u32
}
/// The material bins' work list (header plus 16 bytes per evaluation)
/// bounds retained evaluations independently of the evaluation descriptor table.
fn evaluation_limit(limits: &wgpu::Limits,materials:u32,_words:u32) -> u64 {
    let binding=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    (binding.saturating_sub(u64::from(materials)*24)/16).min(u64::from(u32::MAX))
}
/// Retained evaluations a band reserves before any frame reports demand: a
/// content estimate (slots, or per-sample content) bounded by the band
/// budget. Reported demand is always reserved, with power-of-two headroom
/// within the budget; the caller rejects demand beyond what the view can address.
fn evaluation_capacity(slots:u64, content:u64, demand:u64, budget:u64) -> u64 {
    slots.max(content.max(demand.next_power_of_two()).min(budget)).max(demand)
}
/// At least `records / budget` bands of whole tile rows, each holding about an
/// equal share of `records`: (bands, rows per band). One tile row is the finest band.
fn shading_bands(records:u64, budget:u64, height:u32) -> (u32,u32) {
    let bands=u32::try_from(records.div_ceil(budget.max(1))).unwrap_or(u32::MAX).max(1);
    if bands==1 {return (1,height);}
    let rows=(height/bands/TILE).max(1)*TILE;
    (height.div_ceil(rows),rows.min(height))
}
fn occupancy_row_bytes(width:u32,samples:u32,boundaries:u32)->u64 {
    if boundaries==0 {0} else {u64::from(width)*u64::from(samples)*(1+7*u64::from(boundaries))*4}
}
/// Containment shares a binding with whole-frame metadata, but is regenerated
/// for each band. Refine the evaluation bands against the remaining binding
/// range and the same memory share used for retained evaluations.
fn occupancy_bands(limits:&wgpu::Limits,memory:u64,metadata:u64,row_bytes:u64,height:u32,rows:u32)->Result<(u32,u32)> {
    if row_bytes==0 {return Ok((height.div_ceil(rows),rows));}
    let minimum=height.min(TILE);
    gpu_memory::storage_binding(limits,"camera metadata and one tile row of dielectric occupancy",metadata+row_bytes*u64::from(minimum), "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;
    let available=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)-metadata;
    let budget=available.min(memory/EVALUATION_MEMORY_SHARE);
    // Device limits are hard; the memory target may yield to one tile row.
    let capacity=(budget/row_bytes).max(u64::from(minimum));
    let rows=rows.min(u32::try_from(capacity).unwrap_or(u32::MAX));
    let rows=if rows<height {(rows/TILE).max(1)*TILE} else {rows};
    Ok((height.div_ceil(rows),rows))
}
/// Retained evaluations per band: EVALUATION_BAND_RECORDS, one storage
/// binding, or EVALUATION_MEMORY_SHARE of device memory, whichever is least.
fn evaluation_budget(limits:&wgpu::Limits, device_memory:u64, words:u32) -> u64 {
    EVALUATION_BAND_RECORDS.min(u64::from(evaluation_page_capacity(limits,words))).min(device_memory/EVALUATION_MEMORY_SHARE/(u64::from(words)*16))
}
/// Decoupled shading so far covers opaque and transparent surfaces (material
/// tables and programs, transmission of the world) lit by the scene's lights
/// through triangle visibility, against any background. Any other raster feature
/// is an install error.
pub(crate) fn supported(_plan: &crate::plan::ScenePlan, bounded_transmission: bool) -> Result<()> {
    let unsupported = [
        (bounded_transmission, "transmission with bounded media (the shade does not link VORK_BOUNDED_MEDIUM interior transport)"),
    ];
    if let Some((_, feature)) = unsupported.iter().find(|(used, _)| *used) {
        bail!("decoupled shading does not support {feature} yet");
    }
    Ok(())
}

/// Visibility passes per merge: a batch's keys go to successive layers of one
/// array target and one dispatch merges them. A divisor of 32, so a batch's
/// samples share one coverage mask word (decoupled_merge.slang).
const MERGE_BATCH: u32 = vork_slang::registry::MERGE_BATCH;
const _: () = assert!(32 % MERGE_BATCH == 0);
/// The merges of `passes` visibility passes: (first sample, count) per batch of
/// `batch`, each within one coverage mask word.
fn merge_batches(passes: u32, batch: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..passes).step_by(batch as usize).map(move |first| (first, batch.min(passes - first)))
}
/// decoupled_visibility.slang transparent fragments group 2: the view's fragment list.
pub(crate) fn fragments_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    vec![pipelines::buffer_entry(0,wgpu::ShaderStages::FRAGMENT,wgpu::BufferBindingType::Storage {read_only:false})]
}
/// The same current-pose geometry and sampling table used by retained shading.
pub(crate) fn fragments_geometry_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let stage=wgpu::ShaderStages::FRAGMENT;
    std::iter::once(crate::rt::hit_buffers_entry(0,stage)).chain((1..=3).map(|binding|
        pipelines::buffer_entry(binding,stage,wgpu::BufferBindingType::Storage {read_only:true}))).collect()
}
pub(crate) fn fragments_geometry_layout(d:&wgpu::Device) -> wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("transparent opacity geometry"),entries:&fragments_geometry_entries()})
}
/// decoupled_merge.slang group 0: the batch's keys, slots, overflow, transparent
/// fragments and parameters (Install::merges, at the batch's dynamic offset).
pub(crate) fn merge_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let compute = wgpu::ShaderStages::COMPUTE;
    vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        pipelines::buffer_entry(1, compute, wgpu::BufferBindingType::Storage { read_only: false }),
        pipelines::buffer_entry(2, compute, wgpu::BufferBindingType::Storage { read_only: false }),
        pipelines::buffer_entry(3, compute, wgpu::BufferBindingType::Storage { read_only: true }),
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: compute,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(PARAMETERS),
            },
            count: None,
        },
    ]
}
// The ABI field list also generates camera_config.slang at build time.
macro_rules! camera_config {
    ($($field:ident:$rust:ty=>$slang:ident),*) => {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        pub(crate) struct CameraConfig {$(pub(crate) $field:$rust,)*}
    };
}
crate::camera_native::camera_config_fields!(camera_config);
const _: () = {
    assert!(std::mem::offset_of!(CameraConfig, boundaries) % 16 == 0);
    assert!(std::mem::size_of::<CameraConfig>() % 16 == 0);
};
/// Camera group 4, in dense descriptor order. Geometry lives in group 3.
pub(crate) fn camera_entries(samples: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
    let compute = wgpu::ShaderStages::COMPUTE;
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry { visibility: compute, ..pipelines::environment_texture_entry(1) },
        wgpu::BindGroupLayoutEntry { visibility: compute, ..pipelines::sampler_entry(2, false) },
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: compute, ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly, format: pipelines::resolved_ambient_format(samples), view_dimension: wgpu::TextureViewDimension::D2,
        }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 4, visibility: compute, ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba32Float, view_dimension: wgpu::TextureViewDimension::D2,
        }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 5, visibility: compute, ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba32Float, view_dimension: wgpu::TextureViewDimension::D2,
        }, count: None },
    ];
    entries.push(pipelines::buffer_entry(6, compute, wgpu::BufferBindingType::Storage { read_only: false }));
    entries.push(pipelines::buffer_entry(7, compute, wgpu::BufferBindingType::Storage { read_only: true }));
    // Screen refraction's receivers and opaque image: the opaque pass writes the
    // image's base level, the composite samples its pyramid.
    entries.push(wgpu::BindGroupLayoutEntry { visibility: compute, ..pipelines::texture_entry(8, wgpu::TextureSampleType::Float {filterable:true},false) });
    entries.push(wgpu::BindGroupLayoutEntry { visibility: compute, ..pipelines::sampler_entry(9, false) });
    entries.push(wgpu::BindGroupLayoutEntry { binding: 10, visibility: compute, ty: wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false,
    }, count: None });
    entries.push(wgpu::BindGroupLayoutEntry {binding:crate::camera_native::CONFIG_BINDING,visibility:compute,ty:wgpu::BindingType::Buffer {ty:wgpu::BufferBindingType::Uniform,has_dynamic_offset:true,min_binding_size:std::num::NonZeroU64::new(std::mem::size_of::<CameraConfig>() as u64)},count:None});
    entries.push(wgpu::BindGroupLayoutEntry {binding:12,visibility:compute,ty:wgpu::BindingType::StorageTexture {access:wgpu::StorageTextureAccess::WriteOnly,format:REFRACTION_FORMAT,view_dimension:wgpu::TextureViewDimension::D2},count:None});
    entries.push(pipelines::buffer_entry(13, compute, wgpu::BufferBindingType::Storage {read_only:false}));
    entries.push(pipelines::buffer_entry(14, compute, wgpu::BufferBindingType::Storage {read_only:true}));
    entries
}
/// Screen refraction's opaque image (decoupled.slang refraction_source).
const REFRACTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The opaque image pyramid's levels: down to transmission.slang's widest blur
/// (lod log2(1 + roughness^2 * REG_TRANSMISSION_BLUR_DIAMETER)), or the image's 1x1 level.
fn refraction_levels([w, h]: [u32; 2]) -> u32 {
    let widest = (1. + vork_slang::registry::TRANSMISSION_BLUR_DIAMETER).log2().ceil() as u32 + 1;
    widest.min(w.max(h).ilog2() + 1)
}
/// One merge batch's parameter block (decoupled_merge.slang MergePass).
const PARAMETERS: u64 = 32;
/// Clip offset, sample index and unit-disk lens point (decoupled_visibility.slang).
const VISIBILITY_PARAMETERS: u64 = 32;
pub(crate) fn merge_pipeline(p: &crate::pipelines::CachedPipelines, r: &Renderer) -> wgpu::ComputePipeline {
    p.decoupled_merge.get_or_init(|| {
        let d = &r.device;
        let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("decoupled merge"), entries: &merge_entries() });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decoupled merge"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        compute_pipeline(r, "decoupled merge", &layout, &native::DECOUPLED_MERGE_MERGE)
    }).clone()
}
#[cfg(test)]
pub(crate) fn camera_geometry_entries()->Vec<wgpu::BindGroupLayoutEntry> {camera_geometry_entries_for(1)}
pub(crate) fn camera_geometry_entries_for(evaluation_pages:u32)->Vec<wgpu::BindGroupLayoutEntry> {
    let mut pages=pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false});
    pages.count=std::num::NonZeroU32::new(evaluation_pages);
    let mut entries=vec![crate::rt::hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE),pages];
    for description in &crate::camera_native::TABLES {
        let (binding,count,read_only)=(description.binding,description.resources.len() as u32,description.read_only);
        let mut table=pipelines::buffer_entry(binding,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only});
        table.count=std::num::NonZeroU32::new(count);
        entries.push(table);
    }
    entries.push(crate::passes::reflections::record_fields_entry(5, false));
    entries
}
fn camera_geometry_layout(d:&wgpu::Device,pages:u32)->wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("camera geometry and evaluations"),entries:&camera_geometry_entries_for(pages)})
}
type EvaluationPipeline = pipelines::Drawn<wgpu::ComputePipeline>;
/// Setup has no scene registry identity. The containing device-cache namespace
/// already includes shader sources, compiler inputs, pipeline state and driver.
fn generic_key(shader:&native::Compiled,samples:u32)->blake3::Hash {
    blake3::hash(format!("camera generic {} samples {samples}",shader.name).as_bytes())
}
fn build_evaluation(c:&pipelines::Compiler,p:&crate::pipelines::CachedPipelines,layout:&wgpu::PipelineLayout,kernel:pipelines::ProgramKernel)->Result<wgpu::ComputePipeline> {
    let pipeline=kernel_pipeline(c,p,layout,kernel,&kernel.camera_label().expect("camera label"))?;
    Ok(pipeline)
}
fn evaluation_pipeline(r:&Renderer,p:&std::sync::Arc<crate::pipelines::CachedPipelines>,kernel:pipelines::ProgramKernel,mode:&pipelines::VariantCompile)->Result<EvaluationPipeline> {
    let c=&r.compiler;
    let layout=shade_layout(&r.device,p);
    let slot=p.camera_slot(kernel);
    let generic=p.shader.camera_generic(kernel);
    let defer=generic.is_some_and(|shader|mode.defer(slot.get().is_some() || c.is_known(&p.kernel_content_key(kernel)),c.is_known(&generic_key(shader,p.raster_samples))));
    let fallback=defer.then(|| {let shader=generic.expect("eligible camera generic");compute_pipeline(r,shader.name,&layout,shader)});
    let p=p.clone();
    c.install_variant(mode,slot,fallback,move |worker|build_evaluation(worker,&p,&layout,kernel))
}
fn kernel_pipeline(c:&pipelines::Compiler,p:&crate::pipelines::CachedPipelines,layout:&wgpu::PipelineLayout,kernel:pipelines::ProgramKernel,label:&str)->Result<wgpu::ComputePipeline> {
    let (module,entry)=p.program_kernel(kernel)?;
    let pipeline=c.compute_entry(layout,&module,entry,label);
    // Every successful camera route publishes the exact installed kernel,
    // including Indirect and Occupancy, which bypass build_evaluation.
    c.remember(p.kernel_content_key(kernel));
    Ok(pipeline)
}
/// Every fixed and specialized stage uses the same explicit camera layout.
fn shade_layout(d:&wgpu::Device,p:&crate::pipelines::CachedPipelines)->wgpu::PipelineLayout {
    let group=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("decoupled shade"),entries:&camera_entries(p.raster_samples)});
    d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("decoupled shade"),bind_group_layouts:&[Some(&p.global_layout),Some(&p.mat_layout),Some(&p.shadow_layout),Some(&camera_geometry_layout(d,p.camera_pages)),Some(&group)],immediate_size:0})
}
fn shade_pipeline(r: &Renderer, p: &crate::pipelines::CachedPipelines, kernel: crate::pipelines::ProgramKernel) -> Result<(wgpu::ComputePipeline,Vec<wgpu::ComputePipeline>)> {
    let crate::pipelines::ProgramKernel::Decoupled { stage, .. } = kernel else { unreachable!("decoupled shade kernel") };
    let layout=shade_layout(&r.device,p);
    let classification=if matches!(stage,ShadeStage::Index|ShadeStage::Classify) {
        let make_stage=|stage,label| {
            let mut k=kernel;if let crate::pipelines::ProgramKernel::Decoupled {stage:s,..}=&mut k {*s=stage;}
            kernel_pipeline(&r.compiler,p,&layout,k,label)
        };
        let mut stages=vec![make_stage(ShadeStage::Prefix,"camera material prefix")?];
        if stage==ShadeStage::Classify {stages.push(make_stage(ShadeStage::Scatter,"camera material scatter")?);}
        stages
    } else {Vec::new()};
    Ok((kernel_pipeline(&r.compiler,p,&layout,kernel,&kernel.camera_label().expect("camera kernel label"))?,classification))
}
fn compute_pipeline(r: &Renderer, label: &str, layout: &wgpu::PipelineLayout, shader: &native::Compiled) -> wgpu::ComputePipeline {
    r.compiler.compute_entry(layout,&native::shader_module(&r.device,shader),shader.entry,label)
}

/// Compile the shipped camera entries without installing or rendering a scene.
pub(crate) fn warm_shaders(r:&Renderer) {
    let d=&r.device;
    let rays=d.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    let group=|label,entries:&[_]|d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some(label),entries});
    let globals=group("frame + objects",&pipelines::global_entries());
    let profiles:Vec<_>=[false,true].into_iter().filter(|ray|!*ray || rays).map(|ray| (ray,
        group("material",&pipelines::material_entries(ray)),
        group("shadow read",&pipelines::camera_shadow_entries(ray)))).collect();
    let geometry=camera_geometry_layout(d,1);
    std::thread::scope(|scope| {
        for (rays,materials,shadows) in &profiles {
            for &(msaa,ray,stage,shader) in native::CAMERAS {
                if ray!=u8::from(*rays) {continue;}
                let (globals,geometry)=(&globals,&geometry);
                let samples:&[u32]=if matches!(stage,"evaluate"|"evaluate_opaque") {&[1,4]} else if msaa!=0 {&[4]} else {&[1]};
                for &samples in samples {
                    scope.spawn(move || {
                        let camera=group("decoupled shade",&camera_entries(samples));
                        let layout=d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label:Some("decoupled shade"),bind_group_layouts:&[Some(globals),Some(materials),Some(shadows),Some(geometry),Some(&camera)],immediate_size:0,
                        });
                        let _=compute_pipeline(r,shader.name,&layout,shader);
                        r.compiler.remember(generic_key(shader,samples));
                    });
                }
            }
        }
    });
}

/// The scene's shade kernel, the camera's draw table and the hit buffers over
/// the geometry its records address (rt::draw_table).
pub(crate) struct Install {
    shades: ShadePipelines,
    /// Slots can evaluate every covered sample, including table transparency.
    per_sample: bool,
    evaluation_words: u32,
    composition_flags: u32,
    materials: u32,
    draw_table: wgpu::Buffer,
    /// The draw of every object: visibility keys name the object (decoupled.slang key).
    object_draws: wgpu::Buffer,
    buffers: Vec<crate::rt::HitBuffer>,
}

#[derive(Clone)]
struct ShadePipelines {
    occupancy: Option<wgpu::ComputePipeline>,
    boundaries: u32,
    indirect:Option<wgpu::ComputePipeline>,
    fused:Option<EvaluationPipeline>,
    compose: wgpu::ComputePipeline,
    classification: Option<Vec<wgpu::ComputePipeline>>,
    evaluate: Vec<(u32,EvaluationPipeline)>,
}

/// Visibility offsets and merge budgets belong to a view's resolution and samples.
pub(crate) struct Sampling {
    passes: wgpu::Buffer,
    merges: wgpu::Buffer,
    stride: u32,
}
impl Sampling {
    pub(crate) fn new(d: &wgpu::Device, resolution: [u32; 2], samples: u32) -> Self {
        let batch = MERGE_BATCH.min(samples);
        // The same installed table is a constant vertex stream and the shade's
        // lens sequence. Its first record carries the sequence length.
        let mut passes = Vec::with_capacity(samples as usize + 1);
        passes.push([samples, 0, 0, 0, 0, 0, 0, 0]);
        passes.extend((0..samples).map(|sample| {
            let [x, y] = crate::sampling::clip_offset(resolution, sample as usize, samples as usize);
            let disk = crate::sampling::lens_disk(sample as usize, samples as usize);
            [x.to_bits(), y.to_bits(), sample, 0, disk.x.to_bits(), disk.y.to_bits(),
                (x*resolution[0] as f32*0.5).to_bits(), (y*resolution[1] as f32*0.5).to_bits()]
        }));
        // Whole-slot moments are view invariants, shared by every fully covered
        // pixel instead of walking the visibility lattice in each GPU lane.
        passes[0][1]=(passes.len()*2) as u32;
        let mut moments=[0f32;8];
        for sample in 0..samples {
            let [x,y]=crate::sampling::offset(sample as usize,samples as usize);
            moments[0]+=x;moments[1]+=y;moments[2]+=x*x;
            moments[3]+=y*y;moments[4]+=x*y;
            let disk=crate::sampling::lens_disk(sample as usize,samples as usize);
            moments[5]+=disk.x;moments[6]+=disk.y;
        }
        passes.push(moments.map(f32::to_bits));
        // A byte-mask lookup sums retained film moments without a
        // per-covered-sample loop in every slot.
        if samples.is_power_of_two() && samples <= 128 {
            passes[0][2]=(passes.len()*2) as u32;
            for first in (0..samples).step_by(8) {
                for mask in 0u32..256 {
                    let mut moments=[0f32;8];
                    for bit in 0..8 {
                        if first+bit>=samples || mask&(1<<bit)==0 {continue;}
                        let [x,y]=crate::sampling::offset((first+bit) as usize,samples as usize);
                        moments[0]+=x;moments[1]+=y;moments[2]+=x*x;
                        moments[3]+=y*y;moments[4]+=x*y;
                        let disk=crate::sampling::lens_disk((first+bit) as usize,samples as usize);
                        moments[5]+=disk.x;moments[6]+=disk.y;
                    }
                    passes.push(moments.map(f32::to_bits));
                }
            }
        }
        let passes = gpu_memory::buffer_init(d, "decoupled visibility passes", bytemuck::cast_slice(&passes), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let stride = d.limits().min_uniform_buffer_offset_alignment.max(PARAMETERS as u32);
        let mut blocks = Vec::new();
        for (first, count) in merge_batches(samples, batch) {
            blocks.extend_from_slice(bytemuck::cast_slice(&[first, count, samples, 0, 0, 0, 0, 0]));
            blocks.resize(blocks.len().next_multiple_of(stride as usize), 0);
        }
        let merges = gpu_memory::buffer_init(d, "decoupled merge batches", &blocks, wgpu::BufferUsages::UNIFORM);
        Self { passes, merges, stride }
    }
    pub(crate) fn pass_parameters(&self, sample: u32) -> wgpu::BufferSlice<'_> {
        let offset = (u64::from(sample) + 1) * VISIBILITY_PARAMETERS;
        self.passes.slice(offset..offset + VISIBILITY_PARAMETERS)
    }
}
impl Install {
    pub(crate) fn variants_ready(&self)->bool {self.shades.evaluate.iter().all(|(_,p)|p.ready())}
    pub(crate) fn deferred_variants(&self)->usize {self.shades.evaluate.iter().map(|(_,p)|p).chain(self.shades.fused.iter()).filter(|p|p.deferred()).count()}

    pub(crate) fn new(r: &Renderer, pipelines: &std::sync::Arc<crate::pipelines::CachedPipelines>, draws: &[install::Draw], features: crate::forward_variant::FeatureMask, plan: &crate::plan::ScenePlan, render: &vork_scene::Render, visibility: &[u32], records: &[crate::materials::MaterialRecord], program_features: &[crate::forward_variant::FeatureMask], ray_refraction: bool, boundary_capacity:u32, emitter_profiles:bool, variants:&pipelines::VariantCompile) -> Result<Self> {
        use crate::forward_variant::{FEATURE_CAMERA_RECEIVERS,FEATURE_REFLECTIONS,FEATURE_INDIRECT_CONTROLS};
        if plan.effects.probes && r.adapter_info.backend!=wgpu::Backend::Metal {indirect_arguments_pipeline(r);}
        let features=features
            | if plan.effects.indirect || plan.effects.reflections || plan.screen_refraction || plan.effects.probes {FEATURE_CAMERA_RECEIVERS} else {0}
            | if plan.effects.reflections {FEATURE_REFLECTIONS} else {0}
            | if plan.effects.indirect || plan.effects.probes {FEATURE_INDIRECT_CONTROLS} else {0};
        let d = &r.device;
        let (draw_table, buffers) = crate::rt::draw_table(r, draws)?;
        // Install gives every draw its own objects; a transparent draw's word has the top bit.
        let mut object_draws = vec![0u32; draws.iter().map(|draw| draw.instances.end as usize).max().unwrap_or(0).max(1)];
        for (index, draw) in draws.iter().enumerate() {
            for object in draw.instances.clone() {
                object_draws[object as usize] = index as u32 | u32::from(plan.draws[index].transparent) << 31;
            }
        }
        let object_draws = gpu_memory::buffer_init(d, "decoupled object draws", bytemuck::cast_slice(&object_draws), wgpu::BufferUsages::STORAGE);
        let material_count=records.iter().map(|r|r.tag[0]).max().unwrap_or(0)+2;
        let opacity=vork_scene::v2::OPENPBR.iter().position(|p|p.name=="geometry_opacity").expect("OpenPBR opacity");
        // ClassifyEvaluation also expands ordinary table transparency: a solid
        // rear can be hidden on only some samples, a transmitting silhouette
        // varies across its coverage, and sampled front opacity expands receivers.
        // These are visibility-dependent even with constant material inputs and
        // ray refraction. The installed transparent plan must reserve them before
        // the first exposure, rather than learning the policy through overflow.
        let per_sample=draws.iter().any(|d|d.implicit.is_some()) || material_count!=2 || plan.transparent || plan.screen_refraction || records.iter().any(|r|
            r.tag[2]&(constants::FLAG_VERTEX_ALPHA|vork_slang::registry::FLAG_SAMPLE_MATERIAL)!=0 || (r.inputs[opacity]&constants::MATERIAL_TAP_INDEX_MASK)!=0);
        let evaluation_words=evaluation_words(features,plan.transparent,plan.effects.subsurface,render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness),render.aovs.len() as u32);
        let program_features=|id:u32| program_features[id as usize]
            | (features & (FEATURE_CAMERA_RECEIVERS|FEATURE_REFLECTIONS|FEATURE_INDIRECT_CONTROLS));
        let kernel = |stage| {
            // Ray refraction shades arbitrary receivers and retains their union.
            let features=match stage {
                ShadeStage::EvaluateProgram(id) if !ray_refraction=>program_features(id),
                ShadeStage::EvaluateOpaque=>features & !(crate::forward_variant::FEATURE_TRANSMISSION|crate::forward_variant::FEATURE_SOLID_TRANSMISSION),
                _=>features,
            };
            let features=features | if stage==ShadeStage::Compose && plan.screen_refraction && plan.effects.subsurface {crate::forward_variant::FEATURE_RECEIVER_DIFFUSION} else {0};
            // Only transport evaluators and the containment prepass need the
            // installed active-set capacity. Opaque/classify/compose stages keep
            // their existing zero-transport kernels and generic fallbacks.
            let boundaries=if matches!(stage,ShadeStage::Evaluate|ShadeStage::EvaluateProgram(_)|ShadeStage::Occupancy) {boundary_capacity} else {0};
            pipelines.drawn_kernel(crate::pipelines::ProgramKernel::Decoupled {stage,features,subsurface:render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness),emitter_profiles,boundaries},variants,&r.compiler)
        };
        let compile=|stage|shade_pipeline(r,pipelines,kernel(stage));
        let implicit_slots=!per_sample && !plan.transparent;
        // Table-only camera and secondary transport, with no extra composition stages.
        // Solid scene transport requires the occupancy-backed evaluator; the fused
        // table compositor has no transport capacity. Exact field traversal stays
        // in staged evaluation: fusing its nested ordering/transport walks makes
        // the NVIDIA driver compiler expand the same solver many times.
        let fuse_table=boundary_capacity==0 && !variants.fallback() && !draws.iter().any(|d|d.implicit.is_some())
            && visibility.iter().all(|&n|n==1) && material_count==2 && pipelines.shader.generic_shipped()
            && !plan.effects.probes && !plan.screen_refraction && !plan.effects.subsurface
            && render.aovs.is_empty();
        let needs_classification=material_count!=2 || visibility.iter().any(|&samples|!direct_visibility(samples,plan.transparent,plan.effects.subsurface));
        // Each program owns one bin; table slots split by transparency.
        let used:std::collections::BTreeSet<_>=draws.iter().zip(&plan.draws).map(|(draw,flags)| {
            if records[draw.material].tag[0]!=0 {records[draw.material].tag[0]} else if flags.transparent {0} else {material_count-1}
        }).collect();
        let used:Vec<_>=used.into_iter().collect();
        let cursor=std::sync::atomic::AtomicUsize::new(0);
        let shades=std::thread::scope(|scope| {
            let compose=scope.spawn(||compile(ShadeStage::Compose));
            let classify=needs_classification.then(||scope.spawn(|| {
                let (classify,mut stages)=compile(if implicit_slots {ShadeStage::Index} else {ShadeStage::Classify})?;
                stages.insert(0,classify);anyhow::Ok(stages)
            }));
            let jobs:Vec<_>=(0..used.len().min(std::thread::available_parallelism().map_or(1,usize::from))).map(|_|scope.spawn(|| {
                let mut results=Vec::new();
                loop {
                    let index=cursor.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                    let Some(&tag)=used.get(index) else {break;};
                    let stage=if tag==material_count-1 {ShadeStage::EvaluateOpaque} else if tag==0 {ShadeStage::Evaluate} else {ShadeStage::EvaluateProgram(tag)};
                    let pipeline=evaluation_pipeline(r,pipelines,kernel(stage),variants)?;
                    results.push((tag,pipeline));
                }
                anyhow::Ok(results)
            })).collect();
            let evaluate=jobs.into_iter().map(pipelines::join).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect();
            let (compose,_)=pipelines::join(compose)?;
            let classification=classify.map(pipelines::join).transpose()?;
            let fused=fuse_table.then(|| {
                let key=kernel(ShadeStage::Fused);let slot=pipelines.camera_slot(key);
                let layout=shade_layout(&r.device,pipelines);let p=pipelines.clone();
                let warm=r.compiler.is_known(&p.kernel_content_key(key));
                r.compiler.install_optional(variants,slot,warm,move |worker|build_evaluation(worker,&p,&layout,key))
            }).transpose()?;
            let occupancy=(boundary_capacity>0).then(||compile(ShadeStage::Occupancy).map(|p|p.0)).transpose()?;
            let indirect=plan.effects.probes.then(||compile(ShadeStage::Indirect).map(|p|p.0)).transpose()?;
            anyhow::Ok(ShadePipelines {occupancy,boundaries:boundary_capacity,indirect,fused,compose,evaluate,classification})
        })?;
        let composition_flags= (u32::from(plan.effects.probes)*vork_slang::registry::CAMERA_PROBE_WORK) | (u32::from(features & crate::forward_variant::FEATURE_LIT!=0)*vork_slang::registry::CAMERA_PROBE_LIT) | (u32::from(implicit_slots)*CAMERA_IMPLICIT_SLOTS) | (u32::from(render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness))<<5) | (u32::from(ray_refraction)<<4) | u32::from(features & FEATURE_CAMERA_RECEIVERS != 0)
            | (u32::from(features & FEATURE_REFLECTIONS != 0)<<1)
            | (u32::from(features & FEATURE_INDIRECT_CONTROLS != 0)<<2)
            | (u32::from(features & crate::forward_variant::FEATURE_SOLID_TRANSMISSION != 0)<<3);
        Ok(Self { shades, per_sample, evaluation_words, composition_flags, materials:material_count, draw_table, object_draws, buffers })
    }

}

/// A view's transparent fragment lists (decoupled_slots.slang FragmentLayout): the
/// header, records (one per transparent triangle per pixel per merge batch: at
/// least two per pixel, or the installed reservation if larger) and each batch
/// pass's head per pixel; only the header, of capacity zero, without transparency.
pub(crate) struct Fragments {
    pub(crate) buffer: wgpu::Buffer,
    /// Visibility group 2 over the list, for transparent draws' fragment stage.
    pub(crate) bind: Option<wgpu::BindGroup>,
    pub(crate) geometry: Option<wgpu::BindGroup>,
    /// The heads' byte offset, cleared with the allocations for every merge batch.
    heads: u64,
    explicit_depth: bool,
    lists: u64,
    /// Records reach the device storage binding limit: an overflow cannot grow them.
    pub(crate) full: bool,
}
fn fragment_list_bytes(pixels:u64,samples:u32,transparent:bool,fields:bool)->u64 {
    if transparent {u64::from(MERGE_BATCH.min(samples))*pixels*if fields {8} else {4}} else {0}
}
impl Fragments {
    pub(crate) fn new(s: &InstalledScene, r: &Renderer, [w, h]: [u32; 2], samples: u32, sampling:&Sampling, deforms:&[Option<EvaluatedGeometry>]) -> Result<Self> {
        let d = &r.device;
        let pixels = u64::from(w) * u64::from(h);
        let (header, record) = (u64::from(vork_slang::registry::FRAGMENT_HEADER_WORDS) * 4, u64::from(vork_slang::registry::FRAGMENT_RECORD_WORDS) * 4);
        let explicit_depth = s.plan.transparent && s.pipelines.field_controls;
        let lists = fragment_list_bytes(pixels,samples,s.plan.transparent,explicit_depth);
        let limits = d.limits();
        let limit = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        ensure!(header + lists <= limit, "{}", crate::gpu_memory::limit_message("transparent fragment lists", format_args!("{} bytes",header+lists), format_args!("{limit} bytes"), "reduce resolution or samples with --set samples=N"));
        // A closed transparent solid lists its front and rear face at a pixel;
        // a larger reservation grows to the binding limit.
        let most = ((limit - header - lists) / record).min(u64::from(u32::MAX));
        let capacity = if s.plan.transparent { (2 * pixels).max(u64::from(s.reservations.transparent_fragments).next_power_of_two()).min(most) } else { 0 };
        let heads = header + capacity * record;
        let bytes = heads + lists;
        let buffer = gpu_memory::buffer(d, &wgpu::BufferDescriptor {
            label: Some("transparent fragments"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        r.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[0, 0, u32::try_from(capacity)?, w, h]));
        let bind = Some(d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transparent fragments"),
            layout: &s.pipelines.fragments_layout,
            entries: &[entry(0,buffer.as_entire_binding())],
        }));
        let geometry = bind.as_ref().map(|_| {
            let installed=s.decoupled.as_ref().expect("installed decoupled geometry");
            let buffers=crate::rt::resolve_hit_buffers(&installed.buffers,deforms);
            let bindings=crate::rt::hit_bindings(&buffers);
            d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("transparent opacity geometry"),layout:&s.pipelines.fragments_geometry_layout,entries:&[
                entry(0,wgpu::BindingResource::BufferArray(&bindings)),
                entry(1,installed.draw_table.as_entire_binding()),entry(2,installed.object_draws.as_entire_binding()),
                entry(3,sampling.passes.as_entire_binding()),
            ]})
        });
        Ok(Self { buffer, bind, geometry, heads, explicit_depth, lists, full: capacity == most })
    }
}

pub(crate) fn indirect_arguments_entries()->[wgpu::BindGroupLayoutEntry;4] {
    [pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),
            pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}),
            pipelines::buffer_entry(2,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Uniform),
            pipelines::buffer_entry(3,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})]
}

fn indirect_arguments_pipeline(r:&Renderer)->&wgpu::ComputePipeline {
    r.pipelines.camera_indirect_args.get_or_init(|| {
        let entries=indirect_arguments_entries();
        let bind=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("camera indirect arguments"),entries:&entries});
        let layout=r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("camera indirect arguments"),bind_group_layouts:&[Some(&bind)],immediate_size:0});
        compute_pipeline(r,"size unsupported diffuse connections",&layout,&native::CAMERA_INDIRECT_ARGS_PREPARE)
    })
}

/// One bounded work block's configuration and direct-dispatch extent.
#[derive(Clone, Copy)]
struct IndirectBlock {
    offset:u32,
    groups:u32,
}

/// One camera view's visibility targets, slots and recorded passes.
pub(crate) struct View {
    pub(crate) receivers: wgpu::TextureView,
    pub(crate) normals: wgpu::TextureView,
    /// Install::merges' block stride (the merge's dynamic offset per batch).
    stride: u32,
    /// One view per layer of the batch's key array (render attachments).
    keys: Vec<wgpu::TextureView>,
    depth: wgpu::TextureView,
    pub(crate) slots: wgpu::Buffer,
    pub(crate) subsurface_records: wgpu::Buffer,
    pub(crate) refraction_source: Option<wgpu::TextureView>,
    /// Each slot's reflection record and each pixel's reflection reconstruction
    /// guides, for the reflection passes (passes/reflections.rs).
    pub(crate) reflection_records: Vec<wgpu::Buffer>,
    pub(crate) reflection_field_bytes: u64,
    pub(crate) reflection_fields_per_buffer: u32,
    pub(crate) reflection_guides: wgpu::Buffer,
    /// Bytes of dense slots and list heads; merged views clear these per exposure.
    listed: u64,
    /// One opaque sample already is one slot; shade and reflections need no slot buffer.
    direct_keys: bool,
    /// The overflow pool's allocations this exposure, the frame's demand once it
    /// was full, and its entries (decoupled_merge.slang); frame.rs grows the
    /// reservation to a nonzero demand.
    pub(crate) overflow: wgpu::Buffer,
    capacity_feedback: wgpu::Buffer,
    pub(crate) fragments: Fragments,
    passes: Vec<wgpu::RenderBundle>,
    /// Each pass's transparent fragments, drawn after its opaque keys; empty without transparency.
    transparent_passes: Vec<wgpu::RenderBundle>,
    merge_pipeline: wgpu::ComputePipeline,
    shade_pipeline: wgpu::ComputePipeline,
    evaluation: ShadePipelines,
    evaluation_bins: wgpu::Buffer,
    evaluation_args: Option<wgpu::Buffer>,
    material_count: u32,
    config_stride:u32,
    transport_parts:u32,
    /// Indirect dispatch blocks by band, then coverage word.
    indirect_offsets:Vec<Vec<Vec<IndirectBlock>>>,
    evaluation_allocator:wgpu::Buffer,
    /// GPU-sized bounded work blocks; Metal retains its direct launch policy.
    indirect_dispatch:Option<(wgpu::ComputePipeline,wgpu::BindGroup,wgpu::Buffer,u32)>,
    /// Shading bands of `band_rows` pixel rows (the last may be shorter); each
    /// band's dispatch data starts at block band * materials.
    pub(crate) bands: u32,
    pub(crate) band_rows: u32,
    pub(crate) packets: bool,
    pub(crate) packet_reflections:std::sync::OnceLock<Vec<Vec<crate::graph::Dispatch>>>,
    evaluation_groups:u32,
    merge: wgpu::BindGroup,
    /// The forward pass's scene globals, material and shadow groups, the hit buffers, the shade's own group.
    shade: [wgpu::BindGroup; 5],
    refraction: Option<Refraction>,
    pub(crate) extent: [u32; 2],
}
/// Screen refraction after the opaque shade: the opaque image's pyramid levels
/// (each drawn from the one before) and the transparent composite, whose own
/// group samples the pyramid.
struct Refraction {
    mips: crate::materials::MipChain,
    shade: wgpu::BindGroup,
}
impl View {
    pub(crate) fn samples(&self)->u32 {self.passes.len() as u32}
    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        extent: [u32; 2],
        sampling: &Sampling,
        targets: &crate::view::Targets,
        [passes, transparent_passes]: [Vec<wgpu::RenderBundle>; 2],
        fragments: Fragments,
        deforms: &[Option<EvaluatedGeometry>],
        [global, material, shadow]: [wgpu::BindGroup; 3],
        shader_errors:Option<&wgpu::Buffer>,
        emitter_weights:Option<&wgpu::Buffer>,
    ) -> Result<Self> {
        let d = &r.device;
        let installed = s.decoupled.as_ref().context("decoupled shading install")?;
        let samples = passes.len() as u32;
        let direct_keys=direct_visibility(samples,s.plan.transparent,s.plan.effects.subsurface);
        let [w, h] = extent;
        let limits = d.limits();
        let shape = ViewShape {transparent: s.plan.transparent, diffusion: s.plan.effects.subsurface, reflections: s.plan.effects.reflections, screen_refraction: s.plan.screen_refraction, boundaries:installed.shades.boundaries,
            table_only: installed.materials == 2, reserved: s.reservations.visibility_pool};
        let eligible=packet_eligible(shape,samples) && !s.has_media;
        let memory=r.camera_memory;
        let work_stride=if installed.composition_flags&vork_slang::registry::CAMERA_PROBE_LIT!=0 {24} else if s.plan.effects.subsurface {16} else {8};
        let packet_rows=if eligible {camera_packet_rows(&limits,shape,extent,samples,installed.evaluation_words,memory,installed.materials,
            if s.plan.effects.probes {work_stride} else {0})?} else {None};
        let packets=packet_rows.is_some();
        let packet_rows=packet_rows.unwrap_or(h);
        let storage_extent=[w,packet_rows];
        let storage_pixels=u64::from(w)*u64::from(packet_rows);
        let layers = MERGE_BATCH.min(samples);
        let key_array = s.transients.texture(d, &wgpu::TextureDescriptor {
            label: Some("visibility keys"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: KEY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let keys: Vec<_> = (0..layers).map(|layer| key_array.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        })).collect();
        let batch = key_array.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let depth = s.transients.target(d, "visibility depth", w, h, 1, wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        // decoupled_slots.slang SlotLayout: dense slots and a list head per pixel, then
        // the pool, bounded by all dependent bindings or the measured reservation.
        let pixels = u64::from(w) * u64::from(h);
        let stride = u64::from(VISIBILITY_KEY_WORDS + samples.div_ceil(32));
        let listed = storage_pixels * (u64::from(VISIBILITY_SLOTS) * stride + 1) * 4;
        let pool = if packets {storage_pixels*u64::from(samples.saturating_sub(VISIBILITY_SLOTS))} else {visibility_pool(&limits, shape, extent, samples)};
        let [bytes, metadata_size, capture_bytes, record_bytes, mut guide_bytes] = if packets {
            view_bindings_with_pool(shape,storage_extent,samples,pool).map(|(_,bytes)|bytes)
        } else {check_view(&limits, extent, samples, shape)?};
        if packets {
            guide_bytes=view_bindings_with_pool(shape,extent,samples,0)[4].1;
            for (label,bytes) in view_bindings_with_pool(shape,storage_extent,samples,pool) {gpu_memory::storage_binding(&limits,label,bytes, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;}
            gpu_memory::storage_binding(&limits,"reflection guides",guide_bytes, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;
        }
        let occupancy_row=occupancy_row_bytes(w,samples,shape.boundaries);
        let metadata_size=metadata_size-occupancy_row*u64::from(packet_rows.min(TILE));
        let slots = s.transients.buffer(d, "visibility slots", bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        let overflow = gpu_memory::buffer_init(d, "visibility slot overflow", bytemuck::cast_slice(&[0, 0, u32::try_from(pool)?]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        let merge_pipeline = merge_pipeline(&s.pipelines, r);
        let shade_pipeline = &installed.shades;
        let evaluation=shade_pipeline.clone();
        let shade_pipeline = shade_pipeline.compose.clone();
        let packet_merges=if packets {
            let mut blocks=Vec::new();
            for band in 0..h.div_ceil(packet_rows) {
                for (first,count) in merge_batches(samples,layers) {
                    blocks.extend_from_slice(bytemuck::cast_slice(&[first,count,samples,storage_pixels as u32,band*packet_rows,0,0,0]));
                    blocks.resize(blocks.len().next_multiple_of(sampling.stride as usize),0);
                }
            }
            Some(gpu_memory::buffer_init(d,"packet visibility dispatches",&blocks,wgpu::BufferUsages::UNIFORM))
        } else {None};
        let merge = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoupled merge"),
            layout: &merge_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(&batch)),
                entry(1, slots.as_entire_binding()),
                entry(2, overflow.as_entire_binding()),
                entry(3, fragments.buffer.as_entire_binding()),
                entry(4, wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: packet_merges.as_ref().unwrap_or(&sampling.merges), offset: 0, size: std::num::NonZeroU64::new(PARAMETERS) })),
            ],
        });
        let [aw, ah] = if s.plan.effects.indirect || s.plan.effects.reflections { extent } else { [1, 1] };
        let receivers = s.transients.target(d, "decoupled AO receivers", aw, ah, 1, wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let normals = s.transients.target(d, "decoupled AO normals", aw, ah, 1, wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let refraction_sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("refraction pyramid"), mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let slot_records=storage_pixels * u64::from(VISIBILITY_SLOTS) + pool;
        let subsurface_records = s.transients.buffer(d, "subsurface records", capture_bytes, wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        // Keep each packed word in its own aligned slice of one allocation. Bind
        // slices separately: the complete record array need not fit one binding.
        // SlotLayout.reflection is the common record index in all fields.
        let reflection_field_bytes=record_bytes.div_ceil(u64::from(limits.min_storage_buffer_offset_alignment))*u64::from(limits.min_storage_buffer_offset_alignment);
        // Each packed field is its own binding; pack as many fields per allocation
        // as the device allows so the complete record array can exceed one buffer.
        let reflection_fields_per_buffer=reflection_fields_per_buffer(reflection_field_bytes,limits.max_buffer_size);
        let reflection_records:Vec<_>=(0..VISIBILITY_REFLECTION_WORDS).step_by(reflection_fields_per_buffer as usize).enumerate().map(|(chunk,first)|
            s.transients.buffer(d,&format!("reflection records {chunk}"),reflection_field_bytes*u64::from((VISIBILITY_REFLECTION_WORDS-first).min(reflection_fields_per_buffer)),wgpu::BufferUsages::STORAGE)).collect();
        let reflection_guides = s.transients.buffer(d, "reflection guides", guide_bytes, wgpu::BufferUsages::STORAGE);
        let screen = s.plan.screen_refraction;
        let refraction_receivers = s.transients.buffer(d, "refraction receivers", if screen { pixels * 16 } else { 16 }, wgpu::BufferUsages::STORAGE);
        let [rw, rh] = if screen { extent } else { [1, 1] };
        let levels = refraction_levels([rw, rh]);
        let pyramid = s.transients.texture(d, &wgpu::TextureDescriptor {
            label: Some(if screen { "refraction pyramid" } else { "unused refraction image" }),
            size: wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: REFRACTION_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let level_zero = pyramid.create_view(&wgpu::TextureViewDescriptor { mip_level_count: Some(1), ..Default::default() });
        let implicit_slots=!installed.per_sample && !s.plan.transparent && !direct_keys;
        // One table evaluation per pixel (decoupled.slang direct_evaluations):
        // no slot records, metadata or classification.
        let direct=direct_keys && installed.materials==2;
        let occupancy_offset=u32::try_from(metadata_size/4)?;
        // Every dense/overflow slot can emit a centroid evaluation; per-sample
        // materials can evaluate each of the view's samples, plus a transparent
        // layer's. Their evaluations scale with pixels x samples, so the view
        // classifies, evaluates and composes in bands of rows that fit the band
        // budget; the bands reuse one evaluation store. Replay reserves the
        // reported demand (the frame's, estimated from its largest band's) and
        // adds bands until every band fits.
        let work_stride=u64::from(work_stride);
        let header=u64::from(vork_slang::registry::CAMERA_INDIRECT_HEADER_WORDS);
        let work_limit=if s.plan.effects.probes {
            limits.max_storage_buffer_binding_size.min(limits.max_buffer_size).saturating_sub(header*4)/(work_stride*4)
        } else {u64::MAX};
        let limit=evaluation_limit(&limits,installed.materials,installed.evaluation_words).min(work_limit);
        let memory=r.device_memory;
        #[cfg(test)]
        let memory=r.evaluation_memory.unwrap_or(memory);
        let budget=evaluation_budget(&limits,memory,installed.evaluation_words).min(limit).max(1);
        // The installation's demand may come from another view; this view's own
        // slots bound it (each evaluates at most every sample it covers).
        let demand=s.reservations.shading_evaluations;
        // The estimate includes headroom for rows holding more than an even share.
        let content=if implicit_slots {slot_records} else if installed.per_sample {
            let records=pixels*(u64::from(samples)+if s.plan.transparent {u64::from(VISIBILITY_SLOTS)} else {0});
            records+records/EVALUATION_ESTIMATE_HEADROOM
        } else {0};
        #[cfg(test)]
        let content=r.evaluation_estimate.unwrap_or(content);
        // Every classified route retains compact band-local results, including
        // opaque centroid slots. Containment can refine these same bands.
        let (_,band_rows)=if !direct_keys {shading_bands(content.max(demand.min(slot_records*u64::from(samples))),budget,h)} else {(1,h)};
        let (bands,band_rows)=if packets {(h.div_ceil(packet_rows),packet_rows)} else {occupancy_bands(&limits,memory,metadata_size,occupancy_row,h,band_rows)?};
        // A full band's pixels and slots (the whole pool may fall in one band),
        // and its share of whole-frame counts; the last band may be shorter.
        let band_pixels=u64::from(w)*u64::from(band_rows);
        let band_slots=band_pixels*u64::from(VISIBILITY_SLOTS)+pool;
        let in_band=|records:u64|(records*u64::from(band_rows)).div_ceil(u64::from(h));
        // Direct views evaluate one record per pixel; classification never overflows them.
        let evaluation_capacity=u32::try_from(if packets {storage_pixels*u64::from(samples)} else if direct_keys {pixels} else {
            // The demand is an estimate, possibly another view's: a band reserves at
            // most what its slots and one band's addressing allow; a band that truly
            // needs more stops redo.
            evaluation_capacity(band_slots.min(slot_records).min(budget),in_band(content),in_band(demand).min(band_slots*u64::from(samples)).min(limit),budget)
        }).context("camera evaluation addressing")?;
        let page_capacity=evaluation_page_capacity(&limits,installed.evaluation_words);
        ensure!(evaluation_capacity.div_ceil(page_capacity)<=s.pipelines.camera_pages,"camera evaluation page layout disagrees with its installed word width");
        // Each occupied slot appends at most one ID; evaluation reservations do not
        // enlarge this list. Per-sample bins retain their evaluation capacity.
        let bin_bytes=u64::from(installed.materials)*24+if direct {0} else if implicit_slots {slot_records*4} else {u64::from(evaluation_capacity)*16};
        crate::gpu_memory::storage_binding(&limits,"camera material bins",bin_bytes, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;
        let evaluation_bins=s.transients.buffer(d,"camera material bins",bin_bytes,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC);
        // The allocator holds this view's capacities: its own buffer, never a
        // transient another view of the same size would share.
        let work_words=header+if s.plan.effects.probes {u64::from(evaluation_capacity)*work_stride} else {0};
        crate::gpu_memory::storage_binding(&limits,"camera indirect work",work_words*4, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;
        // Only the header is initial state. Each active work record is written
        // by StoredEvaluation before indirect_material reads it.
        let evaluation_allocator=gpu_memory::buffer(d,&wgpu::BufferDescriptor {
            label:Some("camera evaluation allocator"),size:work_words*4,
            usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
        });
        let mut initial=[0u32;vork_slang::registry::CAMERA_INDIRECT_HEADER_WORDS as usize];initial[1]=evaluation_capacity;
        r.queue.write_buffer(&evaluation_allocator,0,bytemuck::cast_slice(&initial));
        // Table views normally queue a sparse subset of camera receivers. Start
        // from one request per pixel plus the existing estimate headroom. This
        // is a reservation, not a cap: excess demand replays through the same
        // camera evaluation reservation. This estimate covers centroid records
        // only. Single-sample records grow from the other end of the same work
        // buffer and retain its full capacity with fewer, equally bounded launches.
        let centroid_capacity=if installed.materials==2 && r.adapter_info.backend!=wgpu::Backend::Metal {
            let block=u64::from(vork_slang::registry::CAMERA_INDIRECT_BLOCK);
            ((band_pixels+band_pixels/EVALUATION_ESTIMATE_HEADROOM).max(block).max(in_band(demand).next_power_of_two()).div_ceil(block)*block).min(u64::from(evaluation_capacity)) as u32
        } else {evaluation_capacity};
        // One sampled record traces one connection per word, versus up to 32
        // for a centroid. Late authored hooks also run once per record: those
        // scenes retain the original mixed queue and its invocation bound.
        let split=installed.composition_flags&vork_slang::registry::CAMERA_PROBE_LIT==0;
        let sampled_capacity=if !split {0} else if installed.per_sample || samples==1 {evaluation_capacity} else {centroid_capacity};
        let sampled_block=vork_slang::registry::CAMERA_INDIRECT_SAMPLE_BLOCK;
        #[cfg(test)]
        if std::env::var_os("VORK_PROBE_COST_SCENE").is_some() {eprintln!("camera indirect layout: per_sample={} transparent={} evaluation_capacity={evaluation_capacity} centroid_capacity={centroid_capacity} sampled_capacity={sampled_capacity} bands={bands} work_words={work_words}",installed.per_sample,s.plan.transparent);}
        let indirect_dispatch=if s.plan.effects.probes && r.adapter_info.backend!=wgpu::Backend::Metal {
            let centroids=centroid_capacity.div_ceil(vork_slang::registry::CAMERA_INDIRECT_BLOCK);
            let blocks=centroids+sampled_capacity.div_ceil(sampled_block);
            let arguments=s.transients.buffer(d,"camera indirect arguments",u64::from(blocks)*12,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::INDIRECT);
            let pipeline=indirect_arguments_pipeline(r).clone();
            let params=gpu_memory::buffer_init(d,"camera indirect launch capacity",bytemuck::cast_slice(&[centroid_capacity,sampled_capacity,centroids,sampled_block]),wgpu::BufferUsages::UNIFORM);
            let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("camera indirect arguments"),layout:&pipeline.get_bind_group_layout(0),entries:&[
                entry(0,wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&evaluation_allocator,offset:0,size:std::num::NonZeroU64::new(header*4)})),
                entry(1,arguments.as_entire_binding()),entry(2,params.as_entire_binding()),
                entry(3,shader_errors.context("deferred camera work requires capacity diagnostics")?.as_entire_binding()),
            ]});
            Some((pipeline,bind,arguments,blocks))
        } else {None};
        let evaluation_args=(!direct && r.adapter_info.backend!=wgpu::Backend::Metal).then(||s.transients.buffer(d,"camera material dispatches",u64::from(installed.materials)*24,wgpu::BufferUsages::INDIRECT|wgpu::BufferUsages::COPY_DST));
        let evaluation_pages:Vec<_>=(0..evaluation_capacity.div_ceil(page_capacity).max(1)).map(|page| {
            let count=evaluation_capacity.saturating_sub(page*page_capacity).min(page_capacity).max(1);
            s.transients.buffer(d,&format!("camera evaluations {page}"),u64::from(count)*u64::from(installed.evaluation_words)*16,wgpu::BufferUsages::STORAGE)
        }).collect();

        // Reuse this band's per-camera-ray containment across every slot and lobe.
        let occupancy_bytes=occupancy_row*u64::from(band_rows);
        gpu_memory::storage_binding(&limits,"camera evaluation metadata and dielectric occupancy",metadata_size+occupancy_bytes, "reduce resolution or choose --profile std/draft for a lower sample and reflection budget")?;
        let evaluation_metadata=s.transients.buffer(d,"camera evaluation metadata",metadata_size+occupancy_bytes,wgpu::BufferUsages::STORAGE);

        let config_size=std::mem::size_of::<CameraConfig>();
        let config_stride=(config_size as u32).next_multiple_of(d.limits().min_uniform_buffer_offset_alignment);
        // One block per band and material: (band * materials + material) * stride.
        let transport_parts=if installed.composition_flags&16!=0 && s.plan.transparent {
            transport_parts(samples,r.adapter_info.backend)
        } else {1u32};
        let mut configs=vec![0u8;(transport_parts*bands*installed.materials) as usize*config_stride as usize];
        let mut composite_configs=if screen {vec![0u8;configs.len()]} else {Vec::new()};
        for (band,material) in (0..bands).flat_map(|band|(0..installed.materials).map(move |material|(band,material))) {
            let mut config=CameraConfig {
                boundaries:s.rays.as_ref().map_or([0;4],|r|[r.boundaries,r.occupancy_solids,r.object_boundaries,occupancy_offset]),
                materials:installed.materials, material, samples, direct:u32::from(direct_keys),
                diffusion:u32::from(s.plan.effects.subsurface), aovs:s.render.aovs.len() as u32,
                capacity:page_capacity, words:installed.evaluation_words,
                ray_shadows:u32::from(s.pipelines.ray_shadows), lit:u32::from(!screen), screen:u32::from(screen),
                flags:installed.composition_flags, transparent:u32::from(s.plan.transparent),
                transmittance_offset:u32::try_from(slot_records*u64::from(vork_slang::registry::CAMERA_EVALUATION_METADATA_WORDS))?,
                fibre_light_count:if s.fibre.is_some() {s.light_plan.nodes.len() as u32} else {0},
                band:band*band_rows,
                receiver_record_offset:if screen && s.plan.effects.subsurface {u32::try_from(capture_bytes / 8)?} else {0},
                seed:s.render.settings.seed,
                transport_parts, transport_part:0, packet_pixels:if packets {storage_pixels as u32} else {0}, packet_first_pixel:if packets {u32::try_from(u64::from(band)*u64::from(band_rows)*u64::from(w))?} else {0}, packet_pad1:0, packet_pad2:0,

            };
            for part in 0..transport_parts {
                config.transport_part=part;
                let at=((part*bands+band)*installed.materials+material) as usize*config_stride as usize;
                config.lit=u32::from(!screen);
                configs[at..at+config_size].copy_from_slice(bytemuck::bytes_of(&config));
                if screen {
                    config.lit=1;
                    composite_configs[at..at+config_size].copy_from_slice(bytemuck::bytes_of(&config));
                }
            }
        }
        // Work records reuse the evaluation reservation. Each dispatch covers at
        // most one block's connections and one 32-bit coverage word, independently
        // of the complete camera budget. No receiver restarts that budget.
        // Grouped by word: one native pass per word, not per dispatch block.
        let mut indirect_offsets=vec![Vec::new();bands as usize];
        if s.plan.effects.probes {
            for band in 0..bands {
                let first=(band*installed.materials) as usize*config_stride as usize;
                let template:CameraConfig=bytemuck::pod_read_unaligned(&configs[first..first+config_size]);
                let ray_batch=32;
                for word in 0..samples.div_ceil(ray_batch) {
                    let mut blocks=Vec::new();
                    for (sampled,capacity,size) in [(false,centroid_capacity,vork_slang::registry::CAMERA_INDIRECT_BLOCK),(true,sampled_capacity,sampled_block)] {
                        for offset in (0..capacity).step_by(size as usize) {
                            let mut block=template;block.material=word;block.receiver_record_offset=offset;
                            if sampled {block.flags|=vork_slang::registry::CAMERA_INDIRECT_SAMPLED;}
                            blocks.push(IndirectBlock {offset:configs.len() as u32,
                                groups:size.min(capacity-offset).div_ceil(vork_slang::registry::CAMERA_INDIRECT_GROUP)});
                            let append=|bytes:&mut Vec<u8>| {bytes.extend_from_slice(bytemuck::bytes_of(&block));bytes.resize(bytes.len().next_multiple_of(config_stride as usize),0);};
                            append(&mut configs);
                            if screen {block.lit=1;composite_configs.extend_from_slice(bytemuck::bytes_of(&block));composite_configs.resize(composite_configs.len().next_multiple_of(config_stride as usize),0);}
                        }
                    }
                    indirect_offsets[band as usize].push(blocks);
                }
            }
        }
        let config=gpu_memory::buffer_init(d,"camera dispatch data",&configs,wgpu::BufferUsages::UNIFORM);
        let config_entry=entry(crate::camera_native::CONFIG_BINDING,wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&config,offset:0,size:std::num::NonZeroU64::new(std::mem::size_of::<CameraConfig>() as u64)}));
        let page_bindings:Vec<_>=evaluation_pages.iter().map(|buffer|wgpu::BufferBinding {buffer,offset:0,size:None}).collect();
        let geometry=crate::rt::resolve_hit_buffers(&installed.buffers,deforms);
        let geometry_bindings=crate::rt::hit_bindings(&geometry);
        let read_words=[&slots,&installed.draw_table,&installed.object_draws].map(|buffer|wgpu::BufferBinding {buffer,offset:0,size:None});
        let write_words=[&reflection_guides,&subsurface_records,&evaluation_metadata,&evaluation_bins,&evaluation_allocator].map(|buffer|wgpu::BufferBinding {buffer,offset:0,size:None});
        let reflection_fields=crate::passes::reflections::record_fields(&reflection_records,reflection_field_bytes,reflection_fields_per_buffer);
        let write_vectors=[&targets.aov_values,&refraction_receivers].map(|buffer|wgpu::BufferBinding {buffer,offset:0,size:None});
        let hits=d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("camera geometry and evaluations"),layout:&camera_geometry_layout(d,s.pipelines.camera_pages),entries:&[
            entry(0,wgpu::BindingResource::BufferArray(&geometry_bindings)),entry(1,wgpu::BindingResource::BufferArray(&page_bindings)),
            entry(2,wgpu::BindingResource::BufferArray(&read_words)),entry(3,wgpu::BindingResource::BufferArray(&write_words)),entry(4,wgpu::BindingResource::BufferArray(&write_vectors)),entry(5,wgpu::BindingResource::BufferArray(&reflection_fields))]});
        let whole=pyramid.create_view(&Default::default());
        let unused=s.transients.target(d,"unused camera refraction binding",1,1,1,REFRACTION_FORMAT,wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING).create_view(&Default::default());
        let shared = [
                entry(13, targets.nonfinite.as_entire_binding()),
                entry(14,emitter_weights.unwrap_or(&installed.draw_table).as_entire_binding()),
                entry(0, wgpu::BindingResource::TextureView(&targets.resolved)),
                entry(1, wgpu::BindingResource::TextureView(&s.environment.source)),
                entry(2, wgpu::BindingResource::Sampler(&s.environment.image_sampler)),
                entry(3, wgpu::BindingResource::TextureView(&targets.post_ambient)),
                entry(4, wgpu::BindingResource::TextureView(&receivers)),
                entry(5, wgpu::BindingResource::TextureView(&normals)),
                entry(6, targets.diffuse_response.as_entire_binding()),
                entry(7, sampling.passes.as_entire_binding()),

                entry(9, wgpu::BindingResource::Sampler(&refraction_sampler)),
                entry(10, wgpu::BindingResource::TextureView(&batch)),
        ];
        let own = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoupled shade"),
            layout: &shade_pipeline.get_bind_group_layout(4),
            entries: &[&shared[..], &[entry(8,wgpu::BindingResource::TextureView(&unused)),entry(12,wgpu::BindingResource::TextureView(&level_zero)),config_entry]].concat(),
        });
        let refraction = screen.then(|| {
        let composite_config=gpu_memory::buffer_init(d,"camera composite dispatch data",&composite_configs,wgpu::BufferUsages::UNIFORM);
        let composite_config_entry=entry(crate::camera_native::CONFIG_BINDING,wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&composite_config,offset:0,size:std::num::NonZeroU64::new(std::mem::size_of::<CameraConfig>() as u64)}));
            let downsample = crate::materials::mip_resources(r).0[2].clone();
            let mips = crate::materials::MipChain::new(d, &pyramid, None, &downsample, &refraction_sampler);
            let shade = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("refraction composite"),
                layout: &shade_pipeline.get_bind_group_layout(4),
                entries: &[&shared[..], &[entry(8,wgpu::BindingResource::TextureView(&whole)),entry(12,wgpu::BindingResource::TextureView(&unused)),composite_config_entry]].concat(),
            });
            Refraction { mips, shade }
        });
        Ok(Self {
            packets,packet_reflections:Default::default(),
            capacity_feedback: shader_errors.context("camera capacity diagnostics")?.clone(),
            evaluation, evaluation_bins, evaluation_args, evaluation_allocator, indirect_dispatch, indirect_offsets, material_count:installed.materials, config_stride, transport_parts, bands, band_rows,
            evaluation_groups:if direct {w.div_ceil(TILE)*h.div_ceil(TILE)} else {evaluation_capacity.div_ceil(64)},
            subsurface_records, refraction_source: (screen && s.plan.effects.subsurface).then(||level_zero.clone()), direct_keys, receivers, normals, stride: sampling.stride, keys, depth, slots, reflection_records, reflection_field_bytes, reflection_fields_per_buffer, reflection_guides, listed, overflow, fragments, passes, transparent_passes, merge_pipeline, shade_pipeline, merge,
            shade: [global, material, shadow, hits, own],
            refraction,
            extent,
        })
    }
}

/// Each batch's visibility passes into its key layers, then its merge into the slots.
pub(crate) fn visibility(encoder: &mut wgpu::CommandEncoder, f: &crate::view::ViewState, r: &Renderer, segments: &mut crate::command_finish::Segments) -> Result<()> {
    let v = f.decoupled.as_ref().context("decoupled view resources")?;
    if v.packets {return Ok(());}
    visibility_packet(encoder,f,r,segments,0)
}
fn visibility_packet(encoder:&mut wgpu::CommandEncoder,f:&crate::view::ViewState,r:&Renderer,segments:&mut crate::command_finish::Segments,packet:u32)->Result<()> {
    let v=f.decoupled.as_ref().context("packet visibility resources")?;
    let first_row=if v.packets {packet*v.band_rows} else {0};
    let rows=if v.packets {v.band_rows.min(v.extent[1]-first_row)} else {v.extent[1]};
    if !v.direct_keys {
        encoder.clear_buffer(&v.slots, 0, Some(v.listed));
        // The pool restarts each exposure; the frame's demand at its first.
        encoder.clear_buffer(&v.overflow, 0, Some(if f.sample_index == 0 { 8 } else { 4 }));
    }
    let passes = v.passes.len() as u32;
    for (index, (first, count)) in merge_batches(passes, v.keys.len() as u32).enumerate() {
        // A batch's fragments start a fresh list; the frame's demand at its first.
        encoder.clear_buffer(&v.fragments.buffer, 0, Some(if f.sample_index == 0 && index == 0 { 8 } else { 4 }));
        if v.fragments.lists > 0 {encoder.clear_buffer(&v.fragments.buffer, v.fragments.heads, None);}
        for layer in 0..count {
            let sample = first + layer;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("decoupled visibility"),
                color_attachments: &[Some(color_attachment(&v.keys[layer as usize], wgpu::Color::TRANSPARENT))],
                depth_stencil_attachment: Some(depth_attachment(&v.depth, false, true)),
                timestamp_writes: (!v.packets && sample == 0).then(|| wgpu::RenderPassTimestampWrites {
                    query_set: &f.queries,
                    beginning_of_pass_write_index: Some(timing::query("visibility")),
                    end_of_pass_write_index: v.direct_keys.then(|| timing::query("visibility")+1),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if v.packets {pass.set_scissor_rect(0,first_row,v.extent[0],rows);}
            pass.execute_bundles(std::iter::once(&v.passes[sample as usize]));
            if !v.fragments.explicit_depth {
                if let Some(bundle)=v.transparent_passes.get(sample as usize) {pass.execute_bundles(std::iter::once(bundle));}
            }
            drop(pass);
            if let Some(bundle)=v.transparent_passes.get(sample as usize).filter(|_|v.fragments.explicit_depth) {
                let mut pass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("transparent visibility"),
                    color_attachments:&[Some(wgpu::RenderPassColorAttachment {view:&v.keys[layer as usize],resolve_target:None,depth_slice:None,ops:wgpu::Operations {load:wgpu::LoadOp::Load,store:wgpu::StoreOp::Store}})],
                    depth_stencil_attachment:Some(depth_attachment(&v.depth,false,false)),
                    timestamp_writes:None,occlusion_query_set:None,multiview_mask:None,
                });
                pass.execute_bundles(std::iter::once(bundle));
            }
        }
        if v.direct_keys {continue;}
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("decoupled merge"),
            timestamp_writes: (!v.packets && first + count == passes).then(|| wgpu::ComputePassTimestampWrites {

                query_set: &f.queries,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(timing::query("visibility") + 1),
            }),
        });
        pass.set_pipeline(&v.merge_pipeline);
        pass.set_bind_group(0, &v.merge, &[(packet*v.passes.len().div_ceil(v.keys.len()) as u32+index as u32)*v.stride]);
        pass.dispatch_workgroups(v.extent[0].div_ceil(TILE), rows.div_ceil(TILE), 1);
        drop(pass);
        if passes > MERGE_BATCH {
            crate::command_finish::submission_boundary(r, encoder, segments);
        }
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn indirect_work_counts(v:&View,r:&Renderer)->Result<[u32;2]> {
    eprintln!("camera indirect reservation: bytes={} recorded_blocks={}",v.evaluation_allocator.size(),v.indirect_offsets.iter().flatten().map(Vec::len).sum::<usize>());
    let read=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("indirect work counts"),size:12,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
    let mut encoder=r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&v.evaluation_allocator,8,&read,0,12);r.queue.submit([encoder.finish()]);
    let (tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});
    r.device.poll(wgpu::PollType::Wait {submission_index:None,timeout:None})?;rx.recv()??;
    let mapped=read.get_mapped_range(..);let counts:[u32;3]=bytemuck::pod_read_unaligned(&mapped);
    Ok([counts[0]+counts[2],counts[1]])
}

pub(crate) fn classified(v:&View)->bool {!v.direct_keys || v.material_count!=2}
/// The same exposure-boundary decision drives dispatch and timestamp selection.
pub(crate) fn fused_active(f:&crate::view::ViewState)->bool {
    f.fused_shading
}
pub(crate) fn fused_ready(v:&View)->bool {v.evaluation.fused.as_ref().is_some_and(EvaluationPipeline::ready)}

#[cfg(test)]
pub(crate) static TEST_TRANSPORT_PARTS:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
/// Dispatches that share a glass slot's covered samples. One invocation tracing
/// all of them, each with its own unbounded interface chain, outlasts Metal's
/// execution window and aborts intermittently, at any resolution. Each part
/// reruns the slot's surface and adds its samples' transport to the stored record.
/// Other backends keep one dispatch: their watchdogs do not need this.
fn transport_parts(samples:u32,backend:wgpu::Backend)->u32 {
    #[cfg(test)]
    {let parts=TEST_TRANSPORT_PARTS.load(std::sync::atomic::Ordering::Relaxed);if parts!=0 {return parts;}}
    if backend!=wgpu::Backend::Metal {return 1;}
    samples.div_ceil(8).max(4).min(samples)
}
/// One band's dispatch block offset and its pixel extent in tiles.
fn band(v:&View,band:u32)->(u32,u32) {
    let offset=band*v.material_count*v.config_stride;
    let rows=v.band_rows.min(v.extent[1]-band*v.band_rows);
    (offset,rows.div_ceil(TILE))
}
/// Timestamps of a span: its start and its end where flagged. One band times
/// classification, evaluation and composition apart; several bands interleave
/// them, so they time the whole banded shade as "shade" (graph.rs spans).
fn span<'a>(queries:Option<&'a wgpu::QuerySet>,name:&'static str,first:bool,last:bool)->Option<wgpu::ComputePassTimestampWrites<'a>> {
    queries.filter(|_|first || last).map(|q|wgpu::ComputePassTimestampWrites {query_set:q,
        beginning_of_pass_write_index:first.then(||timing::query(name)),end_of_pass_write_index:last.then(||timing::query(name)+1)})
}

/// The view's camera groups, with `own` (the shade's or the refraction
/// composite's group 4) at one dispatch block.
fn bind_camera(pass:&mut wgpu::ComputePass,v:&View,own:&wgpu::BindGroup,block:u32) {
    for (i,b) in v.shade[..4].iter().enumerate() {pass.set_bind_group(i as u32,b,&[]);}
    pass.set_bind_group(4,own,&[block]);
}
/// Count every band's evaluations before shading any of them. A late band's
/// shortage otherwise discards expensive transport already run in earlier bands.
/// One-band views already classify their entire working set before evaluation.
fn preflight_evaluations(encoder:&mut wgpu::CommandEncoder,v:&View,own:&wgpu::BindGroup,queries:Option<&wgpu::QuerySet>) {
    let Some(classification)=v.evaluation.classification.as_ref().filter(|p|v.bands>1 && p.len()>1) else {return;};
    for index in 0..v.bands {
        let (base,tile_rows)=band(v,index);
        encoder.clear_buffer(&v.evaluation_bins,0,Some(u64::from(v.material_count)*24));
        for (stage,pipeline) in classification.iter().take(2).enumerate() {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("camera evaluation capacity preflight"),
                timestamp_writes:span(queries,"shade",index==0 && stage==0,false)});
            pass.set_pipeline(pipeline);bind_camera(&mut pass,v,own,base);
            if stage==1 {pass.dispatch_workgroups(1,1,1);} else {pass.dispatch_workgroups(v.extent[0].div_ceil(TILE),tile_rows,1);}
        }
    }
}
/// One band's classification and material evaluation. The refraction composite
/// reuses the opaque pass's bins when one band covers the view.
fn evaluate(encoder:&mut wgpu::CommandEncoder,v:&View,own:&wgpu::BindGroup,queries:Option<&wgpu::QuerySet>,composite:bool,index:u32,specialized:bool,r:&Renderer) {
    let p=&v.evaluation;
    let direct=!classified(v);
    let (base,tile_rows)=band(v,index);
    let first=index==0;
    if p.indirect.is_some() {encoder.clear_buffer(&v.evaluation_allocator,8,Some(u64::from(vork_slang::registry::CAMERA_INDIRECT_HEADER_WORDS-2)*4));}
    if !direct && (!composite || v.bands>1) {
        encoder.clear_buffer(&v.evaluation_bins,0,Some(u64::from(v.material_count)*24));
        // Implicit slots append record IDs in the index pass; other materials
        // count, prefix and scatter their work. Both publish the same bin args.
        let classification=p.classification.as_ref().expect("visibility classification");
        for (stage,pipeline) in classification.iter().enumerate() {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("classify camera material slots"),
                timestamp_writes:if v.bands>1 {span(queries,"shade",first && stage==0 && (classification.len()==1 || v.packets),false)}
                    else {span(queries,"material classify",stage==0,stage==classification.len()-1)}});
            pass.set_pipeline(pipeline);bind_camera(&mut pass,v,own,base);
            if stage==1 {pass.dispatch_workgroups(1,1,1);} else {pass.dispatch_workgroups(v.extent[0].div_ceil(TILE),tile_rows,1);}
        }
        if let Some(args)=&v.evaluation_args {encoder.copy_buffer_to_buffer(&v.evaluation_bins,0,args,0,u64::from(v.material_count)*24);}
    }
    if let Some(pipeline)=p.occupancy.as_ref().filter(|_|composite || v.refraction.is_none()) {
        let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("camera dielectric occupancy"),timestamp_writes:None});
        pass.set_pipeline(pipeline);bind_camera(&mut pass,v,own,base);
        pass.dispatch_workgroups(v.extent[0].div_ceil(TILE),tile_rows,1);
    }
    let mut evaluations=p.evaluate.iter().filter(|(material,_)|!composite || *material!=v.material_count-1).peekable();
    let timestamps=span(queries.filter(|_|v.bands==1),"shade",true,true);
    // Empty scenes have no evaluation pipelines; a composite can also skip its
    // only (opaque) bin. Do not ask an empty Metal encoder to sample counters.
    if evaluations.peek().is_none() {
        if let Some(timestamps)=timestamps {
            r.timestamp_marker.get_or_init(||timing::Marker::new(&r.device)).encode_timestamps(encoder,timestamps);
        }
    } else {
        let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("evaluate camera material bins"),timestamp_writes:timestamps});
        bind_camera(&mut pass,v,own,base);
        let evaluations:Vec<_>=evaluations.collect();
        for part in 0..v.transport_parts {
            for &(material,pipeline) in &evaluations {
                // Later parts only add transport; the opaque bin has none.
                if part>0 && *material==v.material_count-1 {continue;}
                pass.set_bind_group(4,own,&[base+(part*v.bands*v.material_count+*material)*v.config_stride]);
                pass.set_pipeline(pipeline.current(specialized));
                if let Some(args)=&v.evaluation_args {pass.dispatch_workgroups_indirect(args,u64::from(*material)*24);} else {
                    // Single-bin views and Metal use bounded direct dispatch; shaders reject
                    // invocations beyond each bin before reconstructing a point.
                    let groups=v.evaluation_groups;
                    pass.dispatch_workgroups(groups.min(65535),groups.div_ceil(65535),1);
                }
            }
        }
    }
    if let Some(pipeline)=&p.indirect {
        if let Some((pipeline,bind,_,blocks))=&v.indirect_dispatch {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("size unsupported diffuse connections"),..Default::default()});
            pass.set_pipeline(pipeline);pass.set_bind_group(0,bind,&[]);
            pass.dispatch_workgroups(blocks.div_ceil(vork_slang::registry::CAMERA_INDIRECT_GROUP),1,1);
        }
        // Words of one record accumulate in order, so each word keeps its own
        // pass. Its blocks address disjoint records and share that pass: Metal
        // opens command buffers per pass, and one per block exhausted its queue.
        for blocks in &v.indirect_offsets[index as usize] {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("unsupported diffuse connections"),..Default::default()});
            pass.set_pipeline(pipeline);
            for (i,block) in blocks.iter().enumerate() {
                let offset=block.offset;
                if i==0 {bind_camera(&mut pass,v,own,offset);} else {pass.set_bind_group(4,own,&[offset]);}
                if let Some((_,_,arguments,_))=&v.indirect_dispatch {
                    pass.dispatch_workgroups_indirect(arguments,i as u64*12);
                } else {
                    pass.dispatch_workgroups(block.groups,1,1);
                }
            }
        }
    }
}
/// One band's composition of its pixels.
fn compose(encoder:&mut wgpu::CommandEncoder,v:&View,own:&wgpu::BindGroup,timestamps:Option<wgpu::ComputePassTimestampWrites>,label:&str,index:u32) {
    let (base,tile_rows)=band(v,index);
    let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some(label),timestamp_writes:timestamps});
    pass.set_pipeline(&v.shade_pipeline);
    bind_camera(&mut pass,v,own,base);
    pass.dispatch_workgroups(v.extent[0].div_ceil(TILE), tile_rows, 1);
}

/// Publish shortage feedback in queue order before material evaluation, without
/// a host readback. The regular collection counters still own exact retry demand.
fn capacity_feedback(encoder: &mut wgpu::CommandEncoder, v: &View) {
    encoder.copy_buffer_to_buffer(&v.overflow, 4, &v.capacity_feedback,
        u64::from(vork_slang::registry::CAMERA_VISIBILITY_OVERFLOW_WORD) * 4, 4);
    encoder.copy_buffer_to_buffer(&v.fragments.buffer, 4, &v.capacity_feedback,
        u64::from(vork_slang::registry::CAMERA_FRAGMENT_OVERFLOW_WORD) * 4, 4);
}

/// One evaluation per slot into the resolved colour, background included, band by band.
pub(crate) fn shade(encoder: &mut wgpu::CommandEncoder, f: &crate::view::ViewState, r: &Renderer,segments:&mut crate::command_finish::Segments) -> Result<()> {
    let v = f.decoupled.as_ref().context("decoupled view resources")?;
    capacity_feedback(encoder, v);
    if let Some(pipeline)=v.evaluation.fused.as_ref().filter(|_|fused_active(f) && !v.packets) {
        let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("fused camera shade"),timestamp_writes:span(Some(&f.queries),"shade",true,true)});
        pass.set_pipeline(pipeline.current(true));bind_camera(&mut pass,v,&v.shade[4],0);
        // Fused shading retains no evaluations, so its dispatch spans every band.
        pass.dispatch_workgroups(v.extent[0].div_ceil(TILE),v.extent[1].div_ceil(TILE),1);
        return Ok(());
    }
    if v.packets {
        r.timestamp_marker.get_or_init(||timing::Marker::new(&r.device)).encode_timestamps(encoder,span(Some(&f.queries),"shade",true,false).unwrap());
    } else {preflight_evaluations(encoder,v,&v.shade[4],Some(&f.queries));}
    for index in 0..v.bands {
        if v.packets {visibility_packet(encoder,f,r,segments,index)?;}
        let queries=if v.packets {None} else {Some(&f.queries)};
        evaluate(encoder,v,&v.shade[4],queries,false,index,!f.variants_pending(),r);
        let timestamps=if v.bands>1 {span(queries,"shade",false,index+1==v.bands)} else {span(queries,"material compose",true,true)};
        compose(encoder,v,&v.shade[4],timestamps,"decoupled shade",index);
        if v.packets {
            if let Some(reflections)=v.packet_reflections.get() {
                for chunk in reflections[index as usize].chunks(constants::REFLECTION_SEGMENT_DISPATCHES) {
                    crate::command_finish::submission_boundary(r,encoder,segments);
                    for dispatch in chunk {dispatch.encode_packet(encoder,!f.variants_pending());}
                }
                crate::command_finish::submission_boundary(r,encoder,segments);
            }
            capacity_feedback(encoder,v);
        }
    }
    if v.packets {
        r.timestamp_marker.get_or_init(||timing::Marker::new(&r.device)).encode_timestamps(encoder,span(Some(&f.queries),"shade",false,true).unwrap());
    }
    Ok(())
}
/// Screen refraction: the opaque image's pyramid, then the transparent composite.
pub(crate) fn refraction(encoder: &mut wgpu::CommandEncoder, f: &crate::view::ViewState, r: &Renderer) -> Result<()> {
    let v = f.decoupled.as_ref().context("decoupled view resources")?;
    let refraction = v.refraction.as_ref().context("screen refraction resources")?;
    capacity_feedback(encoder, v);
    refraction.mips.encode(encoder, Some(wgpu::RenderPassTimestampWrites {
        query_set: &f.queries, beginning_of_pass_write_index: Some(timing::query("refraction")), end_of_pass_write_index: None,
    }));
    preflight_evaluations(encoder,v,&refraction.shade,None);
    for index in 0..v.bands {
        evaluate(encoder,v,&refraction.shade,None,true,index,!f.variants_pending(),r);
        let timestamps=span(Some(&f.queries),"refraction",index==0 && refraction.mips.is_empty(),index+1==v.bands);
        compose(encoder,v,&refraction.shade,timestamps,"refraction composite",index);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn reflection_fields_split_at_the_allocation_limit() {
        // 4K records: 8,294,400 pixels x 4 slots x 4 bytes = 132,710,400 B per field.
        let field=132_710_400;
        assert_eq!(super::reflection_fields_per_buffer(field,2_147_483_647),16);
        assert_eq!(super::reflection_fields_per_buffer(field,200<<20),1);
        assert_eq!(super::reflection_fields_per_buffer(u64::MAX,1<<40),1);
        let per=u64::from(super::reflection_fields_per_buffer(field,2_147_483_647));
        assert!(field*per<=2_147_483_647,"one allocation must fit the device limit");
        assert_eq!((f64::from(28u32)/per as f64).ceil() as u32,2,"28 fields take two allocations at 4K");
    }
    #[test]
    fn camera_packets_follow_memory_and_native_limits() -> Result<()> {
        let limits=wgpu::Limits {max_storage_buffer_binding_size:2<<30,max_buffer_size:16<<30,..Default::default()};
        let shape=ViewShape {transparent:false,diffusion:false,reflections:true,screen_refraction:false,table_only:true,boundaries:0,reserved:0};
        assert_eq!(camera_packet_rows(&limits,shape,[1920,1080],8,8,23<<30,2,0)?,None);
        let rows=camera_packet_rows(&limits,shape,[3840,2160],8,8,23<<30,2,0)?.unwrap();
        assert!(rows<2160 && rows%TILE==0,"{rows}");
        assert!(camera_packet_rows(&limits,shape,[1920,1080],8,8,4<<30,2,0)?.unwrap()<1080);
        assert_eq!(camera_packet_rows(&limits,shape,[3840,64],64,8,23<<30,2,0)?,None);
        let reserved=ViewShape {reserved:u32::MAX,..shape};
        assert_eq!(camera_packet_rows(&limits,reserved,[96,64],8,19,128<<20,2,0)?,Some(64),"a proven full-height packet supersedes another view's pool high-water");
        Ok(())
    }

    #[test]
    fn packet_budget_includes_indirect_work_and_balances_the_tail() -> Result<()> {
        let limits=wgpu::Limits {max_storage_buffer_binding_size:2<<30,max_buffer_size:16<<30,..Default::default()};
        let shape=ViewShape {transparent:false,diffusion:false,reflections:true,screen_refraction:false,table_only:false,boundaries:0,reserved:0};
        assert!(packet_eligible(shape,8));
        let rows=camera_packet_rows(&limits,shape,[1080,1920],8,19,8<<30,3,24)?.unwrap();
        let pixels=1080*u64::from(rows);
        let fields=view_bindings_with_pool(shape,[1080,rows],8,pixels*4);
        let bytes=fields[0].1+fields[1].1+fields[2].1+fields[3].1*u64::from(VISIBILITY_REFLECTION_WORDS)
            +pixels*8*(304+16+96)+3*24+u64::from(vork_slang::registry::CAMERA_INDIRECT_HEADER_WORDS)*4;
        assert!(bytes<=2<<30,"camera packet owns {bytes} bytes");
        let packets=1920u32.div_ceil(rows);
        assert_eq!(rows,1920u32.div_ceil(TILE).div_ceil(packets)*TILE);
        assert!(!packet_eligible(ViewShape {transparent:true,..shape},8));
        assert!(!packet_eligible(ViewShape {diffusion:true,..shape},8));
        assert!(!packet_eligible(ViewShape {screen_refraction:true,..shape},8));
        Ok(())
    }

    #[test]
    fn opaque_program_packets_preserve_absolute_samples_and_last_partial_row() -> Result<()> {
        let Some(mut r)=crate::test_renderer()? else {return Ok(());};
        let scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[96,73],"settings":{"integrator":"raster","samples":8,"reflections":"ray","denoiser":null,"fast_gi":null},"display":{"program":"std/standard","inputs":{"dither_intensity":0.}}},
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"attributes":{"POSITION":"p"},"indices":"i"}},
            "programs":{"rings":{"source":"struct Inputs {}; Closure surface(ShadingContext ctx,Inputs inputs) { Closure c={}; c.base_color=float3(frac(length(ctx.POSITION.xy)*8.),0.3,0.7); c.specular_roughness=0.3; return c; }"}},
            "materials":{"rings":{"program":"rings"}},
            "nodes":[{"type":"camera","translation":[0,0,4],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2.5}}},
                {"type":"mesh","geometry":"quad","materials":["rings"]},
                {"type":"light","shape":"sun","energy":3},
                {"type":"light","shape":"environment","color":[0.2,0.3,0.4],"energy":1}]}))?;
        let render=|r:&Renderer,packets:bool|->Result<Vec<[f32;4]>> {
            let mut installed=r.install(scene.clone(),std::path::Path::new("."))?;
            assert!(installed.decoupled.as_ref().unwrap().materials>2);
            let mut frame=installed.create_frame_slot(r,crate::ReadbackPrecision::Float)?;
            let view=frame.view.decoupled.as_ref().unwrap();
            assert_eq!(view.packets,packets);
            if packets {assert!(view.bands>1);assert_ne!(73%view.band_rows,0);}
            let first=installed.render_frame(r,&mut frame,0.)?.linear_rgba.unwrap();
            let warm=installed.render_frame(r,&mut frame,0.)?.linear_rgba.unwrap();
            assert_eq!(bytemuck::cast_slice::<_,u32>(&first),bytemuck::cast_slice::<_,u32>(&warm));
            Ok(first)
        };
        r.camera_memory=u64::MAX;
        let expected=render(&r,false)?;
        r.camera_memory=8<<20;
        let actual=render(&r,true)?;
        assert_eq!(bytemuck::cast_slice::<_,u32>(&expected),bytemuck::cast_slice::<_,u32>(&actual));
        Ok(())
    }

    #[test]
    fn one_full_height_packet_consumes_records_and_completes_queries() -> Result<()> {
        let Some(mut r)=crate::test_renderer()? else {return Ok(());};
        r.camera_memory=128<<20;
        let scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"topology":"triangles","attributes":{"POSITION":"p"},"indices":"i"}},
            "materials":{"glow":{"inputs":{"base_weight":0.,"specular_weight":0.2,"emission_luminance":1.,"emission_color":[0.25,0.5,0.75]}}},
            "nodes":[{"type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","fit":"horizontal","ortho_scale":2.}}},{"type":"mesh","geometry":"quad","materials":["glow"]}],
            "render":{"resolution":[96,64],"settings":{"integrator":"raster","samples":8,"reflections":"ray","denoiser":null,"fast_gi":null},"display":{"program":"std/standard","inputs":{"dither_intensity":0.}}}
        }))?;
        let render=|reserved:u32|->Result<Vec<[f32;4]>> {
            let options=crate::InstallOptions {reservations:crate::Reservations {visibility_pool:reserved,..Default::default()},..Default::default()};
            let mut installed=r.install_with_options(scene.clone(),std::path::Path::new("."),options)?;
            let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            let view=frame.view.decoupled.as_ref().unwrap();
            assert_eq!(view.packets,reserved!=0);assert_eq!(view.bands,1);
            let first=installed.render_frame(&r,&mut frame,0.)?.linear_rgba.unwrap();
            crate::test_linear::assert_linear_eq(&first,&installed.render_frame(&r,&mut frame,0.)?.linear_rgba.unwrap());
            Ok(first)
        };
        crate::test_linear::assert_linear_eq(&render(0)?,&render(u32::MAX)?);
        Ok(())
    }

    #[test]
    fn glass_transport_parts_bound_samples_per_invocation_on_metal() {
        use wgpu::Backend::{Metal,Vulkan};
        for (samples,parts) in [(1,1),(2,2),(4,4),(16,4),(64,8)] {assert_eq!(super::transport_parts(samples,Metal),parts,"{samples}");}
        assert_eq!(super::transport_parts(64,Vulkan),1);
    }
    #[test]
    fn fragment_planes_follow_visibility_consumers() {
        let pixels=1920*1080;
        for samples in [1,8,64] {
            assert_eq!(super::fragment_list_bytes(pixels,samples,false,false),0);
            assert_eq!(super::fragment_list_bytes(pixels,samples,false,true),0);
            let heads=u64::from(samples.min(super::MERGE_BATCH))*pixels*4;
            assert_eq!(super::fragment_list_bytes(pixels,samples,true,false),heads);
            assert_eq!(super::fragment_list_bytes(pixels,samples,true,true),heads*2);
        }
    }

    use super::*;
    #[test]
    fn indirect_arguments_follow_live_requests_and_bound_each_dispatch()->Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let d=&r.device;let pipeline=indirect_arguments_pipeline(&r);
        let block=vork_slang::registry::CAMERA_INDIRECT_BLOCK;
        let group=vork_slang::registry::CAMERA_INDIRECT_GROUP;
        // Pure single-sample connections pack a word's worth of records; late
        // authored hooks retain the original per-record invocation bound.
        for sampled_block in [vork_slang::registry::CAMERA_INDIRECT_SAMPLE_BLOCK,block] {
            let centroid_capacity=2*block+17;let sampled_capacity=2*sampled_block+17;
            let capacity=centroid_capacity+sampled_capacity-1;
            let centroids=centroid_capacity.div_ceil(block);
            let blocks=centroids+sampled_capacity.div_ceil(sampled_block);let bytes=u64::from(blocks)*12;
            let mut header=[0u32;vork_slang::registry::CAMERA_INDIRECT_HEADER_WORDS as usize];header[1]=capacity;
            let allocator=gpu_memory::buffer_init(d,"argument-test allocator",bytemuck::cast_slice(&header),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST);
            let arguments=gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:None,size:bytes,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
            let read=gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:None,size:bytes+4,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            let params=gpu_memory::buffer_init(d,"argument-test capacity",bytemuck::cast_slice(&[centroid_capacity,sampled_capacity,centroids,sampled_block]),wgpu::BufferUsages::UNIFORM);
            let errors=gpu_memory::buffer_init(d,"argument-test errors",bytemuck::cast_slice(&[0u32;4]),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST);
            let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,allocator.as_entire_binding()),entry(1,arguments.as_entire_binding()),entry(2,params.as_entire_binding()),entry(3,errors.as_entire_binding())]});
            // Include empty reuse, partial tails, each launch shortage, and a
            // combined storage shortage whose separate queues both fit.
            for centroid_count in [centroid_capacity+1,centroid_capacity,block+1,block,1,0] {
            for sampled_count in [sampled_capacity+1,sampled_capacity,sampled_block+1,sampled_block,1,0] {
                r.queue.write_buffer(&allocator,8,bytemuck::cast_slice(&[centroid_count,0,sampled_count]));
                r.queue.write_buffer(&errors,12,bytemuck::bytes_of(&0u32));
                let mut encoder=d.create_command_encoder(&Default::default());
                {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(blocks.div_ceil(group),1,1);}
                encoder.copy_buffer_to_buffer(&arguments,0,&read,0,bytes);
                encoder.copy_buffer_to_buffer(&errors,12,&read,bytes,4);
                let submission=r.queue.submit([encoder.finish()]);
                let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});r.wait(Some(submission))?;rx.recv()??;
                let mapped=read.get_mapped_range(..);let values:&[u32]=bytemuck::cast_slice(&mapped);
                let storage_short=centroid_count+sampled_count>capacity;
                let incomplete=storage_short || centroid_count>centroid_capacity || sampled_count>sampled_capacity;
                let mut covered=[0u32;2];
                for i in 0..blocks {
                    let sampled=i>=centroids;let lane=usize::from(sampled);
                    let width=if sampled {sampled_block} else {block};
                    let first=if sampled {i-centroids} else {i};
                    let count=if incomplete {0} else if sampled {sampled_count} else {centroid_count};
                    let records=count.saturating_sub(first*width).min(width);
                    assert_eq!(&values[(i*3) as usize..(i*3+3) as usize],&[records.div_ceil(group),1,1]);
                    assert!(values[(i*3) as usize]*group<=width);
                    covered[lane]+=records;
                }
                assert_eq!(covered,if incomplete {[0,0]} else {[centroid_count,sampled_count]});
                let demand=if !incomplete {0} else if storage_short {centroid_count+sampled_count} else {centroid_count.max(sampled_count)};
                assert_eq!(values[blocks as usize*3],demand);
                drop(mapped);read.unmap();
            }
            }
        }
        r.errors.check()?;Ok(())
    }

    #[test]
    fn receiver_diffusion_storage_is_reserved_only_for_the_combination() {
        let shape=ViewShape {transparent:true,diffusion:false,reflections:false,screen_refraction:false,boundaries:0,table_only:true,reserved:0};
        let plain=view_bindings(&wgpu::Limits::default(),shape,[64,64],4);
        assert_eq!(plain,view_bindings(&wgpu::Limits::default(),ViewShape {screen_refraction:true,..shape},[64,64],4));
        let sss=view_bindings(&wgpu::Limits::default(),ViewShape {diffusion:true,..shape},[64,64],4);
        let both=view_bindings(&wgpu::Limits::default(),ViewShape {diffusion:true,screen_refraction:true,..shape},[64,64],4);
        assert_eq!(sss[..2],both[..2]);
        assert_eq!(both[2].1,sss[2].1.div_ceil(256)*256*2);
    }
    #[test]
    fn invisible_textured_front_preserves_rear_sample_evaluation() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        let temp=std::env::temp_dir().join(format!("vork-zero-front-{}",std::process::id()));
        std::fs::create_dir_all(&temp)?;
        image::RgbaImage::from_fn(64,64,|x,y| {
            let n=(x+y*64).wrapping_mul(1664525).wrapping_add(1013904223);
            image::Rgba([(n>>24) as u8,(n>>16) as u8,(n>>8) as u8,255])
        }).save(temp.join("noise.png"))?;
        for (name,alpha) in [("zero",0),("one",255)] {
            image::RgbaImage::from_pixel(8,8,image::Rgba([255,255,255,alpha])).save(temp.join(format!("{name}.png")))?;
        }
        let mut value=serde_json::json!({"version":2,
            "render":{"resolution":[64,64],"settings":{"integrator":"raster","samples":8}},
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},
                "uv":{"type":"vec2","data":[[0.013,0.023],[1.013,0.023],[1.013,1.023],[0.013,1.023]]},
                "i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"topology":"triangles","attributes":{"POSITION":"p","TEXCOORD_0":"uv"},"indices":"i"}},
            "images":{"noise":{"file":"noise.png"},"zero":{"file":"zero.png","color_space":"Non-Color"},"one":{"file":"one.png","color_space":"Non-Color"}},
            "programs":{"textured":{"source":"import vork; import std.texture; struct Inputs {Image alpha; Image emission_color; float geometry_opacity=1.;float base_weight=0.;float specular_weight=0.;float emission_luminance=0.;}; Closure surface(ShadingContext ctx,Inputs p) {let f=texture_footprint(ctx.TEXCOORD_0,ctx.duv_dx[0],ctx.duv_dy[0],ctx);Closure c={};c.base_weight=p.base_weight;c.specular_weight=p.specular_weight;c.emission_luminance=p.emission_luminance;c.emission_color=texture_sample(p.emission_color,f,ctx).rgb;c.geometry_opacity=p.geometry_opacity*texture_sample(p.alpha,f,ctx).a;return c;}"}},
            "materials":{"rear":{"program":"textured","inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,"emission_color":{"image":"noise"}}},
                "front":{"program":"textured","opacity":"transmittance","inputs":{"alpha":{"image":"zero"}}}},
            "nodes":[{"type":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2}}},
                {"type":"mesh","geometry":"quad","materials":["rear"]},
                {"type":"mesh","geometry":"quad","materials":["front"],"translation":[0,0,0.1]}]});
        let render=|value:serde_json::Value| -> Result<Vec<[f32;4]>> {
            let mut installed=r.install(vork_scene::v2::from_test_value(value)?,&temp)?;
            let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            Ok(installed.render_frame(&r,&mut frame,0.)?.linear_rgba.unwrap())
        };
        // A rejected rear must not change the foreground's texture footprint
        // or switch it from centroid filtering to sample shading.
        let mut behind=value.clone();
        behind["materials"]["rear"]["opacity"]=serde_json::json!("transmittance");
        behind["materials"]["rear"]["inputs"]["geometry_opacity"]=serde_json::json!(0.5);
        behind["nodes"][2]["translation"]=serde_json::json!([0,0,-0.1]);
        let with_invisible_rear=render(behind.clone())?;
        behind["nodes"].as_array_mut().unwrap().pop();
        crate::test_linear::assert_linear_eq(&with_invisible_rear,&render(behind)?);
        for zero_first in [false,true] {
            let mut coplanar=value.clone();
            coplanar["nodes"][2]["translation"]=serde_json::json!([0,0,0]);
            coplanar["materials"]["rear"]["opacity"]=serde_json::json!("transmittance");
            coplanar["materials"]["rear"]["inputs"]["geometry_opacity"]=serde_json::json!(0.5);
            if zero_first {coplanar["nodes"].as_array_mut().unwrap().swap(1,2);}
            let actual=render(coplanar.clone())?;
            coplanar["nodes"].as_array_mut().unwrap().remove(if zero_first {1} else {2});
            if zero_first {
                // An alpha-one map forces sample shading independently of the
                // discarded coplanar front, while preserving 0.5 coverage.
                coplanar["materials"]["rear"]["inputs"]["alpha"]=serde_json::json!({"image":"one"});
            }
            crate::test_linear::assert_linear_eq(&actual,&render(coplanar)?);
        }
        let rejected=render(value.clone())?;
        // An alpha-one textured rear independently forces the same per-sample
        // evaluations, without relying on the invisible front's metadata.
        value["nodes"].as_array_mut().unwrap().pop();
        value["materials"]["rear"]["opacity"]=serde_json::json!("transmittance");
        value["materials"]["rear"]["inputs"]["alpha"]=serde_json::json!({"image":"one"});
        let reference=render(value)?;
        std::fs::remove_dir_all(&temp)?;
        crate::test_linear::assert_linear_eq(&rejected,&reference);
        Ok(())
    }
    #[test]
    fn reflection_allocation_limit_explains_recovery() {
        // One table-glass sample packs four dense records per pixel and no pool,
        // so only the record fields grow with the extent.
        let limits = wgpu::Limits {max_storage_buffer_binding_size:1<<40,max_buffer_size:1<<30,..Default::default()};
        let shape = ViewShape {transparent:false,diffusion:false,reflections:true,screen_refraction:false,boundaries:0,table_only:true,reserved:0};
        // Fields are separate bindings packed across allocations, so the combined
        // 28-field array may exceed max_buffer_size without refusing the view.
        let fields=check_view(&limits,[3840,2160],1,shape).unwrap();
        assert_eq!(fields[3],132_710_400);
        assert!(fields[3]*u64::from(VISIBILITY_REFLECTION_WORDS)>limits.max_buffer_size);
        // One field still has to fit max_buffer_size, and the error explains recovery.
        let small = wgpu::Limits {max_buffer_size:64<<20,..limits};
        let error=check_view(&small,[3840,2160],1,shape).unwrap_err().to_string();
        for text in ["reflection record field", "132710400 bytes", "67108864 bytes", "reduce the rendered resolution"] {
            assert!(error.contains(text), "missing {text:?}: {error}");
        }
        check_view(&small,[1280,720],1,shape).unwrap();
    }

    #[test]
    fn overflow_estimate_fits_all_bindings_and_measured_demand_is_mandatory() {
        let limits = wgpu::Limits {max_storage_buffer_binding_size:1<<29,max_buffer_size:1<<40,..Default::default()};
        let extent = [1920,1080];
        let shape = ViewShape {transparent:false,diffusion:false,reflections:false,screen_refraction:false,boundaries:0,table_only:true,reserved:0};
        let pool = visibility_pool(&limits, shape, extent, 64);
        assert!(pool >= 1920*1080);
        assert!(pool < 5*1920*1080);
        assert!(check_view(&limits, extent, 64, shape).unwrap()[1] <= 1<<29);
        let wide = [32,32];
        assert!(visibility_pool(&limits, shape, wide, 64) >= 1088);
        // A measured demand larger than the hint still fits on a generous device.
        let generous = wgpu::Limits {max_storage_buffer_binding_size:1<<31,max_buffer_size:1<<40,..Default::default()};
        let measured = ViewShape {reserved:6*1920*1080,..shape};
        assert_eq!(visibility_pool(&generous, measured, extent, 64),u64::from(measured.reserved).next_power_of_two());
        check_view(&generous, extent, 64, measured).unwrap();
        // max_buffer_size constrains the same calculation as the binding limit.
        let buffer_limited = wgpu::Limits {max_storage_buffer_binding_size:1<<31,max_buffer_size:1<<29,..Default::default()};
        assert_eq!(check_view(&limits, extent, 64, shape).unwrap(),check_view(&buffer_limited, extent, 64, shape).unwrap());
        // Reflection records are field bindings, so their combined allocation
        // may exceed one binding without refusing the view.
        let reflected = ViewShape {reflections:true,..shape};
        let fields=check_view(&limits, [3840,2160], 1, reflected).unwrap();
        assert!(fields[3]*u64::from(VISIBILITY_REFLECTION_WORDS)>limits.max_storage_buffer_binding_size);
        assert!(fields[3]<=limits.max_storage_buffer_binding_size);
        let l4=wgpu::Limits {max_storage_buffer_binding_size:(1<<31)-4,..limits.clone()};
        check_view(&l4,[3840,2160],64,reflected).unwrap();
        for shape in [shape, ViewShape {diffusion:true,screen_refraction:true,..shape}, ViewShape {reflections:true,..shape}] {
            let extent = [1280,720];
            let pool = visibility_pool(&limits, shape, extent, 64);
            assert!(pool > 0);
            check_view(&limits, extent, 64, shape).unwrap();
            let demand = ViewShape {reserved:pool as u32 + 1,..shape};
            if pool < 5*1280*720 {
                assert!(check_view(&limits, extent, 64, demand).is_err());
            }
            let demand = ViewShape {reserved:20_000_000,..shape};
            assert_eq!(visibility_pool(&limits, demand, extent, 64),20_000_000);
            assert!(check_view(&limits, extent, 64, demand).is_err());
        }
    }
    #[test]
    fn oversized_frames_fail_at_install_with_the_limit_and_what_fits() {
        // Junk Shop at 4K/64 on an L4: transparent per-sample metadata needed 3.36 GB
        // against the 2 GiB storage binding, found only when the first frame was created.
        let limits=wgpu::Limits {max_storage_buffer_binding_size:(1<<31)-4,max_buffer_size:1<<40,..Default::default()};
        let shape=ViewShape {transparent:true,diffusion:false,reflections:false,screen_refraction:false,boundaries:0,table_only:true,reserved:0};
        let check=|extent,samples| check_view(&limits,extent,samples,shape);
        let error=check([3840,2160],64).unwrap_err().to_string();
        assert!(error.contains("camera evaluation metadata") && error.contains("requested ")
            && error.contains("vork help limits") && error.contains("3840x2160 with 64 samples")
            && error.contains("max_storage_buffer_binding_size=2147483644"),"{error}");
        let samples:u32=error.split("camera budget ceiling at this extent=").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        let [w,h]:[u32;2]=error.split("resolution to ").nth(1).unwrap().split(' ').next().unwrap()
            .split('x').map(|n|n.parse().unwrap()).collect::<Vec<_>>().try_into().unwrap();
        assert!(!error.contains("settings.samples") && error.contains("--profile std/draft"), "{error}");
        let draft = vork_scene::Profile::load("std/draft").unwrap().resolve(&Default::default()).unwrap();
        check([3840,2160], draft.samples).unwrap();
        // The internal ceiling and advised resolution fit; one more sample does not.
        check([3840,2160],samples).unwrap();
        assert!(check([3840,2160],samples+1).is_err());
        check([w,h],64).unwrap();
        assert!(w<3840 && h<2160 && (w*2160).abs_diff(h*3840)<=3840);
        check([1920,1080],64).unwrap();
        // Remedies hold for every binding: with a 512 MiB limit the slots fail
        // first, and the advised resolution must also fit the larger metadata.
        let small=wgpu::Limits {max_storage_buffer_binding_size:1<<29,max_buffer_size:1<<40,..Default::default()};
        let error=check_view(&small,[3840,2160],64,shape).unwrap_err().to_string();
        let [w,h]:[u32;2]=error.split("resolution to ").nth(1).unwrap().split(' ').next().unwrap()
            .split('x').map(|n|n.parse().unwrap()).collect::<Vec<_>>().try_into().unwrap();
        check_view(&small,[w,h],64,shape).unwrap();
        // One-sample program shading still allocates whole-frame metadata.
        let programs=ViewShape {transparent:false,diffusion:false,reflections:false,screen_refraction:false,boundaries:0,table_only:false,reserved:0};
        assert!(check_view(&small,[3840,2160],1,programs).unwrap_err().to_string().contains("camera evaluation metadata"));
        check_view(&small,[3840,2160],1,ViewShape {table_only:true,..programs}).unwrap();
    }
    #[test]
    fn evaluation_budget_is_the_band_target_within_device_limits() {
        // Metal: 4 GiB bindings and a 96 GiB working set; L4: 2 GiB bindings, 23 GiB.
        let metal=wgpu::Limits {max_storage_buffer_binding_size:(1<<32)-4,max_buffer_size:1<<40,..Default::default()};
        let l4=wgpu::Limits {max_storage_buffer_binding_size:(1<<31)-1,max_buffer_size:1<<40,..Default::default()};
        for (limits,memory) in [(&metal,96u64<<30),(&l4,23<<30)] {
            assert_eq!(evaluation_budget(limits,memory,2),EVALUATION_BAND_RECORDS,"the target, not the device, sizes a band");
        }
        // One binding of 128-byte records, and a small GPU's memory share, bound it.
        let gib=wgpu::Limits {max_storage_buffer_binding_size:1<<30,..l4.clone()};
        assert_eq!(evaluation_budget(&gib,23<<30,8),(1u64<<30)/128);
        assert_eq!(evaluation_budget(&l4,4<<30,8),(4u64<<30)/EVALUATION_MEMORY_SHARE/128);
    }
    #[test]
    fn per_sample_evaluations_start_within_the_band_budget() {
        // L4, 1080p, 64 samples, transparent per-sample content, 128 B/record.
        let limits=wgpu::Limits {max_storage_buffer_binding_size:(1<<31)-1,max_buffer_size:1<<40,..Default::default()};
        let budget=evaluation_budget(&limits,23<<30,8);
        let pixels=1920u64*1080;
        let slots=pixels*u64::from(VISIBILITY_SLOTS)+pixels;
        let content=pixels*(64+u64::from(VISIBILITY_SLOTS));
        // Bands split the content with headroom; a full band's share fits the budget.
        let estimate=content+content/EVALUATION_ESTIMATE_HEADROOM;
        let (bands,rows)=shading_bands(estimate,budget,1080);
        let share=(estimate*u64::from(rows)).div_ceil(1080);
        assert!(share<=budget,"{bands} bands of {rows} rows");
        assert_eq!(evaluation_capacity(slots,share,0,budget),slots.max(share));
        // Reported demand is always reserved; power-of-two headroom only within the budget.
        assert_eq!(evaluation_capacity(slots,0,budget+1,budget),budget+1);
        assert_eq!(evaluation_capacity(0,0,1000,budget),1024);
        assert_eq!(evaluation_capacity(slots,0,0,budget),slots);
    }
    #[test]
    fn shading_bands_split_records_beyond_the_budget_into_tile_rows() {
        assert_eq!(shading_bands(100,1000,1080),(1,1080));
        // The ocean at 64 samples: 155M records over a 16.7M-record binding.
        let (bands,rows)=shading_bands(155_271_662,((1u64<<31)-1)/128,1080);
        assert_eq!((bands,rows),(11,104));
        // A full band's share of the records stays within the budget, and the
        // bands tile the rows.
        for height in [1080,25,7] {
            for wanted in 1..=height/TILE {
                let (bands,rows)=shading_bands(u64::from(wanted)*1000,1000,height);
                assert!(rows*wanted<=height && (rows%TILE==0 || rows==height) && (bands-1)*rows<height && bands*rows>=height,"{height} {wanted}");
            }
        }
        assert_eq!(shading_bands(2000,1000,25),(4,8),"a 25-row view splits below 16-row bands");
        // Demand beyond one record per tile row still ends at one tile row per band.
        assert_eq!(shading_bands(u64::MAX,1,20),(3,8));
    }
    #[test]
    fn dielectric_occupancy_sizes_bands_independently_of_evaluations()->Result<()> {
        let metal=wgpu::Limits {max_storage_buffer_binding_size:(1<<32)-4,max_buffer_size:1<<40,..Default::default()};
        let extent=[1920,1080];let samples=64;
        // Constant table glass needs no per-sample evaluation bands. Even a
        // single active solid's full-frame containment exceeds Metal's binding.
        let shape=ViewShape {transparent:true,diffusion:false,reflections:false,screen_refraction:false,table_only:true,boundaries:1,reserved:0};
        for limits in [metal.clone(),wgpu::Limits {max_buffer_size:1<<31,..metal}] {
            let row=occupancy_row_bytes(extent[0],samples,shape.boundaries);
            let sizes=check_view(&limits,extent,samples,shape)?;
            let metadata=sizes[1]-row*u64::from(TILE);
            let limit=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
            assert!(metadata+row*u64::from(extent[1])>limit);
            let (bands,rows)=occupancy_bands(&limits,96<<30,metadata,row,extent[1],extent[1])?;
            assert!(bands>1 && rows%TILE==0);
            assert!(metadata+row*u64::from(rows)<=limit);
            assert!((bands-1)*rows<extent[1] && bands*rows>=extent[1]);
            // An already smaller evaluation band is preserved.
            assert_eq!(occupancy_bands(&limits,96<<30,metadata,row,1080,8)?,(135,8));
        }
        let limits=wgpu::Limits {max_storage_buffer_binding_size:1024,max_buffer_size:1024,..Default::default()};
        assert_eq!(occupancy_bands(&limits,u64::MAX,512,64,25,25)?,(4,8));
        assert_eq!(occupancy_bands(&limits,0,512,64,7,7)?,(1,7));
        assert!(occupancy_bands(&limits,u64::MAX,513,64,25,25).unwrap_err().to_string().contains("one tile row"));
        assert_eq!(occupancy_bands(&limits,0,512,0,25,25)?,(1,25));
        Ok(())
    }
    #[test]
    fn dielectric_occupancy_keeps_room_beside_the_visibility_pool()->Result<()> {
        let limits=wgpu::Limits {max_storage_buffer_binding_size:128<<20,max_buffer_size:128<<20,..Default::default()};
        let shape=ViewShape {transparent:true,diffusion:false,reflections:false,screen_refraction:false,table_only:true,boundaries:32,reserved:0};
        let extent=[640,360];let samples=16;
        let pool=visibility_pool(&limits,shape,extent,samples);
        let sizes=check_view(&limits,extent,samples,shape)?;
        let row=occupancy_row_bytes(extent[0],samples,shape.boundaries);
        let metadata=sizes[1]-row*u64::from(TILE);
        let (_,rows)=occupancy_bands(&limits,u64::MAX,metadata,row,extent[1],extent[1])?;
        assert!(metadata+row*u64::from(rows)<=limits.max_storage_buffer_binding_size);
        // Spare speculative pool space must yield to the minimum band.
        assert!(pool<visibility_pool(&limits,ViewShape {boundaries:0,..shape},extent,samples));
        Ok(())
    }
    /// Force containment alone to split table-glass shading; compare all rows,
    /// including a short final band and camera origins inside a tinted solid.
    #[test]
    fn dielectric_occupancy_bands_preserve_camera_transport()->Result<()> {
        let Some(mut r)=crate::test_renderer()? else {return Ok(());};
        if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {return Ok(());}
        let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut input=vork_scene::v2::with_test_settings(include_str!("../../tests/fixtures/analytic.json"),include_str!("../../tests/presets/analytic-7fefd52a.json"))?;
        input["animation"]=serde_json::json!([]);
        input["render"]=serde_json::json!({"resolution":[24,25],"background":"environment","settings":{
            "integrator":"raster","refraction":"ray","reflections":"environment","samples":4,"denoiser":null}});
        input["materials"]=serde_json::json!({"glass":{"inputs":{"transmission_weight":1,"specular_weight":0,"specular_ior":1,
            "specular_roughness":0,"transmission_color":[0.25,0.5,1],"transmission_depth":1}}});
        input["nodes"]=serde_json::json!([
            {"type":"camera","translation":[0,0,0],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":0.2,"clip_start":0.2}}},
            {"type":"mesh","geometry":"shape_box","materials":["glass"]},
            {"type":"light","shape":"environment","color":[1,1,1],"energy":1}]);
        let scene:Scene=vork_scene::v2::from_test_value(input)?;
        let render=|r:&Renderer|->Result<(u32,Vec<[f32;4]>)> {
            let mut installed=r.install(scene.clone(),&root)?;
            let mut frame=installed.create_frame_slot(r,crate::ReadbackPrecision::Float)?;
            let output=installed.render_frame(r,&mut frame,0.)?;
            Ok((frame.view.decoupled.as_ref().unwrap().bands,output.linear_rgba.unwrap()))
        };
        let (one,expected)=render(&r)?;assert_eq!(one,1);
        assert!(expected.iter().all(|p|p[0]>0. && p[0]<p[1] && p[1]<p[2]),"camera containment must tint every row");
        r.evaluation_memory=Some(occupancy_row_bytes(24,4,1)*u64::from(TILE)*EVALUATION_MEMORY_SHARE);
        let (bands,actual)=render(&r)?;
        assert_eq!(bands,4);crate::test_linear::assert_linear_eq(&actual,&expected);
        Ok(())
    }
    #[cfg(all(test, vork_private_tests))]
    include!("decoupled_dielectric_occupancy_bands_preserve_mixed_glass_scene_private.rs");
    /// Per-sample program and blended shading at 96x64, 8 samples.
    #[cfg(vork_private_tests)]
    fn program_transparency() -> Result<(Scene,std::path::PathBuf)> {
        let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let mut scene=crate::load_test_fixture(&root.join("program-transparency.json"), include_str!("../../tests/presets/program-transparency-874d5753.json"))?;
        scene.render.resolution=[96,64];scene.render.settings.samples=8;
        Ok((scene,root))
    }
    /// Per-sample program and blended shading split into bands reproduces the
    /// one-band linear image within float noise: each band classifies, evaluates and composes its rows.
    #[test]
    #[cfg(vork_private_tests)]
    fn shading_bands_reproduce_the_single_band_image() -> Result<()> {
        let Some(mut r) = crate::test_renderer()? else {return Ok(());};
        let (scene,root)=program_transparency()?;
        let render=|r:&Renderer| -> Result<(u32,Vec<[f32;4]>)> {
            let mut installed=r.install(scene.clone(),&root)?;
            let mut frame=installed.create_frame_slot(r,crate::ReadbackPrecision::Float)?;
            let output=installed.render_frame(r,&mut frame,0.5)?;
            Ok((frame.view.decoupled.as_ref().unwrap().bands,output.linear_rgba.unwrap()))
        };
        let (one,expected)=render(&r)?;
        assert_eq!(one,1);
        // A device budget of a quarter of the per-sample records (128 B each).
        r.evaluation_memory=Some(96*64*8/4*128*EVALUATION_MEMORY_SHARE);
        let (bands,banded)=render(&r)?;
        assert!(bands>=3,"{bands} bands");
        crate::test_linear::assert_linear_eq(&expected,&banded);
        Ok(())
    }
    /// Constant table glass has visibility-dependent per-sample work even when
    /// refraction uses rays. Its first submission must already budget that work.
    #[test]
    fn table_transparency_reserves_samples_before_first_submission() -> Result<()> {
        let Some(mut r)=crate::test_renderer()? else {return Ok(());};
        let scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[96,64],"settings":{"integrator":"raster","samples":32,"refraction":"ray",
                "probes":{"gi_diffuse_bounces":1,"volume":{"resolution_x":1,"resolution_y":1,"resolution_z":1,"rays_per_probe":32}}}},
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]}},
            "geometry":{"triangle":{"attributes":{"POSITION":"p"}}},
            "materials":{"glass":{"inputs":{"transmission_weight":1,"geometry_thin_walled":true,"specular_roughness":0.1}}},
            "nodes":[{"type":"camera","translation":[0,0,4],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1.5}}},
                {"type":"mesh","geometry":"triangle","materials":["glass"]},
                {"type":"mesh","geometry":"triangle","translation":[0,0,-1],"scale":[2,2,1]},
                {"type":"light","shape":"sun","energy":3}]}))?;
        let render=|r:&Renderer|->Result<(u32,Vec<[f32;4]>)> {
            let installed=r.install(scene.clone(),std::path::Path::new("."))?;
            assert_eq!(installed.decoupled.as_ref().unwrap().materials,2,"table materials only");
            assert!(installed.decoupled.as_ref().unwrap().per_sample);
            let mut frame=installed.create_frame_slot(r,crate::ReadbackPrecision::Float)?;
            let view=frame.view.decoupled.as_ref().unwrap();
            let bands=view.bands;
            frame.submit(&installed,r,0.)?;
            Ok((bands,frame.collect(r)?.linear_rgba.unwrap()))
        };
        let (one,expected)=render(&r)?;
        assert_eq!(one,1);
        r.evaluation_memory=Some(96*64*32/4*128*EVALUATION_MEMORY_SHARE);
        let (bands,actual)=render(&r)?;
        assert!(bands>1,"sample capacity must split table glass into bands");
        crate::test_linear::assert_linear_eq(&expected,&actual);
        Ok(())
    }
    /// A frame whose evaluations overflow the view's estimate redoes with the
    /// reported demand, destroys the outgrown storage, and keeps its pixels.
    #[test]
    #[cfg(vork_private_tests)]
    fn evaluation_overflow_redoes_and_destroys_the_outgrown_storage() -> Result<()> {
        let Some(mut r) = crate::test_renderer()? else {return Ok(());};
        let (scene,root)=program_transparency()?;
        let expected={
            let mut installed=r.install(scene.clone(),&root)?;
            let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            installed.render_frame(&r,&mut frame,0.5)?.linear_rgba.unwrap()
        };
        // An estimate of one evaluation reserves only the slots; the per-sample
        // evaluations of 8 samples overflow them.
        r.evaluation_estimate=Some(1);
        let mut installed=r.install(scene,&root)?;
        let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        let error=frame.submit(&installed,&r,0.5).and_then(|()|frame.collect(&r)).err().context("the estimate must overflow")?;
        let demand=error.downcast_ref::<crate::FrameRedo>().context("an overflow redoes the frame")?.0;
        assert!(demand.shading_evaluations>0);
        let replaced=frame.view.decoupled.as_ref().context("decoupled view")?.evaluation_bins.clone();
        assert_eq!(installed.redo_frame(&r,std::slice::from_mut(&mut frame),demand,0)?,1);
        // The replaced generation's storage was destroyed, not left to its last handle.
        let scope=r.device.push_error_scope(wgpu::ErrorFilter::Validation);
        r.queue.write_buffer(&replaced,0,&[0;4]);
        r.queue.submit([]);
        assert!(pollster::block_on(scope.pop()).is_some(),"the replaced material bins are still allocated");
        let output=installed.render_frame(&r,&mut frame,0.5)?;
        crate::test_linear::assert_linear_eq(output.linear_rgba.as_ref().unwrap(),&expected);
        Ok(())
    }
    /// Recovery preserves an accepted successor's pixels and GPU-only fence,
    /// but releases the saved result when a different request uses that slot.
    #[test]
    #[cfg(vork_private_tests)]
    fn recovery_retains_an_accepted_queued_exposure() -> Result<()> {
        let Some(mut r) = crate::test_renderer()? else { return Ok(()); };
        let (scene, root) = program_transparency()?;
        let mut installed = r.install(scene, &root)?;
        let good = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        r.evaluation_estimate = Some(1);
        let bad = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        let mut slots = [bad, good];
        slots[0].submit(&installed, &r, 0.5)?;
        slots[1].submit(&installed, &r, 0.75)?;
        let error = slots[0].collect(&r).err().context("undersized slot must reject")?;
        let demand = error.downcast_ref::<crate::FrameRedo>().context("capacity retry")?.0;
        installed.redo_frame(&r, &mut slots, demand, 2)?;
        assert!(slots[1].completed.is_some(), "accepted successor must survive storage growth");
        assert!(slots[0].completed.is_none());
        let first = installed.render_frame(&r, &mut slots[0], 0.5)?;
        // The mapped output lease is still held: an actual resubmission would
        // deadlock on its credit. Recollection must consume the saved result.
        slots[1].submit(&installed, &r, 0.75)?;
        assert!(slots[1].completed.is_some());
        let retained = slots[1].collect(&r)?;
        assert!(slots[1].completed.is_none());
        let expected = installed.render_frame(&r, &mut slots[1], 0.75)?;
        crate::test_linear::assert_linear_eq((retained).linear_rgba.as_ref().expect("linear readback"), (expected).linear_rgba.as_ref().expect("linear readback"));
        crate::test_linear::assert_depth_eq(retained.depth_f32.as_ref().expect("float depth readback"), expected.depth_f32.as_ref().expect("float depth readback"));
        assert!(!first.rgba.is_empty());
        // A different request must release the saved lease and render normally.
        slots[1].submit(&installed, &r, 0.75)?;
        slots[1].collect_for_redo(&r, true)?;
        slots[1].submit_gpu(&installed, &r, 1.)?;
        assert!(slots[1].completed.is_none());
        assert!(slots[1].collect(&r)?.rgba.is_empty());
        slots[1].write_color = true;
        slots[1].submit_gpu(&installed, &r, 1.)?;
        slots[1].collect_for_redo(&r, true)?;
        slots[1].submit_gpu(&installed, &r, 1.)?;
        assert!(slots[1].completed.is_some(), "GPU-only outputs retain their completion fence too");
        assert!(slots[1].collect(&r)?.rgba.is_empty());
        r.errors.check()
    }

    /// A scene whose per-sample evaluations exceed what one material-bin binding
    /// addresses renders in bands within the device limits.
    #[test]
    fn per_sample_records_beyond_one_binding_render_in_bands() -> Result<()> {
        let Some(r) = crate::test_renderer()? else {return Ok(());};
        let limits=r.device.limits();
        let records=limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)/16;
        let samples=64u32;
        let side=((records as f64*1.25/f64::from(samples)).sqrt().ceil() as u32).next_multiple_of(8);
        let scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[side,side],"settings":{"integrator":"raster","samples":samples}},
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"attributes":{"POSITION":"p"},"indices":"i"}},
            "programs":{"rings":{"source":"struct Inputs {}; Closure surface(ShadingContext ctx,Inputs inputs) { Closure c={}; c.base_color=float3(frac(length(ctx.POSITION.xy)*8.)); return c; }"}},
            "materials":{"rings":{"program":"rings"}},
            "nodes":[{"type":"camera","name":"camera","translation":[0,0,4],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1.5}}},
                {"type":"mesh","name":"quad","geometry":"quad","materials":["rings"]},
                {"type":"light","name":"sun","shape":"sun","energy":3}]}))?;
        let mut installed=r.install(scene,std::path::Path::new("."))?;
        let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        let output=installed.render_frame(&r,&mut frame,0.)?;
        let v=frame.view.decoupled.as_ref().unwrap();
        assert!(u64::from(side)*u64::from(side)*u64::from(samples)>records);
        assert!(v.bands>1,"one band");
        // Every band composed its rows: no unshaded (background-black) band.
        let row_bytes=side as usize*4;
        for (band,rows) in output.rgba.chunks(row_bytes*v.band_rows as usize).enumerate() {
            assert!(rows.chunks(4).any(|p|p[0]>32),"band {band} is empty");
        }
        Ok(())
    }
    #[test]
    fn merge_batches_cover_every_pass_within_one_mask_word() {
        for passes in [1, 5, 8, 12, 64, 100] {
            let batch = MERGE_BATCH.min(passes);
            let batches: Vec<_> = merge_batches(passes, batch).collect();
            let mut next = 0;
            for &(first, count) in &batches {
                assert_eq!(first, next);
                assert!(count >= 1 && count <= batch);
                assert!(first % 32 + count <= 32, "{passes} passes: batch ({first}, {count}) crosses a mask word");
                next = first + count;
            }
            assert_eq!(next, passes);
        }
    }
    // Empty SIMD lanes must never reach descriptor loads in either the direct
    // opaque walk or the grouped one-sample transparent walk.
    #[test]
    #[cfg(vork_private_tests)]
    fn single_sample_shading_preserves_sparse_pixel_execution() -> Result<()> {
        let Some(r) = crate::test_renderer()? else {return Ok(());};
        let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let mut scene=crate::load_test_fixture(&root.join("jitter-aa.json"), include_str!("../../tests/presets/jitter-aa-f8d98a5b.json"))?;
        scene.render.resolution=[64,64];scene.render.settings.samples=1;
        let render=|scene:Scene,direct:bool| -> Result<Vec<[f32;4]>> {
            let mut installed=r.install(scene,&root)?;
            let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            assert_eq!(frame.view.decoupled.as_ref().unwrap().direct_keys,direct);
            // Direct work skips classification; timestamp readback must never resolve
            // its unwritten timestamp pair (a driver query wait can hang).
            Ok(installed.render_frame(&r, &mut frame, 0.)?.linear_rgba.unwrap())
        };
        // The authored transmissive sphere exercises the original grouped crash.
        render(scene.clone(),false)?;
        for material in scene.materials.values_mut().filter(|m|m.program=="std/openpbr") {
            material.inputs.insert("transmission_weight".into(),vork_scene::Input::Number(0.));
            material.inputs.insert("geometry_opacity".into(),vork_scene::Input::Number(1.));
        }
        let direct=render(scene.clone(),true)?;
        let mut diffusion=scene.clone();
        for material in diffusion.materials.values_mut().filter(|m|m.program=="std/openpbr") {
            material.inputs.insert("subsurface_weight".into(),vork_scene::Input::Number(1.));
            material.inputs.insert("subsurface_radius".into(),vork_scene::Input::Number(2.));
        }
        let diffused=render(diffusion.clone(),false)?;
        // Keep the local SSS lobe identical; only a resolved diffusion radius
        // can distinguish these images (the tiny radius is below its cutoff).
        let mut tiny_radius=diffusion.clone();
        for material in tiny_radius.materials.values_mut().filter(|m|m.program=="std/openpbr") {
            material.inputs.insert("subsurface_radius".into(),vork_scene::Input::Number(0.001));
        }
        assert_ne!(diffused,render(tiny_radius,false)?,"diffusion must affect visible pixels");
        // An invisible-to-lighting off-screen blend switches grouping on without
        // changing any camera pixel. Both routes must preserve the linear image within float noise.
        scene.materials.insert("offscreen-blend".into(),vork_scene::v2::from_test_value(serde_json::json!({
            "opacity":"transmittance","inputs":{"geometry_opacity":0.5}}))?);
        scene.nodes.push(vork_scene::v2::from_test_value(serde_json::json!({"type":"mesh","name":"offscreen-blend","geometry":"unit_box","materials":["offscreen-blend"],"visible_shadow":false,"visible_diffuse":false,"visible_glossy":false,"translation":[10000,0,0]}))?);
        crate::test_linear::assert_linear_eq(&direct,&render(scene.clone(),false)?);
        diffusion.materials.insert("offscreen-blend".into(),scene.materials["offscreen-blend"].clone());
        diffusion.nodes.push(scene.nodes.last().unwrap().clone());
        crate::test_linear::assert_linear_eq(&diffused,&render(diffusion,false)?);
        Ok(())
    }
    #[test]
    fn opaque_slot_results_preserve_multiple_pixels_with_overflow() -> Result<()> {
        let Some(mut r)=crate::test_renderer()? else {return Ok(());};
        // Force opaque compact results through multiple bands, including a
        // measured overflow pool larger than the retained evaluation store.
        r.evaluation_memory=Some(8*37*4*32*EVALUATION_MEMORY_SHARE);
        let mut nodes=vec![serde_json::json!({"type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","fit":"horizontal","ortho_scale":2.}}})];
        let mut materials=serde_json::Map::new();
        for band in 0..8 {
            let name=format!("band{band}");
            materials.insert(name.clone(),serde_json::json!({"inputs":{"base_weight":0.,"specular_weight":0.,"emission_luminance":1.,"emission_color":[band as f32/7.,0.25,1.-band as f32/7.]}}));
            nodes.push(serde_json::json!({"type":"mesh","geometry":"quad","materials":[name],"translation":[-1.+(band as f32+0.5)*0.25,0,0],"scale":[0.125,1,1]}));
        }
        let scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"topology":"triangles","attributes":{"POSITION":"p"},"indices":"i"}},
            "materials":materials,
            "nodes":nodes,
            "render":{"resolution":[2,2],"settings":{"integrator":"raster","samples":64,"reflections":"environment","fast_gi":null},"display":{"program":"std/standard","inputs":{"dither_intensity":0.}}}
        }))?;
        let render=|scene:Scene,indexed:bool|->Result<Vec<[f32;4]>> {
            let require_overflow=scene.render.resolution[0]==2;
            let options=crate::InstallOptions {reservations:crate::Reservations {visibility_pool:4096,..Default::default()},..Default::default()};
            let mut installed=r.install_with_options(scene,std::path::Path::new("."),options)?;
            let mut frame=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            assert_eq!(frame.view.decoupled.as_ref().unwrap().evaluation.classification.as_ref().map(Vec::len),Some(if indexed {2} else {3}));
            if indexed && installed.render.resolution[1]>8 {
                assert!(frame.view.decoupled.as_ref().unwrap().bands>1);
            }
            let pixels=installed.render_frame(&r, &mut frame, 0.)?.linear_rgba.unwrap();
            if !require_overflow || !indexed {return Ok(pixels);}
            let readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("overflow allocation proof"),size:4,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
            let mut encoder=r.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&frame.view.decoupled.as_ref().unwrap().overflow,0,&readback,0,4);
            let submission=r.queue.submit([encoder.finish()]);let (tx,rx)=std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read,..,move|result|{let _=tx.send(result);});
            r.device.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;rx.recv()??;
            let data=readback.get_mapped_range(..);let allocated=u32::from_le_bytes(data[..4].try_into().unwrap());
            if require_overflow {assert!(allocated>0,"the scene must exercise pooled visibility slots");}
            drop(data);readback.unmap();Ok(pixels)
        };
        for resolution in [[2,2],[2,19],[37,19]] {
            let mut scene=scene.clone();scene.render.resolution=resolution;
            if resolution==[2,19] {
                // Keep all eight bands across the narrow image (horizontal fit)
                // and stretch them over three tile rows.
                for node in &mut scene.nodes[1..] {node.transform.scale[1]=9.5;}
            }
            let indexed=render(scene.clone(),true)?;
            assert!(indexed.iter().any(|p|p[..3]!=indexed[0][..3]),"different camera pixels must retain different bands");
            scene.materials.insert("offscreen-blend".into(),vork_scene::v2::from_test_value(serde_json::json!({"opacity":"transmittance","inputs":{"geometry_opacity":0.5}}))?);
            scene.nodes.push(vork_scene::v2::from_test_value(serde_json::json!({"type":"mesh","geometry":"quad","materials":["offscreen-blend"],"visible_shadow":false,"visible_diffuse":false,"visible_glossy":false,"translation":[10000,0,0]}))?);
            crate::test_linear::assert_linear_eq(&indexed,&render(scene,false)?);
        }
        Ok(())
    }

    // Sparse pixels make neighbouring SIMD lanes finish their gathered-key walks
    // at different times; nested early returns crash llvmpipe here.
    #[test]
    fn merge_sparse_keys_preserves_dense_overflow_and_mask_words() -> Result<()> {
        let Some(r) = crate::test_renderer()? else {return Ok(());};
        let d = &r.device;
        let (w, h, layers, samples) = (8u32, 8u32, MERGE_BATCH, 64u32);
        let mut keys = vec![[0u32; 2]; (w*h*layers) as usize];
        let mut expected = vec![std::collections::BTreeMap::new(); (w*h) as usize];
        for layer in 0..layers {
            for (pixel, key) in [(1, [1, 0]), (2, [2, layer]), (3, [3, layer%3])] {
                keys[(layer*w*h+pixel) as usize] = key;
                *expected[pixel as usize].entry(key).or_insert(0u64) |= (1u64<<layer)|(1u64<<(32+layer));
            }
        }
        let texture = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("sparse merge keys"), size: wgpu::Extent3d {width:w,height:h,depth_or_array_layers:layers},
            mip_level_count:1, sample_count:1, dimension:wgpu::TextureDimension::D2, format:KEY_FORMAT,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST, view_formats:&[],
        });
        r.queue.write_texture(texture.as_image_copy(), bytemuck::cast_slice(&keys),
            wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(w*8),rows_per_image:Some(h)}, texture.size());
        let stride = VISIBILITY_KEY_WORDS+samples.div_ceil(32);
        let dense = w*h*VISIBILITY_SLOTS*stride;
        let pool = w*h;
        let words = dense+w*h+pool*(1+stride);
        let slots = gpu_memory::buffer_init(d,"test slots",bytemuck::cast_slice(&vec![0u32;words as usize]),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let overflow = gpu_memory::buffer_init(d,"test overflow",bytemuck::cast_slice(&[0u32,0,pool]),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let header=vork_slang::registry::FRAGMENT_HEADER_WORDS;
        let record=vork_slang::registry::FRAGMENT_RECORD_WORDS;
        let mut fragment_words=vec![0u32;(header+record+layers*w*h) as usize];
        fragment_words[..header as usize].copy_from_slice(&[0,0,1,w,h]);
        let fragments = gpu_memory::buffer_init(d,"empty fragments",bytemuck::cast_slice(&fragment_words),wgpu::BufferUsages::STORAGE);
        let sampling = Sampling::new(d,[w,h],samples);
        let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:None,entries:&merge_entries()});
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:None,bind_group_layouts:&[Some(&group)],immediate_size:0});
        let pipeline = compute_pipeline(&r,"test merge",&layout,&native::DECOUPLED_MERGE_MERGE);
        let view = texture.create_view(&Default::default());
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&group,entries:&[
            entry(0,wgpu::BindingResource::TextureView(&view)),entry(1,slots.as_entire_binding()),
            entry(2,overflow.as_entire_binding()),entry(3,fragments.as_entire_binding()),
            entry(4,wgpu::BindingResource::Buffer(wgpu::BufferBinding {buffer:&sampling.merges,offset:0,size:std::num::NonZeroU64::new(PARAMETERS)})),
        ]});
        let readback = gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:None,size:u64::from(words+3)*4,
            usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
        let mut encoder = d.create_command_encoder(&Default::default());
        for batch in [0,32/MERGE_BATCH] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[batch*sampling.stride]);pass.dispatch_workgroups(1,1,1);
        }
        encoder.copy_buffer_to_buffer(&slots,0,&readback,0,u64::from(words)*4);
        encoder.copy_buffer_to_buffer(&overflow,0,&readback,u64::from(words)*4,12);
        let submission = r.queue.submit([encoder.finish()]);
        let (tx,rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move|result| {let _=tx.send(result);});
        d.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;rx.recv()??;
        let mapped = readback.get_mapped_range(..);
        let data: &[u32] = bytemuck::cast_slice(&mapped);
        assert_eq!(&data[words as usize..(words+3) as usize], &[4,0,pool], "coverage must deduplicate across mask words");
        for pixel in 0..w*h {
            let mut actual = std::collections::BTreeMap::new();
            let mut add = |at:u32| {
                let at=at as usize;
                if data[at]!=0 {assert!(actual.insert([data[at],data[at+1]],u64::from(data[at+2])|(u64::from(data[at+3])<<32)).is_none());}
            };
            for slot in 0..VISIBILITY_SLOTS {add((pixel*VISIBILITY_SLOTS+slot)*stride);}
            let mut next=data[(dense+pixel) as usize];let mut visited=0;
            while next!=0 {
                assert!(next<=pool && visited<pool,"invalid overflow list");visited+=1;
                let at=dense+w*h+(next-1)*(1+stride);add(at+1);next=data[at as usize];
            }
            assert_eq!(actual,expected[pixel as usize],"pixel {pixel}");
        }
        drop(mapped);readback.unmap();
        Ok(())
    }

}

#[cfg(test)]
mod warm_key_tests {
    use super::*;
    #[test]
    fn direct_camera_constructors_remember_their_complete_kernel() -> Result<()> {
        let Some(r)=crate::test_renderer()? else {return Ok(());};
        if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {return Ok(());}
        let p=r.program_pipelines("",RESOLVE,1,true,crate::slang_forward::SlangScene {
            probes:true,ray_refraction:true,..Default::default()
        })?;
        let root=std::env::temp_dir().join(format!("vork-camera-known-{}",std::process::id()));
        let known=pipelines::Known::load(root.join("keys.entry"),false);
        let c=pipelines::Compiler::new(&r.device,r.cache.as_ref(),known);
        let layout=shade_layout(&r.device,&p);
        for stage in [ShadeStage::Indirect,ShadeStage::Occupancy] {
            let boundaries=if stage==ShadeStage::Occupancy {1} else {0};
            let different=pipelines::ProgramKernel::Decoupled {stage,features:0,subsurface:false,emitter_profiles:false,boundaries};
            let kernel=different.superset().expect("specialized camera stage");
            assert!(!c.is_known(&p.kernel_content_key(kernel)));
            kernel_pipeline(&c,&p,&layout,kernel,&kernel.camera_label().unwrap())?;
            assert!(c.is_known(&p.kernel_content_key(kernel)));
            assert!(!c.is_known(&p.kernel_content_key(different)),"publication must retain the complete feature key");
            assert_eq!(p.drawn_kernel(different,&pipelines::VariantCompile::Fallback,&c),kernel);
        }
        r.errors.check()
    }
}
