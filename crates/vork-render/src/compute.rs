//! Installed Slang dispatches over retained named-buffer pages. Queue submission
//! and the simulation cursor share one lock, so frame slots cannot race state.
use crate::buffers::{AttributeBinding, ProgramBuffers};
use crate::*;
use ::vork_scene::v2::{self, ComputeSchedule};
use std::{collections::BTreeSet, fmt::Write as _, sync::Mutex};

pub(crate) type InputSampler<'a> = dyn FnMut(&mut wgpu::CommandEncoder, ComputeSchedule, &vork_time::Request) -> Result<()> + 'a;

pub(crate) const CONTEXT_BINDINGS: &str = "struct VorkTick { uint frame; float time; float dt; uint reserved; };\n[[vk::binding(0,0)]] ConstantBuffer<VorkTick> vork_tick;\nstruct VorkOffset { uint4 value; };\n[[vk::binding(1,0)]] ConstantBuffer<VorkOffset> vork_offset;\n";

// Integer exponent checks remain valid under native fast-math compilation.
pub(crate) const FINITE_WGSL: &str = r#"
fn finite_f32(v:f32)->bool {return (bitcast<u32>(v) & 0x7f800000u)!=0x7f800000u;}
fn finite3(v:vec3<f32>)->bool {return all((bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
fn finite4(v:vec4<f32>)->bool {return all((bitcast<vec4<u32>>(v) & vec4<u32>(0x7f800000u))!=vec4<u32>(0x7f800000u));}
"#;

/// Declared compute bindings conservatively publish writes to their named source.
pub(crate) struct PreparedProgram {
    pub artifact:vork_slang::ComputeProgram,
    pub header:String,
    pub origin:std::path::PathBuf,
    pub identity:vork_slang::InputIdentity,
    pub input_binding:u32,
    key:blake3::Hash,
    used:BTreeSet<u32>,
    implicit:Vec<GeometryInputBinding>,
}
pub(crate) type Prepared=std::collections::BTreeMap<(String,Option<String>),PreparedProgram>;
pub(crate) fn prepare(r:&Renderer,source:&v2::Scene,base:&std::path::Path,attributes:Option<&ProgramBuffers>,sources:&[Option<&v2::Node>])->Result<Prepared> {
    prepare_with_packed(r,source,base,attributes,sources,&BTreeSet::new())
}
pub(crate) fn prepare_with_packed(r:&Renderer,source:&v2::Scene,base:&std::path::Path,attributes:Option<&ProgramBuffers>,sources:&[Option<&v2::Node>],packed:&BTreeSet<String>)->Result<Prepared> {
    if !source.programs.values().any(|program|program.is_compute()) {return Ok(Default::default());}
    let named=attributes.context("compute requires named buffers")?;
    let mut geometry=std::collections::BTreeMap::new();
    for node in sources.iter().flatten() {
        if let v2::NodeKind::Mesh {geometry:name,..}=&node.kind {
            let Some(shape)=source.geometry.get(name).filter(|shape|crate::computed_geometry::dynamic(source,shape)) else {continue;};
            if geometry.contains_key(name) {continue;}
            let names:BTreeSet<String>=shape.attributes.values().chain(shape.widths.iter()).chain(shape.count.iter()).chain(shape.ids.iter()).chain(shape.status.iter().map(|s| &s.buffer)).cloned().collect();
            let (programs,state)=geometry_solver_dependencies(source,names.clone());
            let plan=private_geometry_plan(named,&state,&names)?;
            geometry.insert(name.clone(),(programs,plan.buffers));
        }
    }
    let global=global_solver_consumers(source,sources);
    let mut prepared=Prepared::new();
    for (name,program) in source.programs.iter().filter(|(_,program)|program.is_compute()) {
        if global.contains(name) || !geometry.values().any(|(programs,_)|programs.contains(name)) {
            prepared.insert((name.clone(),None),prepare_program(r,source,base,named,named,None,false,name,program)?);
        }
    }
    for (geometry,(programs,buffers)) in &geometry {
        // Global consumers compile without a geometry header: they cannot read
        // predecessor pages or per-object constants. Share only whole closures.
        if programs.is_subset(&global) { continue; }
        for name in programs {
            prepared.insert((name.clone(),Some(geometry.clone())),prepare_program(r,source,base,named,buffers,Some(&source.geometry[geometry]),packed.contains(geometry),name,&source.programs[name])?);
        }
    }
    Ok(prepared)
}
/// Declarations only: stage discovery checks the whole module, while only a
/// compute installation supplies implementations and resource bindings.
pub(crate) fn stage_declarations(scene:&v2::Scene, program:&v2::Program)->Result<String> {
    fn accessor(header:&mut String,name:&str,kind:v2::BufferType)->Result<&'static str> {
        let ty=match kind {
            v2::BufferType::F32=>"float",v2::BufferType::I32=>"int",v2::BufferType::U32=>"uint",
            v2::BufferType::Vec2=>"float2",v2::BufferType::Vec3=>"float3",v2::BufferType::Vec4=>"float4",
            v2::BufferType::Uvec4=>"uint4",v2::BufferType::Mat4=>"VorkMat4",
        };
        if ty=="VorkMat4" && !header.contains("struct VorkMat4") {header.push_str("struct VorkMat4 {float4 c0;float4 c1;float4 c2;float4 c3;};\n");}
        writeln!(header,"extern static const uint {name}_count; {ty} {name}_load(uint i);")?;
        Ok(ty)
    }
    if !program.is_compute() {return Ok(String::new());}
    let mut header=String::new();
    for (alias,buffer) in &program.bindings {
        let kind=scene.buffers[buffer].element_type;
        let ty=accessor(&mut header,alias,kind)?;
        if alias=="fluid_status" {header.push_str("#define VORK_FLUID_STATUS_BOUND 1\n");}
        writeln!(header,"void {alias}_store(uint i,{ty} value);")?;
        accessor(&mut header,&format!("previous_{alias}"),kind)?;
        if scene.buffers[buffer].dims.is_some() {writeln!(header,"extern static const uint3 {alias}_dims;")?;}
        if matches!(kind,v2::BufferType::F32|v2::BufferType::U32) {
            writeln!(header,"{ty} {alias}_atomic_add(uint i,{ty} value);")?;
        }
        if kind==v2::BufferType::U32 {
            for operation in ["sub","min","max","and","or","xor","exchange"] {
                writeln!(header,"uint {alias}_atomic_{operation}(uint i,uint value);")?;
            }
            writeln!(header,"uint {alias}_atomic_compare_exchange(uint i,uint expected,uint desired);")?;
        }
    }
    // Use one of this kernel's actual geometry contexts to type-check its
    // unused body. Installation still compiles every referenced context.
    let name=scene.programs.iter().find_map(|(name,p)|std::ptr::eq(p,program).then_some(name));
    if let Some(geometry)=name.and_then(|name|scene.geometry.values().find(|geometry| {
        let names=geometry.attributes.values().chain(geometry.widths.iter()).chain(geometry.count.iter())
            .chain(geometry.ids.iter()).chain(geometry.status.iter().map(|s| &s.buffer)).cloned().collect();
        geometry_solver_dependencies(scene,names).0.contains(name)
    })) {
        for (index,(semantic,buffer)) in geometry.attributes.iter().enumerate() {
            let kind=scene.buffers[buffer].element_type;
            accessor(&mut header,&format!("geometry_attribute_{index}"),kind)?;
            if matches!(semantic.as_str(),"POSITION"|"NORMAL"|"TANGENT")
                || ["TEXCOORD_","COLOR_","JOINTS_","WEIGHTS_"].iter().any(|prefix|
                    semantic.strip_prefix(*prefix).is_some_and(|index|!index.is_empty() && index.bytes().all(|c|c.is_ascii_digit()))) {
                accessor(&mut header,&format!("geometry_{semantic}"),kind)?;
            }
        }
        for (alias,buffer) in [("width",&geometry.widths),("id",&geometry.ids),("count",&geometry.count)] {
            if let Some(buffer)=buffer {accessor(&mut header,&format!("geometry_{alias}"),scene.buffers[buffer].element_type)?;}
        }
        header.push_str("extern static const uint geometry_attribute_count; extern static const uint geometry_vertex_capacity; uint geometry_vertex_count(); uint geometry_source_vertex(uint i);\n");
    }
    header.push_str("uint geometry_vertex_count(Geometry g); uint geometry_object_id(Geometry g); uint geometry_source_vertex_load(Geometry g,uint i); uint geometry_triangle_count(Geometry g); float3 geometry_POSITION_load(Geometry g,uint i); float3 geometry_NORMAL_load(Geometry g,uint i); uint3 geometry_triangle_load(Geometry g,uint i); float3 geometry_to_world(Geometry g,float3 p);\n");
    Ok(header)
}

fn atomic_accessors(header: &mut String, binding: &str, ty: v2::BufferType, layout: &AttributeBinding) -> Result<()> {
    if matches!(ty, v2::BufferType::U32) {
        for operation in ["add", "sub", "min", "max", "and", "or", "xor", "exchange", "compare_exchange"] {
            let (parameters, method, arguments) = if operation == "compare_exchange" {
                ("uint expected, uint desired", "compareExchange", "expected, desired")
            } else { ("uint value", operation, "value") };
            writeln!(header, "uint {binding}_atomic_{operation}(uint i, {parameters}) {{")?;
            for span in &layout.spans {
                let end = span.first + span.count;
                writeln!(header, "if(i >= {}u && i < {end}u) return vork_page_{}[{}u + i - {}u].{method}({arguments});", span.first, span.page, span.offset, span.first)?;
            }
            let missing = if operation == "min" { "0xffffffffu" } else { "0u" };
            writeln!(header, "return {missing}; }}")?;
        }
    } else if matches!(ty, v2::BufferType::F32) {
        // CAS over the existing u32 pages works on every supported compute device;
        // native float-atomic features are not required for the public operation.
        writeln!(header, "float {binding}_atomic_add(uint i, float value) {{")?;
        for span in &layout.spans {
            let end = span.first + span.count;
            writeln!(header, "if(i >= {}u && i < {end}u) {{ uint word={}u+i-{}u; uint old=vork_page_{}[word].load(); for(;;) {{ uint prior=vork_page_{}[word].compareExchange(old,asuint(asfloat(old)+value)); if(prior==old) return asfloat(old); old=prior; }} }}", span.first, span.offset, span.first, span.page, span.page)?;
        }
        header.push_str("return 0.; }\n");
    }
    Ok(())
}

fn prepare_program(r:&Renderer,source:&v2::Scene,base:&std::path::Path,named:&ProgramBuffers,attributes:&ProgramBuffers,geometry:Option<&v2::Geometry>,packed:bool,name:&str,program:&v2::Program)->Result<PreparedProgram> {
    let threads=program.dispatch.context("compute dispatch required")?.threads();
            let used: BTreeSet<_> = program
                .bindings
                .values()
                .flat_map(|b| attributes.bindings[b].spans.iter().map(|s| s.page))
                .collect();
            let storage_limit=crate::pipelines::scalar_storage_limit(&r.device.limits(),r.adapter_info.backend==wgpu::Backend::Metal,2);
            ensure!(
                used.len() as u32 <= storage_limit,
                crate::gpu_memory::device_limit(format_args!("compute {name}"), format_args!("{} storage buffer pages",used.len()), format_args!("{} storage bindings",storage_limit), "reduce compute buffer pool sizes or the number of input buffers")
            );
            let atomic_pages: BTreeSet<_> = program.bindings.values()
                .filter(|name| matches!(source.buffers[*name].element_type, v2::BufferType::U32 | v2::BufferType::F32))
                .flat_map(|name| attributes.bindings[name].spans.iter().map(|span| span.page)).collect();
            let mut header = String::from(CONTEXT_BINDINGS);
            for (i, page) in used.iter().enumerate() {
                writeln!(
                    header,
                    "[[vk::binding({},0)]] RWStructuredBuffer<{}> vork_page_{page};",
                    i + 2, if atomic_pages.contains(page) { "Atomic<uint>" } else { "uint" }
                )?;
            }
            for (binding, buffer) in &program.bindings {
                let layout = &attributes.bindings[buffer];
                if let Some([x,y,z]) = source.buffers[buffer].dims {
                    writeln!(header, "static const uint3 {binding}_dims = uint3({x}u,{y}u,{z}u);")?;
                }
                layout
                    .count
                    .checked_mul(layout.width)
                    .context("compute binding exceeds u32 word addressing")?;
                writeln!(
                    header,
                    "static const uint {binding}_count = {};\nstatic const uint _vork_{binding}_width = {};",
                    layout.count, layout.width
                )?;
                writeln!(header, "uint _vork_{binding}_load(uint word) {{")?;
                for span in &layout.spans {
                    let first = span.first * layout.width;
                    let end = (span.first + span.count) * layout.width;
                    writeln!(
                        header,
                        "if(word >= {first} && word < {end}) return vork_page_{}[{} + word - {first}]{};",
                        span.page, span.offset, if atomic_pages.contains(&span.page) { ".load()" } else { "" }
                    )?;
                }
                writeln!(
                    header,
                    "return 0; }}\nvoid _vork_{binding}_store(uint word, uint value) {{"
                )?;
                for span in &layout.spans {
                    let first = span.first * layout.width;
                    let end = (span.first + span.count) * layout.width;
                    writeln!(
                        header,
                        "if(word >= {first} && word < {end}) {{ vork_page_{}[{} + word - {first}]{}; return; }}",
                        span.page, span.offset, if atomic_pages.contains(&span.page) { ".store(value)" } else { " = value" }
                    )?;
                }
                header.push_str("}\n");
                atomic_accessors(&mut header, binding, source.buffers[buffer].element_type, layout)?;
                typed_accessors(&mut header, binding, source.buffers[buffer].element_type)?;
            }
            let implicit = geometry_input_header(&mut header, source,
                geometry, packed, named, attributes, program, &used, used.len() as u32 + 2)?;
            let input_binding = used.len() as u32 + 2 + implicit.len() as u32;
            let (text, origin) = programs::program_source(base, name, program)?;
            let compiler=vork_slang::Compiler::discover();
            let key = blake3::hash(
                format!(
                    "compute-tiled:{threads:?}:{}:{}:{}:{}:{}:{header}:{text}",
                    programs::admit(&|key|r.disk_cache.path("slang-input",1,key),&compiler)?.version(),
                    vork_slang::ABI_VERSION,
                    constants::PROGRAM_WORKGROUP,
                    origin.display(),
                    name
                )
                .as_bytes(),
            );
            let abi=programs::abi();
            let inputs=programs::bound_program_identity(&r.disk_cache,&compiler,&abi,&text,&origin,&header)?;
            let compiler=compiler.with_identity(&inputs);
            let artifact = programs::cached(
                r,
                key,
                &compiler,
                &abi, &text, &origin,
                |compiler| {
                    compiler.compile_compute(
                        &text,
                        &origin,
                        &header,
                        constants::PROGRAM_WORKGROUP,
                        threads,
                    )
                },
            )?;
    Ok(PreparedProgram {artifact,header,origin,identity:inputs,key,used,implicit,input_binding})
}

pub(crate) fn writes_buffer(scene: &v2::Scene, name: &str) -> bool {
    scene.programs.values().any(|p|p.is_compute() && p.bindings.values().any(|b|b==name))
}

/// Allocate evaluated inputs only for entries that actually read them.
pub(crate) fn predecessor_geometries(prepared: &Prepared) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for ((_, geometry), program) in prepared {
        let Some(geometry) = geometry else { continue; };
        let mut shader = program.artifact.wgsl.clone();
        if !program.artifact.inputs.fields.is_empty() {
            writeln!(shader, "@group(0) @binding({}) var<storage,read> vork_compute_inputs:array<u32>; fn vork_material_word(i:u32)->u32{{return vork_compute_inputs[i];}}", program.input_binding)?;
        }
        let used = used_compute_bindings(&shader)?;
        if program.implicit.iter().any(|binding| used.contains(&binding.binding)
            && matches!(binding.input, GeometryInput::Predecessor(_))) {
            names.insert(geometry.clone());
        }
    }
    Ok(names)
}
/// Named tables use the same reflected entry bindings as predecessor columns.
pub(crate) fn reference_geometries(scene: &v2::Scene, prepared: &Prepared) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for ((program_name, _), program) in prepared {
        let mut shader = program.artifact.wgsl.clone();
        if !program.artifact.inputs.fields.is_empty() {
            writeln!(shader, "@group(0) @binding({}) var<storage,read> vork_compute_inputs:array<u32>; fn vork_material_word(i:u32)->u32{{return vork_compute_inputs[i];}}", program.input_binding)?;
        }
        let used = used_compute_bindings(&shader)?;
        if program.implicit.iter().any(|binding| used.contains(&binding.binding)
            && matches!(binding.input, GeometryInput::Reference)) {
            names.extend(scene.materials.values().filter(|material| material.program == *program_name)
                .flat_map(|material| material.inputs.values()).filter_map(|input| match input {
                    v2::Input::Geometry(binding) => Some(binding.geometry.clone()), _ => None,
                }));
        }
    }
    Ok(names)
}
pub(crate) fn writes_instance_attributes(scene: &v2::Scene, node: &v2::Node, columns: &[&str]) -> bool {
    matches!(&node.kind,v2::NodeKind::Mesh {instances:Some(inst),..} if inst.attributes.iter()
        .any(|(key,name)|columns.contains(&key.as_str()) && writes_buffer(scene,name)))
}
pub(crate) fn spatial_instances(scene: &v2::Scene, node: &v2::Node) -> bool {
    writes_instance_attributes(scene,node,&["TRANSLATION","ROTATION","SCALE"])
}
/// A mesh whose instances a compute program places: it writes their transforms or count
/// on the GPU. Raster, hit shading and native TLAS descriptors share the resulting
/// frame-private object snapshot.
pub(crate) fn places_instances(scene: &v2::Scene, node: &v2::Node) -> bool {
    matches!(&node.kind,v2::NodeKind::Mesh {instances:Some(inst),..} if inst.count.is_some())
        || spatial_instances(scene,node)
}

// State is canonical as a whole: scratch and private ping-pong buffers can
// contain geometry even when only one column is published to a mesh. Never
// split that state into a GPU geometry tier and a partial disk checkpoint.
// Placement-only simulations retain whole-state disk snapshots: those opaque
// bytes are never consumed as CPU geometry or used to construct a draw.
// Named collider inputs are sampled geometry too; their full input history is
// not part of the disk lineage, so those consumers must retain GPU-only state.
fn geometry_checkpoints(scene: &v2::Scene) -> bool {
    scene.simulation.is_some()
        && (scene.geometry.values().any(|g| crate::computed_geometry::dynamic(scene, g))
            || !crate::geometry_reference::names(scene).is_empty())
}

include!("geometry_solver.rs");
include!("geometry_inputs.rs");

struct Dispatch {
    queries: bool,
    pipeline: wgpu::ComputePipeline,
    tiles: Vec<(wgpu::BindGroup, [u32; 3])>,
    schedule: ComputeSchedule,
    inputs: Option<(wgpu::Buffer, Vec<u32>, Option<programs::InputAnimation>)>,
    name: String,
    writes_placement: bool,
    writes: Vec<String>,
}
impl Dispatch {
    fn publish(&self, cursor: &mut Cursor) {
        if self.writes_placement { cursor.publish_placement(); }
        for name in &self.writes { cursor.publish_source(name); }
    }
}
pub(crate) struct Compute {
    geometry_inputs: Option<crate::geometry_sampling::Inputs>,
    trace: Option<crate::compute_trace::Queries>,
    dispatches: Vec<Dispatch>,
    geometry_scopes: Vec<GeometryScope>,
    previous_state: PreviousState,
    pages: Vec<wgpu::Buffer>,
    /// Placement readers bind the original named namespace, not solver pages.
    named_pages: usize,
    initial: Vec<wgpu::Buffer>,
    previous: Vec<wgpu::Buffer>,
    placement_ranges: Vec<(usize, u64, u64)>,
    device: wgpu::Device,
    timestamp_period: f32,
    simulation: Option<v2::Simulation>,
    simulation_ranges: Vec<(usize, u64, u64)>,
    frame_ranges: Vec<(usize, u64, u64)>,
    frame_sources: Vec<String>,
    checkpoint_store: disk_cache::Store,
    gpu_only_checkpoints: bool,
    identity: blake3::Hash,
    tick: wgpu::Buffer,
    fps: f32,
    rate: vork_time::Time,
    frames: u32,
    pub(crate) cursor: Mutex<Cursor>,
    instances: Vec<Instance>,
    /// Installed copies from retained source ranges to object-owned material rows.
    material_attributes: Vec<(usize, u64, u64, u64)>,
    material_attribute_sources: Vec<String>,
    bridge: Option<(
        wgpu::ComputePipeline,
        u32,
        wgpu::Buffer,
    )>,
}
// Policy limits, never scene appearance controls. Only one checkpoint is staged per request.
const CHECKPOINT_INTERVAL: u32 = 64;
pub(crate) struct Checkpoint {
    pub buffers: Vec<wgpu::Buffer>,
    path: PathBuf,
}
impl Checkpoint {
    pub fn publish(self) {
        let mapped: Vec<_> = self.buffers.iter().map(|buffer|buffer.get_mapped_range(..)).collect();
        let parts: Vec<&[u8]> = mapped.iter().map(|view| &view[..]).collect();
        // Cache disabled/full/unwritable is an optimization miss. Stream the
        // retained pages through the existing checksum/atomic cache writer.
        let _ = disk_cache::write_parts(&self.path, &parts);
        drop(parts);drop(mapped);
        for buffer in self.buffers {buffer.unmap();}
    }
}
#[derive(Default)]
pub(crate) struct Cursor {
    #[cfg(test)]
    pub(crate) replay_ticks: u64,
    frame: Option<u32>,
    replay_cost: ReplayCost,
    replay_group: Option<ReplayGroup>,
    simulation: Option<u32>,
    resident: std::collections::VecDeque<(u32, crate::residency::Cache<wgpu::Buffer>)>,
    pub(crate) requested: std::collections::BTreeMap<u32,usize>,
    placement_history: std::collections::BTreeMap<[u32;2], crate::residency::Cache<wgpu::Buffer>>,
    retention_stride: Option<u32>,
    pub(crate) placement_revision: u64,
    pub(crate) publications: std::collections::BTreeMap<String, u64>,
}
impl Cursor {
    fn publish_source(&mut self, name: &str) {
        let revision = self.publications.entry(name.into()).or_default();
        *revision = revision.checked_add(1).expect("geometry source revision exhausted");
    }
    fn publish_placement(&mut self) {
        self.placement_revision = self.placement_revision.checked_add(1).expect("placement revision exhausted");
    }
}
#[derive(Default)]
enum ReplayCost {
    #[default]
    Unknown,
    Pending {buffer:wgpu::Buffer,ready:std::sync::mpsc::Receiver<Result<(),wgpu::BufferAsyncError>>,period:f32},
    Measured(Option<f64>),
}
impl ReplayCost {
    fn value(&mut self) -> Option<f64> {
        if let Self::Pending {buffer,ready,period}=self {
            match ready.try_recv() {
                Ok(Ok(()))=>{
                    let data=buffer.get_mapped_range(..);let values:&[u64]=bytemuck::cast_slice(&data);
                    let cost=crate::timing::elapsed_ms(values[0],values[1],*period)
                        .map(|ms|ms.max(f64::from(*period)/1e6));
                    drop(data);buffer.unmap();*self=Self::Measured(cost);
                }
                Ok(Err(_))|Err(std::sync::mpsc::TryRecvError::Disconnected)=>*self=Self::Measured(None),
                Err(std::sync::mpsc::TryRecvError::Empty)=>{}
            }
        }
        match self {Self::Measured(cost)=>*cost,_=>None}
    }
}
/// Resolve timing for demanded work; mapping starts only after its owning
/// command buffer is submitted. Subsequent normal submissions deliver readiness.
struct TickProbe {queries:wgpu::QuerySet,resolved:wgpu::Buffer,readback:wgpu::Buffer}
impl TickProbe {
    fn new(device:&wgpu::Device)->Self {
        Self {
            queries:device.create_query_set(&wgpu::QuerySetDescriptor {label:Some("simulation tick cost"),ty:wgpu::QueryType::Timestamp,count:2}),
            resolved:crate::gpu_memory::buffer(device,&wgpu::BufferDescriptor {label:Some("simulation timing resolve"),size:wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,usage:wgpu::BufferUsages::QUERY_RESOLVE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false}),
            readback:crate::gpu_memory::buffer(device,&wgpu::BufferDescriptor {label:Some("simulation timing readback"),size:16,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false}),
        }
    }
    fn resolve(self,encoder:&mut wgpu::CommandEncoder,period:f32)->ReplayCost {
        encoder.resolve_query_set(&self.queries,0..2,&self.resolved,0);
        encoder.copy_buffer_to_buffer(&self.resolved,0,&self.readback,0,16);
        let (tx,rx)=std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&self.readback,wgpu::MapMode::Read,..,move |result|{let _=tx.send(result);});
        ReplayCost::Pending {buffer:self.readback,ready:rx,period}
    }
}

/// One paged upload supplies replay clocks and animated numeric inputs. Reusing
/// its source buffer avoids an encoder tracker entry for every tick/pass upload.
struct ReplayUpload {
    buffer: wgpu::Buffer,
    first: u32,
    end: u32,
    stride: u64,
    inputs: Vec<Option<(u64,u64)>>,
}
impl ReplayUpload {
    fn offset(&self, step: u32) -> u64 {u64::from(step-self.first)*self.stride}
}

/// Compare median CPU costs and periodically retry larger groups. Production
/// replay callbacks finish through Segments synchronously (simulation views stream),
/// so this includes native validation and queue publication, not just recording.
struct ReplayGroup {
    tick_ms: Option<f64>,
    gpu_ms: f64,
    ticks: u32,
    target: u32,
    unit_cpu_ms: Option<f64>,
    best_target: u32,
    warming_up: bool,
    observations: Vec<f64>,
    retry_after: u32,
    started: std::time::Instant,
}
impl ReplayGroup {
    fn new(tick_ms: Option<f64>) -> Self {Self {tick_ms,gpu_ms:0.,ticks:0,target:1,unit_cpu_ms:None,best_target:1,warming_up:true,observations:Vec::new(),retry_after:0,started:std::time::Instant::now()}}
    fn tick(&mut self) -> bool {
        self.ticks+=1;
        let Some(cost)=self.tick_ms.filter(|v|v.is_finite() && *v>0.) else {return true};
        self.gpu_ms+=cost;
        self.ticks>=self.target || self.gpu_ms+cost>constants::SAMPLE_GROUP_GPU_MS
            || self.started.elapsed().as_secs_f64()*1000.>=constants::SIMULATION_RECORD_MS
    }
    fn finished(&mut self) {
        self.measured(self.started.elapsed().as_secs_f64()*1000.);
    }
    fn measured(&mut self, elapsed_ms: f64) {
        let unit=elapsed_ms/f64::from(self.ticks);
        if self.warming_up {
            // The first cut also retires the view's upload/setup prologue.
            self.warming_up=false;
        } else if elapsed_ms>constants::SIMULATION_RECORD_MS {
            self.target=(f64::from(self.ticks)*constants::SIMULATION_RECORD_MS/elapsed_ms).floor().max(1.) as u32;
            self.observations.clear();
        } else {
            if self.ticks!=self.target {
                // GPU/host scheduling may cut a trial before its target.
                self.target=self.ticks;self.observations.clear();
            }
            self.observations.push(unit);
            if self.observations.len()==constants::SIMULATION_GROUP_OBSERVATIONS {
                self.observations.sort_by(f64::total_cmp);
                let unit=self.observations[self.observations.len()/2];
                self.observations.clear();
                if self.target==self.best_target || self.unit_cpu_ms.is_none_or(|best|unit<best) {
                    // Refresh the baseline rather than comparing forever against
                    // the luckiest observation from an earlier workload.
                    self.unit_cpu_ms=Some(unit);self.best_target=self.target;
                    if self.retry_after==0 {self.target=self.best_target.saturating_add(1);}
                    else {self.retry_after-=1;}
                } else {
                    self.target=self.best_target;
                    self.retry_after=constants::SIMULATION_GROUP_RETRY;
                }
            }
        }
        self.ticks=0;self.gpu_ms=0.;self.started=std::time::Instant::now();
    }

}

/// Convert a render request to a canonical tick, never rounding to output frames.
fn simulation_tick(sim:&v2::Simulation,time:impl Into<vork_time::Request>)->Result<u32> {
    let time=time.into().exact()?;
    let dt=sim.dt.exact();
    ensure!(dt>vork_time::exact::Rational::from_integer(0.into()),"simulation dt must be positive");
    let tick=time.sub(&vork_time::exact::Germ::point(sim.t0.exact())).scale(&dt.recip()).floor();
    if tick<0.into() {return Ok(0)}
    let tick:u32=tick.try_into().context("simulation time exceeds tick capacity")?;
    ensure!(tick<u32::MAX,"simulation time exceeds tick capacity");
    Ok(tick)
}

struct Instance {
    draw: usize,
    trs: [Option<AttributeBinding>; 3],
    capacity: u32,
    count: Option<AttributeBinding>,
    bounds: (Vec3, Vec3),
    indices: Vec<u32>,
}
pub(crate) struct Slot {
    pub(crate) parents: wgpu::Buffer,
    pub(crate) errors: wgpu::Buffer,
    pub(crate) error_readback: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    binds: Vec<wgpu::BindGroup>,
    groups: u32,
    width: u32,
    objects: Vec<std::ops::Range<u32>>,
}
impl Compute {
    #[cfg(test)]
    pub(crate) fn test_install(r:&Renderer,source:&v2::Scene,base:&std::path::Path,attributes:Option<&ProgramBuffers>,sources:&[Option<&v2::Node>],roots:&[install::MeshRoot],draws:&[install::Draw],rate:vork_time::Time,duration:impl Into<vork_time::Clock>)->Result<Option<Self>> {
        let prepared=prepare(r,source,base,attributes,sources)?;
        Self::install(r,source,base,attributes,&prepared,None,sources,roots,draws,rate,duration,None,None,&Default::default())
    }
    pub(crate) fn install(
        r: &Renderer,
        source: &v2::Scene,
        base: &std::path::Path,
        attributes: Option<&ProgramBuffers>,
        prepared: &Prepared,
        query_scene: Option<crate::compute_trace::Scene<'_>>,
        sources: &[Option<&v2::Node>],
        roots: &[install::MeshRoot],
        draws: &[install::Draw],
        rate: vork_time::Time,
        duration: impl Into<vork_time::Clock>,
        geometry_inputs: Option<crate::geometry_sampling::Inputs>,
        mut sample: Option<&mut InputSampler<'_>>,
        opened: &crate::footage::Opened,
    ) -> Result<Option<Self>> {
        if !source.programs.values().any(|p| p.is_compute()) && !draws.iter().any(|draw|draw.computed) {
            return Ok(None);
        }
        let fps=rate.as_f32();
        let attributes = attributes.context("compute requires named buffers")?;
        let mut pages = attributes.storage.clone();
        let global_programs = global_solver_consumers(source, sources);
        let mut geometry_scopes = Vec::new();
        let mut copy = r.device.create_command_encoder(&Default::default());
        for draw in draws.iter().filter(|draw| draw.computed) {
            let root = &roots[draw.object as usize];
            if let Some(scope) = geometry_scopes.iter_mut().find(|scope: &&mut GeometryScope|
                scope.node == root.node && scope.instance == root.instance_index) {
                scope.objects.insert(draw.object);
                continue;
            }
            let Some(v2::Node { kind: v2::NodeKind::Mesh { geometry, .. }, .. }) = sources[root.node] else { continue; };
            let geometry_name = geometry.clone();
            let geometry = &source.geometry[&geometry_name];
            let names: BTreeSet<String> = geometry.attributes.values().chain(geometry.widths.iter()).chain(geometry.count.iter()).chain(geometry.ids.iter()).chain(geometry.status.iter().map(|s| &s.buffer)).cloned().collect();
            let (programs, state) = geometry_solver_dependencies(source, names.clone());
            if !programs.is_empty() && programs.is_subset(&global_programs) {
                // Keep the geometry view, but alias its geometry-independent named
                // state and schedule the connected programs only once globally.
                geometry_scopes.push(GeometryScope {objects:BTreeSet::from([draw.object]),node:root.node,
                    instance:root.instance_index,geometry:geometry_name,programs:BTreeSet::new(),outputs:names,
                    buffers:ProgramBuffers {storage:attributes.storage.clone(),bindings:attributes.bindings.clone(),initial_identity:attributes.initial_identity.clone()},
                    pages:(0..attributes.storage.len()).collect()});
                continue;
            }
            let (buffers, scope_pages) = private_geometry_buffers(r, attributes, &state, &names, &mut pages, &mut copy)?;
            geometry_scopes.push(GeometryScope { objects: BTreeSet::from([draw.object]), node: root.node,
                instance: root.instance_index, geometry: geometry_name, programs, outputs:names, buffers, pages: scope_pages });
        }
        if !geometry_scopes.is_empty() { crate::command_finish::Segments::submit(&r.queue, copy); }
        let mut initial = Vec::new();
        let mut previous = Vec::new();
        let mut encoder = r.device.create_command_encoder(&Default::default());
        for (index, page) in pages.iter().enumerate() {
            let buffer = crate::gpu_memory::buffer(
                &r.device,
                &wgpu::BufferDescriptor {
                    label: Some("compute initial page"),
                    size: page.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            );
            encoder.copy_buffer_to_buffer(page, 0, &buffer, 0, page.size());
            initial.push(buffer);
            if index < attributes.storage.len() { previous.push(crate::gpu_memory::buffer(
                &r.device,
                &wgpu::BufferDescriptor {
                    label: Some("requested previous placement pose"),
                    size: page.size(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                        | if cfg!(test) { wgpu::BufferUsages::COPY_SRC } else { wgpu::BufferUsages::empty() },
                    mapped_at_creation: false,
                },
            )); }
        }
        let mut previous_state = PreviousState::default();
        let mut dispatches = Vec::new();
        let mut identity = blake3::Hasher::new();
        identity.update(&serde_json::to_vec(&source.simulation)?);
        let materials: std::collections::BTreeMap<_, _> = source.materials.iter()
            .filter(|(_, material)| source.programs.get(&material.program)
                .is_some_and(|p| p.schedule == Some(ComputeSchedule::Simulation))).collect();
        identity.update(&serde_json::to_vec(&materials)?);
        if let Some(sim) = &source.simulation {
            // Use decoded initial bytes, not reopened files or unrelated render inputs.
            for name in &sim.state {
                identity.update(name.as_bytes());
                identity.update(attributes.initial_identity[name].as_bytes());
            }
        }

        let tick = init(
            &r.device,
            "compute tick",
            &[0; 16],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut placement_sources = BTreeSet::new();
        // Reuse the installed draw sources; no second scene-tree walk or capacity registry.
        for node in sources.iter().flatten() {
            if let v2::NodeKind::Mesh { instances: Some(inst), .. } = &node.kind {
                for key in ["TRANSLATION", "ROTATION", "SCALE"] {
                    if let Some(name) = inst.attributes.get(key) { placement_sources.insert(name.clone()); }
                }
                placement_sources.extend(inst.count.iter().cloned());
            }
        }
        let filters:Vec<_>=prepared.values().flat_map(|p|p.artifact.queries.iter().copied()).collect();
        let trace=if filters.is_empty() {None} else {Some(crate::compute_trace::Queries::new(r,&filters,source,query_scene.context("compute trace requires an installed geometry snapshot")?,roots,draws)?)};
        if let Some(trace)=&trace {identity.update(trace.identity.as_bytes());}
        // Global named readers retain their original namespace. Geometry
        // consumers bind only their object's dependency slice.
        let mut programs = Vec::new();
        for (name, program) in source.programs.iter().filter(|(_, p)| p.is_compute()) {
            if global_programs.contains(name) || !geometry_scopes.iter().any(|scope| scope.programs.contains(name)) {
                programs.push((name, program, attributes, None));
            }
            programs.extend(geometry_scopes.iter().enumerate().filter(|(_, scope)| scope.programs.contains(name))
                .map(|(index, scope)| (name, program, &scope.buffers, Some(index))));
        }
        for (name, program, attributes, scope_index) in programs {
            let threads = program
                .dispatch
                .context("compute dispatch required")?
                .threads();
            let groups = [
                threads[0].div_ceil(constants::PROGRAM_WORKGROUP),
                threads[1],
                threads[2],
            ];
            let tiles = dispatch_tiles(
                groups,
                r.device.limits().max_compute_workgroups_per_dimension,
            );
            let scope = scope_index.map(|index| &geometry_scopes[index]);
            let prepared=prepared.get(&(name.clone(),scope.map(|scope|scope.geometry.clone())))
                .with_context(||format!("compute {name}: missing prepared geometry context"))?;
            let artifact=&prepared.artifact;
            let used=&prepared.used;
            let key=prepared.key;
            let implicit=&prepared.implicit;
            let input_binding=prepared.input_binding;
            if program.schedule == Some(ComputeSchedule::Simulation) {
                identity.update(key.as_bytes());
                // Bind retained simulation to the compiled code and reflected
                // defaults, never to dependency bytes reopened after compilation.
                identity.update(&serde_json::to_vec(artifact)?);
            }
            if !artifact.warnings.trim().is_empty() {
                eprintln!("{}", artifact.warnings.trim());
            }
            let mut types = vec![(0, None), (1, None)];
            types.extend(used.iter().enumerate().map(|(index, _)| (index as u32 + 2, Some(false))));
            let mut shader = artifact.wgsl.clone();
            let has_inputs = !artifact.inputs.fields.is_empty();
            if has_inputs {
                let binding = input_binding;
                types.push((binding, Some(true)));
                writeln!(
                    shader,
                    "@group(0) @binding({binding}) var<storage,read> vork_compute_inputs:array<u32>; fn vork_material_word(i:u32)->u32{{return vork_compute_inputs[i];}}"
                )?;
            }
            let active = used_compute_bindings(&shader)?;
            let mut implicit_buffers = Vec::new();
            for binding in implicit.iter().filter(|binding| active.contains(&binding.binding)) {
                let buffer = match binding.input {
                    GeometryInput::Predecessor(page) => scope.and_then(|scope| scope.objects.first())
                        .and_then(|object| geometry_inputs.as_ref().and_then(|inputs| inputs.page(*object,page)))
                        .unwrap_or(&initial[page as usize]).clone(),
                    GeometryInput::Reference => geometry_inputs.as_ref().and_then(|inputs|inputs.reference.as_ref()).context("geometry reference table was not installed")?.buffer.clone(),
                    GeometryInput::Previous(local_page) => {
                        let page = scope.map_or(local_page as usize, |scope| scope.pages[local_page as usize]);
                        let snapshot = previous_state.bind(r, &pages, page).clone();
                        let schedule = program.schedule.context("compute schedule required")?;
                        for layout in program.bindings.values().map(|name| &attributes.bindings[name]) {
                            for span in layout.spans.iter().filter(|span| span.page == local_page) {
                                let range = (schedule, page, u64::from(span.offset)*4, u64::from(span.count)*u64::from(layout.width)*4);
                                if !previous_state.ranges.contains(&range) { previous_state.ranges.push(range); }
                            }
                        }
                        snapshot
                    }
                };
                types.push((binding.binding, Some(true)));
                implicit_buffers.push((binding.binding, buffer));
            }
            let storage_limit=crate::pipelines::scalar_storage_limit(&r.device.limits(),r.adapter_info.backend==wgpu::Backend::Metal,2);
            ensure!(types.iter().filter(|(_, ty)| ty.is_some()).count() as u32 <= storage_limit,
                crate::gpu_memory::device_limit(format_args!("compute {name}: geometry/previous-state inputs and buffer pages"), format_args!("{} storage bindings",types.iter().filter(|(_,ty)|ty.is_some()).count()), format_args!("{} storage bindings",storage_limit), "reduce compute buffer pool sizes or the number of input buffers"));
            let query_program=!artifact.queries.is_empty();
            let group=if query_program {1} else {0};
            let pipeline=if query_program {trace.as_ref().unwrap().pipeline(r,prepared,&types)?}
                else {pipeline_layout_explicit(r,&shader,name,&types,"vork_entry")};
            let materials: Vec<_> = source
                .materials
                .iter()
                .filter(|(_, m)| m.program == *name)
                .collect();
            let instances: Vec<Option<(&String, &v2::Material)>> = if materials.is_empty() {
                vec![None]
            } else {
                materials.into_iter().map(Some).collect()
            };
            for material in instances {
                let values = material
                    .map(|(_, m)| {
                        m.inputs
                            .iter()
                            .map(|(key, value)| Ok((key.clone(), serde_json::to_value(value)?)))
                            .collect::<Result<std::collections::BTreeMap<_, _>>>()
                    })
                    .transpose()?
                    .unwrap_or_default();
                let mut words = artifact.inputs.pack(&values, |name,value| {
                    ensure!(artifact.inputs.fields.iter().any(|field|field.name==name && field.ty==vork_slang::Type::Geometry),"compute input {name}: use a numeric input or a Geometry handle");
                    let target=value.get("geometry").and_then(serde_json::Value::as_str).with_context(||format!("compute input {name}: Geometry requires {{geometry: mesh-node-name}}"))?;
                    crate::geometry_reference::handle(source,target)
                })?;
                let animation = material.map(|(name, _)| programs::InputAnimation::new_in(source, base, name, &artifact.inputs, &mut words, opened)).transpose()?.flatten();
                if program.schedule == Some(ComputeSchedule::Simulation) {
                    if let Some(animation) = &animation { identity.update(animation.identity().as_bytes()); }
                }
                let input = has_inputs.then(|| {
                    init(
                        &r.device,
                        "compute inputs",
                        bytemuck::cast_slice(&words),
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    )
                });
                let mut bindings = Vec::new();
                for &(groups, offset) in &tiles {
                    let offset = init(
                        &r.device,
                        "compute dispatch offset",
                        bytemuck::cast_slice(&offset),
                        wgpu::BufferUsages::UNIFORM,
                    );
                    let mut entries = vec![
                        entry(0, tick.as_entire_binding()),
                        entry(1, offset.as_entire_binding()),
                    ];
                    for (i, page) in used.iter().enumerate() {
                        entries.push(entry(
                            i as u32 + 2,
                            attributes.storage[*page as usize].as_entire_binding(),
                        ));
                    }
                    if has_inputs {
                        entries.push(entry(
                            input_binding,
                            input.as_ref().unwrap().as_entire_binding(),
                        ));
                    }
                    for (binding, buffer) in &implicit_buffers { entries.push(entry(*binding, buffer.as_entire_binding())); }
                    let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(name),
                        layout: &pipeline.get_bind_group_layout(group),
                        entries: &entries,
                    });
                    bindings.push((bind, groups));
                }
                dispatches.push(Dispatch {
                    queries:query_program,
                    name: name.clone(),
                    writes: program.bindings.values().cloned().collect(),
                    writes_placement: program.bindings.values().any(|name| placement_sources.contains(name)),
                    pipeline: pipeline.clone(),
                    tiles: bindings,
                    schedule: program.schedule.context("compute schedule required")?,
                    inputs: input.map(|input| (input, words, animation)),
                });
            }
        }
        let mut instances = Vec::new();
        for (i, draw) in draws.iter().enumerate() {
            let root = &roots[draw.object as usize];
            if let Some(node @ ::vork_scene::v2::Node {
                kind:
                    ::vork_scene::v2::NodeKind::Mesh {
                        instances: Some(inst),
                        ..
                    },
                ..
            }) = sources[root.node]
            {
                if places_instances(source, node) {
                    let binding = |key: &str| -> Result<Option<AttributeBinding>> {
                        inst.attributes.get(key).map(|name| attributes.bindings.get(name).cloned()
                            .context("compute instance column must be a retained binding")).transpose()
                    };
                    let trs = [binding("TRANSLATION")?,binding("ROTATION")?,binding("SCALE")?];
                    let capacity = inst.attributes.values().next().and_then(|name| attributes.bindings.get(name))
                        .context("instance columns require a retained binding")?.count;
                    let count = inst
                        .count
                        .as_ref()
                        .map(|b| {
                            attributes
                                .bindings
                                .get(b)
                                .cloned()
                                .context("instance count must be a compute binding")
                        })
                        .transpose()?;
                    ensure!(
                        count.as_ref().is_none_or(|b| b.count == 1 && b.width == 1),
                        "instance count requires exactly one u32"
                    );
                    let bounds = draw.instances.clone().fold(
                        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                        |(lo, hi), object| {
                            let (a, b) = crate::shadow_envelope::transformed(
                                roots[object as usize].local,
                                draw.bounds,
                            );
                            (lo.min(a), hi.max(b))
                        },
                    );
                    instances.push(Instance {
                        draw: i,
                        trs,
                        capacity,
                        count,
                        bounds,
                        indices: draw
                            .instances
                            .clone()
                            .map(|object| {
                                roots[object as usize]
                                    .instance_index
                                    .expect("compute instance index")
                            })
                            .collect(),
                    });
                }
            }
        }
        // Only TRS and membership participate in the object bridge. In
        // particular a cloth/particle solver's unrelated dense scratch must
        // not be copied for every shutter sample's historical placement.
        let mut placement_ranges: Vec<_> = instances.iter()
            .flat_map(|instance| instance.trs.iter().flatten().chain(instance.count.iter()))
            .flat_map(|binding| binding.spans.iter().map(move |span|
                (span.page as usize, u64::from(span.offset)*4,
                 u64::from(span.count)*u64::from(binding.width)*4)))
            .collect();
        placement_ranges.sort_unstable();
        placement_ranges.dedup();
        let count=if dispatches.iter().any(|d|d.schedule==ComputeSchedule::Frame) {
            (duration.into().exact()*vork_time::exact::decimal(&rate.to_string())?).ceil().to_integer()
        }else{1.into()};
        let frames:u32=count.try_into().context("compute frame grid exceeds u32 capacity")?;
        ensure!(frames>0,"compute frame grid must be nonempty");
        // Geometry adds immutable retained columns to these pages. Only producer-
        // owned ranges reset and publish on seeks; sharing a page is not a write.
        let frame_sources: Vec<String> = dispatches.iter()
            .filter(|d| d.schedule != ComputeSchedule::Simulation)
            .flat_map(|d| d.writes.iter().cloned()).collect::<BTreeSet<_>>().into_iter().collect();
        let mut simulation_ranges = Vec::new();
        let mut frame_ranges = Vec::new();
        for (attributes, page_map) in std::iter::once((attributes, None))
            .chain(geometry_scopes.iter().map(|scope| (&scope.buffers, Some(&scope.pages)))) {
            for (name, binding) in &attributes.bindings {
                let ranges = if source.simulation.as_ref().is_some_and(|s| s.state.contains(name)) {
                    &mut simulation_ranges
                } else if frame_sources.contains(name) { &mut frame_ranges } else { continue };
                for span in &binding.spans {
                    let page = page_map.map_or(span.page as usize, |pages| pages[span.page as usize]);
                    ranges.push((page, u64::from(span.offset) * 4, u64::from(span.count) * u64::from(binding.width) * 4));
                }
            }
        }
        simulation_ranges.sort_unstable();
        simulation_ranges.dedup();
        frame_ranges.sort_unstable();
        frame_ranges.dedup();
        let result = Self {
            geometry_inputs,
            trace,
            dispatches,
            geometry_scopes,
            previous_state,
            named_pages: attributes.storage.len(),
            pages,
            initial,
            previous,
            placement_ranges,
            device: r.device.clone(),
            timestamp_period: r.queue.get_timestamp_period(),
            simulation: source.simulation.clone(),
            simulation_ranges,
            frame_ranges,
            frame_sources,
            checkpoint_store: r.disk_cache.clone(),
            gpu_only_checkpoints: geometry_checkpoints(source),
            identity: identity.finalize(),
            tick,
            fps,
            rate,
            frames,
            cursor: Mutex::new(Cursor::default()),
            instances,
            material_attributes: Vec::new(),
            material_attribute_sources: Vec::new(),
            bridge: None,
        };
        // Install dispatches run now; seeks replay them from the same initial bytes.
        let mut cursor = result.cursor.lock().expect("compute cursor poisoned");
        if result.dispatches.iter().any(|dispatch| dispatch.schedule == ComputeSchedule::Install) {
            if let Some(inputs) = &result.geometry_inputs {
                inputs.sample(&mut encoder, &vork_time::Request::Frame(0, result.rate))?;
            }
            if let Some(sample) = sample.as_mut() {
                sample(&mut encoder, ComputeSchedule::Install, &vork_time::Request::Frame(0, result.rate))?;
            }
        }
        result.run(&mut encoder, &mut cursor, ComputeSchedule::Install, 0);
        if let Some(sim) = &result.simulation {
            if let Some(inputs) = &result.geometry_inputs {
                inputs.sample(&mut encoder, &vork_time::Request::Tick(0, sim.t0.clone(), sim.dt.clone()))?;
            }
            if let Some(sample) = sample.as_mut() {
                sample(&mut encoder, ComputeSchedule::Install, &vork_time::Request::Tick(0, sim.t0.clone(), sim.dt.clone()))?;
            }
            result.set_tick(&mut encoder, 0, sim.t0.as_f64(), 0., sim.seed);
            result.run_named(&mut encoder, &mut cursor, &sim.init, sim.t0.as_f64() as f32);
        }
        crate::command_finish::Segments::submit(&r.queue, encoder);
        drop(cursor);
        Ok(Some(result))
    }
    pub(crate) fn geometry_buffers(&self, object: u32) -> Option<&ProgramBuffers> {
        self.geometry_scopes.iter().find(|scope| scope.objects.contains(&object)).map(|scope| &scope.buffers)
    }
    pub(crate) fn bounds(&self, draw: usize) -> Option<(Vec3, Vec3)> {
        self.instances
            .iter()
            .find(|i| i.draw == draw)
            .map(|i| i.bounds)
    }
    pub(crate) fn varying(&self, sources: &BTreeSet<String>) -> bool {
        self.dispatches.iter().any(|d|d.schedule != ComputeSchedule::Install && d.writes.iter().any(|n|sources.contains(n)))
    }
    pub(crate) fn geometry_uses_simulation_clock(&self, sources: &[String]) -> bool {
        self.simulation.is_some() && !self.dispatches.iter().any(|d|d.schedule==ComputeSchedule::Frame && d.writes.iter().any(|n|sources.contains(n)))
    }
    pub(crate) fn geometry_time_key(&self,sources:&[String],time:impl Into<vork_time::Request>)->Result<[u32;2]> {
        let time=time.into();
        let simulation=self.simulation.as_ref().filter(|_|self.dispatches.iter().any(|d|d.schedule==ComputeSchedule::Simulation && d.writes.iter().any(|n|sources.contains(n))))
            .map(|sim|simulation_tick(sim,&time)).transpose()?.unwrap_or(0);
        let frame=if self.dispatches.iter().any(|d|d.schedule==ComputeSchedule::Frame && d.writes.iter().any(|n|sources.contains(n))) {
            nominal_frame(&time,self.rate,self.frames)?
        }else{0};
        Ok([simulation,frame])
    }
    pub(crate) fn request_key(&self,time:&vork_time::Request)->Result<[u32;2]> {
        let tick=self.simulation.as_ref().map(|sim|simulation_tick(sim,time)).transpose()?.unwrap_or(0);
        let frame=if self.dispatches.iter().any(|d|d.schedule==ComputeSchedule::Frame) {
            nominal_frame(&time,self.rate,self.frames)?
        }else{0};
        Ok([tick,frame])
    }
    pub(crate) fn validate_request(&self,time:&vork_time::Request)->Result<()> {
        time.numeric()?;
        if let Some(sim)=&self.simulation {
            let tick=simulation_tick(sim,time)?;
            vork_time::Request::Tick(tick,sim.t0.clone(),sim.dt.clone()).numeric()?;
        }
        Ok(())
    }
    pub(crate) fn bind_material_attributes(&mut self, buffers: &ProgramBuffers, rows: Vec<(String, u32, u32)>) -> Result<()> {
        for (source, index, destination) in rows {
            let mut owners=self.geometry_scopes.iter().filter(|scope|scope.outputs.contains(&source));
            let owner=owners.next().filter(|_|owners.next().is_none());
            let binding = &owner.map_or(buffers,|scope| &scope.buffers).bindings[&source];
            ensure!(index < binding.count, "material attribute row exceeds source capacity");
            let span = binding.spans.iter().find(|span| index >= span.first && index < span.first + span.count)
                .context("material attribute row has no installed range")?;
            let page=owner.map_or(span.page as usize,|scope|scope.pages[span.page as usize]);
            self.material_attributes.push((page,
                u64::from(span.offset + (index - span.first) * binding.width) * 4,
                u64::from(destination) * 4, u64::from(binding.width) * 4));
            self.material_attribute_sources.push(source);
        }
        self.material_attribute_sources.sort();
        self.material_attribute_sources.dedup();
        Ok(())
    }
    pub(crate) fn has_material_attributes(&self) -> bool { !self.material_attributes.is_empty() }
    pub(crate) fn publish_material_attributes(&self, encoder: &mut wgpu::CommandEncoder,
        cursor: &Cursor, changes: &mut values::ChangeTracker, records: &wgpu::Buffer) {
        let versions: Vec<u64> = self.material_attribute_sources.iter()
            .map(|source| cursor.publications.get(source).copied().unwrap_or(0)).collect();
        if changes.changed(values::ChangeKey::MaterialAttributes, bytemuck::cast_slice(&versions)) {
            for &(page, source, destination, size) in &self.material_attributes {
                encoder.copy_buffer_to_buffer(&self.pages[page], source, records, destination, size);
            }
        }
    }
    pub(crate) fn publish_queries(&self,r:&Renderer,encoder:&mut wgpu::CommandEncoder,destination:&wgpu::Buffer) {
        if let Some(trace)=&self.trace {trace.publish(r,encoder,destination);}
        if let Some(inputs)=&self.geometry_inputs {inputs.publish_errors(r,encoder,destination);}
    }
    pub(crate) fn has_simulation(&self) -> bool { self.simulation.is_some() }
    pub(crate) fn has_instances(&self) -> bool {
        !self.instances.is_empty()
    }
    pub(crate) fn computed_draw(&self, draw: usize) -> bool {
        self.instances.iter().any(|i| i.draw == draw)
    }
    fn run(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, schedule: ComputeSchedule, frame: u32) {
        self.previous_state.capture(encoder, &self.pages, schedule);
        for dispatch in self.dispatches.iter().filter(|d| d.schedule == schedule) {
            dispatch.publish(cursor);
            self.inputs(encoder, dispatch, frame as f32 / self.fps);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&dispatch.pipeline);
            if dispatch.queries {self.trace.as_ref().unwrap().bind(&mut pass);}
            for (bind, groups) in &dispatch.tiles {
                pass.set_bind_group(if dispatch.queries {1} else {0}, bind, &[]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
        }
    }
    fn inputs(&self, encoder: &mut wgpu::CommandEncoder, dispatch: &Dispatch, time: f32) {
        if let Some((input, words, Some(animation))) = &dispatch.inputs {
            let mut words = words.clone();
            animation.words(time, &mut words);
            let upload = init(&self.device, "compute sampled inputs", bytemuck::cast_slice(&words), wgpu::BufferUsages::COPY_SRC);
            encoder.copy_buffer_to_buffer(&upload, 0, input, 0, upload.size());
        }
    }
    fn set_tick(&self, encoder: &mut wgpu::CommandEncoder, index: u32, time: f64, dt: f64, seed: u32) {
        let bytes = [index, (time as f32).to_bits(), (dt as f32).to_bits(), seed];
        let upload = init(&self.device, "simulation clock", bytemuck::cast_slice(&bytes), wgpu::BufferUsages::COPY_SRC);
        encoder.copy_buffer_to_buffer(&upload, 0, &self.tick, 0, 16);
    }
    fn run_named(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, names: &[String], time: f32) {
        self.run_named_uploaded(encoder,cursor,names,time,None,None);
    }
    fn run_named_uploaded(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, names: &[String], time: f32, upload: Option<(&ReplayUpload,u32)>, queries: Option<&wgpu::QuerySet>) {
        self.previous_state.capture(encoder, &self.pages, ComputeSchedule::Simulation);
        let mut remaining=names.iter().map(|name|self.dispatches.iter().filter(|d|&d.name==name).count()).sum::<usize>();
        let mut first=true;
        for name in names {
            for (index,dispatch) in self.dispatches.iter().enumerate().filter(|(_,d)| &d.name == name) {
                dispatch.publish(cursor);
                if let Some((page,step))=upload.filter(|(p,_)|p.inputs[index].is_some()) {
                    let (offset,bytes)=page.inputs[index].unwrap();
                    let input=&dispatch.inputs.as_ref().unwrap().0;
                    encoder.copy_buffer_to_buffer(&page.buffer,page.offset(step)+offset,input,0,bytes);
                } else {self.inputs(encoder, dispatch, time);}
                // Separate compute passes supply the storage-write -> read/write barrier.
                remaining-=1;
                let timestamp_writes=queries.map(|query_set|wgpu::ComputePassTimestampWrites {
                    query_set,beginning_of_pass_write_index:first.then_some(0),end_of_pass_write_index:(remaining==0).then_some(1),
                }).filter(|_|first || remaining==0);
                first=false;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:None,timestamp_writes});
                pass.set_pipeline(&dispatch.pipeline);
            if dispatch.queries {self.trace.as_ref().unwrap().bind(&mut pass);}
                for (bind, groups) in &dispatch.tiles {
                    pass.set_bind_group(if dispatch.queries {1} else {0}, bind, &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
            }
        }
    }
    fn replay_upload(&self, sim: &v2::Simulation, first: u32, target: u32, page_bytes: u64) -> ReplayUpload {
        let limit=page_bytes.min(self.device.limits().max_buffer_size).max(16);
        let mut words_per_tick=4usize;
        let mut inputs=vec![None;self.dispatches.len()];
        for (index,dispatch) in self.dispatches.iter().enumerate().filter(|(_,d)|sim.steps.contains(&d.name)) {
            if let Some((_,words,Some(_)))=&dispatch.inputs {
                inputs[index]=Some((words_per_tick as u64*4,words.len() as u64*4));
                words_per_tick+=words.len();
            }
        }
        // A single tick may contain more numeric inputs than one upload page.
        // Keep that valid scene on its ordinary per-dispatch input upload path.
        if words_per_tick as u64*4>limit {words_per_tick=4;inputs.fill(None);}
        let stride=words_per_tick as u64*4;
        let count=u64::from(target-first).min(limit/stride) as u32;
        let end=first+count;
        let mut data=Vec::with_capacity(count as usize*words_per_tick);
        for step in first..end {
            let time=sim.t0.as_f64()+f64::from(step)*sim.dt.as_f64();
            data.extend_from_slice(&[step,(time as f32).to_bits(),(sim.dt.as_f64() as f32).to_bits(),sim.seed]);
            for (index,dispatch) in self.dispatches.iter().enumerate() {
                if inputs[index].is_none() {continue;}
                let (_,words,Some(animation))=dispatch.inputs.as_ref().unwrap() else {unreachable!()};
                let start=data.len();data.extend_from_slice(words);
                animation.words(time as f32,&mut data[start..]);
            }
        }
        ReplayUpload {buffer:init(&self.device,"simulation replay clocks and inputs",bytemuck::cast_slice(&data),wgpu::BufferUsages::COPY_SRC),first,end,stride,inputs}
    }
    fn checkpoint_path(&self, tick: u32) -> PathBuf {
        let mut key = blake3::Hasher::new();
        key.update(self.identity.as_bytes());
        key.update(&tick.to_le_bytes());
        self.checkpoint_store.path("simulation", 1, key.finalize())
    }
    fn checkpoint_size(&self) -> u64 {
        self.simulation_ranges.iter().map(|r| r.2).sum()
    }
    fn restore_checkpoint(&self, encoder: &mut wgpu::CommandEncoder, target: u32, minimum: u32) -> Option<u32> {
        if self.gpu_only_checkpoints { return None; }
        let size = self.checkpoint_size();
        if size == 0 { return None; }
        for tick in (1..=target / CHECKPOINT_INTERVAL).rev().map(|n| n * CHECKPOINT_INTERVAL) {
            if tick <= minimum {break;}
            let path=self.checkpoint_path(tick);
            if std::fs::metadata(&path).ok().is_none_or(|m|m.len()!=size+blake3::OUT_LEN as u64) {continue;}
            let Some(bytes) = disk_cache::read_mapped(&path) else {continue;};
            if bytes.len() as u64 != size {continue;}
            // Allocate every upload before mutating state: optional allocation
            // failure must leave canonical replay's starting state untouched.
            let mut src=0;
            let uploads: Option<Vec<_>>=self.simulation_ranges.iter().map(|&(_,_,count)| {
                let upload=crate::gpu_memory::optional_buffer_init(&self.device,"simulation checkpoint restore",
                    &bytes[src as usize..(src+count) as usize],wgpu::BufferUsages::COPY_SRC);
                src+=count;upload
            }).collect();
            let uploads=uploads?;
            for (&(page,offset,count),upload) in self.simulation_ranges.iter().zip(&uploads) {
                encoder.copy_buffer_to_buffer(upload,0,&self.pages[page],offset,count);
            }
            return Some(tick);
        }
        None
    }
    // Resident and disk snapshots represent the same canonical declared state.
    // Cursor ownership and queue order make restoration safe across frame slots.
    pub(crate) fn plan_requests(&self, cursor: &mut Cursor, times: impl Iterator<Item = vork_time::Request>) -> Result<()> {
        cursor.requested.clear();
        cursor.retention_stride=None;
        if let Some(sim) = &self.simulation {
            for time in times { *cursor.requested.entry(simulation_tick(sim,time)?).or_default() += 1; }
        }
        Ok(())
    }
    pub(crate) fn consume_request(&self, cursor: &mut Cursor, time: impl Into<vork_time::Request>) -> Result<()> {
        if let Some(sim) = &self.simulation {
            let tick=simulation_tick(sim,time)?;
            if let Some(count)=cursor.requested.get_mut(&tick) {
                *count-=1;
                if *count==0 { cursor.requested.remove(&tick); }
            }
        }
        Ok(())
    }
    /// Historical placement is a sampled output, not another solver advance.
    pub(crate) fn plan_placement_requests(&self, cursor: &mut Cursor, times: &[vork_time::Request], cut: bool) -> Result<()> {
        let previous=times.iter().map(|time|if cut {Ok(time.clone())}else{time.previous(self.rate)})
            .collect::<Result<Vec<_>>>()?;
        let wanted=times.iter().chain(&previous).map(|time|self.request_key(time)).collect::<Result<BTreeSet<_>>>()?;
        cursor.placement_history.retain(|key,cache|wanted.contains(key) && cache.is_resident());
        self.plan_requests(cursor,times.iter().chain(&previous).cloned())
    }
    fn placement_history_buffer(&self, cursor: &Cursor) -> Option<wgpu::Buffer> {
        let size=self.placement_ranges.iter().map(|range|range.2).sum::<u64>();
        if size==0 || size>self.device.limits().max_buffer_size {return None;}
        let headroom=cursor.placement_history.values().find_map(|cache|cache.get()
            .and_then(|buffer|crate::residency::buffer_available(&self.device,&buffer)));
        if headroom.is_some_and(|free|free<size) {return None;}
        crate::gpu_memory::retention_buffer(&self.device,&wgpu::BufferDescriptor {
            label:Some("historical placement columns"),size,
            usage:wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
        })
    }
    fn save_placement_history(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, key: [u32;2], buffer: wgpu::Buffer) {
        let mut destination=0;
        for &(page,offset,size) in &self.placement_ranges {
            encoder.copy_buffer_to_buffer(&self.pages[page],offset,&buffer,destination,size);
            destination+=size;
        }
        let cache=crate::residency::Cache::new(&crate::gpu_memory::ledger(&self.device));
        cache.insert(buffer,destination);
        cursor.placement_history.insert(key,cache);
    }
    pub(crate) fn retain_placement_history(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, time: &vork_time::Request) -> Result<()> {
        let key=self.request_key(time)?;
        if cursor.placement_history.get(&key).is_some_and(|cache|cache.is_resident()) {return Ok(());}
        if let Some(buffer)=self.placement_history_buffer(cursor) {self.save_placement_history(encoder,cursor,key,buffer);}
        Ok(())
    }
    pub(crate) fn restore_placement_history(&self, encoder: &mut wgpu::CommandEncoder, cursor: &Cursor, time: &vork_time::Request) -> Result<bool> {
        let Some(buffer)=cursor.placement_history.get(&self.request_key(time)?).and_then(|cache|cache.get()) else {return Ok(false);};
        let mut source=0;
        for &(page,offset,size) in &self.placement_ranges {
            encoder.copy_buffer_to_buffer(&buffer,source,&self.previous[page],offset,size);
            source+=size;
        }
        Ok(true)
    }
    /// Gather missing historical columns in clock order before current shutter
    /// evaluation. Cache hits never move the canonical simulation/frame cursor.
    pub(crate) fn prepare_placement_history(&self, r: &Renderer, cursor: &mut Cursor, times: &[vork_time::Request], cut: bool, checkpoint: &mut Option<Checkpoint>) -> Result<()> {
        if cut {return Ok(());}
        let mut times=times.iter().map(|time| {
            let previous=time.previous(self.rate)?;
            Ok((previous.exact()?,previous))
        }).collect::<Result<Vec<_>>>()?;
        times.sort_by(|a,b|a.0.compare(&b.0));
        let mut encoder=r.device.create_command_encoder(&Default::default());
        for (_,time) in times {
            let key=self.request_key(&time)?;
            if cursor.placement_history.get(&key).is_some_and(|cache|cache.is_resident()) {continue;}
            // A residency miss falls back at the consumer, without first doing
            // an unretainable historical replay in this preparation pass.
            let Some(buffer)=self.placement_history_buffer(cursor) else {continue;};
            let saved=self.advance_with(&mut encoder,cursor,&time,|encoder| {
                let next=r.device.create_command_encoder(&Default::default());
                crate::command_finish::Segments::submit(&r.queue,std::mem::replace(encoder,next));
            })?;
            if checkpoint.is_none() {*checkpoint=saved;}
            self.save_placement_history(&mut encoder,cursor,key,buffer);
        }
        crate::command_finish::Segments::submit(&r.queue,encoder);
        Ok(())
    }
    fn restore_resident(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, target: u32) -> Option<u32> {
        let minimum = cursor.simulation.filter(|&tick|tick<=target);
        let (index, _) = cursor.resident.iter().enumerate()
            .filter(|(_, (tick, cache))| minimum.is_none_or(|minimum|*tick>minimum) && *tick <= target && cache.is_resident())
            .max_by_key(|(_, (tick, _))| *tick)?;
        let entry=cursor.resident.remove(index)?;
        let buffer=entry.1.get()?;
        let mut src=0;
        for &(page,offset,bytes) in &self.simulation_ranges {
            encoder.copy_buffer_to_buffer(&buffer,src,&self.pages[page],offset,bytes);src+=bytes;
        }
        let tick=entry.0;
        cursor.resident.push_back(entry);
        Some(tick)
    }
    fn retain_resident(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, target: u32) {
        if !cursor.requested.contains_key(&target) { return; }
        cursor.resident.retain(|(_,cache)|cache.is_resident());
        if cursor.resident.iter().any(|(tick,_)|*tick==target) {return;}
        let size=self.checkpoint_size();
        if size==0 || size>self.device.limits().max_buffer_size {return;}
        let headroom=cursor.resident.iter().find_map(|(_,cache)|cache.get()
            .and_then(|buffer|crate::residency::buffer_available(&self.device,&buffer)));
        if cursor.retention_stride.is_none() {
            if let Some(free)=headroom {
                let capacity=(free/size).saturating_add(cursor.resident.len() as u64).max(1);
                let first=*cursor.requested.first_key_value().unwrap().0;
                let last=*cursor.requested.last_key_value().unwrap().0;
                cursor.retention_stride=Some(if cursor.requested.len() as u64<=capacity {1} else {
                    (u64::from(last-first)+1).div_ceil(capacity).clamp(1,u64::from(u32::MAX)) as u32
                });
            }
        }
        // Preserve prefixes across the exposure when every tick cannot fit.
        // The first retained request supplies the heap identity and an anchor.
        if target%cursor.retention_stride.unwrap_or(1)!=0
            && cursor.requested.first_key_value().is_none_or(|(&first,_)|target!=first) {return;}
        // Copies and consumers use the same ordered queue. A retired request's
        // buffer can be overwritten after its previous restore, without a wait.
        let reusable=cursor.resident.iter().position(|(tick,_)|!cursor.requested.contains_key(tick))
            .and_then(|index|cursor.resident.remove(index))
            .and_then(|(_,cache)|cache.take()).and_then(|buffer|std::sync::Arc::try_unwrap(buffer).ok());
        let buffer = reusable.or_else(|| {
            // A successful virtual allocation does not guarantee that queued
            // copies can commit it. Check the identified native heap first.
            if headroom.is_some_and(|free|free<size) {return None;}
            crate::gpu_memory::retention_buffer(&self.device,&wgpu::BufferDescriptor {
                label:Some("resident simulation checkpoint"),size,
                usage:wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
            })
        });
        let Some(buffer)=buffer else {return;};
        let cache=crate::residency::Cache::new(&crate::gpu_memory::ledger(&self.device));
        let mut dst=0;
        for &(page,offset,bytes) in &self.simulation_ranges {
            encoder.copy_buffer_to_buffer(&self.pages[page],offset,&buffer,dst,bytes);dst+=bytes;
        }
        cache.insert(buffer,size);
        cursor.resident.push_back((target,cache));
    }
    fn checkpoint(&self, encoder: &mut wgpu::CommandEncoder, tick: u32) -> Option<Checkpoint> {
        if self.gpu_only_checkpoints { return None; }
        let size = self.checkpoint_size();
        if size == 0 {return None;}
        // Match retained storage ranges: page size is transfer granularity, not
        // a checkpoint or scene-size ceiling. Avoid a monolithic pinned allocation.
        // No copies are recorded until every page has been admitted. Unknown
        // or unrelated heap telemetry cannot suppress a viable disk checkpoint.
        let buffers=crate::gpu_memory::optional_pages(self.simulation_ranges.len(), |index| {
            let count=self.simulation_ranges[index].2;
            crate::gpu_memory::retention_buffer(&self.device,&wgpu::BufferDescriptor {
                label:Some("simulation checkpoint staging"),size:count,
                usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,
            })
        }, |index,buffer| {
            let (page,offset,count)=self.simulation_ranges[index];
            encoder.copy_buffer_to_buffer(&self.pages[page],offset,buffer,0,count);
        })?;
        Some(Checkpoint {buffers,path:self.checkpoint_path(tick)})
    }
    #[cfg(test)]
    fn advance(&self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor, time: impl Into<vork_time::Request>) -> Result<Option<Checkpoint>> {
        self.advance_with(encoder, cursor, time, |_| {})
    }
    pub(crate) fn advance_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cursor: &mut Cursor,
        time: impl Into<vork_time::Request>,
        boundary: impl FnMut(&mut wgpu::CommandEncoder),
    ) -> Result<Option<Checkpoint>> {
        self.advance_paged(encoder,cursor,time,constants::SCENE_BUFFER_PAGE_BYTES as u64,boundary)
    }
    /// Use the existing cursor while refreshing external geometry at canonical
    /// dispatch times. Installation receives the same sampler separately.
    pub(crate) fn advance_observing(
        &self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor,
        time: impl Into<vork_time::Request>,
        boundary: impl FnMut(&mut wgpu::CommandEncoder),
        sample: Option<&mut InputSampler<'_>>,
        observe: impl FnMut(&mut wgpu::CommandEncoder,u32),
    ) -> Result<Option<Checkpoint>> {
        self.advance_paged_sampling(encoder,cursor,time,constants::SCENE_BUFFER_PAGE_BYTES as u64,boundary,sample,observe)
    }
    fn advance_paged(
        &self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor,
        time: impl Into<vork_time::Request>, upload_page_bytes: u64,
        boundary: impl FnMut(&mut wgpu::CommandEncoder),
    ) -> Result<Option<Checkpoint>> {
        self.advance_paged_sampling(encoder,cursor,time,upload_page_bytes,boundary,None,|_,_| {})
    }
    /// Refresh read-only geometry inputs before a canonical dispatch group.
    /// Sampling happens at the tick's left endpoint, including replay; observing
    /// happens after the group. Neither callback owns or advances a clock.
    fn advance_paged_sampling(
        &self, encoder: &mut wgpu::CommandEncoder, cursor: &mut Cursor,
        time: impl Into<vork_time::Request>, upload_page_bytes: u64,
        mut boundary: impl FnMut(&mut wgpu::CommandEncoder),
        mut sample: Option<&mut InputSampler<'_>>,
        mut observe: impl FnMut(&mut wgpu::CommandEncoder,u32),
    ) -> Result<Option<Checkpoint>> {
        let time=time.into();
        let mut checkpoint = None;
        if let Some(sim) = &self.simulation {
            let target = simulation_tick(sim, &time)?;
            if let Some(tick) = self.restore_resident(encoder,cursor,target) {
                for name in &sim.state {cursor.publish_source(name);}
                if self.dispatches.iter().any(|d|d.schedule==ComputeSchedule::Simulation && d.writes_placement) {cursor.publish_placement();}
                cursor.simulation=Some(tick);
            }
            if cursor.simulation.is_some_and(|last| target < last) {
                if let Some(inputs) = &self.geometry_inputs {
                    inputs.sample(encoder, &vork_time::Request::Tick(0, sim.t0.clone(), sim.dt.clone()))?;
                }
                if let Some(sample) = sample.as_mut() {
                    sample(encoder, ComputeSchedule::Install, &vork_time::Request::Tick(0, sim.t0.clone(), sim.dt.clone()))?;
                }
                for &(page, offset, bytes) in &self.simulation_ranges {
                    encoder.copy_buffer_to_buffer(&self.initial[page], offset, &self.pages[page], offset, bytes);
                }
                for name in &sim.state { cursor.publish_source(name); }
                if self.dispatches.iter().any(|d| d.schedule == ComputeSchedule::Simulation && d.writes_placement) { cursor.publish_placement(); }
                self.set_tick(encoder, 0, sim.t0.as_f64(), 0., sim.seed);
                self.run_named(encoder, cursor, &sim.init, sim.t0.as_f64() as f32);
                cursor.simulation = None;
            }
            if let Some(tick) = self.restore_checkpoint(encoder, target, cursor.simulation.unwrap_or(0)) {
                for name in &sim.state { cursor.publish_source(name); }
                if self.dispatches.iter().any(|d| d.schedule == ComputeSchedule::Simulation && d.writes_placement) { cursor.publish_placement(); }
                cursor.simulation = Some(tick);
            }
            self.retain_resident(encoder,cursor,cursor.simulation.unwrap_or(0));
            let mut group=cursor.replay_group.take().unwrap_or_else(||ReplayGroup::new(cursor.replay_cost.value()));
            group.tick_ms=cursor.replay_cost.value();
            group.ticks=0;group.gpu_ms=0.;group.started=std::time::Instant::now();group.warming_up=true;
            let mut upload: Option<ReplayUpload>=None;
            for step in cursor.simulation.unwrap_or(0)..target {
                #[cfg(test)] { cursor.replay_ticks += 1; }
                if group.tick_ms.is_none() {
                    if let Some(cost)=cursor.replay_cost.value() {group=ReplayGroup::new(Some(cost));}
                }
                let probe=if matches!(cursor.replay_cost,ReplayCost::Unknown) {
                    cursor.replay_cost=ReplayCost::Measured(None);
                    self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(||TickProbe::new(&self.device))
                } else {None};
                if upload.as_ref().is_none_or(|page|step>=page.end) {
                    // Materialize only the next measured group; the upload
                    // policy below bounds bytes independently of the clock.
                    let count=group.target;
                    upload=Some(self.replay_upload(sim,step,target.min(step.saturating_add(count)),upload_page_bytes));
                }
                let page=upload.as_ref().unwrap();
                if let Some(inputs) = &self.geometry_inputs {
                    inputs.sample(encoder, &vork_time::Request::Tick(step, sim.t0.clone(), sim.dt.clone()))?;
                }
                if let Some(sample) = sample.as_mut() {
                    sample(encoder, ComputeSchedule::Simulation, &vork_time::Request::Tick(step, sim.t0.clone(), sim.dt.clone()))?;
                }
                encoder.copy_buffer_to_buffer(&page.buffer,page.offset(step),&self.tick,0,16);
                self.run_named_uploaded(encoder, cursor, &sim.steps, (sim.t0.as_f64() + f64::from(step) * sim.dt.as_f64()) as f32,Some((page,step)),probe.as_ref().map(|p|&p.queries));
                cursor.simulation = Some(step + 1);
                if let Some(probe)=probe {cursor.replay_cost=probe.resolve(encoder,self.timestamp_period);}
                observe(encoder,step+1);
                self.retain_resident(encoder,cursor,step+1);
                if step + 1 == target / CHECKPOINT_INTERVAL * CHECKPOINT_INTERVAL && step + 1 > 0 {
                    checkpoint = self.checkpoint(encoder, step + 1);
                }
                if group.tick() {boundary(encoder);group.finished();}
            }
            cursor.replay_group=Some(group);
            cursor.simulation = Some(target);
            self.retain_resident(encoder,cursor,target);
            if cursor.requested.is_empty() { cursor.resident.clear(); }
        }
        if !self.dispatches.iter().any(|d| d.schedule == ComputeSchedule::Frame) {
            return Ok(checkpoint);
        }
        let frame=nominal_frame(&time,self.rate,self.frames)?;

        if cursor.frame.is_some_and(|last| frame < last) {
            if let Some(inputs) = &self.geometry_inputs {
                inputs.sample(encoder, &vork_time::Request::Frame(0, self.rate))?;
            }
            if let Some(sample) = sample.as_mut() {
                sample(encoder, ComputeSchedule::Install, &vork_time::Request::Frame(0, self.rate))?;
            }
            for &(page, offset, bytes) in &self.frame_ranges {
                encoder.copy_buffer_to_buffer(&self.initial[page], offset, &self.pages[page], offset, bytes);
            }
            for name in &self.frame_sources { cursor.publish_source(name); }
            if self.dispatches.iter().any(|d| d.schedule != ComputeSchedule::Simulation && d.writes_placement) { cursor.publish_placement(); }
            self.set_tick(encoder,0,0.,0.,0);
            self.run(encoder, cursor, ComputeSchedule::Install, 0);
            cursor.frame = None;
        }
        let start = cursor.frame.map_or(0, |f| f + 1);
        for step in start..=frame {
            if let Some(inputs) = &self.geometry_inputs {
                inputs.sample(encoder, &vork_time::Request::Frame(step, self.rate))?;
            }
            if let Some(sample) = sample.as_mut() {
                sample(encoder, ComputeSchedule::Frame, &vork_time::Request::Frame(step, self.rate))?;
            }
            self.set_tick(encoder,step,f64::from(step)*self.rate.denominator() as f64/self.rate.numerator() as f64,if step==0 {0.}else{self.rate.denominator() as f64/self.rate.numerator() as f64},0);
            self.run(encoder, cursor, ComputeSchedule::Frame, step);
            cursor.frame = Some(step);
        }
        Ok(checkpoint)
    }
    /// Snapshot an explicitly requested output pose, never an incidental producer
    /// tick. The submission owner holds the single cursor lock through the copy,
    /// subsequent current-pose evaluation and bridge consumption.
    pub(crate) fn snapshot_placement_previous(&self, encoder: &mut wgpu::CommandEncoder) {
        for &(page, offset, size) in &self.placement_ranges {
            encoder.copy_buffer_to_buffer(&self.pages[page], offset, &self.previous[page], offset, size);
        }
    }
    pub(crate) fn commands(r: &Renderer, draws: &[install::Draw]) -> wgpu::Buffer {
        let args: Vec<_> = draws
            .iter()
            .map(|draw| wgpu::util::DrawIndexedIndirectArgs {
                index_count: draw.count,
                instance_count: draw.instances.len() as u32,
                first_index: 0,
                base_vertex: 0,
                first_instance: draw.instances.start,
            })
            .collect();
        let empty = wgpu::util::DrawIndexedIndirectArgs::default();
        init(
            &r.device,
            "compute draw commands",
            if args.is_empty() {
                bytemuck::bytes_of(&empty)
            } else {
                bytemuck::cast_slice(&args)
            },
            wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
        )
    }
    pub(crate) fn slot(
        &self,
        r: &Renderer,
        s: &InstalledScene,
        objects: &wgpu::Buffer,
        commands: wgpu::Buffer,
    ) -> Result<Slot> {
        ensure!(
            r.device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            "compute instances require INDIRECT_FIRST_INSTANCE"
        );
        let parent_values: Vec<_> = self
            .instances
            .iter()
            .map(|instance| s.objects[s.draws[instance.draw].instances.start as usize])
            .collect();
        let parents = init(
            &r.device,
            "compute parent objects",
            bytemuck::cast_slice(&parent_values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let count = self.named_pages;
        let (pipeline, groups, indices) = self
            .bridge
            .as_ref()
            .context("missing installed instance bridge")?;
        let errors=init(&r.device,"compute placement errors",&[0;8],wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let error_readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("compute placement errors readback"),size:8,usage:wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let mut binds = Vec::new();
        for output in [&commands] {
            let mut entries = Vec::new();
            for i in 0..count {
                entries.push(entry(i as u32, self.pages[i].as_entire_binding()));
                entries.push(entry(
                    (count + i) as u32,
                    self.previous[i].as_entire_binding(),
                ));
            }
            entries.push(entry((count * 2) as u32, parents.as_entire_binding()));
            entries.push(entry((count * 2 + 1) as u32, objects.as_entire_binding()));
            entries.push(entry((count * 2 + 2) as u32, output.as_entire_binding()));
            entries.push(entry((count * 2 + 3) as u32, indices.as_entire_binding()));
            entries.push(entry((count * 2 + 4) as u32, errors.as_entire_binding()));
            binds.push(r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute instance bridge"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            }));
        }
        Ok(Slot {
            parents, errors, error_readback,
            pipeline: pipeline.clone(),
            binds,
            groups: *groups,
            width: (*groups).min(r.device.limits().max_compute_workgroups_per_dimension),
            objects: self
                .instances
                .iter()
                .map(|i| s.draws[i.draw].instances.clone())
                .collect(),
        })
    }
    pub(crate) fn prepare_bridge(
        &mut self,
        r: &Renderer,
        draws: &[install::Draw],
        objects: &[Object],
    ) -> Result<()> {
        if self.has_instances() {
            self.bridge = Some(self.build_bridge(r, draws, objects)?);
        }
        Ok(())
    }
    fn build_bridge(
        &self,
        r: &Renderer,
        draws: &[install::Draw],
        objects: &[Object],
    ) -> Result<(
        wgpu::ComputePipeline,
        u32,
        wgpu::Buffer,
    )> {
        let mut code = String::from(FINITE_WGSL);
        let count = self.named_pages;
        let storage_limit=crate::pipelines::scalar_storage_limit(&r.device.limits(),r.adapter_info.backend==wgpu::Backend::Metal,0);
        ensure!(
            2 * count as u32 + 5 <= storage_limit,
            crate::gpu_memory::device_limit("compute instance bridge", format_args!("{} storage bindings",2*count+5), format_args!("{} storage bindings",storage_limit), "reduce compute buffer pool size or the number of named buffers")
        );
        for i in 0..count {
            writeln!(
                code,
                "@group(0) @binding({}) var<storage,read> current_{i}:array<u32>;\n@group(0) @binding({}) var<storage,read> previous_{i}:array<u32>;",
                i,
                count + i
            )?;
        }
        code.push_str("struct Object { model:mat4x4<f32>, previous_model:mat4x4<f32>, next_model:mat4x4<f32>, opacity:vec4<f32>, material_index:vec4<u32>, metadata:vec4<u32> };\nstruct Args { indices:u32, instances:u32, first:u32, base:i32, object:u32 };\n");
        code.push_str(&crate::object_abi::wgsl("GpuObject"));
        code.push_str("fn affine_rows(m:mat4x4<f32>)->mat3x4<f32> { let t=transpose(m); return mat3x4<f32>(t[0],t[1],t[2]); }\n");
        writeln!(
            code,
            "@group(0) @binding({}) var<storage,read> parents:array<Object>;\n@group(0) @binding({}) var<storage,read_write> objects:array<GpuObject>;\n@group(0) @binding({}) var<storage,read_write> commands:array<Args>;",
            count * 2,
            count * 2 + 1,
            count * 2 + 2
        )?;
        writeln!(
            code,
            "@group(0) @binding({}) var<storage,read> source_indices:array<u32>;",
            count * 2 + 3
        )?;
        let indices: Vec<u32> = self
            .instances
            .iter()
            .flat_map(|instance| {
                instance
                    .indices
                    .iter()
                    .zip(draws[instance.draw].instances.clone())
                    .flat_map(|(&source, object)| {
                        [source, objects[object as usize].opacity[1] as u32, objects[object as usize].metadata[2]]
                    })
            })
            .collect();
        let indices = init(
            &r.device,
            "compute source instance indices",
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::STORAGE,
        );
        writeln!(code,"@group(0) @binding({}) var<storage,read_write> placement_errors:array<atomic<u32>>;",count*2+4)?;
        let mut offset = 0;
        for instance in &self.instances {
            // Pose/material grouping partitions the original transform domain.
            writeln!(
                code,
                "fn source_{}(i:u32)->u32 {{ return source_indices[({offset}u+i)*3u]; }}",
                instance.draw
            )?;
            offset += instance.indices.len();
            for previous in [false, true] {
                let prefix = if previous { "previous" } else { "current" };
                for (column, binding) in instance.trs.iter().enumerate() {
                    if let Some(binding) = binding {
                        writeln!(code,"fn load_{prefix}_draw_{}_{}(word:u32)->f32 {{",instance.draw,column)?;
                        for span in &binding.spans {
                            let first = span.first * binding.width;
                            let end = (span.first + span.count) * binding.width;
                            writeln!(code,"if(word >= {first}u && word < {end}u) {{ return bitcast<f32>({prefix}_{}[{}u+word-{first}u]); }}",span.page,span.offset)?;
                        }
                        code.push_str("return 0.; }\n");
                    }
                }
                let vector = |column:usize,width:u32,default:&str| {
                    if instance.trs[column].is_none() { return default.to_owned(); }
                    format!("vec{width}<f32>({})",(0..width).map(|component|
                        format!("load_{prefix}_draw_{}_{}(i*{width}u+{component}u)",instance.draw,column)).collect::<Vec<_>>().join(","))
                };
                let t=vector(0,3,"vec3<f32>(0.)");
                let q=vector(1,4,"vec4<f32>(0.,0.,0.,1.)");
                let scale=vector(2,3,"vec3<f32>(1.)");
                writeln!(code,"fn matrix_{prefix}_{}(i:u32)->mat4x4<f32> {{
                    let t={t}; let raw_q={q}; let s={scale};
                    let q_length=dot(raw_q,raw_q);
                    if(!finite3(t) || !finite3(s) || !finite4(raw_q) || !finite_f32(q_length) || !(abs(q_length-1.)<=0.001)) {{
                        atomicOr(&placement_errors[0],2u);
                        return mat4x4<f32>(vec4(1.,0.,0.,0.),vec4(0.,1.,0.,0.),vec4(0.,0.,1.,0.),vec4(0.,0.,0.,1.));
                    }}
                    let q=normalize(raw_q);
                    let x=q.x; let y=q.y; let z=q.z; let w=q.w;
                    return mat4x4<f32>(
                        vec4<f32>(vec3<f32>(1.-2.*(y*y+z*z),2.*(x*y+w*z),2.*(x*z-w*y))*s.x,0.),
                        vec4<f32>(vec3<f32>(2.*(x*y-w*z),1.-2.*(x*x+z*z),2.*(y*z+w*x))*s.y,0.),
                        vec4<f32>(vec3<f32>(2.*(x*z+w*y),2.*(y*z-w*x),1.-2.*(x*x+y*y))*s.z,0.),
                        vec4<f32>(t,1.)); }}",instance.draw)?;
            }
        }
        writeln!(code, "fn update(i:u32) {{")?;
        let mut groups = 1;
        let mut id_offset = 0;
        for (parent_index, instance) in self.instances.iter().enumerate() {
            let draw = &draws[instance.draw];
            let capacity = draw.instances.end - draw.instances.start;
            groups = groups.max(capacity.div_ceil(constants::PROGRAM_WORKGROUP));
            let identity = instance.indices.iter().copied().eq(0..capacity);
            let source_live = instance
                .count
                .as_ref()
                .map(|b| {
                    let sp = &b.spans[0];
                    format!(
                        "min(current_{}[{}u],{}u)",
                        sp.page, sp.offset, instance.capacity
                    )
                })
                .unwrap_or(format!("{}u", instance.capacity));
            if let Some(binding)=&instance.count {
                let span=&binding.spans[0];
                writeln!(code,"if(i==0u && current_{}[{}u]>{}u) {{ atomicOr(&placement_errors[0],1u); }}",span.page,span.offset,instance.capacity)?;
            }
            let previous_live=source_live.replace("current_","previous_");
            let live = if identity {
                format!("min({source_live},{capacity}u)")
            } else {
                format!("{capacity}u")
            };
            writeln!(
                code,
                "if(i < {capacity}u) {{ let o={}u+i; let parent=parents[{parent_index}u]; var value:GpuObject; value.opacity=vec2(parent.opacity.x,f32(source_indices[({id_offset}u+i)*3u+1u])); value.material_index=parent.material_index.x; value.metadata=parent.metadata.xy; value.reserved=source_indices[({id_offset}u+i)*3u+2u]; let source=source_{}(i); value.padding=vec2(parent.material_index.y,source); if(source >= {source_live}) {{ value.metadata.y=0u; }} var model=parent.model; if(source < {source_live}) {{ model=parent.model * matrix_current_{}(source); }} if(!finite4(model[0]) || !finite4(model[1]) || !finite4(model[2]) || !finite4(model[3])) {{ atomicOr(&placement_errors[0],2u); value.metadata.y=0u; model=mat4x4<f32>(vec4(1.,0.,0.,0.),vec4(0.,1.,0.,0.),vec4(0.,0.,1.,0.),vec4(0.,0.,0.,1.)); }} value.model=affine_rows(model); value.next_model=affine_rows(model); value.previous_model=value.model; if(source < {source_live} && source < {previous_live}) {{ value.previous_model=affine_rows(parent.previous_model * matrix_previous_{}(source)); }} let det=dot(model[0].xyz,cross(model[1].xyz,model[2].xyz)); if(!finite_f32(det)) {{ atomicOr(&placement_errors[0],2u); }} if(det==0. || !finite_f32(det)) {{ value.metadata.y=0u; }} objects[o]=value; if(i==0u) {{ commands[{}u].instances={live}; }} }}",
                draw.instances.start, instance.draw, instance.draw, instance.draw, instance.draw
            )?;
            id_offset += instance.indices.len();
        }
        let width = groups.min(r.device.limits().max_compute_workgroups_per_dimension);
        writeln!(
            code,
            "}}\n@compute @workgroup_size({0}) fn vork_entry(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {{ let g=group.y*{width}u+group.x; if(g<{groups}u) {{ update(g*{0}u+lane); }} }}",
            constants::PROGRAM_WORKGROUP
        )?;
        let mut types = vec![Some(true); count * 2 + 1];
        types.extend([Some(false), Some(false), Some(true), Some(false)]);
        let pipeline = pipeline_layout(r, &code, "compute instance bridge", &types, "vork_entry");
        Ok((pipeline, groups, indices))
    }
}
impl Slot {
    pub(crate) fn parent_offset(&self, object: usize) -> Option<u64> {
        self.objects
            .iter()
            .position(|range| range.start == object as u32)
            .map(|index| (index * std::mem::size_of::<Object>()) as u64)
    }
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for bind in &self.binds {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            let groups = self.groups;
            let width = groups.min(self.width);
            pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
        }
    }
}
// Explicit layouts preserve declared bindings even when Slang eliminates unused accessors.
pub(crate) fn pipeline_layout(
    r: &Renderer,
    source: &str,
    label: &str,
    types: &[Option<bool>],
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let bindings: Vec<_> = types.iter().enumerate().map(|(index, ty)| (index as u32, *ty)).collect();
    pipeline_layout_explicit(r, source, label, &bindings, entry_point)
}
fn pipeline_layout_explicit(
    r: &Renderer, source: &str, label: &str, types: &[(u32, Option<bool>)], entry_point: &str,
) -> wgpu::ComputePipeline {
    let entries: Vec<_> = types
        .iter()
        .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding: *binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: ty.map_or(wgpu::BufferBindingType::Uniform, |read_only| {
                    wgpu::BufferBindingType::Storage { read_only }
                }),
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let group = r
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
    let layout = r
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
    let module = crate::timing::shader_module(&r.device, wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    })
}

// Offset each physical dispatch in the authored thread domain. Every tile is
// independent; the entry guard runs before addition to avoid u32 wrap at the edge.
fn dispatch_tiles(groups: [u32; 3], limit: u32) -> Vec<([u32; 3], [u32; 4])> {
    let mut tiles = Vec::new();
    for z in (0..groups[2]).step_by(limit as usize) {
        for y in (0..groups[1]).step_by(limit as usize) {
            for x in (0..groups[0]).step_by(limit as usize) {
                tiles.push((
                    [
                        (groups[0] - x).min(limit),
                        (groups[1] - y).min(limit),
                        (groups[2] - z).min(limit),
                    ],
                    [x * constants::PROGRAM_WORKGROUP, y, z, 0],
                ));
            }
        }
    }
    tiles
}

fn nominal_frame(time:&vork_time::Request,fps:vork_time::Time,frames:u32)->Result<u32> {
    Ok(time.exact()?.scale(&vork_time::exact::decimal(&fps.to_string())?).floor_clamped(0,i64::from(frames)-1)? as u32)
}

/// Typed element access over the shared tightly packed storage pages.
fn typed_accessors(
    header: &mut String,
    binding: &str,
    kind: vork_scene::v2::BufferType,
) -> Result<()> {
    if binding == "fluid_status" {
        header.push_str("#define VORK_FLUID_STATUS_BOUND 1\n");
    }
    typed_element_accessors(header, binding, kind, true)
}
fn typed_element_accessors(header: &mut String, binding: &str, kind: v2::BufferType, writable: bool) -> Result<()> {
    use vork_scene::v2::BufferType::*;
    let (ty, width, decode, encode) = match kind {
        F32 => ("float", 1, "asfloat", "asuint"),
        I32 => ("int", 1, "asint", "asuint"),
        U32 => ("uint", 1, "", ""),
        Vec2 => ("float2", 2, "asfloat", "asuint"),
        Vec3 => ("float3", 3, "asfloat", "asuint"),
        Vec4 => ("float4", 4, "asfloat", "asuint"),
        Uvec4 => ("uint4", 4, "", ""),
        Mat4 => ("VorkMat4", 16, "asfloat", "asuint"),
    };
    // Column vectors preserve the scene buffer's column-major matrix layout.
    if matches!(kind, Mat4) && !header.contains("struct VorkMat4") {
        header.push_str("struct VorkMat4 { float4 c0; float4 c1; float4 c2; float4 c3; };\n");
    }
    writeln!(
        header,
        "{ty} {binding}_load(uint i) {{ {ty} value; if(i >= {binding}_count) return ({ty})0;"
    )?;
    for word in 0..width {
        let field = if width == 1 {
            String::new()
        } else if width == 16 {
            format!(".c{}[{}]", word / 4, word % 4)
        } else {
            format!("[{word}]")
        };
        writeln!(
            header,
            "value{field} = {decode}(_vork_{binding}_load(i*{width}+{word}));"
        )?;
    }
    writeln!(header, "return value; }}")?;
    if !writable { return Ok(()); }
    writeln!(header, "void {binding}_store(uint i, {ty} value) {{ if(i >= {binding}_count) return;")?;
    for word in 0..width {
        let field = if width == 1 {
            String::new()
        } else if width == 16 {
            format!(".c{}[{}]", word / 4, word % 4)
        } else {
            format!("[{word}]")
        };
        writeln!(
            header,
            "_vork_{binding}_store(i*{width}+{word}, {encode}(value{field}));"
        )?;
    }
    header.push_str("}\n");
    Ok(())
}

#[cfg(all(test, vork_private_tests))]
include!("compute_tests_private.rs");

#[cfg(test)]
#[test]
#[ignore = "water4 phase timing on a real GPU; VORK_WATER_SCENE names the frozen scene"]
fn water_simulation_and_reconstruction_timing() -> Result<()> {
    let path=std::env::var_os("VORK_WATER_SCENE").map(PathBuf::from)
        .unwrap_or_else(||Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/water/scene.json"));
    let mut scene=Scene::from_json(&std::fs::read(&path)?)?;
    scene.render.resolution=[32,32];
    let r=Renderer::new()?;
    let s=r.install(scene,path.parent().unwrap())?;
    let compute=s.compute.as_ref().context("water simulation")?;
    let sim=compute.simulation.as_ref().context("water clock")?;
    // This example's tick first advances the fluid/markers, then reconstructs
    // their display and pressure fields. Keep the actual ordered programs.
    let split=sim.steps.iter().position(|name|name=="marker_advect").context("water marker advancement")?+1;
    let mut cursor=compute.cursor.lock().unwrap();
    r.wait(None)?;
    let mut totals=[0.;3];
    for tick in 0..6 {
        for (phase,names) in [&sim.steps[..split],&sim.steps[split..]].into_iter().enumerate() {
            let start=std::time::Instant::now();
            let mut encoder=r.device.create_command_encoder(&Default::default());
            compute.set_tick(&mut encoder,tick,sim.t0.as_f64()+f64::from(tick)*sim.dt.as_f64(),sim.dt.as_f64(),sim.seed);
            compute.run_named(&mut encoder,&mut cursor,names,(sim.t0.as_f64()+f64::from(tick)*sim.dt.as_f64()) as f32);
            let submission=r.queue.submit([encoder.finish()]);r.wait(Some(submission))?;
            let seconds=start.elapsed().as_secs_f64();
            eprintln!("water phase tick={tick} phase={} wall_s={seconds:.6}",if phase==0 {"simulation"} else {"reconstruction"});
            if tick>0 {totals[phase]+=seconds;}
        }
        let start=std::time::Instant::now();
        let mut encoder=r.device.create_command_encoder(&Default::default());
        for draw in &s.draws {if let Some(field)=&draw.implicit {field.acceleration.encode(&mut encoder);}}
        let submission=r.queue.submit([encoder.finish()]);r.wait(Some(submission))?;
        let seconds=start.elapsed().as_secs_f64();
        eprintln!("water phase tick={tick} phase=occupancy wall_s={seconds:.6}");
        if tick>0 {totals[2]+=seconds;}
    }
    eprintln!("water warm five-tick frame (48 fps): simulation_wall_s={:.6} reconstruction_wall_s={:.6} occupancy_five_publications_wall_s={:.6}",totals[0],totals[1],totals[2]);
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "GPU phase timing through bench/bench; VORK_WATER_SCENE names the frozen fluid scene"]
fn fluid_step_timing() -> Result<()> {
    let path=PathBuf::from(std::env::var_os("VORK_WATER_SCENE").context("VORK_WATER_SCENE is required")?);
    let scene=Scene::from_json(&std::fs::read(&path)?)?;
    let base=path.parent().unwrap();let r=Renderer::new()?;
    ensure!(r.device.features().contains(wgpu::Features::TIMESTAMP_QUERY),"fluid timing needs GPU timestamps");
    let decoded=buffers::SceneBuffers::decode(&scene,base)?;
    let attrs=ProgramBuffers::install(&r,&scene,&decoded)?;
    let compute=Compute::test_install(&r,&scene,base,Some(&attrs),&[],&[],&[],vork_time::Time::integer(24),1.)?.context("fluid compute")?;
    let sim=compute.simulation.as_ref().context("fluid simulation")?;
    let split=sim.steps.iter().position(|name|name=="marker_advect").context("fluid marker advancement")?+1;
    let mut cursor=compute.cursor.lock().unwrap();r.wait(None)?;
    let mut totals=[0.;2];
    for tick in 0..6 {
        for (phase,names) in [&sim.steps[..split],&sim.steps[split..]].into_iter().enumerate() {
            let mut encoder=r.device.create_command_encoder(&Default::default());
            compute.set_tick(&mut encoder,tick,sim.t0.as_f64()+f64::from(tick)*sim.dt.as_f64(),sim.dt.as_f64(),sim.seed);
            let probe=TickProbe::new(&r.device);
            compute.run_named_uploaded(&mut encoder,&mut cursor,names,(sim.t0.as_f64()+f64::from(tick)*sim.dt.as_f64()) as f32,None,Some(&probe.queries));
            let mut cost=probe.resolve(&mut encoder,compute.timestamp_period);
            let submission=r.queue.submit([encoder.finish()]);r.wait(Some(submission))?;
            let gpu_ms=cost.value().context("missing fluid phase GPU timestamps")?;
            if tick>0 {totals[phase]+=gpu_ms;}
        }
    }
    // Failed publication is not a fast simulation. Check both generations of
    // the fluid library's receipts before reporting any measurement.
    for (name,element,expected) in [("correction",3,1_f32.to_bits()),("fluid_status",0,0)] {
        let Some(binding)=attrs.bindings.get(name) else {continue};
        let span=binding.spans.iter().find(|span|element>=span.first && element<span.first+span.count).context("fluid timing receipt")?;
        let readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("fluid timing validity"),size:4,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&attrs.storage[span.page as usize],u64::from(span.offset+element-span.first)*4,&readback,0,4);
        let submission=r.queue.submit([encoder.finish()]);let(tx,rx)=std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move |value|{let _=tx.send(value);});r.wait(Some(submission))?;rx.recv()??;
        let data=readback.get_mapped_range(..);let actual=u32::from_le_bytes(data[..4].try_into().unwrap());drop(data);readback.unmap();
        ensure!(actual==expected,"{name} refused fluid timing: {actual:#x}");
    }
    eprintln!("FLUID_STEP_TIMING {}",serde_json::json!({"simulation_gpu_ms":totals[0]/5.,"reconstruction_gpu_ms":totals[1]/5.,"step_gpu_ms":(totals[0]+totals[1])/5.,"steps":5,"warmup_steps":1}));
    Ok(())
}

#[cfg(test)]
mod atomic_contract_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang"]
    fn scalar_atomic_helpers_compile_through_compute_validation() -> Result<()> {
        let layout = AttributeBinding { count: 2, width: 1, spans: vec![
            buffers::AttributeSpan { page: 0, offset: 7, first: 0, count: 1 },
            buffers::AttributeSpan { page: 1, offset: 3, first: 1, count: 1 },
        ] };
        let mut header = String::from(CONTEXT_BINDINGS);
        header.push_str("[[vk::binding(2,0)]] RWStructuredBuffer<Atomic<uint>> vork_page_0; [[vk::binding(3,0)]] RWStructuredBuffer<Atomic<uint>> vork_page_1;");
        atomic_accessors(&mut header,"u",v2::BufferType::U32,&layout)?;
        atomic_accessors(&mut header,"f",v2::BufferType::F32,&layout)?;
        let source = "struct Inputs {}; void compute(ComputeContext c, Inputs i) {uint j=c.id.x; uint x=u_atomic_add(j,1u); x+=u_atomic_sub(j,1u); x+=u_atomic_min(j,3u); x+=u_atomic_max(j,4u); x+=u_atomic_and(j,7u); x+=u_atomic_or(j,8u); x+=u_atomic_xor(j,1u); x+=u_atomic_exchange(j,9u); x+=u_atomic_compare_exchange(j,9u,x); f_atomic_add(j,float(x));}";
        vork_slang::Compiler::discover().compile_compute(source,Path::new("atomic-contract.slang"),&header,64,[3,1,1])?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "cloth_tests.rs"]
mod cloth_rim_tests;

#[cfg(test)]
#[path = "compute_declarations_tests.rs"]
mod compute_declarations_tests;
