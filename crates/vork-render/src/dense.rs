//! Dense fields publish retained compute pages into the existing medium table.
//! Reduction reads every cell, not probes. Only its finite range crosses to the
//! CPU certificate interpreter; voxel values and transport remain on the GPU.
use super::*;
use crate::buffers::ProgramBuffers;

pub(crate) struct Certificate {
    pub program: vork_slang::MediumProgram,
    pub words: Vec<Option<u32>>,
    pub grids: Vec<(String, usize)>,
    pub destination: u32,
    pub bounds: (Vec3, Vec3),
    pub times: [f32; 2],
}
pub(crate) struct Dense {
    fields: Vec<Field>,
    pipeline: wgpu::ComputePipeline,
    initial: wgpu::Buffer,
    readback: wgpu::Buffer,
    certificates: Vec<Certificate>,
}
struct Field {
    name: String,
    spans: Vec<(wgpu::Buffer, u64, u64, u64)>,
    reductions: Vec<(wgpu::BindGroup,u32)>,
    range: wgpu::Buffer,
    destinations: Vec<u64>,
}
impl Dense {
    pub(crate) fn install(r:&Renderer,scene:&Scene,buffers:Option<&ProgramBuffers>,destinations:Vec<(String,u32)>,certificates:Vec<Certificate>)->Result<Option<Self>> {
        if destinations.is_empty() {return Ok(None);}
        let buffers=buffers.context("dense fields require retained buffers")?;
        let shader=crate::timing::shader_module(&r.device,wgpu::ShaderModuleDescriptor {label:Some("dense field range"),source:wgpu::ShaderSource::Wgsl(REDUCE.replace("DENSE_GROUP", &constants::DENSE_REDUCTION_GROUP.to_string()).into())});
        let pipeline=r.compiler.compute(&wgpu::ComputePipelineDescriptor {label:Some("dense field range"),layout:None,module:&shader,entry_point:Some("reduce"),compilation_options:Default::default(),cache:None});
        let layout=pipeline.get_bind_group_layout(0);
        let initial=init(&r.device,"dense range identity",bytemuck::cast_slice(&[u32::MAX,0u32,0u32]),wgpu::BufferUsages::COPY_SRC);
        let mut fields=Vec::new();
        for (name,_) in scene.buffers.iter().filter(|(_,b)|b.dims.is_some()) {
            let destinations=destinations.iter().filter(|(n,_)|n==name).map(|(_,at)|u64::from(*at)*4).collect::<Vec<_>>();
            if destinations.is_empty() {continue;}
            let range=init(&r.device,"dense finite range",&[0;12],wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST);
            let binding=&buffers.bindings[name];
            let mut spans=Vec::new(); let mut reductions=Vec::new();
            for span in &binding.spans {
                let count=span.count*binding.width;
                let params=init(&r.device,"dense range span",bytemuck::cast_slice(&[span.offset,count,0,0]),wgpu::BufferUsages::UNIFORM);
                let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("dense range span"),layout:&layout,entries:&[
                    wgpu::BindGroupEntry {binding:0,resource:buffers.storage[span.page as usize].as_entire_binding()},
                    wgpu::BindGroupEntry {binding:1,resource:range.as_entire_binding()},
                    wgpu::BindGroupEntry {binding:2,resource:params.as_entire_binding()},
                ]});
                // Each workgroup loops over a strided span, so arbitrarily large
                // retained pages do not exceed backend dispatch dimensions.
                reductions.push((bind,count.div_ceil(constants::DENSE_REDUCTION_GROUP).min(r.device.limits().max_compute_workgroups_per_dimension)));
                spans.push((buffers.storage[span.page as usize].clone(),u64::from(span.offset)*4,u64::from(span.first*binding.width)*4,u64::from(count)*4));
            }
            fields.push(Field {name:name.clone(),spans,reductions,range,destinations});
        }
        let readback=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("dense finite ranges"),size:fields.len() as u64*12,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
        Ok(Some(Self {fields,pipeline,initial,readback,certificates}))
    }
    pub(crate) fn publish(&self,r:&Renderer,encoder:&mut wgpu::CommandEncoder,segments:&mut crate::command_finish::Segments,table:&wgpu::Buffer)->Result<()> {
        for (i,field) in self.fields.iter().enumerate() {
            encoder.copy_buffer_to_buffer(&self.initial,0,&field.range,0,12);
            for (bind,groups) in &field.reductions {
                let mut pass=encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,bind,&[]);pass.dispatch_workgroups(*groups,1,1);
            }
            for &destination in &field.destinations {
                for (source,offset,first,bytes) in &field.spans {
                    encoder.copy_buffer_to_buffer(source,*offset,table,destination+8+first,*bytes);
                }
            }
            encoder.copy_buffer_to_buffer(&field.range,0,&self.readback,i as u64*12,12);
        }
        // The same submission owner drains earlier simulation chunks and value
        // uploads before range collection. Dense scenes use streaming samples.
        let (submission,_)=segments.interop(r,encoder,"after dense publication");
        let (tx,rx)=std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read,..,move |result|{let _=tx.send(result);});
        r.device.poll(wgpu::PollType::Wait {submission_index:submission,timeout:None})?;rx.recv()??;
        let ranges=bytemuck::cast_slice::<u8,u32>(&self.readback.get_mapped_range(..)).to_vec();
        self.readback.unmap();
        let decode=|ordered:u32|f32::from_bits(if ordered&0x80000000!=0 {ordered^0x80000000}else{!ordered});
        let mut values=std::collections::BTreeMap::new();
        for (field,range) in self.fields.iter().zip(ranges.chunks_exact(3)) {
            ensure!(range[2]==0,"dense grid {} published nonfinite values",field.name);
            let (lo,hi)=(decode(range[0]),decode(range[1]));
            ensure!(lo.is_finite() && hi.is_finite() && lo<=hi,"dense grid {} has no finite range",field.name);
            let words=[lo.to_bits(),hi.to_bits()];
            for &destination in &field.destinations {r.queue.write_buffer(table,destination,bytemuck::cast_slice(&words));}
            values.insert(field.name.as_str(),words);
        }
        for certificate in &self.certificates {
            let mut words=certificate.words.clone();
            for (name,slot) in &certificate.grids {
                let range=values[name.as_str()];words[slot+4]=Some(range[0]);words[slot+5]=Some(range[1]);
            }
            let bound=crate::medium::bound::certify(&certificate.program,&words,certificate.bounds,certificate.times)?;
            r.queue.write_buffer(table,u64::from(certificate.destination)*4,bytemuck::bytes_of(&bound));
        }
        Ok(())
    }
}
const REDUCE:&str=r#"
@group(0) @binding(0) var<storage,read> cells:array<u32>;
struct Range {lo:atomic<u32>,hi:atomic<u32>,invalid:atomic<u32>}
@group(0) @binding(1) var<storage,read_write> range:Range;
@group(0) @binding(2) var<uniform> span:vec4<u32>;
@compute @workgroup_size(DENSE_GROUP)
fn reduce(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {
    for(var i=id.x;i<span.y;i+=groups.x*DENSE_GROUP) {
        let bits=cells[span.x+i];
        if (bits&0x7f800000u)==0x7f800000u {atomicOr(&range.invalid,1u);continue;}
        let ordered=select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);
        atomicMin(&range.lo,ordered);atomicMax(&range.hi,ordered);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use vork_scene::v2;
    #[test]
    #[ignore="requires pinned Slang and a real GPU"]
    fn surface_dense_scalar_vector_publication_and_seek()->Result<()> {
        let r=Renderer::new()?;
        for (vector,animated) in [(false,false),(true,false),(false,true),(true,true)] {
            let (ty,grid,sample,store)=if vector {("vec3","VectorGrid","sample_grid(i.field,float3(0))","float3(q,q*.5,q*.25)")}
                else {("f32","FloatGrid","float3(sample_grid(i.field,float3(0)))","q")};
            let mut value=serde_json::json!({"version":2,
                "buffers":{"field":{"type":ty,"dims":[2,2,2]}},
                "programs":{
                    "surface":{"source":format!("import vork;import std.grid;struct Inputs {{{grid} field;float gain=1.;}};Closure surface(ShadingContext c,Inputs i) {{return unlit(({sample})*i.gain);}}")},
                    "write":{"source":format!("struct Inputs {{}};void compute(ComputeContext c,Inputs i) {{float q=c.dt==0?.75:2.;field_store(c.id.x,{store});}}"),"bindings":{"field":"field"},"dispatch":8,"schedule":"simulation"}},
                "materials":{"m":{"program":"surface","inputs":{"field":{"grid":"field"}}}},
                "geometry":{"quad":{"attributes":{"POSITION":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]}},"indices":{"type":"u32","data":[0,1,2,0,2,3]}}},
                "nodes":[{"type":"mesh","name":"screen","geometry":"quad","materials":["m"]},{"type":"camera","name":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2.2}}}],
                "composition":{"space":"3d","view":{"program":"std/standard"}},"render":{"resolution":[32,32],"duration":2},
                "animation":[{"tracks":[{"path":["materials","m","inputs","gain"],"keys":[[0,1],[1,0.5]]}]}],
                "simulation":{"t0":0,"dt":0.125,"seed":155,"state":["field","other"],"init":["write"],"steps":["write"]}});
            // Initial range [0,1]: .75 detects stale cells, 2 detects stale range.
            value["buffers"]["field"]["data"]=if vector {serde_json::json!([[0,0,0],[1,1,1],[0,0,0],[1,1,1],[0,0,0],[1,1,1],[0,0,0],[1,1,1]])} else {serde_json::json!([0,1,0,1,0,1,0,1])};
            // Different mutable grids with identical initial bytes must not alias.
            value["buffers"]["other"]=value["buffers"]["field"].clone();
            value["programs"]["write"]["bindings"]["other"]=serde_json::json!("other");
            let write=value["programs"]["write"]["source"].as_str().unwrap().replace(";}",&format!(";q*=.25;other_store(c.id.x,{store});}}"));
            value["programs"]["write"]["source"]=serde_json::json!(write);
            value["materials"]["other"]=value["materials"]["m"].clone();
            value["materials"]["other"]["inputs"]["field"]["grid"]=serde_json::json!("other");
            value["nodes"][0]["translation"]=serde_json::json!([-0.5,0,0]);
            value["nodes"][0]["scale"]=serde_json::json!([0.45,0.9,1]);
            let mut other=value["nodes"][0].clone();
            other["name"]=serde_json::json!("other");other["translation"]=serde_json::json!([0.5,0,0]);other["materials"]=serde_json::json!(["other"]);
            value["nodes"].as_array_mut().unwrap().push(other);
            if !animated {value.as_object_mut().unwrap().remove("animation");}
            let scene:Scene=vork_scene::v2::from_test_value(value.clone())?;
            let installed=r.install(scene,Path::new("."))?;
            assert!(installed.dense.is_some(),"surface-only grids need publication");
            let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            for (time,q) in [(0.,0.75),(0.125,2.),(0.,0.75)] {
                slot.submit(&installed,&r,time)?;
                let actual=slot.collect(&r)?.linear_rgba.unwrap();
                let mut reference=value.clone();
                reference.as_object_mut().unwrap().remove("simulation");
                reference["programs"].as_object_mut().unwrap().remove("write");
                reference["buffers"]["field"]["data"]=if vector {serde_json::json!(vec![[q,q*0.5,q*0.25];8])} else {serde_json::json!(vec![q;8])};
                let q=q*0.25;
                reference["buffers"]["other"]["data"]=if vector {serde_json::json!(vec![[q,q*0.5,q*0.25];8])} else {serde_json::json!(vec![q;8])};
                let reference=r.install(vork_scene::v2::from_test_value(reference)?,Path::new("."))?;
                let mut expected=reference.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
                expected.submit(&reference,&r,time)?;
                crate::test_linear::assert_linear_eq(&actual,&expected.collect(&r)?.linear_rgba.unwrap());
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires pinned Slang and a real GPU"]
    fn dense_scalar_vector_publication_and_seek_replay()->Result<()> {
        let r=Renderer::new()?;
        let value=serde_json::json!({"version":2,"composition":{"space":"3d","view":{"program":"std/standard"}},"render":{"resolution":[32,32],"duration":2},
            "buffers":{"d":{"type":"f32","dims":[2,2,2]},"v":{"type":"vec3","dims":[2,2,2]}},
            "geometry":{"box":{"topology":"volume","attributes":{"density":"d","colour":"v"}}},
            "programs":{
                "init":{"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {float q=.1+float(c.id.x)*.01;d_store(c.id.x,q);v_store(c.id.x,float3(q,-q*2.,q*3.));}","bindings":{"d":"d","v":"v"},"schedule":"simulation","dispatch":8},
                "step":{"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {d_store(c.id.x,d_load(c.id.x)+c.dt);v_store(c.id.x,v_load(c.id.x)+float3(c.dt,-2.*c.dt,3.*c.dt));}","bindings":{"d":"d","v":"v"},"schedule":"simulation","dispatch":8},
                "medium":{"source":"import std.grid;struct Inputs {FloatGrid d;VectorGrid v;}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;float d=sample_grid(i.d,c.position);m.absorption=float3(d*(1.-d));m.emission=abs(sample_grid(i.v,c.position));return m;}"}},
            "materials":{"m":{"program":"medium","inputs":{"d":{"grid":"d"},"v":{"grid":"v"}}}},
            "simulation":{"t0":0,"dt":0.125,"seed":7,"state":["d","v"],"init":["init"],"steps":["step"]},
            "nodes":[{"type":"camera","name":"camera","translation":[0,0,4],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2.5,"clip_end":10}}},{"type":"mesh","name":"volume","geometry":"box","materials":["m"],"scale":[1.2,0.8,0.5]}]});
        // Stage-1 medium admission has no bounded work proof for resource
        // inputs, so a dense-grid medium is refused before publication under
        // either camera owner.
        for path in [false,true] {
            let mut scene:Scene=vork_scene::v2::from_test_value(value.clone())?;
            if path {scene.render.settings=v2::Settings::with_integrator(v2::Integrator::Path(Default::default()));}
            let error=r.install(scene,Path::new(".")).err().context("dense-grid medium installed")?;
            let error=format!("{error:#}");
            assert!(error.contains("has no bounded callback work proof") && error.contains("medium work proof does not yet admit resource inputs"),"{error}");
        }
        // The same compute-written scalar and vector grids publish to a surface
        // consumer, the supported owner of dense snapshots.
        let mut surface=value;
        surface["programs"].as_object_mut().unwrap().remove("medium");
        surface["programs"]["surface"]=serde_json::json!({"source":"import vork;import std.grid;struct Inputs {FloatGrid d;VectorGrid v;};Closure surface(ShadingContext c,Inputs i) {return unlit(float3(sample_grid(i.d,float3(0)))+abs(sample_grid(i.v,float3(0))));}"});
        surface["materials"]["m"]["program"]=serde_json::json!("surface");
        surface["geometry"]=serde_json::json!({"quad":{"attributes":{"POSITION":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]}},"indices":{"type":"u32","data":[0,1,2,0,2,3]}}});
        surface["nodes"][1]=serde_json::json!({"type":"mesh","name":"screen","geometry":"quad","materials":["m"]});
        let mut scene:Scene=vork_scene::v2::from_test_value(surface)?;
        scene.render.settings.samples=4;
        let installed=r.install(scene.clone(),Path::new("."))?;
        assert!(installed.dense.is_some(),"surface-only grids need publication");
        let mut slot=installed.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        slot.submit(&installed ,&r,0.5)?;let first=slot.collect(&r)?.linear_rgba.unwrap();
        slot.submit(&installed,&r,1.)?;let later=slot.collect(&r)?.linear_rgba.unwrap();
        assert_ne!(first,later,"published cells must change the picture");
        slot.submit(&installed ,&r,0.5)?;crate::test_linear::assert_linear_eq(&first,&slot.collect(&r)?.linear_rgba.unwrap());
        let fresh=r.install(scene,Path::new("."))?;let mut other=fresh.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        other.submit(&fresh ,&r,0.5)?;crate::test_linear::assert_linear_eq(&first,&other.collect(&r)?.linear_rgba.unwrap());
        Ok(())
    }
}
