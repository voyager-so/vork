//! A threaded bounds index for initial shadow occupancy. Immutable topology is
//! installed once; each frame slot refits against the same object snapshot as
//! its TLAS. Open/unknown domains stay in the separate unbounded list; closed deformation
//! reuses the renderer's evaluated geometry bounds.
use crate::*;

pub(crate) struct Plan {
    pub template:wgpu::Buffer,
    pub count:u32,
    pub slot:u32,
    pipeline:std::sync::OnceLock<wgpu::ComputePipeline>,
    dynamic:Vec<(u64,Vec<usize>)>,
}
pub(crate) struct Slot {pub buffer:wgpu::Buffer,pipeline:wgpu::ComputePipeline,bind:wgpu::BindGroup,count:u32,width:u32}

fn topology(solids:&[passes::path::PathSolid], eligible:&std::collections::BTreeSet<u32>, objects:&[Object]) -> Vec<u32> {
    let mut ids:Vec<_>=eligible.iter().copied().filter(|&id|solids[id as usize].shadow_bounds.is_some()).collect();
    if ids.is_empty() {return Vec::new();}
    let centers:Vec<_>=solids.iter().map(|s|objects[s.row as usize].model.transform_point3(s.shadow_bounds.map_or(Vec3::ZERO,|(lo,hi)|(lo+hi)*0.5))).collect();
    fn build(words:&mut Vec<u32>,ids:&mut [u32],first:usize,centers:&[Vec3]) {
        let start=words.len();words.resize(start+constants::SHADOW_BOUND_NODE_WORDS,0);
        words[start+7]=if ids.len()==1 {ids[0]} else {u32::MAX};
        words[start+8]=first as u32;words[start+9]=(first+ids.len()) as u32;
        if ids.len()>1 {
            let (lo,hi)=ids.iter().fold((Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY)),|(lo,hi),&id|(lo.min(centers[id as usize]),hi.max(centers[id as usize])));
            let size=hi-lo;let axis=if size.x>=size.y && size.x>=size.z {0} else if size.y>=size.z {1} else {2};
            ids.sort_by(|a,b|centers[*a as usize][axis].total_cmp(&centers[*b as usize][axis]));
            let middle=ids.len()/2;let (left,right)=ids.split_at_mut(middle);
            build(words,left,first,centers);build(words,right,first+middle,centers);
        }
        words[start+3]=words.len() as u32;
    }
    let mut words=Vec::new();build(&mut words,&mut ids,0,&centers);
    let nodes=words.len();
    for node in words.chunks_exact_mut(constants::SHADOW_BOUND_NODE_WORDS) {
        node[8]=nodes as u32+node[8]*constants::PATH_SOLID_WORDS as u32;
        node[9]=nodes as u32+node[9]*constants::PATH_SOLID_WORDS as u32;
    }
    for id in ids {let s=&solids[id as usize];let (lo,hi)=s.shadow_bounds.unwrap();words.extend([lo.x.to_bits(),lo.y.to_bits(),lo.z.to_bits(),hi.x.to_bits(),hi.y.to_bits(),hi.z.to_bits(),s.row]);}
    words
}
impl Plan {
    pub(crate) fn new(r:&Renderer,solids:&[passes::path::PathSolid],eligible:&std::collections::BTreeSet<u32>,objects:&[Object],draws:&[install::Draw],owners:&[u32]) -> Result<Option<Self>> {
        let words=topology(solids,eligible,objects);if words.is_empty() {return Ok(None);}
        let count=words[3]/constants::SHADOW_BOUND_NODE_WORDS as u32;
        let mut by_solid=std::collections::HashMap::<u32,Vec<usize>>::new();
        for (i,draw) in draws.iter().enumerate() {
            for object in draw.instances.clone() {
                let Some(&solid)=owners.get(object as usize).filter(|&&s|s!=u32::MAX) else {continue;};
                let members=by_solid.entry(solid).or_default();if members.last()!=Some(&i) {members.push(i);}
            }
        }
        let mut dynamic=Vec::new();
        for node in words[..words[3] as usize].chunks_exact(constants::SHADOW_BOUND_NODE_WORDS) {
            let solid=node[7];if solid==u32::MAX {continue;}
            let members=by_solid.remove(&solid).unwrap_or_default();
            if members.iter().any(|&i|draws[i].evaluated()) {dynamic.push((u64::from(node[8])*4,members));}
        }
        crate::gpu_memory::storage_binding(&r.device.limits(),"shadow containment index",words.len() as u64*4, "reduce solid-boundary geometry or the number of shadow lights")?;
        Ok(Some(Self {template:init(&r.device,"shadow containment topology",bytemuck::cast_slice(&words),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC),count,slot:0,pipeline:Default::default(),dynamic}))
    }
}
impl Plan {
    pub(crate) fn dynamic(&self)->bool {!self.dynamic.is_empty()}
    pub(crate) fn upload(&self,slot:&Slot,local:&[(Vec3,Vec3)],changes:&mut values::ChangeTracker) {
        for (offset,members) in &self.dynamic {
            let (lo,hi)=members.iter().fold((Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY)),|(lo,hi),&i|(lo.min(local[i].0),hi.max(local[i].1)));
            changes.upload(&slot.buffer,*offset,bytemuck::cast_slice(&[lo.x,lo.y,lo.z,hi.x,hi.y,hi.z]));
        }
    }
}
impl Slot {
    pub(crate) fn new(r:&Renderer,plan:&Plan,objects:&wgpu::Buffer) -> Result<Self> {
        let d=&r.device;
        let width=plan.count.div_ceil(64).min(d.limits().max_compute_workgroups_per_dimension);
        let pipeline=plan.pipeline.get_or_init(|| {
            let source=format!(r#"
{object}
{finite}
@group(0) @binding(0) var<storage,read> objects:array<Object>;
@group(0) @binding(1) var<storage,read_write> words:array<u32>;
fn v(at:u32)->vec3<f32> {{return bitcast<vec3<f32>>(vec3<u32>(words[at],words[at+1u],words[at+2u]));}}
@compute @workgroup_size(64) fn refit(@builtin(global_invocation_id) tid:vec3<u32>) {{
    // No scene size in the text: the root node's skip word is the end of the nodes, and the
    // dispatch row width follows from the count and the device's workgroup limit.
    let count=words[3u]/{stride}u;
    let index=tid.x+tid.y*min((count+63u)/64u,{row_groups}u)*64u;
    if(index>=count) {{return;}}
    let node=index*{stride}u;
    let huge=bitcast<f32>(0x7f800000u);
    var low=vec3<f32>(huge);var high=vec3<f32>(-huge);
    // Nodes read only immutable local bounds, never another invocation's output.
    // O(n log n) placement work replaces O(n) discovery on every connection.
    for(var at=words[node+8u];at<words[node+9u];at+={solid_stride}u) {{
        let obj=objects[words[at+6u]];
        if((obj.metadata.y & {shadow}u)==0u) {{continue;}}
        let rows=obj.model;
        let determinant=dot(rows[0].xyz,cross(rows[1].xyz,rows[2].xyz));
        if(determinant==0.) {{continue;}}
        let linear=transpose(mat3x3<f32>(rows[0].xyz,rows[1].xyz,rows[2].xyz));
        let inv=mat3x3<f32>(cross(rows[1].xyz,rows[2].xyz),cross(rows[2].xyz,rows[0].xyz),cross(rows[0].xyz,rows[1].xyz))*(1./determinant);
        let a=mat3x3<f32>(abs(linear[0]),abs(linear[1]),abs(linear[2]));
        let b=mat3x3<f32>(abs(inv[0]),abs(inv[1]),abs(inv[2]));
        let t=vec3<f32>(rows[0].w,rows[1].w,rows[2].w);
        let lo=v(at);let hi=v(at+3u);let reach=max(abs(lo),abs(hi));
        let norm_rows=b*a*vec3<f32>({gamma});let norm=max(norm_rows.x,max(norm_rows.y,norm_rows.z));
        if(!(norm<0.5)) {{low=vec3<f32>(-huge);high=vec3<f32>(huge);break;}}
        let c=b*(a*reach+2.*abs(t))*{gamma};
        let e=max(c.x,max(c.y,c.z))/(1.-norm);
        let margin=(a*reach+abs(t))*{gamma}+a*vec3<f32>(e);
        let center=linear*((lo+hi)*0.5)+t;let radius=a*((hi-lo)*0.5)+margin;
        let lower=center-radius;let upper=center+radius;
        if(!finite3(lower) || !finite3(upper)) {{low=vec3<f32>(-huge);high=vec3<f32>(huge);break;}}
        low=min(low,lower);high=max(high,upper);
    }}
    for(var axis=0u;axis<3u;axis++) {{words[node+axis]=bitcast<u32>(low[axis]);words[node+4u+axis]=bitcast<u32>(high[axis]);}}
}}
"#,object=crate::object_abi::wgsl("Object"),finite=crate::compute::FINITE_WGSL,row_groups=d.limits().max_compute_workgroups_per_dimension,stride=constants::SHADOW_BOUND_NODE_WORDS,solid_stride=constants::PATH_SOLID_WORDS,shadow=constants::RAY_SHADOW,gamma=vork_slang::registry::FP32_GAMMA7);
            crate::compute::pipeline_layout(r,&source,"shadow containment refit",&[Some(true),Some(false)],"refit")
        }).clone();
        let buffer=crate::gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:Some("frame shadow containment bounds"),size:plan.template.size(),usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let mut encoder=d.create_command_encoder(&Default::default());encoder.copy_buffer_to_buffer(&plan.template,0,&buffer,0,buffer.size());crate::command_finish::Segments::submit(&r.queue,encoder);
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("shadow containment refit"),layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,objects.as_entire_binding()),entry(1,buffer.as_entire_binding())]});
        Ok(Self {buffer,pipeline,bind,count:plan.count,width})
    }
    pub(crate) fn encode(&self,encoder:&mut wgpu::CommandEncoder) {
        let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&self.pipeline);pass.set_bind_group(0,&self.bind,&[]);pass.dispatch_workgroups(self.width,self.count.div_ceil(64).div_ceil(self.width),1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn selected(words:&[u32],point:Vec3)->(Vec<u32>,usize) {
        let mut found=Vec::new();let mut visited=0;let mut at=0usize;let end=words[3] as usize;
        while at<end {
            visited+=1;
            let lo=Vec3::new(f32::from_bits(words[at]),f32::from_bits(words[at+1]),f32::from_bits(words[at+2]));
            let hi=Vec3::new(f32::from_bits(words[at+4]),f32::from_bits(words[at+5]),f32::from_bits(words[at+6]));
            if point.cmpge(lo).all() && point.cmple(hi).all() {
                if words[at+7]==u32::MAX {at+=constants::SHADOW_BOUND_NODE_WORDS;continue;}
                found.push(words[at+7]);
            }
            at=words[at+3] as usize;
        }
        found.sort_unstable();(found,visited)
    }
    #[test]
    fn empty_and_unbounded_domains_allocate_no_index() {
        assert!(topology(&[],&Default::default(),&[]).is_empty());
        let solids=[passes::path::PathSolid {lo:Vec3::splat(f32::NEG_INFINITY),hi:Vec3::splat(f32::INFINITY),row:0,contact_bounds:(Vec3::splat(f32::NEG_INFINITY),Vec3::splat(f32::INFINITY)),shadow_bounds:None,rest_bounds:None}];
        assert!(topology(&solids,&[0].into(),&[Object::new(Mat4::IDENTITY)]).is_empty());
    }
    #[test]
    #[ignore="requires real GPU"]
    fn bounds_refit_prunes_off_ray_solids_and_tracks_moving_placements() -> Result<()> {
        let r=Renderer::new()?;
        for far in [0u32,64,1024] {
            let solids:Vec<_>=(0..=far).map(|row|passes::path::PathSolid {lo:Vec3::splat(-1.),hi:Vec3::splat(1.),row,contact_bounds:(Vec3::splat(f32::NEG_INFINITY),Vec3::splat(f32::INFINITY)),shadow_bounds:Some((Vec3::splat(-1.),Vec3::splat(1.))),rest_bounds:None}).collect();
            let mut objects:Vec<_>=(0..=far).map(|i|Object::new(Mat4::from_translation(if i==0 {Vec3::ZERO} else {Vec3::new(10.+i as f32*4.,0.,0.)}))).collect();
            let plan=Plan::new(&r,&solids,&(0..=far).collect(),&objects,&[],&[])?.unwrap();
            let object_buffer=init(&r.device,"bounds test objects",bytemuck::cast_slice(&gpu_objects(&objects)),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST);
            let slot=Slot::new(&r,&plan,&object_buffer)?;
            let readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:slot.buffer.size(),usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            for phase in 0..4 {
                let moved=phase>0;
                if moved && far>0 {objects[far as usize].model=Mat4::from_scale_rotation_translation(Vec3::new(2.,0.5,3.),glam::Quat::from_rotation_y(0.7),Vec3::ZERO);}
                if phase==2 {objects[0].model=Mat4::from_scale_rotation_translation(Vec3::new(0.1,100.,2.),glam::Quat::from_rotation_z(0.9),Vec3::new(1e6,-1e6,3.));}
                if phase==3 {objects[0].model=Mat4::from_scale(Vec3::ZERO);}
                r.queue.write_buffer(&object_buffer,0,bytemuck::cast_slice(&gpu_objects(&objects)));
                let mut encoder=r.device.create_command_encoder(&Default::default());slot.encode(&mut encoder);encoder.copy_buffer_to_buffer(&slot.buffer,0,&readback,0,slot.buffer.size());crate::command_finish::Segments::submit(&r.queue,encoder);
                let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move |result| {let _=tx.send(result);});r.wait(None)?;rx.recv()??;
                let mapped=readback.get_mapped_range(..);let words:&[u32]=bytemuck::cast_slice(&mapped);
                let (found,visited)=selected(words,Vec3::ZERO);
                let mut expected=Vec::new();if phase<2 {expected.push(0);}if moved && far>0 {expected.push(far);}
                assert_eq!(found,expected);
                for id in [0,far] {
                    let model=objects[id as usize].model;
                    if model.determinant()==0. {continue;}
                    for corner in [Vec3::splat(-1.),Vec3::splat(1.)] {
                        assert!(selected(words,model.transform_point3(corner)).0.contains(&id),"refit lost a transformed solid corner");
                    }
                }
                assert!(visited<=4*((far+1).ilog2()+2) as usize,"off-ray discovery traversed {visited} nodes for {far} solids");
                eprintln!("{far} off-ray solids, phase={phase}: {visited} spatial nodes, {} containing domains",found.len());
                drop(mapped);readback.unmap();
            }
        }
        r.errors.check()?;Ok(())
    }
}
