//! Typed named streams gathered into the shared evaluated geometry at sample time.
//! Bounds remain GPU-resident; validation receipts are checked at output collection.
use crate::*;
use std::{collections::BTreeSet, fmt::Write};
use vork_scene::v2;

pub(crate) fn dynamic(scene: &Scene, g: &v2::Geometry) -> bool {
    if matches!(g.topology, v2::Topology::Volume | v2::Topology::Implicit) { return false; }
    g.status.is_some() || g.count.is_some() || g.attributes.values().chain(g.widths.iter()).chain(g.ids.iter())
        .any(|name|crate::compute::writes_buffer(scene,name))
}
/// Membership and shading streams cannot invalidate authored spatial bounds.
pub(crate) fn spatial(scene: &Scene, g: &v2::Geometry) -> bool {
    g.attributes.get("POSITION").into_iter().chain(g.widths.iter()).chain(g.status.iter().map(|s| &s.buffer)).any(|name|crate::compute::writes_buffer(scene,name))
}

pub(crate) struct Mesh {
    pipeline: [wgpu::ComputePipeline; 2],
    inputs: Vec<wgpu::Buffer>,
    count: u32,
    point_capacity: u32,
    point_template: Option<Vec<u32>>,
    point_vertices: u32,
    work_count: u32,
    sources: Vec<String>,
    position_sources: BTreeSet<String>,
    history_ranges: Vec<Vec<(u64, u64, u64)>>,
    id_pipeline: Option<wgpu::ComputePipeline>,
    identity_words: u64,
    id_count: u32,
    historical_attributes: bool,
    status_message: Option<String>,
}
pub(crate) struct Frame {
    pub geometry: EvaluatedGeometry,
    bind: wgpu::BindGroup,
    bounds: wgpu::Buffer,
    published: std::sync::atomic::AtomicBool,
    initial: wgpu::Buffer,
    identity: Option<wgpu::Buffer>,
    history: std::sync::Mutex<std::collections::BTreeMap<[u32;2], Vec<wgpu::Buffer>>>,
}
impl Mesh {
    pub(crate) fn source_names(&self) -> &[String] { &self.sources }
    pub fn new(
        r: &Renderer,
        g: &v2::Geometry,
        draw: &install::Draw,
        names: &[String],
        buffers: &buffers::ProgramBuffers,
        historical_attributes: bool,
    ) -> Result<Self> {
        let points = matches!(g.topology, v2::Topology::Points);
        let identities=points && g.ids.is_some();
        let point_capacity=buffers.bindings[&g.attributes["POSITION"]].count;
        if let Some(ids)=&g.ids {ensure!(buffers.bindings[ids].count==point_capacity && buffers.bindings[ids].width==1,"point ids must match POSITION capacity");}
        let id_slots=if identities {u64::from(point_capacity).checked_next_power_of_two().and_then(|n|n.checked_mul(2)).context("point identity table overflow")?} else {0};
        let pose_base=id_slots*2;
        let identity_words=pose_base+if identities {u64::from(point_capacity)*5} else {0};
        if let Some(count) = &g.count {
            ensure!(
                buffers.bindings[count].count == 1 && buffers.bindings[count].width == 1,
                "geometry.count requires one u32 element"
            );
        }
        if let Some(status) = &g.status {
            let b = &buffers.bindings[&status.buffer];
            ensure!(b.count == 1 && b.width == 1, "geometry.status requires one u32 element");
        }
        let mut sources: Vec<_> = g
            .attributes
            .values()
            .chain(g.widths.iter())
            .chain(g.count.iter())
            .chain(g.ids.iter())
            .chain(g.status.iter().map(|s| &s.buffer))
            .cloned()
            .collect();
        sources.sort();
        sources.dedup();
        let pages: BTreeSet<_> = sources
            .iter()
            .flat_map(|n| buffers.bindings[n].spans.iter().map(|s| s.page))
            .collect();
        let mut inputs: Vec<_> = pages
            .iter()
            .map(|&i| buffers.storage[i as usize].clone())
            .collect();
        let history_sources: BTreeSet<_> = g.attributes.iter()
            .filter(|(name,_)| historical_attributes || name.as_str()=="POSITION" || (!points && matches!(name.as_str(),"NORMAL"|"TANGENT")))
            .map(|(_,buffer)|buffer.clone())
            .chain(g.widths.iter().cloned()).chain(g.count.iter().cloned()).chain(g.ids.iter().cloned()).chain(g.status.iter().map(|s| s.buffer.clone())).collect();
        let history_ranges: Vec<Vec<(u64,u64,u64)>> = pages.iter().map(|page| {
            let mut spans: Vec<_> = history_sources.iter().flat_map(|n| buffers.bindings[n].spans.iter()
                .filter(move |s| s.page == *page).map(move |span| {
                    let first=u64::from(span.offset)*4;
                    (first,first+u64::from(span.count)*u64::from(buffers.bindings[n].width)*4)
                })).collect();
            spans.sort_unstable();
            let mut merged: Vec<(u64,u64)> = Vec::new();
            for (first,end) in spans {
                if let Some(last)=merged.last_mut().filter(|last|first<=last.1) {last.1=last.1.max(end);}
                else {merged.push((first,end));}
            }
            let mut destination=0;
            merged.into_iter().map(|(first,end)| {
                let size=end-first;let range=(first,destination,size);destination+=size;range
            }).collect()
        }).collect();
        let mut code = String::from(crate::compute::FINITE_WGSL);
        for (binding, page) in pages.iter().enumerate() {
            writeln!(
                code,
                "@group(0) @binding({binding}) var<storage,read> page_{page}:array<u32>;"
            )?;
        }
        for (index, name) in sources.iter().enumerate() {
            let b = &buffers.bindings[name];
            writeln!(code, "fn load_{index}(i:u32,c:u32)->u32 {{")?;
            for span in &b.spans {
                writeln!(
                    code,
                    "if(i>={}u && i<{}u) {{return page_{}[{}u+(i-{}u)*{}u+c];}}",
                    span.first,
                    span.first + span.count,
                    span.page,
                    span.offset,
                    span.first,
                    b.width
                )?;
            }
            code.push_str("return 0u; }\n");
        }
        let load = |name: &str, i: &str, c: u32| {
            format!(
                "load_{}({i},{c}u)",
                sources.iter().position(|n| n == name).unwrap()
            )
        };
        let float = |name: &str, i: &str, c: u32| format!("bitcast<f32>({})", load(name, i, c));
        let vector = |name: &str, i: &str| {
            format!(
                "vec3<f32>({},{},{})",
                float(name, i, 0),
                float(name, i, 1),
                float(name, i, 2)
            )
        };
        let count = if points {point_capacity} else {draw.vertex_count};
        let limits = r.device.limits();
        let bytes = u64::from(count) * 80;
        let attributes = u64::from(count) * draw.layout.attribute_stride() * if historical_attributes {2} else {1};
        let limit = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        ensure!(
            bytes <= limit && attributes <= limit,
            crate::gpu_memory::device_limit("computed geometry", format_args!("{bytes} motion bytes and {attributes} attribute bytes"), format_args!("{limit} bytes per binding"), "reduce geometry density or point pool size")
        );
        let topology = draw
            .exit_geometry
            .as_ref()
            .context("computed geometry requires retained source topology")?;
        let point_vertices=if points {(topology.vertices.len()/topology.point_centers.len()) as u32} else {0};
        let point_template=points.then(|| {
            let mut words=vec![point_capacity,point_vertices];words.resize(2+point_capacity as usize*10,0);
            for vertex in &topology.vertices[..point_vertices as usize] {words.extend(vertex.NORMAL.into_iter().chain(vertex.TEXCOORD_0).map(f32::to_bits));}
            words
        });
        if let Some(template)=&point_template {gpu_memory::storage_binding(&limits,"compact point geometry",template.len() as u64*4, "reduce the point pool size")?;}
        // Point normals come from the immutable sphere. They never read triangle
        // adjacency; constructing it would replicate the shape for every point.
        let mut incidents = vec![Vec::new(); if points {0} else {count as usize}];
        if !points {
            for tri in topology.indices.chunks_exact(3) {
                for &v in tri {
                    incidents[v as usize].extend_from_slice(tri);
                }
            }
        }
        let mut adjacency = Vec::with_capacity(incidents.len() + 1);
        let mut offset = incidents.len() as u32 + 1;
        for v in &incidents {
            adjacency.push(offset);
            offset += v.len() as u32;
        }
        adjacency.push(offset);
        adjacency.extend(incidents.into_iter().flatten());
        // Flat-normal corners keep the authored stream row. Store the gather
        // map beside adjacency so all attributes and incident positions use
        // that identity without adding another GPU binding.
        if points {
            adjacency=topology.point_centers.iter().zip(&topology.point_widths).flat_map(|(p,w)|[p[0],p[1],p[2],*w].map(f32::to_bits)).collect();
        }
        let rows = adjacency.len();
        if !points && !topology.vertex_ids.is_empty() {
            ensure!(
                topology.vertex_ids.len() == count as usize,
                "computed vertex identity count mismatch"
            );
            for &id in &topology.vertex_ids {
                let row = u32::try_from(id)
                    .context("computed vertex identity exceeds u32 addressing")?;
                ensure!(
                    row < buffers.bindings[&g.attributes["POSITION"]].count,
                    "computed vertex identity exceeds source capacity"
                );
                adjacency.push(row);
            }
            writeln!(
                code,
                "fn source_row(i:u32)->u32 {{return incidents[{rows}u+i];}}"
            )?;
        } else {
            code.push_str("fn source_row(i:u32)->u32 {return i;}\n");
        }
        inputs.extend([
            draw.vertices.clone(),
            draw.attributes.clone(),
            init(
                &r.device,
                "computed vertex incidents",
                bytemuck::cast_slice(&adjacency),
                wgpu::BufferUsages::STORAGE,
            ),
        ]);
        let base = pages.len();
        if historical_attributes {
            writeln!(code,"@group(0) @binding({}) var<storage,read_write> attribute_history:array<u32>;",base+7+usize::from(identities))?;
        }
        if identities {
            writeln!(code,"@group(0) @binding({}) var<storage,read_write> identity:array<atomic<u32>>;",base+7)?;
            code.push_str("fn point_identity_hash(value:u32)->u32 {var x=value;x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;return x^(x>>16u); }\n");
        }

        if points {
            writeln!(code, "@group(0) @binding({}) var<storage,read_write> point_poses:array<vec4<f32>>;", base+7+usize::from(identities)+usize::from(historical_attributes))?;
        }

        for (i, (access, name, ty)) in [
            ("read", "source", "u32"),
            ("read", "source_attributes", "u32"),
            ("read", "incidents", "u32"),
            ("read_write", "vertices", "u32"),
            ("read_write", "endpoints", "u32"),
            ("read_write", "attributes", "u32"),
            ("read_write", "bounds", "atomic<u32>"),
        ]
        .iter()
        .enumerate()
        {
            writeln!(
                code,
                "@group(0) @binding({}) var<storage,{access}> {name}:array<{ty}>;",
                base + i
            )?;
        }
        let storage_limit=crate::pipelines::scalar_storage_limit(&r.device.limits(),r.adapter_info.backend==wgpu::Backend::Metal,0);
        ensure!(
            base + 7 + usize::from(identities) + usize::from(historical_attributes) + usize::from(points) <= storage_limit as usize,
            crate::gpu_memory::device_limit("computed geometry", format_args!("{} storage bindings",base+7+usize::from(identities)+usize::from(historical_attributes)+usize::from(points)), format_args!("{} storage bindings",storage_limit), "reduce geometry attribute buffer inputs")
        );
        code.push_str("fn ordered(v:f32)->u32 {let b=bitcast<u32>(v);return select(b^0x80000000u,~b,(b&0x80000000u)!=0u);}\n");
        let position = &g.attributes["POSITION"];
        writeln!(
            code,
            "fn position(i:u32)->vec3<f32> {{return {};}}",
            vector(position, "source_row(i)")
        )?;
        let capacity = buffers.bindings[position].count;
        let live = g
            .count
            .as_ref()
            .map_or(format!("{capacity}u"), |n| load(n, "0u", 0));
        let stride = draw.layout.attribute_stride() / 4;
        let work_count=if identities {capacity} else {count};
        let groups = work_count.div_ceil(constants::DEFORM_GROUP);
        let width = groups.min(r.device.limits().max_compute_workgroups_per_dimension);
        writeln!(
            code,
            "fn gather(index:u32,previous:bool) {{ if(index>={work_count}u) {{return;}} let live=min({live},{capacity}u); if(index==0u) {{if({live}>{capacity}u) {{atomicOr(&bounds[6],1u);}} if(previous) {{atomicStore(&bounds[7],live);}} }}"
        )?;
        if let Some(status) = &g.status {
            writeln!(code,"if(index==0u) {{atomicMax(&bounds[8],{});}}",load(&status.buffer,"0u",0))?;
        }
        if !identities {code.push_str("let i=index;\n");}
        if identities {code.push_str("let row=index;\n");}
        else if points {
            code.push_str("let row=i;\n");
        } else {
            code.push_str("let row=source_row(i);\n");
        }
        if let Some(ids)=g.ids.as_ref().filter(|_|identities) {
            let id=load(ids,"row",0);let width=float(g.widths.as_ref().unwrap(),"row",0);
            writeln!(code,"if(previous) {{if(row<live) {{let o={pose_base}u+row*5u;let center=position(row);if(!finite3(center) || !finite_f32({width}) || {width}<0.) {{atomicOr(&bounds[6],2u);}} atomicStore(&identity[o],{id});atomicStore(&identity[o+1u],bitcast<u32>(center.x));atomicStore(&identity[o+2u],bitcast<u32>(center.y));atomicStore(&identity[o+3u],bitcast<u32>(center.z));atomicStore(&identity[o+4u],bitcast<u32>({width}));}} return;}} var prior_center=position(row);var prior_width={width};")?;
            writeln!(code,"if(row<live) {{let id={id};var slot=point_identity_hash(id)&{}u;for(var probe=0u;probe<{id_slots}u;probe++) {{let entry=atomicLoad(&identity[slot]);if(entry==0u) {{break;}} let o={pose_base}u+(entry-1u)*5u;if(atomicLoad(&identity[o])==id) {{prior_center=vec3<f32>(bitcast<f32>(atomicLoad(&identity[o+1u])),bitcast<f32>(atomicLoad(&identity[o+2u])),bitcast<f32>(atomicLoad(&identity[o+3u])));prior_width=bitcast<f32>(atomicLoad(&identity[o+4u]));break;}} slot=(slot+1u)&{}u;}}}}",id_slots-1,id_slots-1)?;
            let other=load(ids,"claim.old_value-1u",0);
            writeln!(code,"if(row<live) {{let id={id};var slot=point_identity_hash(id)&{}u;for(var probe=0u;probe<{id_slots}u;) {{let claim=atomicCompareExchangeWeak(&identity[{id_slots}u+slot],0u,row+1u);if(claim.exchanged) {{break;}} if(claim.old_value==0u) {{continue;}} if({other}==id) {{atomicOr(&bounds[6],4u);break;}} slot=(slot+1u)&{}u;probe++;}}}}",id_slots-1,id_slots-1)?;
        }
        if points {
            let guard="true";
            let diameter=float(g.widths.as_ref().unwrap(), "row", 0);
            writeln!(code,"if({guard}) {{var pose=vec4<f32>(0.); if(row<live) {{let center=position(row);let diameter={diameter};if(finite3(center) && finite_f32(diameter) && diameter>=0.) {{pose=vec4<f32>(center,diameter);}}}} if(previous) {{point_poses[row*2u]=pose;}} else {{point_poses[row*2u+1u]=pose;")?;
            if identities {
                code.push_str("point_poses[row*2u]=select(vec4<f32>(0.),vec4<f32>(prior_center,prior_width),row<live);\n");
            } else {
                code.push_str("if(row>=atomicLoad(&bounds[7])) {point_poses[row*2u]=pose;}\n");
            }
            code.push_str("}}\n");
        }
        if identities {code.push_str("let i=row;\n");}
        code.push_str("var p=vec3<f32>(0.);var n=vec3<f32>(0.,1.,0.);var t=vec4<f32>(0.);\n");
        if points {
            let shape=&topology.vertices[..point_vertices as usize];
            let (lo,hi)=shape.iter().fold((Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY)),|(lo,hi),v|(lo.min(Vec3::from(v.NORMAL)),hi.max(Vec3::from(v.NORMAL))));
            let vector=|v:Vec3|format!("vec3<f32>({:?},{:?},{:?})",v.x,v.y,v.z);
            writeln!(code,"if(row<live) {{let diameter={};let center=position(row);if(!finite_f32(diameter) || diameter<0. || !finite3(center) || !finite3(center+{}*diameter*0.5) || !finite3(center+{}*diameter*0.5)) {{atomicOr(&bounds[6],2u);}} else {{p=center;n=vec3<f32>(diameter,0.,0.);}}}}",float(g.widths.as_ref().unwrap(),"row",0),vector(lo),vector(hi))?;
        } else {
            code.push_str("p=position(i);\n");
            if let Some(normal) = g.attributes.get("NORMAL") {
                writeln!(code, "n={};", vector(normal, "row"))?;
            } else {
                code.push_str("n=vec3<f32>(0.); for(var j=incidents[i];j<incidents[i+1u];j+=3u) {let a=position(incidents[j]);let b=position(incidents[j+1u]);let c=position(incidents[j+2u]);n+=cross(b-a,c-a);} if(dot(n,n)>0.) {n=normalize(n);} else {n=vec3<f32>(0.,1.,0.);}\n");
            }
            if let Some(tangent) = g.attributes.get("TANGENT") {
                writeln!(
                    code,
                    "t=vec4<f32>({},{},{},{});",
                    float(tangent, "row", 0),
                    float(tangent, "row", 1),
                    float(tangent, "row", 2),
                    float(tangent, "row", 3)
                )?;
            }
        }
        code.push_str("if(!finite3(p) || !finite3(n) || !finite4(t)) {atomicOr(&bounds[6],2u);p=vec3<f32>(0.);n=vec3<f32>(0.,1.,0.);t=vec4<f32>(0.);}\n");
        if points {
            code.push_str("if(previous) {point_poses[row*2u]=vec4<f32>(p,n.x);} else {point_poses[row*2u+1u]=vec4<f32>(p,n.x);\n");
            if !identities {code.push_str("if(row>=atomicLoad(&bounds[7])) {point_poses[row*2u]=vec4<f32>(p,n.x);}\n");}
            code.push_str("}\n");
        }
        if identities {code.push_str("var prior=p; if(row<live) {prior=prior_center;}\n");}
        let position_base=if points {2} else {0};
        // Current endpoints occupy the second interleaved record; newly born rows
        // also replace the first, so a dead slot never invents historical motion.
        for (offset, expr) in [
            "p.x", "p.y", "p.z", "n.x", "n.y", "n.z", "t.x", "t.y", "t.z", "t.w",
        ]
        .iter()
        .enumerate()
        {
            writeln!(
                code,
                "if(previous) {{endpoints[i*20u+{offset}u]=bitcast<u32>({expr});}} else {{vertices[{position_base}u+i*10u+{offset}u]=bitcast<u32>({expr});endpoints[i*20u+{}u]=bitcast<u32>({expr});",
                offset + 10
            )?;
            if identities {
                let prior=if offset<3 {format!("prior.{}",["x","y","z"][offset])} else if offset==3 {"prior_width".to_owned()} else {expr.to_string()};
                writeln!(code,"endpoints[i*20u+{offset}u]=bitcast<u32>({prior});")?;
            } else if points {
                writeln!(code,"if(row>=atomicLoad(&bounds[7])) {{endpoints[i*20u+{offset}u]=bitcast<u32>({expr});}}")?;
            }
            code.push_str("}\n");
        }
        let source_attribute=if points {format!("i*{point_vertices}u")} else {"i".to_owned()};
        if historical_attributes {
            writeln!(code,"let attribute_at=i*{stride}u*2u+select({stride}u,0u,previous); for(var c=0u;c<{stride}u;c++) {{let value=source_attributes[({source_attribute})*{stride}u+c];attribute_history[attribute_at+c]=value;if(!previous) {{attributes[i*{stride}u+c]=value;}}}}")?;
        } else {
            writeln!(code,"if(previous) {{return;}} for(var c=0u;c<{stride}u;c++) {{attributes[i*{stride}u+c]=source_attributes[({source_attribute})*{stride}u+c];}}")?;
        }
        if points {
            if historical_attributes {writeln!(code,"for(var c=0u;c<4u;c++) {{let value=incidents[row*4u+c];attribute_history[attribute_at+c]=value;if(!previous) {{attributes[i*{stride}u+c]=value;}}}}")?;}
            else {writeln!(code,"for(var c=0u;c<4u;c++) {{attributes[i*{stride}u+c]=incidents[row*4u+c];}}")?;}
        }
        let mut columns = vec![
            ("TEXCOORD_0".to_owned(), 3u64, 2u32),
            ("TEXCOORD_1".to_owned(), 5, 2),
            ("COLOR_0".to_owned(), 7, 4),
        ];
        columns.extend(
            names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), 11 + i as u64 * 4, 4)),
        );
        for (name, offset, dest_width) in columns {
            if let Some(buffer) = g.attributes.get(&name) {
                let source_width = buffers.bindings[buffer].width;
                for c in 0..dest_width {
                    let value = if c < source_width {
                        load(buffer, "row", c)
                    } else if v2::is_color_attribute(&name) && c == 3 {
                        "0x3f800000u".into()
                    } else {
                        "0u".into()
                    };
                    if historical_attributes {
                        let at=offset+u64::from(c);
                        writeln!(code,"if(row<live) {{let value={value};if(!finite_f32(bitcast<f32>(value))) {{atomicOr(&bounds[6],2u);}} attribute_history[attribute_at+{at}u]=value;if(!previous) {{attributes[i*{stride}u+{at}u]=value;}}}}")?;
                    } else { writeln!(
                        code,
                        "if(row<live) {{ let value={value}; if(!finite_f32(bitcast<f32>(value))) {{atomicOr(&bounds[6],2u);}} attributes[i*{stride}u+{}u]=value; }}",
                        offset + u64::from(c)
                    )?; }
                }
            }
        }
        // The complete tessellated sphere lies inside center +/- radius. Reduce
        // once per point instead of contending on six atomics for every vertex.
        // Keep the per-vertex finite checks above: finite centers and widths can
        // still overflow during reconstruction.
        let reduction = if points {
            let diameter=float(g.widths.as_ref().unwrap(), "row", 0);
            format!("if(row<live) {{let center=position(row);let radius={diameter}*0.5;for(var c=0u;c<3u;c++) {{atomicMin(&bounds[c],ordered(max(center[c]-radius,-3.402823466e+38)));atomicMax(&bounds[c+3u],ordered(min(center[c]+radius,3.402823466e+38)));}}}}")
        } else {
            "if(row<live) {for(var c=0u;c<3u;c++) {atomicMin(&bounds[c],ordered(p[c]));atomicMax(&bounds[c+3u],ordered(p[c]));}}".to_owned()
        };
        let point_loop_end="";
        writeln!(
            code,
            "{reduction} {point_loop_end} }}\n@compute @workgroup_size({}) fn current(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{gather((g.y*{width}u+g.x)*{}u+lane,false);}}\n@compute @workgroup_size({}) fn previous(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{gather((g.y*{width}u+g.x)*{}u+lane,true);}}",
            constants::DEFORM_GROUP,
            constants::DEFORM_GROUP,
            constants::DEFORM_GROUP,
            constants::DEFORM_GROUP
        )?;
        if identities {
            writeln!(code,"@compute @workgroup_size({}) fn map_ids(@builtin(global_invocation_id) g:vec3<u32>) {{let row=g.x;if(row>=atomicLoad(&bounds[7])) {{return;}} let id=atomicLoad(&identity[{pose_base}u+row*5u]);var slot=point_identity_hash(id)&{}u;for(var probe=0u;probe<{id_slots}u;) {{let claim=atomicCompareExchangeWeak(&identity[slot],0u,row+1u);if(claim.exchanged) {{return;}} if(claim.old_value==0u) {{continue;}} if(atomicLoad(&identity[{pose_base}u+(claim.old_value-1u)*5u])==id) {{atomicOr(&bounds[6],4u);return;}} slot=(slot+1u)&{}u;probe++;}} atomicOr(&bounds[6],4u);}}",constants::DEFORM_GROUP,id_slots-1,id_slots-1)?;
        }
        let mut history_code = code.clone();
        for name in &sources {
            for span in &buffers.bindings[name].spans {
                let page = pages.iter().position(|p| *p == span.page).unwrap();
                let old = format!("return page_{}[{}u+",span.page,span.offset);
                if !history_sources.contains(name) {
                    let expression=format!("{old}(i-{}u)*{}u+c];",span.first,buffers.bindings[name].width);
                    history_code=history_code.replace(&expression,"return 0u;");
                    continue;
                }
                let source=u64::from(span.offset)*4;
                let &(first,destination,_) = history_ranges[page].iter().find(|&&(first,_,size)|source>=first && source<first+size).unwrap();
                let new = format!("return historical_page_{}[{}u+",span.page,(destination+source-first)/4);
                history_code = history_code.replace(&old,&new);
            }
        }
        let history_code = history_code.replace("historical_page_", "page_");
        let history_module = crate::timing::shader_module(&r.device, wgpu::ShaderModuleDescriptor {
            label: Some("historical geometry gather"), source: wgpu::ShaderSource::Wgsl(history_code.into()),
        });
        let module = crate::timing::shader_module(&r.device, wgpu::ShaderModuleDescriptor {
            label: Some("typed geometry gather"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        // Both entries use the same explicit layout, including otherwise unused streams.
        let entries: Vec<_> = (0..base + 7 + usize::from(identities) + usize::from(historical_attributes) + usize::from(points))
            .map(|i| {
                pipelines::buffer_entry(
                    i as u32,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage {
                        read_only: i < base + 3,
                    },
                )
            })
            .collect();
        let layout = r
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("typed geometry streams"),
                entries: &entries,
            });
        let pl = r
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("typed geometry"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = ["current", "previous"].map(|entry| {
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("typed geometry gather"),
                layout: Some(&pl),
                module: if entry == "previous" { &history_module } else { &module },
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: r.cache.as_ref(),
            })
        });
        let id_pipeline=identities.then(||r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:Some("point identity correspondence"),layout:Some(&pl),module:&module,entry_point:Some("map_ids"),compilation_options:Default::default(),cache:r.cache.as_ref(),
        }));
        let position_sources = g
            .attributes
            .get("POSITION")
            .into_iter()
            .chain(g.widths.iter())
            .chain(g.count.iter())
            .chain(g.status.iter().map(|s| &s.buffer))
            .cloned()
            .collect();
        Ok(Self {
            historical_attributes,
            status_message: g.status.as_ref().map(|s| s.message.clone()),
            point_capacity: if points {point_capacity} else {0},
            point_template,
            point_vertices,
            pipeline,
            inputs,
            count,
            work_count,
            sources,
            position_sources,
            history_ranges,
            id_pipeline,
            identity_words,
            id_count: if identities {point_capacity} else {0},
        })
    }
}
pub(crate) fn resources(r: &Renderer, s: &InstalledScene) -> Result<Vec<Option<Frame>>> {
    s.computed_geometry
        .iter()
        .enumerate()
        .map(|(i, mesh)| {
            let Some(mesh) = mesh else { return Ok(None) };
            let make = |label, size, usage| {
                gpu_memory::buffer(
                    &r.device,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size,
                        usage,
                        mapped_at_creation: false,
                    },
                )
            };
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC;
            let output = if let Some(template)=&mesh.point_template {
                init(&r.device,"compact point geometry",bytemuck::cast_slice(template),usage|wgpu::BufferUsages::COPY_SRC)
            } else {make("computed vertices",u64::from(mesh.count)*40,usage|rt::geometry_usage(s.rays.is_some()))};
            let previous = make(
                "computed motion endpoints",
                u64::from(mesh.count) * 80,
                usage,
            );
            let attributes = make(
                "computed surface attributes",
                u64::from(mesh.count) * s.draws[i].layout.attribute_stride(),
                usage,
            );
            let point_poses=(mesh.point_capacity>0).then(||make("compact point endpoints",u64::from(mesh.point_capacity)*32,wgpu::BufferUsages::STORAGE));
            let previous_attributes=mesh.historical_attributes.then(||make("computed historical attributes",attributes.size()*2,usage));
            let bounds = make(
                "computed bounds",
                36,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            let initial = init(
                &r.device,
                "empty computed bounds",
                bytemuck::cast_slice(&[u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0, 0, 0]),
                wgpu::BufferUsages::COPY_SRC,
            );
            let identity=(mesh.identity_words>0).then(||make("point identity correspondence",mesh.identity_words*4,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST));
            let buffers: Vec<_> = mesh
                .inputs
                .iter()
                .chain([&output, &previous, &attributes, &bounds])
                .chain(identity.iter())
                .chain(previous_attributes.iter())
                .chain(point_poses.iter())
                .collect();
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| entry(i as u32, b.as_entire_binding()))
                .collect();
            let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("computed geometry snapshot"),
                layout: &mesh.pipeline[0].get_bind_group_layout(0),
                entries: &entries,
            });
            Ok(Some(Frame {
                geometry: EvaluatedGeometry {
                    point_poses,
                    point_domain: (mesh.point_capacity>0).then_some((mesh.point_capacity,mesh.point_vertices)),
                    gpu_bounds: Some(bounds.clone()),
                    output: geometry::VertexStream {
                        size: output.size(),
                        buffer: output,
                        offset: 0,
                    },
                    previous_output: geometry::VertexStream {
                        size: previous.size(),
                        buffer: previous,
                        offset: 0,
                    },
                    attributes: Some(attributes),
                    previous_attributes,
                },
                bind,
                bounds,
                published: std::sync::atomic::AtomicBool::new(false),
                initial,
                identity,
                history: Default::default(),
            }))
        })
        .collect()
}

fn snapshot_sources(r: &Renderer, mesh: &Mesh, frame: &Frame, encoder: &mut wgpu::CommandEncoder, key: [u32;2]) {
    let mut history = frame.history.lock().unwrap();
    if history.contains_key(&key) {return;}
    let snapshots = mesh.history_ranges.iter().enumerate().map(|(page,ranges)| {
        let size=ranges.iter().map(|&(_,_,size)|size).sum::<u64>().max(4);
        let buffer = gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
            label: Some("historical geometry columns"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        for &(source,destination,size) in ranges {
            encoder.copy_buffer_to_buffer(&mesh.inputs[page],source,&buffer,destination,size);
        }
        buffer
    }).collect();
    history.insert(key,snapshots);
}

/// Retain only geometry source columns at the exposure's historical times.
/// The submission owner calls this before any current shutter state is consumed.
pub(crate) fn prepare_history(r: &Renderer, s: &InstalledScene, frames: &[Option<Frame>], times: &[vork_time::Request], checkpoint: &mut Option<compute::Checkpoint>) -> Result<()> {
    if !frames.iter().any(Option::is_some) { return Ok(()); }
    let compute = s.compute.as_ref().context("computed geometry schedule")?;
    let mut cursor = compute.cursor.lock().unwrap();
    for (i,frame) in frames.iter().enumerate() {
        let Some(frame) = frame else {continue};
        let mesh = s.computed_geometry[i].as_ref().unwrap();
        let keys: BTreeSet<_> = times.iter().map(|t| compute.geometry_time_key(&mesh.sources,t)).collect::<Result<_>>()?;
        frame.history.lock().unwrap().retain(|key,_| keys.contains(key));
    }
    let mut times=times.iter().map(|t|Ok((t.exact()?,t))).collect::<Result<Vec<_>>>()?;
    times.sort_by(|a,b|a.0.compare(&b.0));times.dedup_by(|a,b|a.0==b.0);
    let mut encoder = r.device.create_command_encoder(&Default::default());
    for (_,time) in times {
        let missing = frames.iter().enumerate().any(|(i,frame)| frame.as_ref().is_some_and(|frame| {
            let mesh = s.computed_geometry[i].as_ref().unwrap();
            !frame.history.lock().unwrap().contains_key(&compute.geometry_time_key(&mesh.sources,time).unwrap())
        }));
        if !missing {continue;}
        let saved = compute.advance_with(&mut encoder,&mut cursor,time,|encoder| {
            let old = std::mem::replace(encoder,r.device.create_command_encoder(&Default::default()));
            crate::command_finish::Segments::submit(&r.queue,old);
        })?;
        if checkpoint.is_none() { *checkpoint = saved; }
        for (i,frame) in frames.iter().enumerate() {
            let Some(frame) = frame else {continue};
            let mesh = s.computed_geometry[i].as_ref().unwrap();
            let key = compute.geometry_time_key(&mesh.sources,time)?;
            snapshot_sources(r,mesh,frame,&mut encoder,key);

        }
    }
    crate::command_finish::Segments::submit(&r.queue,encoder);
    Ok(())
}

pub(crate) fn evaluate(
    r: &Renderer,
    s: &InstalledScene,
    frames: &[Option<Frame>],
    cached: &[Option<(Vec3, Vec3)>],
    changes: &mut values::ChangeTracker,
    time: &vork_time::Request,
    previous: &vork_time::Request,
    checkpoint: &mut Option<compute::Checkpoint>,
) -> Result<Vec<Option<(Vec3, Vec3)>>> {
    if !frames.iter().any(Option::is_some) {
        return Ok(Vec::new());
    }
    let compute = s
        .compute
        .as_ref()
        .context("computed geometry requires a compute schedule")?;
    let mut cursor = compute.cursor.lock().expect("compute cursor poisoned");
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("computed geometry before visibility"),
        });
    let mut advance =
        |encoder: &mut wgpu::CommandEncoder, cursor: &mut compute::Cursor, t: &vork_time::Request| -> Result<()> {
            // Retain the two generations bracketing each current publication.
            // Previous-frame arithmetic can request the preceding canonical tick
            // at a float boundary. Copies observe only ticks already executing.
            let mut wanted: BTreeSet<_> = cursor.requested.keys().flat_map(|&tick|[tick.saturating_sub(1),tick]).collect();
            for mesh in s.computed_geometry.iter().flatten().filter(|mesh|compute.geometry_uses_simulation_clock(&mesh.sources)) {
                let tick=compute.geometry_time_key(&mesh.sources,t)?[0];
                wanted.extend([tick.saturating_sub(1),tick]);
            }
            let saved = compute.advance_observing(encoder, cursor, t, |encoder| {
                let next = r.device.create_command_encoder(&Default::default());
                let old = std::mem::replace(encoder, next);
                crate::command_finish::Segments::submit(&r.queue, old);
            }, None, |encoder,tick| {
                if !wanted.contains(&tick) {return;}
                for (i,frame) in frames.iter().enumerate() {
                    let Some(frame)=frame else {continue};
                    let mesh=s.computed_geometry[i].as_ref().unwrap();
                    if compute.geometry_uses_simulation_clock(&mesh.sources) {snapshot_sources(r,mesh,frame,encoder,[tick,0]);}
                }
            })?;
            if checkpoint.is_none() {
                *checkpoint = saved;
            }
            Ok(())
        };
    advance(&mut encoder, &mut cursor, time)?;
    for (i,frame) in frames.iter().enumerate() {
        let Some(frame) = frame else {continue};
        let mesh = s.computed_geometry[i].as_ref().unwrap();
        snapshot_sources(r,mesh,frame,&mut encoder,compute.geometry_time_key(&mesh.sources,time)?);
    }
    for (name, version) in &cursor.publications {
        changes.observe_stream(name, *version);
    }
    let mut active = Vec::new();
    let mut receipts=Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let Some(frame) = frame else { continue };
        let mesh = s.computed_geometry[i].as_ref().unwrap();
        let mut key = Vec::new();
        for n in &mesh.sources {
            key.extend(changes.stream_version(n).to_le_bytes());
        }
        if compute.varying(&mesh.position_sources) {
            key.extend(bytemuck::bytes_of(&compute.geometry_time_key(&mesh.sources,time)?));
            key.extend(bytemuck::bytes_of(&compute.geometry_time_key(&mesh.sources,previous)?));
        }
        let dirty =
            changes.changed(values::ChangeKey::ComputedGeometry(i), &key) || !frame.published.load(std::sync::atomic::Ordering::Relaxed);
        let mut geometry_key = Vec::new();
        for n in &mesh.position_sources {
            geometry_key.extend(changes.stream_version(n).to_le_bytes());
        }
        changes.changed(values::ChangeKey::ComputedPositions(i), &geometry_key);
        if dirty {
            active.push(i);
            encoder.copy_buffer_to_buffer(&frame.initial, 0, &frame.bounds, 0, 36);
            if let Some(identity)=&frame.identity {encoder.clear_buffer(identity,0,None);}
        }
    }
    if !active.is_empty() {
        for pose in [1,0] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gather named geometry streams"), timestamp_writes: None,
            });
            for &i in &active {
                let mesh = s.computed_geometry[i].as_ref().unwrap();
                let frame = frames[i].as_ref().unwrap();
                let history = frame.history.lock().unwrap();
                let bind;
                let selected = if pose == 1 {
                    let buffers = history.get(&compute.geometry_time_key(&mesh.sources,previous)?).context("missing planned geometry pose")?;
                    let inputs: Vec<_> = buffers.iter().chain(mesh.inputs[mesh.history_ranges.len()..].iter())
                        .chain([&frame.geometry.output.buffer,&frame.geometry.previous_output.buffer,frame.geometry.attributes.as_ref().unwrap(),&frame.bounds]).chain(frame.identity.iter()).chain(frame.geometry.previous_attributes.iter()).chain(frame.geometry.point_poses.iter()).collect();
                    let entries: Vec<_> = inputs.iter().enumerate().map(|(i,b)| entry(i as u32,b.as_entire_binding())).collect();
                    bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("historical geometry source"), layout: &mesh.pipeline[1].get_bind_group_layout(0), entries: &entries,
                    });
                    &bind
                } else { &frame.bind };
                pass.set_pipeline(&mesh.pipeline[pose]);
                pass.set_bind_group(0, selected, &[]);
                let groups = mesh.work_count.div_ceil(constants::DEFORM_GROUP);
                let width = groups.min(r.device.limits().max_compute_workgroups_per_dimension);
                pass.dispatch_workgroups(width, groups.div_ceil(width), 1);
            }
            drop(pass);
            if pose==1 {
                let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("match point identities"),timestamp_writes:None});
                for &i in &active {
                    let mesh=s.computed_geometry[i].as_ref().unwrap();let Some(pipeline)=&mesh.id_pipeline else {continue};
                    pass.set_pipeline(pipeline);pass.set_bind_group(0,&frames[i].as_ref().unwrap().bind,&[]);
                    pass.dispatch_workgroups(mesh.id_count.div_ceil(constants::DEFORM_GROUP),1,1);
                }
            }
        }
        for &i in &active {
            let f = frames[i].as_ref().unwrap();
            let readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("computed geometry validation receipt"),size:4,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            encoder.copy_buffer_to_buffer(&f.bounds,24,&readback,0,4);
            if let Some(message)=&s.computed_geometry[i].as_ref().unwrap().status_message {
                let receipt=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("solver status receipt"),size:4,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
                encoder.copy_buffer_to_buffer(&f.bounds,32,&receipt,0,4);
                receipts.push((crate::graph::PublicationDiagnostic::SolverStatus(message.clone()),receipt));
            }
            receipts.push((crate::graph::PublicationDiagnostic::ComputedGeometry,readback));
        }
    }
    // Seeking to the previous pose and back may publish fresh generations. Freeze
    // those revisions now so the next same-time sample can retain this snapshot.
    for (name, version) in &cursor.publications {
        changes.observe_stream(name, *version);
    }
    for &i in &active {
        let mesh = s.computed_geometry[i].as_ref().unwrap();
        let mut key = Vec::new();
        for n in &mesh.sources {
            key.extend(changes.stream_version(n).to_le_bytes());
        }
        if compute.varying(&mesh.position_sources) {
            key.extend(bytemuck::bytes_of(&compute.geometry_time_key(&mesh.sources,time)?));
            key.extend(bytemuck::bytes_of(&compute.geometry_time_key(&mesh.sources,previous)?));
        }
        changes.changed(values::ChangeKey::ComputedGeometry(i), &key);
        let mut key = Vec::new();
        for n in &mesh.position_sources {
            key.extend(changes.stream_version(n).to_le_bytes());
        }
        changes.changed(values::ChangeKey::ComputedPositions(i), &key);
    }
    let submission = crate::command_finish::Segments::submit(&r.queue, encoder);
    drop(cursor);
    r.errors.check()?;
    if !receipts.is_empty() {
        s.transients.publication_checks.lock().unwrap().push((submission,receipts));
    }
    for &i in &active {frames[i].as_ref().unwrap().published.store(true,std::sync::atomic::Ordering::Relaxed);}
    // Unknown CPU bounds disable camera rejection. Shading consumes the reduced
    // GPU bounds below, while containment conservatively retains unknown solids.
    Ok(vec![None;cached.len()])
}

pub(crate) fn validation_message(flags:u32)->&'static str {
    if flags&1!=0 {"live count exceeds installed capacity"}
    else if flags&4!=0 {"duplicate point identities in the live prefix"}
    else {"nonfinite attribute or negative/nonfinite width"}
}

/// Publish reduced bounds after the sample's material uploads. Rest/Generated
/// bounds remain untouched; only the evaluated-bound fields change.
pub(crate) struct BoundsPublication(Vec<(wgpu::ComputePipeline,wgpu::BindGroup,u32,u32)>);
impl BoundsPublication {
    pub(crate) fn new(r:&Renderer,s:&InstalledScene,outputs:&[Option<EvaluatedGeometry>],materials:Option<&materials::FrameMaterials>)->Self {
        let mut jobs=Vec::new();
        for (draw,output) in s.draws.iter().zip(outputs) {
            let Some(bounds)=output.as_ref().and_then(|o|o.gpu_bounds.as_ref()) else {continue};
            let records=&materials.expect("computed geometry material rows").records;
            let addresses:Vec<u32>=draw.instances.clone().map(|object|s.objects[object as usize].metadata[2]+6).collect();
            let count=addresses.len() as u32;
            if count==0 {continue;}
            let width=count.div_ceil(64).min(r.device.limits().max_compute_workgroups_per_dimension);
            let code=r#"
@group(0) @binding(0) var<storage,read> bounds:array<u32>;
@group(0) @binding(1) var<storage,read> addresses:array<u32>;
@group(0) @binding(2) var<storage,read_write> materials:array<u32>;
fn decode(v:u32)->u32 {return select(~v,v^0x80000000u,(v&0x80000000u)!=0u);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let index=gid.x+gid.y*DISPATCH_WIDTH;
    if(index>=arrayLength(&addresses)) {return;}
    let base=addresses[index];
    for(var c=0u;c<6u;c++) {materials[base+c]=select(decode(bounds[c]),0u,bounds[0]==0xffffffffu);}
    materials[base+6u]=bitcast<u32>(1.);
}
"#.replace("DISPATCH_WIDTH",&format!("{}u",width*64));
            let pipeline=crate::compute::pipeline_layout(r,&code,"computed bound publication",&[Some(true),Some(true),Some(false)],"main");
            let addresses=init(&r.device,"computed bound destinations",bytemuck::cast_slice(&addresses),wgpu::BufferUsages::STORAGE);
            let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("computed bound publication"),layout:&pipeline.get_bind_group_layout(0),entries:&[
                entry(0,bounds.as_entire_binding()),entry(1,addresses.as_entire_binding()),entry(2,records.as_entire_binding())]});
            jobs.push((pipeline,bind,count,width));
        }
        Self(jobs)
    }
    pub(crate) fn encode(&self,encoder:&mut wgpu::CommandEncoder) {
        if self.0.is_empty() {return;}
        let mut pass=encoder.begin_compute_pass(&Default::default());
        for (pipeline,bind,count,width) in &self.0 {pass.set_pipeline(pipeline);pass.set_bind_group(0,bind,&[]);pass.dispatch_workgroups(*width,count.div_ceil(64).div_ceil(*width),1);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    fn scene(points: bool) -> Value {
        let luminance = if points {
            json!({"attribute":"heat"})
        } else {
            json!({"attribute":"TEXCOORD_0","channel":"r"})
        };
        let g = if points {
            json!({"topology":"points","attributes":{"POSITION":"p","COLOR_0":"color","heat":"heat"},"widths":"width","count":"live"})
        } else {
            json!({"attributes":{"POSITION":"p","NORMAL":"normal","TEXCOORD_0":"uv","COLOR_0":"color","heat":"heat"}})
        };
        json!({"version":2,"render":{"resolution":[40,32],"fps":4,"duration":2,"background":[0.02,0.03,0.04]},
            "buffers":{"p":{"type":"vec3","data":[[-8,0,0],[-7,0,0],[-8,1,0]]},
                "normal":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1]]},
                "uv":{"type":"vec2","data":[[0,0],[1,0],[0,1]]},
                "color":{"type":"vec3","data":[[1,1,1],[1,1,1],[1,1,1]]},
                "heat":{"type":"f32","data":[0.5,0.5,0.5]},
                "width":{"type":"f32","data":[0.5,0.5,0.5]},"live":{"type":"u32","data":[3]}},
            "geometry":{"g":g},"materials":{"m":{"double_sided":true,"inputs":{"base_color":{"attribute":"COLOR_0"},"emission_color":{"attribute":"COLOR_0"},"emission_luminance":luminance,"specular_roughness":{"attribute":"heat"}}}},
            "programs":{"move":{"schedule":"frame","dispatch":3,"bindings":{"p":"p","color":"color","uv":"uv","heat":"heat","width":"width","live":"live"},"source":"struct Inputs {}; void compute(ComputeContext ctx, Inputs inputs) {uint i=ctx.id.x;float3 value=i==0u ? float3(-0.8,-0.6,0) : (i==1u ? float3(0.8,-0.6,0) : float3(0,0.8,0));p_store(i,value+float3(ctx.time*0.4,0,0));color_store(i,float3(0.2,0.5,0.8));uv_store(i,float2(0.25,0.75));heat_store(i,0.25);if(i==0u) live_store(0u,ctx.time>=1. ? 0u : (ctx.time>=0.5 ? 1u : 3u));}"}},
            "nodes":[{"type":"mesh","name":"moving","geometry":"g","materials":["m"]},{"type":"camera","name":"camera","translation":[0,0,4]}]})
    }
    fn oracle(mut value: Value, points: bool, time: f32) -> Value {
        value.as_object_mut().unwrap().remove("programs");
        // Unreferenced stream geometry must not require an installed schedule.
        value["geometry"]["unused"] = json!({"topology":"points","attributes":{"POSITION":"p"},"widths":"width","count":"live"});
        value["buffers"]["p"]["data"] = json!([
            [-0.8 + time * 0.4, -0.6, 0.],
            [0.8 + time * 0.4, -0.6, 0.],
            [time * 0.4, 0.8, 0.]
        ]);
        value["buffers"]["color"]["data"] = json!(vec![[0.2, 0.5, 0.8]; 3]);
        value["buffers"]["uv"]["data"] = json!(vec![[0.25, 0.75]; 3]);
        value["buffers"]["heat"]["data"] = json!([0.25, 0.25, 0.25]);
        if points {
            value["geometry"]["g"]
                .as_object_mut()
                .unwrap()
                .remove("count");
            value["buffers"]["width"]["data"] = json!(if time >= 1. {
                [0., 0., 0.]
            } else if time >= 0.5 {
                [0.5, 0., 0.]
            } else {
                [0.5, 0.5, 0.5]
            });
        }
        value
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn computed_mesh_and_live_points_match_authored_raster_and_rays() -> Result<()> {
        let r = crate::test_renderer()?;
        let rays = r
            .device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES);
        for settings in [
            json!({"integrator":"raster","samples":2,"shadow_method":"ray","reflections":"environment","fast_gi":null}),
            json!({"integrator":"raster","samples":2,"shadow_method":"ray","reflections":"ray","fast_gi":{"method":"ray"}}),
            json!({"integrator":"path","samples":4,"denoiser":null}),
        ] {
            if (settings["integrator"] == "path" || settings["shadow_method"] == "ray") && !rays {
                continue;
            }
            for points in [false, true] {
                let mut value = scene(points);
                v2::replace_integrator(&mut value, settings.clone());
                let installed =
                    r.install(serde_json::from_value(value.clone())?, Path::new("."))?;
                let mut slot = installed.create_frame_slot(&r, Default::default())?;
                if points {
                    assert_eq!(slot.deforms[0].as_ref().unwrap().point_poses.as_ref().unwrap().size(),3*32,
                        "point endpoints scale with source capacity, not sphere tessellation");
                    let geometry=slot.deforms[0].as_ref().unwrap();
                    let (capacity,shape_vertices)=geometry.point_domain.unwrap();
                    assert_eq!(geometry.output.size,(2+u64::from(capacity)*10+u64::from(shape_vertices)*5)*4);
                    assert_eq!(geometry.previous_output.size,u64::from(capacity)*80);
                    assert_eq!(installed.rays.as_ref().unwrap().instances.len(),3,
                        "each source point needs a placement of the shared shape");
                }
                // Computed-only geometry must complete with diagnostic query
                // resolves enabled, including retained and backward requests.
                slot.profile = true;
                for time in [0., 0.5, 1., 0.25, 0.] {
                    let expected = r.install(
                        serde_json::from_value(oracle(value.clone(), points, time))?,
                        Path::new("."),
                    )?;
                    if points {if let Some(rays)=&expected.rays {assert_eq!(rays.instances.len(),3,"authored points must retain shared-shape placements");}}
                    let mut reference = expected.create_frame_slot(&r, Default::default())?;
                    slot.submit(&installed, &r, time)?;
                    reference.submit(&expected, &r, time)?;
                    let actual = slot.collect(&r)?;
                    let expected = reference.collect(&r)?;
                    let error = actual
                        .rgba
                        .iter()
                        .zip(&expected.rgba)
                        .map(|(a, b)| a.abs_diff(*b) as u64)
                        .sum::<u64>() as f64
                        / actual.rgba.len() as f64;
                    ensure!(
                        error < 1.5,
                        "computed {points} at {time}, {settings}: mean byte error {error}"
                    );
                    assert!(slot.geometry_bounds[0].is_none(), "computed bounds must remain GPU-resident");
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn simulation_geometry_holds_subframes_and_replays() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut value = scene(false);
        value["programs"]["move"]["schedule"] = json!("simulation");
        value["programs"]["initialize"] = value["programs"]["move"].clone();
        value["simulation"] = json!({"t0":0,"dt":0.125,"seed":0,"state":["p","color","uv","heat","width","live"],"init":["initialize"],"steps":["move"]});
        v2::replace_integrator(
            &mut value,
            json!({"integrator":"raster","samples":1,"fast_gi":null,"reflections":"environment"}),
        );
        let s = r.install(serde_json::from_value(value.clone())?, Path::new("."))?;
        let mut f = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        for time in [0.25f32, 0.24, 0.375, 0., 0.25] {
            let state_time = ((time / 0.125).floor() - 1.).max(0.) * 0.125;
            let mut authored = oracle(value.clone(), false, state_time);
            authored.as_object_mut().unwrap().remove("simulation");
            let expected = r.install(serde_json::from_value(authored)?, Path::new("."))?;
            let mut reference = expected.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
            f.submit(&s, &r, time)?;
            reference.submit(&expected, &r, time)?;
            crate::test_linear::assert_linear_eq(f.collect(&r)?.linear_rgba.as_ref().unwrap(),reference.collect(&r)?.linear_rgba.as_ref().unwrap());
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn point_identities_preserve_motion_through_death_compaction_and_rebirth() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut value=scene(true);
        value["buffers"]["identity"]=json!({"type":"u32","data":[10,20,30]});
        value["geometry"]["g"]["ids"]=json!("identity");
        value["programs"]["move"]["bindings"]["identity"]=json!("identity");
        value["programs"]["move"]["source"]=json!(r#"
struct Inputs {};
void compute(ComputeContext ctx,Inputs inputs) {
 uint i=ctx.id.x;float t=ctx.time;uint id;
 if(t<.125) id=i==0u ? 10u : (i==1u ? 20u : 30u);
 else if(t<.375) id=i==0u ? 20u : 30u;
 else id=i==0u ? (t<.625 ? 40u : 50u) : (i==1u ? 20u : 30u);
 float3 p=id==20u ? float3(.65,-.5,0.) : (id==30u ? float3(0.,.6,0.) : (id==50u ? float3(0.,-.5,0.) : float3(-.65,-.5,0.)));
 p_store(i,p);identity_store(i,id);
 if(i==0u) live_store(0u,t>=.125 && t<.375 ? 2u : 3u);
}
"#);
        v2::replace_integrator(&mut value,json!({"integrator":"raster","samples":1,"fast_gi":null,"reflections":"environment"}));
        let s=r.install(serde_json::from_value(value.clone())?,Path::new("."))?;
        let mut frame=s.create_frame_slot(&r,ReadbackPrecision::Float)?;
        for time in [0.,0.25,0.5,0.75] {
            frame.submit_outputs(&s,&r,time,ReadbackOutputs {float:FloatOutputs {color:true,motion:true,..Default::default()},..Default::default()})?;
            let output=frame.collect(&r)?;
            assert!(output.linear_rgba.as_ref().unwrap().iter().any(|p|p[0]>0.1),"no point reached the motion oracle");
            assert!(output.motion_f32.unwrap().iter().flatten().all(|v|v.abs()<1e-4),"stationary survivor or newborn acquired motion at {time}");
        }
        // Duplicate IDs in the current prefix must also reject, even when the
        // previous prefix was valid and every point has finite geometry.
        value["programs"]["move"]["source"]=value["programs"]["move"]["source"].as_str().unwrap().replace("identity_store(i,id);","identity_store(i,ctx.time>=.125 ? 20u : id);").into();
        let s=r.install(serde_json::from_value(value)?,Path::new("."))?;
        let mut frame=s.create_frame_slot(&r,Default::default())?;
        frame.submit(&s,&r,0.25)?;
        let error=frame.collect(&r).err().context("duplicate current point identities were admitted")?;
        assert!(format!("{error:#}").contains("duplicate point identities"));
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn historical_columns_exclude_scratch_and_do_not_replay_previous_poses() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut value=scene(false);
        value["buffers"]["padding"]=json!({"type":"u32","dims":[65536,1,1]});
        value["programs"]["move"]["bindings"]["padding"]=json!("padding");
        value["programs"]["move"]["schedule"]=json!("simulation");
        value["programs"]["initialize"]=value["programs"]["move"].clone();
        value["simulation"]=json!({"t0":0,"dt":0.125,"seed":0,"state":["p","color","uv","heat","width","live","padding"],"init":["initialize"],"steps":["move"]});
        v2::replace_integrator(&mut value,json!({"integrator":"raster","samples":1,"fast_gi":null,"reflections":"environment"}));
        let s=r.install(serde_json::from_value(value)?,Path::new("."))?;
        let frames=resources(&r,&s)?;let mut checkpoint=None;
        prepare_history(&r,&s,&frames,&[0_f32.into(),0.125_f32.into(),0.25_f32.into()],&mut checkpoint)?;
        for mesh in s.computed_geometry.iter().flatten() {
            assert_eq!(mesh.history_ranges.iter().flatten().map(|&(_,_,size)|size).sum::<u64>(),72,"history should retain only three positions and normals");
        }
        for frame in frames.iter().flatten() {
            let history=frame.history.lock().unwrap();
            assert_eq!(history.len(),3);
            assert!(history.values().all(|pages|pages.iter().map(wgpu::Buffer::size).sum::<u64>()<1024),"unrelated scratch leaked into geometry snapshots");
        }
        let compute=s.compute.as_ref().unwrap();
        let before=compute.cursor.lock().unwrap().replay_ticks;
        let mut changes=values::ChangeTracker::default();
        let mut cached=vec![None;frames.len()];
        for (time,previous) in [(0.375,0.),(0.5,0.125),(0.625,0.25)] {
            cached=evaluate(&r,&s,&frames,&cached,&mut changes,&vork_time::Request::Numeric(time),&vork_time::Request::Numeric(previous),&mut checkpoint)?;
        }
        assert_eq!(compute.cursor.lock().unwrap().replay_ticks-before,3,"historical gathers replayed the simulation");
        evaluate(&r,&s,&frames,&cached,&mut changes,&vork_time::Request::Numeric(1.),&vork_time::Request::Numeric(0.25),&mut checkpoint)?;
        let crossed=compute.cursor.lock().unwrap().replay_ticks;
        prepare_history(&r,&s,&frames,&[0.875_f32.into()],&mut checkpoint)?;
        assert_eq!(compute.cursor.lock().unwrap().replay_ticks,crossed,"the preceding published generation was replayed");
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn shading_only_stream_updates_retain_acceleration_structures() -> Result<()> {
        let r = crate::test_renderer()?;
        if !r
            .device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES)
        {
            return Ok(());
        }
        let mut value = scene(false);
        value["programs"]["move"]["bindings"] = json!({"color":"color"});
        value["programs"]["move"]["source"] = json!(
            "struct Inputs {}; void compute(ComputeContext ctx, Inputs inputs) {color_store(ctx.id.x,float3(ctx.time,0.2,0.5));}"
        );
        v2::replace_integrator(
            &mut value,
            json!({"integrator":"path","samples":1,"denoiser":null}),
        );
        let s = r.install(serde_json::from_value(value)?, Path::new("."))?;
        let mut f = s.create_frame_slot(&r, Default::default())?;
        f.submit(&s, &r, 0.)?;
        f.collect(&r)?;
        let first = f.rays.as_ref().unwrap().builds;
        f.submit(&s, &r, 0.5)?;
        f.collect(&r)?;
        ensure!(
            f.rays.as_ref().unwrap().builds == first,
            "shading-only stream rebuilt AS"
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn rejected_geometry_requires_reinstall_before_recovery() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut value = scene(true);
        let source = value["programs"]["move"]["source"]
            .as_str()
            .unwrap()
            .to_owned();
        value["programs"]["move"]["source"]=source.replace("p_store(i,value+float3(ctx.time*0.4,0,0));", "p_store(i,ctx.frame==1u ? float3(asfloat(0x7f800000u),0,0) : value+float3(ctx.time*0.4,0,0));").replace("if(i==0u) live_store(0u,ctx.time>=1. ? 0u : (ctx.time>=0.5 ? 1u : 3u));", "if(i==0u) live_store(0u,3u);").into();
        value["animation"] = json!([{"tracks":[{"path":["materials","m","inputs","emission_luminance"],"keys":[[0,0.1],[0.25,0.5],[1,0.5]]}]}]);
        v2::replace_integrator(
            &mut value,
            json!({"integrator":"raster","samples":1,"fast_gi":null,"reflections":"environment"}),
        );
        let s = r.install(serde_json::from_value(value.clone())?, Path::new("."))?;
        let mut reused = s.create_frame_slot(&r, Default::default())?;
        reused.submit(&s, &r, 0.25)?;
        let error = reused.collect(&r).err().context("invalid geometry was delivered")?;
        ensure!(format!("{error:#}").contains("nonfinite"), "invalid geometry was admitted: {error:#}");
        // Recording may already have submitted canonical state or shared view
        // prefixes. Recovery belongs to the installation, not one slot.
        for slot in [&mut reused, &mut s.create_frame_slot(&r, Default::default())?] {
            let error = slot.submit(&s, &r, 0.75).unwrap_err();
            ensure!(format!("{error:#}").contains("reinstall"), "{error:#}");
        }
        let recovered = r.install(serde_json::from_value(value.clone())?, Path::new("."))?;
        let mut fresh = recovered.create_frame_slot(&r, Default::default())?;
        fresh.submit(&recovered, &r, 0.75)?;
        let actual = fresh.collect(&r)?;
        let mut authored = oracle(value, true, 0.75);
        // This producer keeps all three points live throughout the witness.
        authored["buffers"]["width"]["data"] = json!([0.5, 0.5, 0.5]);
        let expected = r.install(serde_json::from_value(authored)?, Path::new("."))?;
        let mut reference = expected.create_frame_slot(&r, Default::default())?;
        reference.submit(&expected, &r, 0.75)?;
        let expected = reference.collect(&r)?;
        let error = actual.rgba.iter().zip(&expected.rgba)
            .map(|(a,b)| a.abs_diff(*b) as u64).sum::<u64>() as f64 / actual.rgba.len() as f64;
        ensure!(error < 1.5, "reinstalled geometry differs from authored reference: mean byte error {error}");
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn solver_status_reports_authored_failure_without_nonfinite_geometry() -> Result<()> {
        let r = crate::test_renderer()?;
        for points in [false,true] {
            let mut value = scene(points);
            if points {value["buffers"]["live"]["data"]=json!([0]);}
            value["buffers"]["status"] = json!({"type":"u32","data":[0]});
            value["geometry"]["g"]["status"] = json!({"buffer":"status","message":"cloth node shirt: self-contact capacity 512; reduce collision_margin"});
            value["programs"]["move"]["bindings"]["status"] = json!("status");
            value["programs"]["move"]["source"] = json!("struct Inputs {}; void compute(ComputeContext ctx, Inputs inputs) {if(ctx.id.x==0u && ctx.time>=0.5) status_store(0u,513u);}");
            let s = r.install(v2::from_test_value(value.clone())?, Path::new("."))?;
            let mut frame = s.create_frame_slot(&r, Default::default())?;
            frame.submit(&s, &r, 0.)?; frame.collect(&r)?;
            for time in [0.5,0.75] {
                let error = frame.submit(&s, &r, time).and_then(|()|frame.collect(&r).map(|_|())).unwrap_err();
                if time==0.5 {ensure!(crate::failure::exit_code(&error)==3,"solver failure must remain a scene error");}
                let message = format!("{error:#}");
                for part in ["cloth node shirt", "self-contact capacity 512", "513", "reduce collision_margin"] {
                    ensure!(message.contains(part), "{message}");
                }
                ensure!(!message.contains("nonfinite"), "{message}");
            }
            // Failed recording requires reinstall; initial state must clear status.
            let reset = r.install(v2::from_test_value(value)?, Path::new("."))?;
            let mut frame = reset.create_frame_slot(&r, Default::default())?;
            frame.submit(&reset, &r, 0.)?; frame.collect(&r)?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn point_overflow_and_bad_width_refuse_output() -> Result<()> {
        let r = crate::test_renderer()?;
        for (source, message) in [
            ("if(i==0u) live_store(0u,4u);", "capacity"),
            ("if(i==0u) live_store(0u,0xffffffffu);", "capacity"),
            ("width_store(i,-1.);", "width"),
            ("p_store(i,float3(asfloat(0x7f800000u),0,0));", "nonfinite"),
        ] {
            let mut value = scene(true);
            value["programs"]["move"]["source"] = format!(
                "struct Inputs {{}}; void compute(ComputeContext ctx, Inputs inputs) {{uint i=ctx.id.x;{source}}}"
            )
            .into();
            let s = r.install(serde_json::from_value(value)?, Path::new("."))?;
            let mut f = s.create_frame_slot(&r, Default::default())?;
            for _ in 0..2 {
                let err = match f.submit(&s, &r, 0.) {Ok(())=>f.collect(&r).err().context("invalid geometry was delivered")?,Err(err)=>err};
                ensure!(format!("{err:#}").contains(message), "{err:#}");
            }
        }
        Ok(())
    }
}
