use super::*;
#[test]
#[ignore = "requires real GPU and pinned Slang"]
fn cloth_shared_rim_patch_freefalls_without_false_face_contact() -> Result<()> {
    let r=Renderer::new()?;
    let mut source=v2::Scene::from_json(include_bytes!("../tests/fixtures/cloth/rim-patch.json"))?;
    source.render.settings.samples=4;
    let base=std::env::temp_dir().join(format!("vork-rim-patch-{}",std::process::id()));
    std::fs::create_dir_all(&base)?;
    std::fs::write(base.join("rim-patch.bin"),include_bytes!("../tests/fixtures/cloth/rim-patch.bin"))?;
    let s=r.install(source,&base)?;
    let compute=s.compute.as_ref().unwrap();
    let scope=&compute.geometry_scopes[0];
    let prefix=scope.buffers.bindings.keys().find_map(|s|s.strip_suffix("rest_position")).unwrap();
    let column=|name:&str| -> Result<Vec<f32>> {
        let b=&scope.buffers.bindings[&format!("{prefix}{name}")];
        let mut result=Vec::new();
        for span in &b.spans {result.extend(read_words(&r,&compute.pages[scope.pages[span.page as usize]],u64::from(span.offset)*4,(span.count*b.width) as usize)?.into_iter().map(f32::from_bits));}
        Ok(result)
    };
    let rest=column("rest_position")?;
    let mut frame=s.create_frame_slot(&r,Default::default())?;
    for tick in [0,1] {
        frame.submit_outputs(&s,&r,vork_time::Request::Frame(tick,vork_time::Time::integer(480)),ReadbackOutputs {color:true,..Default::default()})?;
        let pixels=frame.collect(&r)?;
        if let Some(out)=std::env::var_os("CLOTH_OUT") {
            let out=std::path::PathBuf::from(out);std::fs::create_dir_all(&out)?;
            image::save_buffer(out.join(format!("rim-patch-{tick}.png")),&pixels.rgba,320,240,image::ColorType::Rgba8)?;
        }
        let actual=column("p")?;
        for (j,(&p,&initial)) in actual.iter().zip(&rest).enumerate() {
            let expected=initial-if j%3==1 {tick as f32*9.81/(480.*480.)} else {0.};
            ensure!((p-expected).abs()<2e-6,"rim patch tick {tick}, component {j}: {p} != {expected}");
        }
    }
    Ok(())
}

fn read_words(
    r: &Renderer,
    buffer: &wgpu::Buffer,
    offset: u64,
    count: usize,
) -> Result<Vec<u32>> {
    // GPU-produced counts can be invalid (including UINT_MAX sentinels).
    // Validate the source range before allocating or recording a readback.
    let bytes = u64::try_from(count)?.checked_mul(4).context("compute readback size overflow")?;
    ensure!(offset % wgpu::COPY_BUFFER_ALIGNMENT == 0, "compute readback offset is unaligned");
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.size()),
        "compute readback range exceeds source buffer: offset={offset}, words={count}, size={}", buffer.size());
    if bytes == 0 { return Ok(Vec::new()); }
    let read = crate::gpu_memory::buffer(
        &r.device,
        &wgpu::BufferDescriptor {
            label: Some("compute test readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, offset, &read, 0, bytes);
    let submission = r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    r.device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    rx.recv()??;
    let result = bytemuck::cast_slice(&read.slice(..).get_mapped_range()).to_vec();
    read.unmap();
    Ok(result)
}

// Keep the numeric contracts on the actual shipped predicates, including their
// zero-scale branches, instead of maintaining a second solver in Rust.
fn cloth_function(library: &str, name: &str) -> String {
    let start = library.find(&format!(" {name}(")).unwrap();
    let start = library[..start].rfind('\n').map_or(0, |i| i + 1);
    let open = start + library[start..].find('{').unwrap();
    let mut depth = 0;
    for (i, c) in library[open..].char_indices() {
        match c { '{' => depth += 1, '}' => depth -= 1, _ => {} }
        if depth == 0 { return library[start..=open+i].to_owned(); }
    }
    panic!("unterminated cloth function {name}");
}
fn cloth_numeric_contract(functions: &[(&str, &str)], setup: &str, checks: &str) -> Result<()> {
    let r = crate::test_renderer()?;
    let mut source = setup.to_owned();
    for &(library, name) in functions { source.push_str(&cloth_function(library, name)); }
    source.push_str(&format!(r#"
        struct Inputs {{}};
        void compute(ComputeContext ctx, Inputs inputs) {{
            for(uint i=0;i<6;i++) {{
                bool ok=true;
                float scale=i==0?2e-6:(i==1?1e-5:(i==2?8e-5:(i==3?.01:(i==4?1.:1000.))));
                {checks}
                result_store(i,ok?1u:0u);
            }}
        }}"#));
    let scene: v2::Scene = serde_json::from_value(serde_json::json!({
        "version":2,"nodes":[],"buffers":{"result":{"type":"u32","data":[0,0,0,0,0,0]}},
        "programs":{"probe":{"source":source,"schedule":"install","dispatch":1,"bindings":{"result":"result"}}}
    }))?;
    let decoded = buffers::SceneBuffers::decode(&scene, Path::new("."))?;
    let attrs = ProgramBuffers::install(&r, &scene, &decoded)?;
    let _compute = Compute::test_install(&r, &scene, Path::new("."), Some(&attrs), &[], &[], &[], vork_time::Time::integer(24), 1.)?;
    let span = &attrs.bindings["result"].spans[0];
    assert_eq!(read_words(&r, &attrs.storage[span.page as usize], u64::from(span.offset)*4, 6)?, [1;6], "cloth numeric contract failed");
    Ok(())
}
const CLOTH_SOLVER: &str = include_str!("../../../libraries/cloth/solver.slang");
const CLOTH_SELF: &str = include_str!("../../../libraries/cloth/self.slang");
const CLOTH_BODY: &str = include_str!("../../../libraries/cloth/body.slang");

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_unit_handles_lengths_areas_and_zero() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_unit")], "", r#"
        float3 direction=normalize(float3(1,2,-3));
        ok=ok&&length(cloth_unit(scale*direction)-direction)<1e-6;
        ok=ok&&length(cloth_unit(scale*scale*direction)-direction)<1e-6;
        ok=ok&&all(cloth_unit(float3(0))==float3(0,1,0));
        ok=ok&&length(cloth_unit(1e-30*direction)-direction)<1e-6;
        ok=ok&&length(cloth_unit(1e30*direction)-direction)<1e-6;
    "#)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_closest_preserves_regions_and_collapsed_triangles_across_units() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_closest")], "", r#"
        float3 a=0,b=float3(1,0,0)*scale,c=float3(0,1,0)*scale,w;
        float3 q=cloth_closest(float3(.2,.3,1)*scale,a,b,c,w);
        ok=ok&&length(q/scale-float3(.2,.3,0))<1e-5&&length(w-float3(.5,.2,.3))<1e-5;
        q=cloth_closest(float3(.5,-1,0)*scale,a,b,c,w);
        ok=ok&&length(q/scale-float3(.5,0,0))<1e-5&&w.z==0;
        q=cloth_closest(float3(.2,.3,1)*scale,a,a,c,w);
        ok=ok&&length(q/scale-float3(0,.3,0))<1e-5;
        q=cloth_closest(b,a,a,a,w);ok=ok&&all(q==a)&&all(w==float3(1,0,0));
    "#)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_body_cross_preserves_intersection_parallel_and_zero_across_units() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_BODY,"cloth_body_cross")], "", r#"
        float3 a=0,b=float3(1,0,0)*scale,c=float3(0,1,0)*scale,w;float t;
        bool hit=cloth_body_cross(float3(.2,.3,-1)*scale,float3(.2,.3,1)*scale,a,b,c,t,w);
        ok=ok&&hit&&abs(t-.5)<1e-5&&length(w-float3(.5,.2,.3))<1e-5;
        ok=ok&&!cloth_body_cross(float3(2,2,-1)*scale,float3(2,2,1)*scale,a,b,c,t,w);
        ok=ok&&!cloth_body_cross(float3(.2,.3,1)*scale,float3(.8,.3,1)*scale,a,b,c,t,w);
        ok=ok&&!cloth_body_cross(a,a,a,b,c,t,w);
        ok=ok&&!cloth_body_cross(-c,c,a,b,b,t,w);
        // Keep an accepted shallow crossing representable at small scales;
        // squaring the cubic determinant would flush this case to zero.
        hit=cloth_body_cross(float3(.2,.3,-1e-6)*scale,float3(.8,.3,1e-6)*scale,a,b,c,t,w);
        ok=ok&&hit&&abs(t-.5)<1e-5&&length(w-float3(.2,.5,.3))<1e-5;
        ok=ok&&!cloth_body_cross(float3(.2,.3,-.5e-7)*scale,float3(.8,.3,.5e-7)*scale,a,b,c,t,w);
    "#)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_self_normal_preserves_crossings_and_gap_direction_across_units() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_unit"),(CLOTH_SELF,"cloth_self_normal")],
        "groupshared float SELF_MARGIN;", r#"
        SELF_MARGIN=.03*scale;float distance;
        float3 delta=float3(0,0,-.001)*scale,old_delta=-delta,area=float3(0,0,1)*scale*scale;
        float3 n=cloth_self_normal(delta,old_delta,area,true,distance);
        ok=ok&&n.z>.999&&abs(distance/scale+.001)<1e-6;
        n=cloth_self_normal(delta,old_delta,area,false,distance);
        ok=ok&&n.z<-.999&&distance>0;
        n=cloth_self_normal(float3(0),old_delta,float3(0),true,distance);
        ok=ok&&n.z>.999&&distance==0;
    "#)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_face_region_tolerance_tracks_thickness_across_units() -> Result<()> {
    // Execute the shipped face-region expression, with projected points just
    // inside, at the boundary and well outside a finite triangular face.
    let epsilon = CLOTH_BODY.lines().find(|s| s.contains("float face_epsilon=")).unwrap();
    let predicate = CLOTH_BODY.lines().find(|s| s.contains("bool face_region=")).unwrap();
    let checks = format!(r#"
        THICKNESS=.01*scale;
        float3 a=0,b=float3(1,0,0)*scale,c=float3(0,1,0)*scale,w;
        for(uint j=0;j<3;j++) {{
            float3 projected=float3(.5,j==0?.1:(j==1?0.:-.001),0)*scale;
            float3 closest=cloth_closest(projected,a,b,c,w);
            {epsilon}
            {predicate}
            ok=ok&&(face_region==(j<2));
        }}
        // Vary thickness independently of triangle size: the contact band
        // follows the fabric, rather than collider tessellation or world units.
        for(uint j=0;j<6;j++) {{
            THICKNESS=(j/2==0?.001:(j/2==1?.01:.1))*scale;
            float3 projected=float3(.5*scale,(j%2==0?-1e-6:-1e-3)*THICKNESS,0);
            float3 closest=cloth_closest(projected,a,b,c,w);
            {epsilon}
            {predicate}
            ok=ok&&(face_region==(j%2==0));
        }}"#);
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_closest")], "groupshared float THICKNESS;", &checks)
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_stretch_uses_rest_length_for_collapsed_edges() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_stretch")], r#"
        groupshared float3 cloth_x[4];groupshared float rest_length;
        uint edges_load(uint i){return i;}
        float mass_load(uint i){return 1;}
        float rest_load(uint i){return rest_length;}
        float stretch_compliance_load(uint i){return 0;}
        float lambda_load(uint i){return 0;}
        void lambda_store(uint i,float x){}
    "#, r#"
        rest_length=scale;cloth_x[0]=0;cloth_x[1]=float3(1.2,0,0)*scale;
        cloth_stretch(0,1.);
        ok=ok&&abs(cloth_x[0].x/scale-.1)<1e-5&&abs(cloth_x[1].x/scale-1.1)<1e-5;
        cloth_x[0]=0;cloth_x[1]=float3(1e-7,0,0)*scale;
        cloth_stretch(0,1.);ok=ok&&all(cloth_x[0]==float3(0));
        cloth_x[1]=0;cloth_stretch(0,1.);ok=ok&&all(cloth_x[1]==float3(0));
    "#)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_bend_keeps_small_hinges_and_rejects_collapsed_faces() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_bend")], r#"
        groupshared float3 cloth_x[4];
        uint hinges_load(uint i){return i;}
        float mass_load(uint i){return 1;}
        float bend_rest_load(uint i){return 0;}
        float bend_compliance_load(uint i){return 0;}
        float bend_lambda_load(uint i){return 0;}
        void bend_lambda_store(uint i,float x){}
    "#, r#"
        cloth_x[0]=0;cloth_x[1]=float3(1,0,0)*scale;
        cloth_x[2]=float3(0,1,0)*scale;cloth_x[3]=float3(0,-1,.4)*scale;
        cloth_bend(0,1.);
        float3 e=cloth_x[1]-cloth_x[0];
        float3 n1=normalize(cross(e,cloth_x[2]-cloth_x[0]));
        float3 n2=normalize(cross(cloth_x[3]-cloth_x[0],e));
        ok=ok&&dot(n1,n2)>.99&&length(cloth_x[0])/scale>.01;
        cloth_x[0]=0;cloth_x[1]=float3(1,0,0)*scale;
        cloth_x[2]=cloth_x[1];cloth_x[3]=float3(0,-1,.4)*scale;
        cloth_bend(0,1.);ok=ok&&all(cloth_x[0]==float3(0));
        cloth_x[2]=float3(0,1,0)*scale;cloth_x[3]=cloth_x[1];
        cloth_bend(0,1.);ok=ok&&all(cloth_x[0]==float3(0));
        cloth_x[1]=0;cloth_bend(0,1.);ok=ok&&all(cloth_x[0]==float3(0));
    "#)
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_contact_projection_and_friction_preserve_mass_and_length_units() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_unit"),(CLOTH_SELF,"cloth_contacts_gauss_seidel")], r#"
        static const uint CLOTH_VERTICES=4,CONTACT_CAPACITY=1;
        groupshared float3 cloth_x[4];groupshared float inverse_mass;
        float contact_count_load(uint i){return i==0?1:0;}
        float contact_owner_load(uint i){return 0;}
        float4 contact_ids_load(uint i){return float4(0,1,2,3);}
        float4 contact_weights_load(uint i){return float4(1,0,0,0);}
        float4 contact_plane_load(uint i){return float4(0,1,0,0);}
        float mass_load(uint i){return inverse_mass;}
        float contact_lambda_load(uint i){return 0;}
        void contact_lambda_store(uint i,float x){}
        float contact_kind_load(uint i){return 2;}
        float3 contact_tangent_lambda_load(uint i){return float3(0);}
        void contact_tangent_lambda_store(uint i,float3 x){}
        float3 old_load(uint i){return float3(0,cloth_x[i].y,0);}
        float3 contact_motion_load(uint i){return float3(0);}
    "#, r#"
        for(uint j=0;j<3;j++) {
            inverse_mass=j==0?1e-15:(j==1?1.:1e15);
            for(uint k=0;k<4;k++)cloth_x[k]=float3(0);
            cloth_x[0]=float3(1,-1,0)*scale;
            cloth_contacts_gauss_seidel();
            ok=ok&&length(cloth_x[0]/scale-float3(.4,0,0))<1e-5;
        }
    "#)
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_tangent_fallback_is_relative_to_the_local_tangent() -> Result<()> {
    let predicate = CLOTH_SOLVER.lines().find(|s| s.contains("float tangent_length2=")).unwrap();
    let checks = format!(r#"
        for(uint j=0;j<3;j++) {{
            float3 n=float3(1,2,3)*scale*scale;
            float3 t=j==0?float3(1,0,0)*scale:(j==1?float3(1,2,3)*scale:float3(0));
            {predicate}
            ok=ok&&length(t)>0&&abs(dot(cloth_unit(t),n))<1e-5;
        }}"#);
    cloth_numeric_contract(&[(CLOTH_SOLVER,"cloth_unit")], "", &checks)
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn cloth_segment_degeneracy_tracks_thickness_for_either_edge() -> Result<()> {
    cloth_numeric_contract(&[(CLOTH_SELF,"cloth_segments")], "groupshared float SELF_MARGIN;", r#"
        SELF_MARGIN=.03*scale;float s,t;
        float3 p=float3(-.5,0,0)*scale,q=-p,a=float3(0,-.5,0)*scale,b=-a;
        cloth_segments(p,q,a,b,s,t);ok=ok&&abs(s-.5)<1e-5&&abs(t-.5)<1e-5;
        float3 tiny=float3(1e-9,0,0)*scale;
        a=float3(.5e-9,-.5,0)*scale;b=float3(.5e-9,.5,0)*scale;
        cloth_segments(float3(0),tiny,a,b,s,t);ok=ok&&s==0&&abs(t-.5)<1e-5;
        cloth_segments(a,b,float3(0),tiny,s,t);ok=ok&&t==0&&abs(s-.5)<1e-5;
        cloth_segments(float3(0),float3(0),float3(0),float3(0),s,t);ok=ok&&s==0&&t==0;
        cloth_segments(p,q,p+float3(0,scale,0),q+float3(0,scale,0),s,t);
        ok=ok&&abs(length(lerp(p,q,s)-lerp(p+float3(0,scale,0),q+float3(0,scale,0),t))/scale-1)<1e-5;
    "#)
}
