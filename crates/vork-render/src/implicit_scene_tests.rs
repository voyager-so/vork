use super::*;

fn sphere_document() -> serde_json::Value {
    serde_json::json!({"version":2,
        "buffers":{"field":{"type":"f32","dims":[24,24,24]}},
        "geometry":{"sphere":{"topology":"implicit","attributes":{"phi":"field"}}},
        "programs":{"seed":{"schedule":"install","dispatch":13824,"bindings":{"field":"field"},
            "source":"struct Inputs {}; void compute(ComputeContext ctx,Inputs i) { uint n=ctx.id.x; uint3 c=uint3(n%24,(n/24)%24,n/576); float3 p=(float3(c)+.5)/12.-1.; field_store(n,length(p)-.65); }"}},
        "materials":{"m":{"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,"emission_color":[1,0,0]}}},
        "nodes":[{"type":"mesh","geometry":"sphere","materials":["m"]},{"type":"camera","name":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"lens":50}}}],
        "render":{"camera":"camera","resolution":[32,32],"background":[0,0,0]}})
}

#[test]
fn implicit_scene_contract() -> Result<()> {
    let source=sphere_document();
    let check=|v:&serde_json::Value| -> Result<()> {let s:Scene=vork_scene::v2::from_test_value(v.clone())?;s.validate()?;Ok(())};
    check(&source)?;
    for invalid in [serde_json::json!([1,24,24]),serde_json::json!([0,24,24])] {
        let mut s=source.clone();s["buffers"]["field"]["dims"]=invalid;assert!(check(&s).is_err());
    }
    let mut s=source.clone();s["geometry"]["sphere"]["attributes"]["other"]="field".into();assert!(check(&s).is_err());
    let mut s=source.clone();s["buffers"]["field"]["type"]="vec3".into();assert!(check(&s).is_err());
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn implicit_solver_status_reports_node_and_remedy() -> Result<()> {
    let renderer = crate::test_renderer()?;
    let mut document=sphere_document();
    document["nodes"][0]["name"]="falling drop".into();
    document["buffers"]["status"]=serde_json::json!({"type":"u32","data":[7]});
    document["geometry"]["sphere"]["status"]=serde_json::json!({"buffer":"status","message":"fluid reconstruction capacity exceeded; increase surface_grid"});
    // An invalid field must not obscure its producer's earlier failure.
    document["programs"]["seed"]["source"]="struct Inputs {}; void compute(ComputeContext c,Inputs i) { field_store(c.id.x,0.); }".into();
    let installed=renderer.install(crate::test_scene(document,Path::new("."))?,Path::new("."))?;
    let mut frame=installed.create_frame_slot(&renderer,Default::default())?;
    let error=frame.submit(&installed,&renderer,0.).and_then(|()|frame.collect(&renderer).map(|_|())).unwrap_err();
    let message=format!("{error:#}");
    for part in ["falling drop","fluid reconstruction","increase surface_grid","7"] {ensure!(message.contains(part),"{message}");}
    ensure!(!message.contains("implicit grid has"),"{message}");
    ensure!(crate::failure::exit_code(&error)==3,"solver failure must remain a scene error");
    let error=frame.submit(&installed,&renderer,0.).unwrap_err();
    ensure!(format!("{error:#}").contains("reinstall"),"{error:#}");
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_sphere_profiles_and_invalid_publication() -> Result<()> {
    let renderer=Renderer::new()?;
    let mut reference:Option<Vec<[f32;4]>>=None;
    for level in 0..=2 {
        let mut scene:Scene=vork_scene::v2::from_test_value(sphere_document())?;
        scene.render.settings=vork_scene::Settings::for_level(level,Default::default());
        scene.render.settings.samples=1;
        let installed=renderer.install(scene,Path::new("."))?;
        assert_eq!(installed.render.settings.integrator.is_path(),level==2,"implicit geometry must preserve the selected renderer");
        assert_eq!(installed.render.settings.samples,1,"implicit geometry must preserve the call budget");
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {float:FloatOutputs {color:true,normals:true,..Default::default()},..Default::default()})?;
        let collected=frame.collect(&renderer)?;
        let normals=collected.normals_f32.unwrap();
        assert!(normals[0].iter().all(|v|v.abs()<0.001),"bounds must not leak into normal controls: {:?}",normals[0]);
        assert!(normals[16*32+16][2]>0.9,"sphere front normal: {:?}",normals[16*32+16]);
        let pixels=collected.linear_rgba.unwrap();
        let center=pixels[16*32+16];
        assert!(center[0]>0.9 && center[1]<0.001,"field center must hit the red surface: {center:?}");
        assert!(pixels[0][0]<0.001,"the bounds must not be visible");
        if let Some(previous)=&reference {
            let mismatch=pixels.iter().zip(previous).filter(|(a,b)| (a[0]>0.5)!=(b[0]>0.5)).count();
            assert!(mismatch<32,"raster/path silhouettes disagree at {mismatch} pixels");
        } else {reference=Some(pixels);}
    }
    let mut invalid=sphere_document();invalid["programs"]["seed"]["source"]="struct Inputs {}; void compute(ComputeContext c,Inputs i) { field_store(c.id.x,0.); }".into();
    let mut scene:Scene=vork_scene::v2::from_test_value(invalid)?;scene.render.settings.samples=1;
    let installed=renderer.install(scene,Path::new("."))?;let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
    frame.submit(&installed,&renderer,0.)?;
    let error=frame.collect(&renderer).err().context("constant-zero publication must fail")?;
    assert!(error.to_string().contains("implicit grid"),"{error:#}");
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_dielectric_preserves_touching_triangle_receiver() -> Result<()> {
    let renderer=Renderer::new()?;
    for level in [1,2] {
    for (scale,translation,receiver) in [(1.,0.,-0.6),(1.,0.,-0.601),(1.,0.,-0.599),(2.,0.2,-1.)] {
        let mut document=sphere_document();
        document["programs"]["seed"]["source"]="struct Inputs {}; void compute(ComputeContext c,Inputs i) { uint k=c.id.x; float3 p=(float3(k%24,(k/24)%24,k/576)+.5)/12.-1.; field_store(k,max(abs(p.x)-.6,max(abs(p.y)-.6,abs(p.z)-.6))); }".into();
        document["buffers"]["positions"]=serde_json::json!({"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]});
        document["buffers"]["indices"]=serde_json::json!({"type":"u32","data":[0,1,2,0,2,3]});
        document["geometry"]["receiver"]=serde_json::json!({"attributes":{"POSITION":"positions"},"indices":"indices"});
        document["materials"]["glass"]=serde_json::json!({"inputs":{"base_weight":0,"specular_weight":0,"transmission_weight":1,"specular_ior":1,"specular_roughness":0,"transmission_color":[1,1,1]}});
        document["nodes"][0]["materials"]=serde_json::json!(["glass"]);
        document["nodes"][0]["scale"]=serde_json::json!([1,1,scale]);
        document["nodes"][0]["translation"]=serde_json::json!([0,0,translation]);
        document["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","geometry":"receiver","materials":["m"],"translation":[0,0,receiver]}));
        let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
        scene.render.settings=vork_scene::Settings::for_level(level,Default::default());
        scene.render.settings.samples=1;
        let installed=renderer.install(scene,Path::new("."))?;
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        let pixels=frame.collect(&renderer)?.linear_rgba.unwrap();
        for y in 14..18 {for x in 14..18 {
            let red=pixels[y*32+x][0];
            assert!(red>0.95,"index-matched field must preserve its opaque receiver: scale={scale}, translation={translation}, receiver={receiver}, red={red}");
        }}
    }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_liquid_state_and_seek() -> Result<()> {
    let renderer=Renderer::new()?;
    let mut document:serde_json::Value=serde_json::from_str(include_str!("../../../examples/water/scene.json"))?;
    for program in document["programs"].as_object_mut().unwrap().values_mut() {
        let Some(file)=program["file"].as_str() else {continue;};
        let source=match file {
            "table.slang" => include_str!("../../../examples/water/table.slang"),
            "ceramic.slang" => include_str!("../../../examples/water/ceramic.slang"),
            file => panic!("unhandled water fixture program {file}"),
        }.to_owned();
        program.as_object_mut().unwrap().remove("file");
        program["source"]=source.into();
    }
    // Keep the software CI workload independent of the presentation grid.
    // The explicit opt-in also runs identical invariants at production scale on L4.
    let production = std::env::var_os("VORK_TEST_WATER_PRODUCTION").is_some();
    if !production {
        let mut dispatch_sizes = std::collections::BTreeMap::new();
        let proposals=(11264.0_f64*(1.0/960.0)/(0.052_f64/58.86).sqrt()).ceil() as u64;
        for buffer in document["buffers"].as_object_mut().unwrap().values_mut() {
            if let Some(data)=buffer.get_mut("data").and_then(serde_json::Value::as_array_mut) {
                if data.len()==8192 {data.truncate(128);}
            }
            let Some(dims) = buffer.get("dims").and_then(serde_json::Value::as_array) else {continue;};
            let dims: Vec<_> = dims.iter().map(|v|v.as_u64().unwrap()).collect();
            let resized = match dims.as_slice() {
                [512,1,512] => [128,1,128],
                [512,3,512] => [128,3,128],
                [2880,1,1] => [44,1,1],
                [336,112,336] => [72,24,72],
                [768,256,768] => [144,48,144],
                [768,1,768] => [144,1,144],
                [128,45,128] => [32,11,32],
                [737280,1,1] => [11264,1,1],
                [192,64,192] => [48,16,48],
                [2304,1,1] => [81,1,1],
                [9,1,1] => [1,1,1],
                [1024,1,1] => [64,1,1],
                [4,1,1] => [1,1,1],
                [8192,1,1] => [128,1,1],
                [51678,1,1] => [proposals,1,1],
                [202,1,1] => [proposals.div_ceil(256),1,1],
                [1,1,1] => [1,1,1],
                [32,1,1] => [1,1,1],
                _ => panic!("unclassified liquid test grid: {dims:?}"),
            };
            dispatch_sizes.insert(dims.iter().product::<u64>(), resized.iter().product::<u64>());
            buffer["dims"] = serde_json::json!(resized);
        }
        for program in document["programs"].as_object_mut().unwrap().values_mut() {
            if let Some(count) = program.get("dispatch").and_then(serde_json::Value::as_u64) {
                if count != 1 {program["dispatch"] = dispatch_sizes[&count].into();}
            }
            let source=program["source"].as_str().unwrap()
                .replace("BIN_RES = 192u", "BIN_RES = 48u")
                .replace("uint3(192,64,192)", "uint3(48,16,48)")
                .replace("MARKER_SPACING = 0.013", "MARKER_SPACING = 0.052")
                .replace("KERNEL_SUPPORT = 0.035", "KERNEL_SUPPORT = 0.14")
                .replace("CONTACT_OFFSET = 0.005", "CONTACT_OFFSET = 0.02");
            program["source"]=source.into();
        }
    }
    let extent=document["nodes"][0]["scale"][0].as_f64().unwrap_or(1.);
    let y_extent=document["nodes"][0]["scale"][1].as_f64().unwrap_or(1.) as f32;
    let y_origin=document["nodes"][0]["translation"][1].as_f64().unwrap_or(0.) as f32;
    // This numerical fixture exercises retained liquid state, independently of
    // the example's external mesh/image assets. The full scene is rendered too.
    document["geometry"]=serde_json::json!({});
    document["images"]=serde_json::json!({});
    document["nodes"].as_array_mut().unwrap().retain(|node| node["type"]=="camera");
    document["render"].as_object_mut().unwrap().remove("camera");
    let base=Path::new(".").to_path_buf();
    let scene=crate::test_scene(document,&base)?;
    scene.validate()?;
    let decoded=buffers::SceneBuffers::decode(&scene,&base)?;
    let attrs=if production {buffers::ProgramBuffers::install(&renderer,&scene,&decoded)?}
        else {buffers::ProgramBuffers::install_with_page_limit(&renderer,&scene,&decoded,1024*1024)?};
    let compute=compute::Compute::test_install(&renderer,&scene,&base,Some(&attrs),&[],&[],&[],vork_time::Time::integer(24),1.5)?.unwrap();
    assert!(attrs.bindings["surface"].spans.len()>1,"fine field must exercise retained storage across pages");
    let dims=scene.buffers["surface"].dims.unwrap();
    assert!(dims[0]>scene.buffers["phi"].dims.unwrap()[0], "surface resolution must be independent of the pressure grid");
    let bytes=dims.iter().map(|&d|u64::from(d)).product::<u64>()*4;
    let marker_bytes=scene.buffers["markers"].len().unwrap() as u64*12;
    let wetness_bytes=scene.buffers["wetness"].len().unwrap() as u64*4;
    let landed_bytes=scene.buffers["landed"].len().unwrap() as u64*4;
    let white_count=scene.buffers["white_position"].len().unwrap() as u64;
    let white_offset=bytes+2*marker_bytes+4+wetness_bytes+landed_bytes;
    let mut saw_spray=false;
    let readback=renderer.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:white_offset+white_count*20,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
    let mut reference_energy=None;
    let mut reference_count=None;
    let mut snapshots=Vec::new();
    let mut volume_errors=Vec::new();
    let mut cursor=compute.cursor.lock().unwrap();
    for time in [0.,0.32,0.5,1.5,0.32] {
        let mut encoder=renderer.device.create_command_encoder(&Default::default());
        compute.advance_with(&mut encoder,&mut cursor,time,|encoder| {
            let done=std::mem::replace(encoder,renderer.device.create_command_encoder(&Default::default()));
            renderer.queue.submit([done.finish()]);
        })?;
        for (name,offset) in [("surface",0),("markers",bytes),("marker_velocity",bytes+marker_bytes),("overflow",bytes+2*marker_bytes),("wetness",bytes+2*marker_bytes+4),("landed",bytes+2*marker_bytes+4+wetness_bytes),("white_position",white_offset),("white_life",white_offset+white_count*12),("white_kind",white_offset+white_count*16)] {
            let binding=&attrs.bindings[name];
            for span in &binding.spans {
                encoder.copy_buffer_to_buffer(&attrs.storage[span.page as usize],u64::from(span.offset)*4,&readback,
                    offset+u64::from(span.first*binding.width)*4,u64::from(span.count*binding.width)*4);
            }
        }
        let submitted=renderer.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|r|{let _=tx.send(r);});
        renderer.wait(Some(submitted))?;rx.recv()??;
        let words=readback.get_mapped_range(..).to_vec();readback.unmap();
        let values:&[f32]=bytemuck::cast_slice(&words[..bytes as usize]);let n=dims[0] as usize;let ny=dims[1] as usize;
        let positions:&[[f32;3]]=bytemuck::cast_slice(&words[bytes as usize..(bytes+marker_bytes) as usize]);
        let velocities:&[[f32;3]]=bytemuck::cast_slice(&words[(bytes+marker_bytes) as usize..(bytes+2*marker_bytes) as usize]);
        let failure=u32::from_le_bytes(words[(bytes+2*marker_bytes) as usize..(bytes+2*marker_bytes+4) as usize].try_into().unwrap());
        let wetness:&[f32]=bytemuck::cast_slice(&words[(bytes+2*marker_bytes+4) as usize..(bytes+2*marker_bytes+4+wetness_bytes) as usize]);
        let landed:&[u32]=bytemuck::cast_slice(&words[(bytes+2*marker_bytes+4+wetness_bytes) as usize..white_offset as usize]);
        let white_p:&[f32]=bytemuck::cast_slice(&words[white_offset as usize..(white_offset+white_count*12) as usize]);
        let white_w:&[f32]=bytemuck::cast_slice(&words[(white_offset+white_count*12) as usize..(white_offset+white_count*16) as usize]);
        let white_k:&[u32]=bytemuck::cast_slice(&words[(white_offset+white_count*16) as usize..]);
        assert!(white_p.iter().all(|v|v.is_finite()),"nonfinite whitewater position");
        assert!(white_w.iter().zip(white_k).all(|(&w,&k)| w.is_finite() && w>=0. && k<=3 && (w>0.)==(k>0)),"invalid secondary state");
        let kinds=[1,2,3].map(|k|white_k.iter().filter(|&&v|v==k).count());
        eprintln!("whitewater at {time}: spray/foam/bubbles={kinds:?}");
        saw_spray|=kinds[0]>0;
        assert!(landed.iter().all(|&v|v<=3),"invalid deposited-marker state");
        assert!(positions.iter().zip(landed).all(|(p,&state)| {
            let contact=if production {0.005} else {0.02};
            let floor_contact=p[1]>-1. && p[1]<=-0.93+contact*1.02 && p[0]*p[0]+p[2]*p[2]>0.90*0.90;
            !floor_contact || state>0
        }),"water resting on the exposed floor must enter the film at {time}");
        let deposited=landed.iter().filter(|&&v|v==1 || v==2).count();
        let spacing: f64=if production {0.013} else {0.052};
        let film_cell=6./scene.buffers["wetness"].dims.unwrap()[0] as f64;
        let deposited_volume=deposited as f64*spacing.powi(3);
        let film_volume=wetness.iter().map(|&v|f64::from(v)).sum::<f64>()*film_cell*film_cell;
        assert!((film_volume-deposited_volume).abs()<1e-7+deposited_volume*0.0001,
            "film must receive each landed marker exactly once: t={time}, film={film_volume}, deposited={deposited_volume}");
        assert!(wetness.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),"invalid wet coverage at {time}");
        if time==0. {assert!(wetness.iter().all(|&v|v==0.),"floor starts dry");}
        if production && time==1.5 {assert!(wetness.iter().any(|&v|v>0.0001),"spilled water must wet the floor");}
        let mut bounds=[0.0_f32;3];
        for p in positions.iter().filter(|p|p[1]>-1.) {
            for axis in 0..3 {bounds[axis]=bounds[axis].max(p[axis].abs());}
        }
        eprintln!("marker absolute bounds at {time}: {bounds:?}");
        assert_eq!(failure,u32::MAX,"particle publication error at {time}: particle escaped the reconstruction domain");
        let mut energy=0.0_f64;let mut count=0;
        for ((p,v),state) in positions.iter().zip(velocities).zip(landed) {
            assert!(p.iter().chain(v).all(|x|x.is_finite()),"nonfinite marker at {time}");
            if *state==3 {continue;}
            count+=1;
            energy+=v.iter().map(|&x|0.5*f64::from(x).powi(2)).sum::<f64>()+58.86*(f64::from(p[1])+0.875);
        }
        energy/=count as f64;
        eprintln!("marker count={count}, mean mechanical energy={energy}");
        if let Some(initial)=reference_energy {
            assert!(energy<initial*1.15,"particle energy grew at {time}: {initial} -> {energy}");
            assert_eq!(Some(count),reference_count,"lost liquid markers");
        } else {reference_energy=Some(energy);reference_count=Some(count);}

        assert!(values.iter().all(|v|v.is_finite()), "invalid particle bin or field publication at {time}");
        let wet:Vec<_>=values.iter().enumerate().filter(|(_,v)|**v<0.).collect();
        let top=wet.iter().map(|(i,_)|((((i/n)%ny) as f32+0.5)*2./ny as f32-1.)*y_extent+y_origin).fold(-1.,f32::max);
        eprintln!("liquid t={time}: negative cells={}, top={top}, range={:?}",wet.len(),values.iter().copied().fold((f32::INFINITY,f32::NEG_INFINITY),|(a,b),v|(a.min(v),b.max(v))));
        let dx=2.0*extent/n as f64;
        // Surface values and film depths are solver lengths; node scale converts to SI.
        let metres_per_unit=extent/3.;
        let liquid=fluid_surface_volume(values,dims,16);
        let film_dx=2.*extent/(scene.buffers["wetness"].dims.unwrap()[0] as f64);
        let film=wetness.iter().map(|&v|f64::from(v)).sum::<f64>()*metres_per_unit*film_dx*film_dx/(dx*dx*dx);
        let volume=liquid+film;
        let target=count as f64*(spacing*metres_per_unit/dx).powi(3);
        let error=(volume/target-1.).abs();
        eprintln!("interpolated volume at {time}: {volume}, rest target={target}, relative error={error}");
        volume_errors.push((time,error));
        snapshots.push(words);
    }
    // The coarse fixture cannot resolve the authored crown; emission modes
    // also have a controlled GPU oracle independent of that resolution.
    if production {assert!(saw_spray,"splash must produce secondary spray");}
    assert_ne!(snapshots[0],snapshots[1],"liquid must advance");
    assert_eq!(&snapshots[1],snapshots.last().unwrap(),"backward seek must replay identical field bytes");
    assert!(volume_errors.iter().all(|(_,error)|*error<0.002),"visible liquid volume exceeds 0.2% physical ceiling: {volume_errors:?}");
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_emitter_does_not_light_its_own_surface() -> Result<()> {
    let mut renderer=Renderer::new()?;
    // A tiny convex octahedron has no surface-to-surface light paths. Its
    // diffuse response must not add energy to its constant unit emission.
    let mut values=Vec::new();
    for z in 0..5 {for y in 0..5 {for x in 0..5 {
        let p=[x,y,z].map(|c| (c as f32-2.)*0.4);
        values.push(p.iter().map(|v|v.abs()).sum::<f32>()-0.6);
    }}}
    for direct_only in [false,true] {
    // Keep the production path render, then isolate next-event energy from
    // later BSDF hits at the octahedron's reconstructed smooth normals.
    renderer.reference_transport_bounces=direct_only.then_some(0);
    for rotation in [[0.,0.,0.],[0.21,0.37,0.13]] {
        let document=serde_json::json!({"version":2,
            "buffers":{"field":{"type":"f32","dims":[5,5,5],"data":values}},
            "geometry":{"source":{"topology":"implicit","attributes":{"phi":"field"}}},
            "materials":{"source":{"inputs":{"base_color":[0.8,0.8,0.8],"specular_weight":0,
                "emission_luminance":1,"emission_color":[1,1,1]}}},
            "nodes":[
                {"type":"mesh","geometry":"source","materials":["source"],
                    "scale":[0.8,1.1,0.6],"translation":[0.1,0.2,0.3],"rotation":{"euler":rotation}},
                {"type":"camera","name":"camera","translation":[0.1,0.2,3.3],"target":[0.1,0.2,0.3],
                    "material":{"program":"std/camera","inputs":{"lens":50}}},
                {"type":"light","shape":"environment","color":[0,0,0]}],
            "render":{"camera":"camera","resolution":[24,24],"background":[0,0,0]}});
        let mut scene=crate::test_scene(document,Path::new("."))?;
        scene.render.settings=vork_scene::Settings::for_level(2,Default::default());
        scene.render.settings.samples=16;
        scene.render.settings.disable_denoiser();
        let installed=renderer.install(scene,Path::new("."))?;
        assert_eq!(installed.rays.as_ref().unwrap().implicit[3],1);
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit(&installed,&renderer,0.)?;
        let pixels=frame.collect(&renderer)?.linear_rgba.context("linear frame")?;
        assert!(pixels.iter().flatten().all(|v|v.is_finite()),"rotation {rotation:?}");
        let peak=pixels.iter().flat_map(|p|&p[..3]).copied().fold(0_f32,f32::max);
        eprintln!("implicit emitter: direct_only={direct_only}, rotation={rotation:?}, peak={peak}");
        assert!(peak>0.9,"emissive grid must be visible: {peak}, rotation {rotation:?}");
        if direct_only {
            assert!(peak<=1.00001,"convex emitter must not illuminate itself: {peak}, rotation {rotation:?}");
        }
    }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_emitter_connections_light_receivers() -> Result<()> {
    let r=Renderer::new()?;
    for level in [1,2] {
    for (emission,program,glow) in [(0.,false,false),(10.,false,false),(10.,true,false),(10.,true,true)] {
        let mut document=sphere_document();
        document["render"]["resolution"]=serde_json::json!([32,24]);
        document["buffers"]["floor_p"]=serde_json::json!({"type":"vec3","data":[[-5,0,-5],[5,0,-5],[5,0,5],[-5,0,5]]});
        document["buffers"]["floor_i"]=serde_json::json!({"type":"u32","data":[0,2,1,0,3,2]});
        document["geometry"]["floor"]=serde_json::json!({"attributes":{"POSITION":"floor_p"},"indices":"floor_i"});
        document["materials"]["floor"]=serde_json::json!({"inputs":{"base_color":[0.8,0.8,0.8],"specular_weight":0}});
        document["materials"]["m"]["inputs"]["emission_luminance"]=emission.into();
        if program {
            let source=if glow {
                "struct Inputs {float strength;}; Closure surface(ShadingContext c,Inputs i) {return unlit(float3(0.));} Closure lit(ShadingContext c,Inputs i,Lighting l) {return unlit(float3(i.strength,0.,0.));}"
            } else {
                "struct Inputs {float strength;}; Closure surface(ShadingContext c,Inputs i) {return unlit(float3(i.strength,0.,0.));}"
            };
            document["programs"]["emitter"]=serde_json::json!({"source":source});
            document["materials"]["m"]=serde_json::json!({"program":"emitter","inputs":{"strength":emission}});
        }
        document["nodes"]=serde_json::json!([
            {"type":"mesh","geometry":"sphere","materials":["m"],"translation":[0,1.6,0],"scale":[0.1,0.12,0.1]},
            {"type":"mesh","geometry":"floor","materials":["floor"]},
            {"type":"camera","name":"camera","translation":[4,3,6],"target":[0,0.5,0],"material":{"program":"std/camera","inputs":{"lens":45}}},
            {"type":"light","shape":"environment","color":[0,0,0]}]);
        if glow {
            for event in ["visible_diffuse","visible_glossy","visible_transmission"] {document["nodes"][0][event]=false.into();}
        }
        let mut scene=crate::test_scene(document,Path::new("."))?;
        scene.render.settings=vork_scene::Settings::for_level(level,Default::default());
        scene.render.settings.samples=64;
        let installed=r.install(scene,Path::new("."))?;
        assert_eq!(installed.rays.as_ref().unwrap().implicit[3],u32::from(emission>0. && !glow));
        let mut frame=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        frame.submit(&installed,&r,0.)?;
        let pixels=frame.collect(&r)?.linear_rgba.context("linear frame")?;
        // The receiver region excludes the directly visible source.
        let mut energy=0.;let mut lit=0;
        for y in 18..23 {for x in 10..22 {let red=pixels[y*32+x][0];energy+=red;lit+=usize::from(red>0.);}}
        assert!(pixels.iter().flatten().all(|v|v.is_finite()));
        // The source subtends too little solid angle for 64 BSDF-only samples
        // to illuminate most pixels. Source connections must resolve its light.
        if emission>0. && !glow {assert!(lit>=54,"small implicit source reached only {lit}/60 receivers (energy {energy})");}
        else {assert_eq!(energy,0.,"proposal bounds must not emit");}
    }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_simulation_samples_the_shutter() -> Result<()> {
    let renderer=Renderer::new()?;
    let mut variances=Vec::new();
    for shutter in [0.,1.] {
        let mut document=sphere_document();
        document["programs"]["seed"]["schedule"]="simulation".into();
        document["programs"]["seed"]["source"]="struct Inputs {}; void compute(ComputeContext c,Inputs i) {uint k=c.id.x;uint3 cell=uint3(k%24,(k/24)%24,k/576);float3 p=(float3(cell)+.5)/12.-1.;float x=.5*(c.time+c.dt-1.);field_store(k,length(p-float3(x,0,0))-.2);}".into();
        document["simulation"]=serde_json::json!({"t0":0,"dt":0.125,"seed":7,"state":["field"],"init":["seed"],"steps":["seed"]});
        document["render"]["resolution"]=serde_json::json!([64,64]);
        document["render"]["fps"]=1.into();document["render"]["duration"]=2.into();
        document["render"]["motion_blur"]=serde_json::json!({"shutter":shutter});
        let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
        scene.render.settings.samples=16;
        let installed=renderer.install(scene,Path::new("."))?;
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit_outputs(&installed,&renderer,1.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        let pixels=frame.collect(&renderer)?.linear_rgba.unwrap();
        let mut sum=0.0_f64;let mut first=0.;let mut second=0.;
        for (index,pixel) in pixels.iter().enumerate() {
            let weight=f64::from(pixel[0]);let x=(index%64) as f64;
            sum+=weight;first+=weight*x;second+=weight*x*x;
        }
        assert!(sum>1.,"the moving field must be visible");
        variances.push(second/sum-(first/sum).powi(2));
    }
    assert!(variances[1]>variances[0]*1.4,"shutter samples must spread the moving implicit silhouette: {variances:?}");
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn raster_implicit_transparency_respects_opaque_depth() -> Result<()> {
    let renderer=Renderer::new()?;
    for mirrored in [1.,-1.] {
        let mut document=sphere_document();
        document["materials"]["m"]["opacity"]="transmittance".into();
        document["materials"]["m"]["inputs"]["geometry_opacity"]=0.5.into();
        document["nodes"][0]["scale"]=serde_json::json!([mirrored,1,1]);
        document["buffers"]["positions"]=serde_json::json!({"type":"vec3","data":[[-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]]});
        document["buffers"]["indices"]=serde_json::json!({"type":"u32","data":[0,1,2,0,2,3]});
        document["geometry"]["occluder"]=serde_json::json!({"attributes":{"POSITION":"positions"},"indices":"indices"});
        document["materials"]["green"]=serde_json::json!({"inputs":{"base_weight":0,"specular_weight":0,"emission_luminance":1,"emission_color":[0,1,0]}});
        document["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","geometry":"occluder","materials":["green"]}));
        let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
        scene.render.settings=vork_scene::Settings::for_level(1,Default::default());scene.render.settings.samples=1;
        let installed=renderer.install(scene,Path::new("."))?;
        assert!(!installed.render.settings.integrator.is_path());
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        let pixels=frame.collect(&renderer)?.linear_rgba.unwrap();
        let center=pixels[16*32+16];
        assert!(center[0]<0.001 && center[1]>0.9,"hidden field leaked into transparency: {center:?}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn raster_implicit_alpha_composes_both_crossings() -> Result<()> {
    let renderer=Renderer::new()?;
    for mirrored in [1.,-1.] {
        let mut document=sphere_document();
        document["materials"]["m"]["opacity"]="transmittance".into();
        document["materials"]["m"]["inputs"]["geometry_opacity"]=0.5.into();
        document["nodes"][0]["scale"]=serde_json::json!([mirrored,1,1]);
        let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
        scene.render.settings=vork_scene::Settings::for_level(1,Default::default());scene.render.settings.samples=1;
        let installed=renderer.install(scene,Path::new("."))?;
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        let pixels=frame.collect(&renderer)?.linear_rgba.unwrap();
        let center=pixels[16*32+16];
        assert!((center[0]-0.75).abs()<0.02 && center[1]<0.001,"two half-opacity interfaces must compose: {center:?}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn raster_implicit_alpha_orders_interleaved_fields() -> Result<()> {
    let renderer=Renderer::new()?;
    for samples in [1,64] {for reverse in [false,true] {
        let mut document=sphere_document();
        document["materials"]["m"]["opacity"]="transmittance".into();
        document["materials"]["m"]["inputs"]["geometry_opacity"]=0.5.into();
        document["materials"]["blue"]=document["materials"]["m"].clone();
        document["materials"]["blue"]["inputs"]["emission_color"]=serde_json::json!([0,0,1]);
        document["nodes"][0]["translation"]=serde_json::json!([0,0,0.2]);
        document["nodes"].as_array_mut().unwrap().push(serde_json::json!({
            "type":"mesh","geometry":"sphere","materials":["blue"],
            "translation":[0,0,-0.2],"scale":[-1,1,1]}));
        if reverse {document["nodes"].as_array_mut().unwrap().swap(0,2);}
        let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
        scene.render.settings=vork_scene::Settings::for_level(1,Default::default());
        scene.render.settings.samples=samples;
        let mut installed=renderer.install(scene,Path::new("."))?;
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        // Interleaved sample records can exceed the initial reservation.
        let pixels=installed.render_frame(&renderer,&mut frame,0.)?.linear_rgba.unwrap();
        let center=pixels[16*32+16];
        // Along the camera ray: red front, blue front, red rear, blue rear.
        // Each interface has opacity one half; object installation order is irrelevant.
        assert!((center[0]-0.625).abs()<0.02 && center[1]<0.001 && (center[2]-0.3125).abs()<0.02,
            "interleaved field alpha order ({samples} samples, reverse={reverse}): {center:?}");
    }}
    Ok(())
}



#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_invalid_publication_is_checked_offscreen_and_retained() -> Result<()> {
    let renderer=Renderer::new()?;
    for level in [1,2] {for invalid in [
        "if(n==7200u) v=asfloat(0x7fc00000u);",
        "if(all(c>=uint3(10)) && all(c<=uint3(11))) v=0.;",
    ] {
        let mut document=sphere_document();
        document["programs"]["seed"]["source"]=format!("struct Inputs {{}}; void compute(ComputeContext ctx,Inputs i) {{ uint n=ctx.id.x; uint3 c=uint3(n%24,(n/24)%24,n/576); float v=1.; {invalid} field_store(n,v); }}").into();
        document["nodes"][0]["translation"]=serde_json::json!([100,0,0]);
        // Fatal publication errors poison the installation, not just its slot
        // (media.md). Each exposure therefore needs a fresh invalid install.
        for time in [0.,1.] {
            let mut scene:Scene=vork_scene::v2::from_test_value(document.clone())?;
            scene.render.settings=vork_scene::Settings::for_level(level,Default::default());
            scene.render.settings.samples=2;
            let installed=renderer.install(scene,Path::new("."))?;
            let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
            frame.submit(&installed,&renderer,time)?;
            let error=frame.collect(&renderer).err().context("offscreen invalid publication must fail every exposure")?;
            assert!(error.to_string().contains("implicit grid"),"{error:#}");
            let require_poison=|error:&anyhow::Error| {
                let message=format!("{error:#}");
                assert!(message.contains("reinstall") &&
                    (message.contains("previously failed") || message.contains("invalidated")),
                    "expected installation poison diagnostic: {message}");
            };
            let error=frame.submit(&installed,&renderer,time+0.25).err()
                .context("old slot must refuse a poisoned installation")?;
            require_poison(&error);
            match installed.create_frame_slot(&renderer,ReadbackPrecision::Float) {
                Err(error)=>require_poison(&error),
                Ok(mut other)=> {
                    let error=other.submit(&installed,&renderer,time+0.25).err()
                        .context("new slot must refuse the same poisoned installation")?;
                    require_poison(&error);
                }
            }
            // Poison belongs to the failed install, not the shared Renderer.
            let mut valid:Scene=vork_scene::v2::from_test_value(sphere_document())?;
            valid.render.settings=vork_scene::Settings::for_level(level,Default::default());
            valid.render.settings.samples=2;
            let recovered=renderer.install(valid,Path::new("."))?;
            let mut recovery=recovered.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
            recovery.submit(&recovered,&renderer,time)?;
            let pixels=recovery.collect(&renderer)?.linear_rgba.context("reinstalled valid sphere must return linear output")?;
            assert!(pixels.iter().flatten().all(|value|value.is_finite()),"reinstalled output must be finite");
            assert!(pixels[16*32+16][0]>0.9,"reinstalled sphere must remain visible");
        }
    }}
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_carrier_survives_camera_clip_planes() -> Result<()> {
    let renderer=Renderer::new()?;
    for projection in ["persp","ortho"] {for (eye,far) in [(3.,2.5),(0.8,0.3)] {
        let mut reference:Option<(f32,u16)>=None;
        for (level,clip_end) in [(2,10.),(1,10.),(2,far),(1,far)] {
            let mut document=sphere_document();
            document["nodes"][1]["translation"]=serde_json::json!([0,0,eye]);
            document["nodes"][1]["material"]["inputs"]=serde_json::json!({"lens":50,"projection":projection,"ortho_scale":2,"clip_start":0.01,"clip_end":clip_end});
            let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
            scene.render.settings=vork_scene::Settings::for_level(level,Default::default());scene.render.settings.samples=1;
            let installed=renderer.install(scene,Path::new("."))?;
            let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
            frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {depth_ids:true,float:FloatOutputs {color:true,depth:true,normals:true,..Default::default()},..Default::default()})?;
            let result=frame.collect(&renderer)?;let at=16*32+16;
            let red=result.linear_rgba.as_ref().unwrap()[at][0];
            let depth=result.depth_f32.as_ref().unwrap()[at];
            let normal=result.normals_f32.as_ref().unwrap()[at];
            let id=result.ids[at];
            assert!(red>0.9 && normal[2]>0.9,"{projection} eye={eye} far={clip_end} level={level}: red={red}, normal={normal:?}");
            if let Some((d,i))=reference {assert!((depth-d).abs()<0.03 && id==i,"clipped carrier changed depth/ID: {depth} {id}, expected {d} {i}");}
            else {reference=Some((depth,id));}
        }
    }}
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_index_is_shared_and_tracks_source_publications() -> Result<()> {
    let renderer=Renderer::new()?;
    let mut document=sphere_document();
    document["materials"]["second"]=document["materials"]["m"].clone();
    document["materials"]["second"]["inputs"]["emission_color"]=serde_json::json!([0,1,0]);
    document["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","geometry":"sphere","materials":["second"],"translation":[2,0,0]}));
    let mut scene:Scene=vork_scene::v2::from_test_value(document)?;scene.render.settings.samples=1;
    let installed=renderer.install(scene,Path::new("."))?;
    assert_eq!(installed.implicit_fields.len(),1);
    let fields:Vec<_>=installed.draws.iter().filter_map(|d|d.implicit.as_ref()).collect();
    assert_eq!(fields.len(),2);
    assert!(std::sync::Arc::ptr_eq(&fields[0].acceleration,&fields[1].acceleration));
    let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
    for time in [0.,1.,2.] {
        frame.submit(&installed,&renderer,time)?;frame.collect(&renderer)?;
        assert_eq!(fields[0].acceleration.builds.load(std::sync::atomic::Ordering::Relaxed),1,"static publications must retain the index across frames");
    }
    let mut changes=values::ChangeTracker::default();
    let mut encoder=renderer.device.create_command_encoder(&Default::default());
    let index=&installed.implicit_fields[0].acceleration;
    assert!(index.publish(&mut encoder,&mut changes,1));
    assert!(!index.publish(&mut encoder,&mut changes,1),"held publication must reuse its index");
    assert!(index.publish(&mut encoder,&mut changes,2),"restore or a new producer publication must rebuild");
    assert!(!index.publish(&mut encoder,&mut changes,2));
    renderer.queue.submit([encoder.finish()]);
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_controls_continue_after_rejected_crossing() -> Result<()> {
    let renderer=Renderer::new()?;
    for opacity in ["coverage","transmittance"] {for mirrored in [1.,-1.] {
        let mut reference:Option<(f32,u16,[f32;3])>=None;
        for level in [2,1] {
            let mut document=sphere_document();
            document["programs"]["cut"]=serde_json::json!({"source":"struct Inputs {}; Closure surface(ShadingContext c,Inputs i) {var s=unlit(float3(1.,0.,0.));s.geometry_opacity=c.POSITION.z<0. ? 1. : 0.;return s;}"});
            document["materials"]["m"]=serde_json::json!({"program":"cut","opacity":opacity});
            document["nodes"][0]["scale"]=serde_json::json!([mirrored,1,1]);
            let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
            scene.render.settings=vork_scene::Settings::for_level(level,Default::default());scene.render.settings.samples=1;
            let installed=renderer.install(scene,Path::new("."))?;
            let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
            frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {depth_ids:true,float:FloatOutputs {color:true,depth:true,normals:true,..Default::default()},..Default::default()})?;
            let result=frame.collect(&renderer)?;let at=16*32+16;
            let red=result.linear_rgba.as_ref().unwrap()[at][0];
            let depth=result.depth_f32.as_ref().unwrap()[at];let mut normal=result.normals_f32.as_ref().unwrap()[at];let id=result.ids[at];
            // Path exports the outward vertex normal; raster controls export the
            // incident-facing shading normal. This accepted root is the exit.
            assert!(if level==2 {normal[2]< -0.9} else {normal[2]>0.9},"accepted exit normal: level={level}, {normal:?}");
            if level==2 {normal=normal.map(|v|-v);}
            assert!(red>0.9 && depth>3.5 && depth<3.8,"{opacity}, mirrored={mirrored}, level={level}: red={red} depth={depth}");
            if let Some((d,i,n))=reference {
                assert!((depth-d).abs()<0.03 && id==i,"accepted depth/ID mismatch");
                for channel in 0..3 {assert!((normal[channel]-n[channel]).abs()<0.1,"accepted normal mismatch: {normal:?} {n:?}");}
            } else {reference=Some((depth,id,normal));}
        }
    }}
    Ok(())
}

#[test]
#[ignore = "requires a real ray-query GPU and pinned Slang"]
fn implicit_caster_shadows_medium_illumination() -> Result<()> {
    let renderer=Renderer::new()?;
    let mut ratios=Vec::new();
    let mut profile_energies=Vec::new();
    for level in [1,2] {
        let mut energies=Vec::new();
        for caster in [false,true] {
            let mut document=sphere_document();
            document["materials"]["m"]=serde_json::json!({"inputs":{"base_color":[0,0,0],"specular_weight":0}});
            document["nodes"][0]["visible_camera"]=false.into();
            document["nodes"][0]["visible_shadow"]=caster.into();
            document["nodes"][1]["translation"]=serde_json::json!([0,0,4]);
            document["nodes"][1]["material"]["inputs"]=serde_json::json!({"projection":"ortho","ortho_scale":4,"clip_start":0.1,"clip_end":6});
            document["nodes"].as_array_mut().unwrap().extend([
                serde_json::json!({"type":"light","shape":"environment","color":[0,0,0],"energy":0,"medium":{"program":{"source":include_str!("../../vork-scene/tests/fixtures/medium.slang")},"inputs":{"density":0.05,"scattering":[1,1,1]}}}),
                // Keep the point source outside the image: unresolved 1/r falloff
                // at the image edge must not dominate this field-shadow oracle.
                serde_json::json!({"type":"light","shape":"point","translation":[-3,0,0],"energy":100,"radius":0}),
            ]);
            let mut scene:Scene=vork_scene::v2::from_test_value(document)?;
            scene.render.settings=vork_scene::Settings::for_level(level,Default::default());scene.render.settings.samples=4096;
            let installed=renderer.install(scene,Path::new("."))?;
            let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
            frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            let pixels=frame.collect(&renderer).with_context(||format!("fog shadow level={level}, caster={caster}"))?.linear_rgba.unwrap();
            assert!(pixels.iter().flatten().all(|v|v.is_finite()));
            let mut energy=0.;
            // Fog to the right of the caster, away from its camera silhouette.
            for y in 13..19 {for x in 23..29 {energy+=pixels[y*32+x][0];}}
            let total_energy=pixels.iter().map(|p|p[0]).sum::<f32>();
            eprintln!("fog energy level={level}, caster={caster}: region={energy}, total={total_energy}");
            energies.push((energy,total_energy));
        }
        assert!(energies[0].0>0.01 && energies[1].0<energies[0].0*0.97,"level={level}: fog behind the caster must darken: {energies:?}");
        ratios.push(energies[1].0/energies[0].0);
        profile_energies.push(energies);
    }
    for caster in 0..2 {
        let relative=(profile_energies[0][caster].1/profile_energies[1][caster].1-1.).abs();
        assert!(relative<0.05,"selected/promoted path fog energy differs by {relative}: {profile_energies:?}");
    }
    assert!((ratios[0]-ratios[1]).abs()<0.075,"selected/promoted path medium shadow ratios disagree: {ratios:?}");
    Ok(())
}

fn fluid_fixture(kind:&str,n:u32) -> Result<serde_json::Value> {
    if let Some(root)=std::env::var_os("VORK_TEST_FLUID_SCENES") {
        return Ok(serde_json::from_slice(&std::fs::read(Path::new(&root).join(format!("{kind}-{n}.json")))?)?);
    }
    let output=std::process::Command::new("python3")
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .args(["-m","libraries.fluid.gpu_fixture",kind,&n.to_string()]).output()?;
    ensure!(output.status.success(),"fluid fixture: {}",String::from_utf8_lossy(&output.stderr));
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn fluid_impossible_container_reports_its_cause() -> Result<()> {
    let renderer = crate::test_renderer()?;
    let document=fluid_fixture("drop_blocked",120)?;
    let installed=renderer.install(crate::test_scene(document,Path::new("."))?,Path::new("."))?;
    let mut frame=installed.create_frame_slot(&renderer,Default::default())?;
    let error=frame.submit(&installed,&renderer,0.).and_then(|()|frame.collect(&renderer).map(|_|())).unwrap_err();
    let message=format!("{error:#}");
    for part in ["small liquid drop_blocked","capacity","surface_grid","reported value: 2"] {ensure!(message.contains(part),"{message}");}
    ensure!(!message.contains("implicit grid has"),"{message}");
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn fluid_zero_cell_reports_its_cause() -> Result<()> {
    let renderer = crate::test_renderer()?;
    let mut document=fluid_fixture("drop_empty",120)?;
    // A zero wall over an empty domain publishes identically-zero interior
    // cells. The library must reject them through its own receipt, even when
    // the zero rest-volume residual is accepted on the first evaluation.
    for program in document["programs"].as_object_mut().unwrap().values_mut() {
        if let Some(source)=program["source"].as_str() {
            program["source"]=source.replace("return max(-p.y,max(abs(p.x)-.148,abs(p.z)-.048));", "return 0.;").into();
        }
    }
    let installed=renderer.install(crate::test_scene(document,Path::new("."))?,Path::new("."))?;
    let mut frame=installed.create_frame_slot(&renderer,Default::default())?;
    let error=frame.submit(&installed,&renderer,0.).and_then(|()|frame.collect(&renderer).map(|_|())).unwrap_err();
    let message=format!("{error:#}");
    for part in ["small liquid drop_empty","zero cell","surface_grid","reported value: 5"] {ensure!(message.contains(part),"{message}");}
    ensure!(!message.contains("implicit grid has"),"{message}");
    Ok(())
}

// Independent midpoint XZ quadrature, exact along Y. This reads the final public
// field, never the solver's reductions or correction objective.
fn fluid_surface_volume(values:&[f32],dims:[u32;3],samples:usize) -> f64 {
    let [nx,ny,nz]=dims.map(|v|v as usize);
    let mut sum=0.;
    for z in -1..nz as isize {for y in -1..ny as isize {for x in -1..nx as isize {
        let f:[f64;8]=std::array::from_fn(|k| {
            let xx=(x+(k&1) as isize).clamp(0,nx as isize-1) as usize;
            let yy=(y+((k>>1)&1) as isize).clamp(0,ny as isize-1) as usize;
            let zz=(z+(k>>2) as isize).clamp(0,nz as isize-1) as usize;
            f64::from(values[xx+nx*(yy+ny*zz)])
        });
        let weight=if x<0 || x+1==nx as isize {0.5} else {1.} * if y<0 || y+1==ny as isize {0.5} else {1.} * if z<0 || z+1==nz as isize {0.5} else {1.};
        if f.iter().all(|&v|v>=0.) {continue;}
        if f.iter().all(|&v|v<=0.) {sum+=weight;continue;}
        for iz in 0..samples {for ix in 0..samples {
            let u=(ix as f64+0.5)/samples as f64;let v=(iz as f64+0.5)/samples as f64;
            let a=(f[0]*(1.-u)+f[1]*u)*(1.-v)+(f[4]*(1.-u)+f[5]*u)*v;
            let b=(f[2]*(1.-u)+f[3]*u)*(1.-v)+(f[6]*(1.-u)+f[7]*u)*v;
            sum+=weight*(if a<0. {if b<=0. {1.} else {-a/(b-a)}} else if b<0. {-b/(a-b)} else {0.})/(samples*samples) as f64;
        }}
    }}}
    sum
}

#[test]
#[ignore = "requires a real GPU and pinned Slang"]
fn fluid_edge_publication_and_physical_volume() -> Result<()> {
    let renderer=Renderer::new()?;
    for kind in ["edge","corner","interior","edge_thin","thin","sparse","dam","drop","drop_shift","drop_tiny","drop_pair","drop_thin","drop_empty"] {
        if std::env::var("VORK_TEST_FLUID_KIND").is_ok_and(|v|v!=kind) {continue;}
        let mut errors=Vec::new();
        for n in if kind=="thin" || kind=="sparse" {vec![32,64]} else if kind=="dam" {vec![128]} else if kind.starts_with("drop") {vec![120]} else {vec![32]} {
            let document=fluid_fixture(kind,n)?;
            let scene=crate::test_scene(document,Path::new("."))?;
            let decoded=buffers::SceneBuffers::decode(&scene,Path::new("."))?;
            let attrs=buffers::ProgramBuffers::install(&renderer,&scene,&decoded)?;
            let compute=compute::Compute::test_install(&renderer,&scene,Path::new("."),Some(&attrs),&[],&[],&[],vork_time::Time::integer(24),1.)?.unwrap();
            let mut cursor=compute.cursor.lock().unwrap();
            let mut encoder=renderer.device.create_command_encoder(&Default::default());
            compute.advance_with(&mut encoder,&mut cursor,0.,|encoder| {
                let done=std::mem::replace(encoder,renderer.device.create_command_encoder(&Default::default()));
                renderer.queue.submit([done.finish()]);
            })?;
            let name=if ["edge","corner","interior","edge_thin"].contains(&kind) {"film_surface"} else {"surface"};
            let binding=&attrs.bindings[name];let bytes=u64::from(binding.count*binding.width)*4;
            let readback=renderer.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:bytes,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
            for span in &binding.spans {
                encoder.copy_buffer_to_buffer(&attrs.storage[span.page as usize],u64::from(span.offset)*4,&readback,u64::from(span.first*binding.width)*4,u64::from(span.count*binding.width)*4);
            }
            let submitted=renderer.queue.submit([encoder.finish()]);
            let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|r|{let _=tx.send(r);});
            renderer.wait(Some(submitted))?;rx.recv()??;
            let words=readback.get_mapped_range(..).to_vec();readback.unmap();
            if kind=="drop" {
                // Advance through actual solver ticks, then restore/replay the
                // initial publication. The fallback is retained simulation state.
                let mut encoder=renderer.device.create_command_encoder(&Default::default());
                for time in [0.002,0.] {
                    compute.advance_with(&mut encoder,&mut cursor,time,|encoder| {
                        let done=std::mem::replace(encoder,renderer.device.create_command_encoder(&Default::default()));
                        renderer.queue.submit([done.finish()]);
                    })?;
                }
                for span in &binding.spans {
                    encoder.copy_buffer_to_buffer(&attrs.storage[span.page as usize],u64::from(span.offset)*4,&readback,u64::from(span.first*binding.width)*4,u64::from(span.count*binding.width)*4);
                }
                let submitted=renderer.queue.submit([encoder.finish()]);
                let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|r|{let _=tx.send(r);});
                renderer.wait(Some(submitted))?;rx.recv()??;
                let restored=readback.get_mapped_range(..).to_vec();readback.unmap();
                ensure!(words==restored,"small-drop field changed after backward seek");
            }
            let values:&[f32]=bytemuck::cast_slice(&words);
            assert!(values.iter().all(|v|v.is_finite()),"failed volume solve for {kind} n={n}");
            if name=="film_surface" {
                let dims=scene.buffers["film_surface"].dims.unwrap();
                let volume=fluid_surface_volume(values,dims,32)*12./dims.iter().map(|&v|v as f64).product::<f64>();
                let depth=if kind=="edge_thin" {1e-8} else {0.04};let target=depth*0.25;
                eprintln!("film {kind}: published={volume}, ledger={target}");
                assert!((volume-target).abs()<target*0.00001,"film boundary water vanished");
                let at=match kind {"edge"|"edge_thin"=>8,"corner"=>0,_=>10};
                let index=at%4+4*(1+3*(at/4));
                assert!((f64::from(values[index])+depth).abs()<depth*0.0001,"wetness centre lost its published depth");
            } else {
                let drop=kind.starts_with("drop");
                let volume=fluid_surface_volume(values,scene.buffers["surface"].dims.unwrap(),if drop {512} else {128})*(if drop {0.3} else if kind=="dam" {4.} else {2.}/n as f64).powi(3);
                let target=if drop {
                    // Independent authored-lattice count, not the GPU ledger.
                    let nx=if kind=="drop_empty" {0} else if kind=="drop_pair" {20} else {10};
                    let mut count=0;
                    for z in 0..10 {for y in 0..10 {for x in 0..nx {
                        let dx=(x as f64-0.5*(nx-1) as f64)*0.002;
                        let dy=(y as f64-4.5)*0.002;let dz=(z as f64-4.5)*0.002;
                        let r2=if kind=="drop_pair" {(dx-0.014).powi(2).min((dx+0.014).powi(2))+dy*dy+dz*dz}
                            else {dx*dx+(dy/if kind=="drop_thin" {0.4} else {1.}).powi(2)+dz*dz};
                        if r2<(if kind=="drop_tiny" || kind=="drop_pair" {0.0045_f64} else {0.009_f64}).powi(2) {count+=1;}
                    }}}
                    count as f64*0.002_f64.powi(3)
                } else if kind=="dam" {38400.*0.04_f64.powi(3)} else if kind=="sparse" {0.12_f64.powi(3)} else {64.*0.08_f64.powi(3)};
                if let Some(root)=std::env::var_os("VORK_TEST_FLUID_OUT") {
                    std::fs::write(Path::new(&root).join(format!("{kind}-{n}.f32")),&words)?;
                }
                let error=if target==0. {volume.abs()} else {(volume/target-1.).abs()};errors.push(error);
                eprintln!("bulk {kind} n={n}: published={volume}, rest={target}, relative error={error}");
                assert!(error<if drop {0.0001} else {0.002},"{kind} independent rendered volume error {error}");
            }
        }
        if errors.len()==2 {assert!(errors[1]<=errors[0]+0.0002,"resolution worsened volume: {kind} {errors:?}");}
    }
    Ok(())
}

// Frozen fdac96eb9 water authoring contract. External mesh/HDR/ceramic bytes are
// unchanged in examples/water; preserve the old table module import as well.
fn water_040_fixture() -> Result<(Scene,PathBuf)> {
    let mut document:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/water-0.4.0/scene.json"))?;
    assert!(!document["buffers"].as_object().unwrap().contains_key("fluid_status"));
    for program in document["programs"].as_object_mut().unwrap().values_mut() {
        assert!(program["bindings"].get("fluid_status").is_none());
        if program["file"] == "table.slang" {
            program.as_object_mut().unwrap().remove("file");
            program["source"]=include_str!("../tests/fixtures/water-0.4.0/table.slang").into();
        }
    }
    let base=std::env::var_os("VORK_TEST_WATER_ROOT").map(PathBuf::from)
        .unwrap_or_else(||Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/water"));
    Ok((crate::test_scene(document,&base)?,base))
}

#[test]
#[ignore = "requires pinned Slang and production-size fluid buffers; no GPU"]
fn water_040_without_status_checks_on_cpu() -> Result<()> {
    let (scene,base)=water_040_fixture()?;
    let mut renderer=pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
    renderer.check_only=true;
    renderer.install(scene,&base)?;
    Ok(())
}

#[test]
#[ignore = "requires real GPU, pinned Slang, and production-size fluid buffers"]
fn water_040_without_status_loads_and_renders() -> Result<()> {
    let renderer=Renderer::new()?;
    let (mut scene,base)=water_040_fixture()?;
    scene.render.resolution=[160,90];
    scene.render.settings=vork_scene::Settings::for_level(0,Default::default());
    scene.render.settings.samples=1;
    let installed=renderer.install(scene,&base)?;
    let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
    frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
    let pixels=frame.collect(&renderer)?.linear_rgba.context("missing water frame")?;
    assert_eq!(pixels.len(),160*90);
    assert!(pixels.iter().flatten().all(|v|v.is_finite()));
    assert!(pixels.iter().any(|p|p[..3].iter().any(|v|*v>0.01)),"empty water frame");
    Ok(())
}

// Greenlight's authored-size water frame, with only the internal admission
// policy changed. Keep the exact strategy available to each profile as its oracle.
fn water_boundary_admission_preserves_picture(level:u8) -> Result<()> {
    let base=std::env::var_os("VORK_TEST_WATER_ROOT").map(PathBuf::from)
        .unwrap_or_else(||Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/water"));
    let document=serde_json::from_str(include_str!("../../../examples/water/scene.json"))?;
    let scene=crate::test_scene(document,&base)?;
    let resolution=scene.render.resolution;
    let renderer=Renderer::new()?;
    let mut expected:Option<(Vec<u8>,Vec<u32>)>=None;
    for approximate in [false,true] {
        let mut scene=scene.clone();
        let intent=serde_json::from_value(serde_json::json!({"samples":8,"seed":11,"denoise":false}))?;
        scene.render.settings=vork_scene::Settings::for_level(level,intent);
        scene.render.settings.approximate_field_boundaries=approximate;
        let installed=renderer.install(scene,&base)?;
        let mut frame=installed.create_frame_slot(&renderer,ReadbackPrecision::Float)?;
        frame.submit_outputs(&installed,&renderer,0.,ReadbackOutputs {
            color:true,float:FloatOutputs {color:true,..Default::default()},..Default::default()
        })?;
        let output=frame.collect(&renderer)?;
        if let Some(directory)=std::env::var_os("VORK_TEST_OUTPUT_DIR") {
            std::fs::create_dir_all(&directory)?;
            image::save_buffer(Path::new(&directory).join(if approximate {"approximate.png"} else {"exact.png"}),
                &output.rgba,resolution[0],resolution[1],image::ColorType::Rgba8)?;
        }
        let linear=output.linear_rgba.context("missing water linear frame")?;
        ensure!(linear.len()==(resolution[0]*resolution[1]) as usize,"incomplete water frame");
        ensure!(linear.iter().flatten().all(|v|v.is_finite()),"nonfinite water frame");
        ensure!(linear.iter().any(|p|p[..3].iter().any(|v|*v>0.01)),"empty water frame");
        if let Some(directory)=std::env::var_os("VORK_TEST_OUTPUT_DIR") {
            exr::prelude::write_rgba_file(Path::new(&directory).join(if approximate {"approximate.exr"} else {"exact.exr"}),
                resolution[0] as usize,resolution[1] as usize,|x,y| {
                    let p=linear[y*resolution[0] as usize+x];(p[0],p[1],p[2],p[3])
                })?;
        }
        let bits:Vec<_>=linear.iter().flatten().map(|v|v.to_bits()).collect();
        if let Some((display,reference))=&expected {
            ensure!(output.rgba==*display,"profile level {level}: water display pixels changed");
            let changed=bits.iter().zip(reference.iter()).filter(|(a,b)|a!=b).count();
            let max_error=bits.iter().zip(reference.iter()).map(|(&a,&b)|(f32::from_bits(a)-f32::from_bits(b)).abs()).fold(0.,f32::max);
            ensure!(changed==0,"profile level {level}: water changed {changed} scene-linear components; max-abs={max_error}");
        } else {expected=Some((output.rgba,bits));}
    }
    Ok(())
}

#[test]
#[ignore = "requires real GPU, pinned Slang, and production-size fluid buffers"]
fn draft_water_boundary_admission_preserves_picture() -> Result<()> {
    water_boundary_admission_preserves_picture(0)
}

#[test]
#[ignore = "requires real GPU, pinned Slang, and production-size fluid buffers"]
fn previs_water_boundary_admission_preserves_picture() -> Result<()> {
    water_boundary_admission_preserves_picture(3)
}

#[test]
#[ignore = "requires real GPU, pinned Slang, and production-size fluid buffers"]
fn final_water_boundary_admission_preserves_picture() -> Result<()> {
    water_boundary_admission_preserves_picture(1)
}
