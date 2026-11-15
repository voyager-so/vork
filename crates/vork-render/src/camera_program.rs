//! Camera programs use the retained material-word buffer and its animation plan.
use super::*;
use vork_scene::v2;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(crate) struct State {
    pub program: std::sync::Arc<vork_slang::CameraProgram>,
    pub words: Vec<u32>,
    pub offset: usize,
    pub focus: f32,
    pub time: f32,
    pub evaluated: Evaluated,
}
type Certificate=Option<[[f32;4];4]>;
#[derive(Clone,Default)]
struct SampleCalibration {
    words:Vec<u32>, focus:u32, time:u32, aspects:Vec<(u32,Certificate)>,
}
/// Evaluated optics belong to this sampled node, alongside its evaluated words.
#[derive(Default)]
pub(crate) struct Evaluated(std::sync::Mutex<SampleCalibration>);
impl Clone for Evaluated {
    fn clone(&self)->Self {Self(std::sync::Mutex::new(self.0.lock().unwrap().clone()))}
}
impl State {
    pub fn calibration(&self, aspect: f32) -> Result<Option<[[f32;4];4]>> {
        let mut evaluated=self.evaluated.0.lock().unwrap();
        let time=self.time.to_bits();
        if evaluated.words!=self.words || evaluated.focus!=self.focus.to_bits() || evaluated.time!=time {
            *evaluated=SampleCalibration {words:self.words.clone(),focus:self.focus.to_bits(),time,aspects:Vec::new()};
        }
        if let Some((_,certificate))=evaluated.aspects.iter().find(|(a,_)|*a==aspect.to_bits()) {return Ok(*certificate);}
        let certificate=self.evaluate_calibration(aspect)?;
        evaluated.aspects.push((aspect.to_bits(),certificate));
        Ok(certificate)
    }
    fn evaluate_calibration(&self,aspect:f32)->Result<Certificate> {
        let c=self.program.calibration(&programs::abi(), &self.words, aspect, self.focus, self.time)?;
        let Some(c)=c.filter(|c|c[1][3]!=0.) else {return Ok(None)};
        ensure!(c.iter().flatten().all(|v|v.is_finite()) && c[0][0]>0. && c[0][1]>0.
            && c[1][3]==1. && c[1][0]>0. && c[1][1]>c[1][0] && (c[1][2]==0. || c[1][2]==1.)
            && c[2][0]>=0. && (c[2][0]==0. || (c[2][1]>0. && c[3][0]>0.
                && c[2][2]>=0. && c[2][2].fract()==0.)), "invalid projective camera calibration");
        Ok(Some(c))
    }
}

pub(crate) struct Instance {
    pub standard_camera: bool,
    pub node: usize,
    pub material: String,
    pub id: u32,
    pub offset: usize,
    pub spectral: bool,
    pub unit_weight: bool,
    pub uses_time: bool,
    pub words: Vec<u32>,
    pub program: std::sync::Arc<vork_slang::CameraProgram>,
}
pub(crate) use crate::camera_dispatch_native::Native;

/// Resolve all possible view cameras once with the same default and cut semantics.
/// Aspects belong to views, not to the camera's position in the node list.
pub(crate) fn views(scene: &v2::Scene, nodes: &[EvalNode], animation: &animation::Plan, initial_time: impl Into<vork_time::Request>) -> BTreeMap<String,Vec<[u32;2]>> {
    let default=nodes.iter().find(|n|matches!(n.node.kind,NodeKind::Camera {..})).map(|n|n.node.name.as_str());
    let start=scene.render.camera.clone().or_else(||animation.start_camera(&scene.render,initial_time));
    let fallback=start.as_deref().or(default);
    let primary:Vec<_>=animation.camera_names().chain(fallback).collect();
    let mut views:BTreeMap<String,Vec<[u32;2]>>=BTreeMap::new();
    for name in &primary {views.entry((*name).into()).or_default().push(scene.render.resolution);}
    for image in scene.images.values().filter_map(|i|i.render.as_ref()) {
        let names=image.camera.as_deref().or(fallback).into_iter().collect::<Vec<_>>();
        for name in names {views.entry(name.into()).or_default().push(image.resolution);}
    }
    for resolutions in views.values_mut() {resolutions.sort_unstable();resolutions.dedup();}
    views
}

pub(crate) fn install(r: &Renderer, scene: &v2::Scene, nodes: &mut [EvalNode], base: &Path, views: &BTreeMap<String,Vec<[u32;2]>>) -> Result<Vec<Instance>> {
    let dynamic=materials::animated_inputs(scene)?;
    let abi=programs::abi();
    let mut result=Vec::new();
    for (node,n) in nodes.iter_mut().enumerate() {
        if n.camera.is_some() || !views.contains_key(&n.node.name) {continue;}
        let NodeKind::Camera {material:name,focus_object}=&n.node.kind else {continue;};
        let default: v2::Material=serde_json::from_value(serde_json::json!({"program":"std/camera"}))?;
        let material=name.as_ref().map_or(&default,|name|&scene.materials[name]);
        let name=name.clone().unwrap_or_default();
        let (source,origin)=if let Some(source)=vork_slang::module_source(&material.program) {
            (source.to_owned(),PathBuf::from(format!("{}.slang",material.program)))
        } else {programs::program_source(base,&material.program,scene.programs.get(&material.program).context("unknown camera program")?)?};
        let mut hash=blake3::Hasher::new();
        hash.update(b"camera-optics-v3");hash.update(abi.cache_identity()?.as_bytes());hash.update(source.as_bytes());hash.update(origin.to_string_lossy().as_bytes());
        let key=hash.finalize();let tag=key.to_hex();
        let program=programs::cached(r,key,&vork_slang::Compiler::discover(),
        &abi, &source, &origin,
            |compiler|compiler.compile_camera(&abi,&source,&origin,&tag[..12]))?;
        let values=material.inputs.iter().map(|(k,v)|Ok((k.clone(),serde_json::to_value(v)?))).collect::<Result<BTreeMap<_,_>>>()?;
        let words=program.inputs.pack(&values,|name,_|bail!("camera input {name} must be numeric"))?;
        let varying=program.inputs.varying_words(&words,dynamic.get(&name).unwrap_or(&BTreeSet::new()));
        let spectral=program.needs_spectral(&abi,&varying)?;
        let unit_weight=program.unit_weight(&abi,&varying)?;
        let uses_time=program.uses_time(&abi)?;
        let program=std::sync::Arc::new(program);
        n.camera=Some(State {program:program.clone(),words:words.clone(),offset:0,focus:if focus_object.is_some() {f32::MAX} else {-1.},time:0.,evaluated:Default::default()});
        n.dynamic=true;
        // Sampled clones inherit certificates for their actual view aspects.
        for &[w,h] in &views[&n.node.name] {
            n.camera.as_ref().unwrap().calibration(w as f32/h as f32)?;
        }
        result.push(Instance {standard_camera:material.program=="std/camera",node,material:name.clone(),id:result.len() as u32+1,offset:0,spectral,unit_weight,uses_time,words,program});
    }
    Ok(result)
}

/// The radius about the camera position that holds every ray origin std/camera
/// (libraries/std/camera.slang) produces under certificate `c` at `resolution`,
/// in camera-axis units. A ray starts at its near-plane point: the position,
/// plus the lens point, plus clip_start times (forward + aim); an orthographic
/// ray starts at the film point pushed clip_start forward. Film coordinates
/// reach the screen edge plus the path film filter's 1.5 pixel support.
/// Propagated rays start at the lens or film point itself, inside the same ball.
fn origin_radius(c: &[[f32;4];4], resolution: [u32;2]) -> f32 {
    let [screen, clip, aperture, _] = c;
    let film = |axis: usize| (1. + 3. / resolution[axis] as f32 + screen[2 + axis].abs()) / screen[axis];
    let film = film(0).hypot(film(1));
    // The shared raster lens point and this sample's own are each within the aperture.
    let lens = 2. * aperture[0];
    if clip[2] != 0. {lens + film + clip[0]} else {lens + clip[0] * (1. + film + if aperture[0] > 0. {aperture[0] / aperture[1]} else {0.})}
}
/// Balls holding every origin that any sample of any reachable camera can
/// start from, for the life of the install; None when some camera's are not
/// known. A camera is known when it is std/camera with constant inputs, its
/// position is at rest, and an open aperture does not focus on an object (the
/// aim then scales with a distance evaluated each frame). `resolutions` are
/// all view sizes: every camera is bounded at each.
pub(crate) fn origin_balls(instances: &[Instance], nodes: &[EvalNode], animated: &BTreeMap<String,BTreeSet<String>>, resolutions: &[[u32;2]], at_rest: impl Fn(usize)->bool) -> Result<Option<Vec<(Vec3,f32)>>> {
    if instances.is_empty() || resolutions.is_empty() {return Ok(None);}
    let mut balls=Vec::new();
    for p in instances {
        let node=&nodes[p.node];
        if !p.standard_camera || animated.contains_key(&p.material) || !at_rest(p.node) {return Ok(None);}
        let state=node.camera.as_ref().context("camera program installed")?;
        let focus_object=matches!(node.node.kind,NodeKind::Camera {focus_object:Some(_),..});
        let mut radius=0_f32;
        for &[w,h] in resolutions {
            let Some(c)=state.calibration(w as f32/h as f32)? else {return Ok(None);};
            if c[2][0]>0. && focus_object {return Ok(None);}
            radius=radius.max(origin_radius(&c,[w,h]));
        }
        // The view frame is orthonormal about the node's world position
        // (values::node_view), so camera-axis units are world units.
        let centre=node.world.w_axis.truncate();
        // f32 evaluation of the origin in the kernel stays far inside this slack.
        let radius=radius*1.001+1e-5*(centre.abs().max_element()+radius);
        if !centre.is_finite() || !radius.is_finite() {return Ok(None);}
        balls.push((centre,radius));
    }
    Ok(Some(balls))
}

pub(crate) fn native(instances: &[Instance]) -> Native {
    let mut native=crate::camera_dispatch_native::generate(&programs::abi(),instances.iter().map(|i|(i.id,i.program.as_ref())));
    native.standard_camera=instances.len()==1 && instances[0].id==1 && instances[0].standard_camera;
    native
}

pub(crate) fn selection(scene: &InstalledScene, node: usize, state: &EvalNode, aspect: f32) -> [u32;4] {
    let state=state.camera.as_ref().expect("camera program installed");
    let uncalibrated=state.calibration(aspect).expect("validated calibration").is_none();
    scene.camera_programs.iter().find(|p|p.node==node).map_or([0;4],|p|[p.id,p.offset as u32,u32::from(p.spectral || scene.spectral)|(u32::from(uncalibrated)<<1)|(u32::from(p.program.sensor_entry.is_some())<<2),state.focus.to_bits()])
}

/// Views inherit transport settings. Camera cuts and render images keep every
/// reachable camera in the same traced registry; wavelength mode remains per view.
pub(crate) fn promote(r: &Renderer, scene: &mut v2::Scene, base: &Path, initial_time: impl Into<vork_time::Request>) -> Result<bool> {
    promote_in(r, scene, base, initial_time, &Default::default())
}
/// [`promote`] within a document open, reading media through the sources that open read.
pub(crate) fn promote_in(r: &Renderer, scene: &mut v2::Scene, base: &Path, initial_time: impl Into<vork_time::Request>, opened: &crate::footage::Opened) -> Result<bool> {
    fn collect(nodes: &[v2::Node], out: &mut Vec<EvalNode>) {
        for node in nodes {
            if matches!(node.kind,NodeKind::Camera {..}) {out.push(EvalNode {node:node.without_children(),camera:None,parent:None,world:Mat4::IDENTITY,dynamic:false,embedded:None});}
            collect(&node.children,out);
        }
    }
    let mut nodes=Vec::new();collect(&scene.nodes,&mut nodes);
    let animation=animation::Plan::compile_in(scene,base,None,opened)?;
    let views=views(scene,&nodes,&animation,initial_time);
    let animated=materials::animated_inputs(scene)?;
    let programs=install(r,scene,&mut nodes,base,&views)?;
    let mut uncalibrated=false;
    let mut aperture=false;
    for p in &programs {
        let state=nodes[p.node].camera.as_ref().unwrap();
        let varying=p.program.inputs.varying_words(&p.words,animated.get(&p.material).unwrap_or(&BTreeSet::new()));
        let focus=if matches!(nodes[p.node].node.kind,NodeKind::Camera {focus_object:Some(_),..}) {None} else {Some(state.focus)};
        for &[w,h] in &views[&nodes[p.node].node.name] {
            let aspect=w as f32/h as f32;
            uncalibrated |= state.calibration(aspect)?.is_none();
            aperture |= p.program.aperture_possible(&programs::abi(),&varying,aspect,focus)?;
        }
    }
    let requires_path=uncalibrated || programs.iter().any(|p|p.spectral || !p.unit_weight || p.program.sensor_entry.is_some());
    if !scene.render.settings.integrator.is_path() {
        // Media ownership is already resolved. Dry aperture tracing remains a
        // quality choice and preserves its existing denoise policy.
        let aperture_path = aperture && !requires_path && !scene.render.settings.previs
            && r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES);
        if requires_path || aperture_path {crate::rt::promote_to_path(scene);}
    }
    Ok(uncalibrated)
}

/// Only material instances reachable from geometry enter transport. An unused
/// program declaration cannot select spectra. Inputs use the same varying-word
/// mask as the material compiler; unknown values conservatively retain effects.
pub(crate) fn scene_spectral(r: &Renderer, scene: &v2::Scene, base: &Path) -> Result<bool> {
    fn visit(nodes: &[v2::Node], names: &mut BTreeSet<String>) {
        for n in nodes {
            match &n.kind {
                NodeKind::Mesh {materials,..}=>names.extend(materials.iter().cloned()),
                NodeKind::Asset {materials,..}=>names.extend(materials.values().cloned()),
                NodeKind::Light {medium:Some(m),..}=>{names.insert(m.clone());},
                _=>{},
            }
            visit(&n.children,names);
        }
    }
    let mut names=BTreeSet::new();visit(&scene.nodes,&mut names);
    let dynamic=materials::animated_inputs(scene)?;
    let mut abi=programs::abi();abi.aov_count=scene.render.aovs.len();
    let stages=programs::mesh_stages(r,scene,base)?;
    let names: Vec<_> = names.into_iter().collect();
    // Keep nested results in key order: speculative workers must not change
    // serial error precedence or the first positive spectral proof.
    let analyzed = programs::parallel_map_with_limit(&names, 4, |name| Ok((|| -> Result<bool> {
        let medium=stages.get(name).is_none_or(|s|s.medium);
        let surface=stages.get(name).is_some_and(|s|s.surface);
        let material=&scene.materials[name];
        let varying=dynamic.get(name).cloned().unwrap_or_default();
        if material.program=="std/openpbr" {
            let active=|field:&str| varying.contains(field) || match material.inputs.get(field) {
                None=>false,Some(v2::Input::Number(v))=>*v!=0.,_=>true,
            };
            if active("transmission_weight") && active("transmission_dispersion_scale") {return Ok(true);}
            return Ok(false);
        }
        let (source,origin)=programs::material_source(scene,base,&material.program)?;
        if medium {
            let mut hash=blake3::Hasher::new();hash.update(b"spectral-surface-v1");hash.update(abi.cache_identity()?.as_bytes());hash.update(source.as_bytes());hash.update(origin.to_string_lossy().as_bytes());
            let tag=hash.finalize().to_hex();
            let mut medium_key=hash.clone();medium_key.update(b"medium");
            let program=programs::cached(r,medium_key.finalize(),&vork_slang::Compiler::discover(),
            &abi, &source, &origin,
                |compiler|compiler.compile_medium(&abi,&source,&origin,&tag[..12]))?;
            let values=material.inputs.iter().map(|(k,v)|Ok((k.clone(),serde_json::to_value(v)?))).collect::<Result<BTreeMap<_,_>>>()?;
            if program.needs_spectral(&abi,&program.inputs.effect_words(&values,&varying)?)? {return Ok(true);}
            if !surface {return Ok(false);}
        }
        let program=programs::compile(r,&vork_slang::Compiler::discover(),&abi,&source,&origin)?;
        let values=material.inputs.iter().map(|(k,v)|Ok((k.clone(),serde_json::to_value(v)?))).collect::<Result<BTreeMap<_,_>>>()?;
        if program.needs_spectral(&abi,&program.inputs.effect_words(&values,&varying)?)? {return Ok(true);}
        Ok(false)
    })()))?;
    for spectral in analyzed {
        if spectral? { return Ok(true); }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn aperture_refinement_respects_views_and_world_transport() -> Result<()> {
        let r = crate::test_renderer()?;
        let available=r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES);
        let base:v2::Scene=v2::from_test_value(json!({"version":2,
            "render":{"camera":"main","resolution":[8,8]},
            "materials":{"fog":{"program":{"source":include_str!("../../vork-scene/tests/fixtures/medium.slang")},"inputs":{"height_falloff":0.05}}},
            "nodes":[{"type":"camera","name":"main"},
                {"type":"camera","name":"lens","material":{"program":"std/camera","inputs":{"use_dof":true}}}]}))?;
        let mut unused=base.clone();promote(&r,&mut unused,Path::new("."),0.)?;
        assert!(!unused.render.settings.integrator.is_path());
        let mut active=base.clone();active.render.camera=Some("lens".into());
        promote(&r,&mut active,Path::new("."),0.)?;
        assert_eq!(active.render.settings.integrator.is_path(),available);
        let mut world=base.clone();world.render.camera=Some("lens".into());
        world.nodes.push(v2::from_test_value(json!({"type":"light","shape":"environment","medium":"fog"}))?);
        promote(&r,&mut world,Path::new("."),0.)?;
        assert_eq!(world.render.settings.integrator.is_path(),available,"compiler-proven exponential world supports aperture path transport");
        let mut capture=base.clone();capture.images.insert("view".into(),v2::from_test_value(json!({"render":{"camera":"lens"}}))?);
        promote(&r,&mut capture,Path::new("."),0.)?;
        assert_eq!(capture.render.settings.integrator.is_path(),available);
        let mut cuts=base;cuts.animation=v2::from_test_value(json!([{"tracks":[{"path":["render","camera"],"keys":[[0,"main"],[1,"lens"]]}]}]))?;
        promote(&r,&mut cuts,Path::new("."),0.)?;
        assert_eq!(cuts.render.settings.integrator.is_path(),available);
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn aperture_refinement_covers_clock_and_capture_aspect() -> Result<()> {
        let r = crate::test_renderer()?;
        let available=r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES);
        for (condition,capture,expected) in [("time>0.5",false,true),("aspect<1.",false,false),("aspect<1.",true,true),("false",true,false)] {
            let source=include_str!("../../../libraries/std/camera.slang")
                .replace("i.use_dof && !ortho",condition);
            let mut scene:v2::Scene=v2::from_test_value(json!({"version":2,
                "render":{"resolution":[16,8],"camera":"lens"},
                "programs":{"lens":{"source":source}},
                "nodes":[{"type":"camera","name":"lens","material":{"program":"lens"}}]}))?;
            if capture {scene.images.insert("portrait".into(),v2::from_test_value(json!({"render":{"camera":"lens","resolution":[8,16]}}))?);}
            promote(&r,&mut scene,Path::new("."),0.)?;
            assert_eq!(scene.render.settings.integrator.is_path(),available && expected,"{condition}, capture={capture}");
        }
        Ok(())
    }

    fn scene(source: &str, medium: bool) -> Value {
        let mut value=json!({"version":2,
            "composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[16,16],"background":[0,0,0]},
            "programs":{"test":{"source":source}},
            "materials":{"test":{"program":"test"},"black":{"inputs":{"base_weight":0,"specular_weight":0}}},
            "buffers":{"p":{"type":"vec3","data":[[-10,-10,-2],[10,-10,-2],[0,10,-2]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p"}}},
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1,"clip_start":0.01}}},
                {"type":"mesh","geometry":"g","materials":["test"]}]});
        if medium {
            value["nodes"][1]["materials"]=json!(["black"]);
            value["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","energy":0,"medium":"test"}));
        }
        value
    }

    fn render(r: &Renderer, value: Value, spectral: bool, path: bool) -> Result<[f32;3]> {
        let mut scene: v2::Scene=vork_scene::v2::from_test_value(value)?;
        scene.render.settings=vork_scene::v2::from_test_value(json!({"integrator":if path {"path"} else {"raster"},"samples":32,"denoiser":null}))?;
        let installed=r.install(scene,Path::new("."))?;
        assert_eq!(installed.spectral,spectral,"wavelength output dependency must select transport");
        let mut slot=installed.create_frame_slot(r,ReadbackPrecision::Float)?;
        slot.submit_outputs(&installed,r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        let pixels=slot.collect(r)?.linear_rgba.context("linear readback")?;
        let mut mean=[0.;3];
        for p in &pixels {for c in 0..3 {assert!(p[c].is_finite(),"nonfinite radiance: {p:?}");mean[c]+=p[c]/pixels.len() as f32;}}
        Ok(mean)
    }

    #[test]
    fn sampled_calibration_retains_only_its_true_input_values()->Result<()> {
        let program=std::sync::Arc::new(vork_slang::Compiler::discover().compile_camera(&programs::abi(),
            "struct Inputs {float gain=2.;}; CameraRay camera(CameraContext c,Inputs i){CameraRay r;r.origin=c.origin;r.direction=c.forward;r.t_min=.1;r.t_max=1000.;return r;} CameraCalibration camera_calibration(float aspect,float focus,float time,Inputs i){CameraCalibration c;c.screen=float4(i.gain/aspect,i.gain,time,0.);c.clip=float4(.1,1000.,0.,1.);c.aperture=float4(.1,focus,0.,0.);return c;}",
            Path::new("sampled-calibration.slang"),"sampledcalibration")?);
        let words=program.inputs.pack(&Default::default(),|_,_|unreachable!())?;
        let mut state=State {program,words,offset:0,focus:10.,time:0.,evaluated:Default::default()};
        let first=state.calibration(1.)?.unwrap();
        assert_eq!(first[0],[2.,2.,0.,0.]);
        assert_eq!(state.calibration(1.)?,Some(first));
        assert_eq!(state.evaluated.0.lock().unwrap().aspects.len(),1);
        assert_eq!(state.calibration(2.)?.unwrap()[0],[1.,2.,0.,0.]);
        assert_eq!(state.evaluated.0.lock().unwrap().aspects.len(),2);
        let original=state.clone();
        state.time=3.;state.focus=20.;
        state.words=state.program.inputs.pack(&BTreeMap::from([("gain".into(),json!(4.))]),|_,_|unreachable!())?;
        let changed=state.calibration(2.)?.unwrap();
        assert_eq!(changed[0],[2.,4.,3.,0.]);assert_eq!(changed[2][1],20.);
        assert_eq!(state.evaluated.0.lock().unwrap().aspects.len(),1);
        assert_eq!(original.calibration(1.)?,Some(first));
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn only_reachable_camera_programs_admit_transport() -> Result<()> {
        let r = crate::test_renderer()?;
        let source="struct Inputs {}; CameraRay camera(CameraContext c,Inputs i){CameraRay r;r.origin=c.origin;r.direction=c.forward;r.t_min=.1;r.t_max=1000.;return r;}";
        let base=json!({"version":2,"render":{"resolution":[8,8],"camera":"ordinary"},
            "programs":{"custom":{"source":source}},"materials":{"custom":{"program":"custom"}},
            "nodes":[{"type":"camera","name":"ordinary"},{"type":"camera","name":"custom","material":"custom"}]});
        for (selected,cut,capture,calibrated,path) in [("ordinary",false,false,false,false),("custom",false,false,false,true),("ordinary",true,false,false,true),("ordinary",false,true,false,true),("custom",false,false,true,false)] {
            let mut value=base.clone();value["render"]["camera"]=json!(selected);
            if calibrated {value["programs"]["custom"]["source"]=json!(vork_slang::module_source("std/camera").unwrap());}
            if cut {value["animation"]=json!([{"tracks":[{"path":["render","camera"],"keys":[[0,"ordinary"],[1,"custom"]]}]}]);}
            if capture {value["images"]=json!({"capture":{"render":{"camera":"custom","resolution":[8,8]}}});}
            let installed=r.install(v2::from_test_value(value)?,Path::new("."))?;
            assert_eq!(installed.render.settings.integrator.is_path(),path,"selected={selected},cut={cut},capture={capture}");
            assert_eq!(installed.plan.effects.implicit,path);
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            slot.submit_outputs(&installed,&r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().flatten().all(|v|v.is_finite()));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn installation_validates_only_reachable_camera_aspects() -> Result<()> {
        let r = crate::test_renderer()?;
        let source="struct Inputs {}; CameraRay camera(CameraContext c,Inputs i){CameraRay r;r.origin=c.origin;r.direction=c.forward;r.t_min=.1;r.t_max=1000.;return r;} CameraCalibration camera_calibration(float aspect,float focus,float time,Inputs i){CameraCalibration c;c.screen=float4(aspect==2. && time<1.?1.:-1.,1.,0.,0.);c.clip=float4(.1,1000.,0.,1.);return c;}";
        let base=json!({"version":2,"render":{"resolution":[8,8],"camera":"main"},
            "programs":{"aspect":{"source":source}},"materials":{"aspect":{"program":"aspect"}},
            "buffers":{"p":{"type":"vec3","data":[[0,0,-2],[0,1,-2]]},"w":{"type":"f32","data":[0.2,0.2]},"c":{"type":"u32","data":[2]}},
            "geometry":{"hair":{"topology":"curves","curve_vertex_counts":"c","widths":"w","attributes":{"POSITION":"p"}}},
            "nodes":[{"type":"camera","name":"main"},{"type":"camera","name":"capture","material":"aspect"},{"type":"mesh","geometry":"hair"}]});
        let unused=r.install(v2::from_test_value(base.clone())?,Path::new("."))?;
        assert_eq!(unused.camera_programs.len(),1);
        assert!(unused.nodes.iter().find(|n|n.node.name=="capture").unwrap().camera.is_none());
        let mut capture=base.clone();
        capture["images"]=json!({"view":{"render":{"camera":"capture","resolution":[16,8]}}});
        // Exercise installation directly, before curve sizing or any sampled view
        // can populate the retained certificate. No primary-aspect prerequisite.
        let input:v2::Scene=v2::from_test_value(capture.clone())?;
        let mut nodes=input.nodes.iter().map(|node|EvalNode {node:node.clone(),camera:None,parent:None,world:Mat4::IDENTITY,dynamic:false,embedded:None}).collect::<Vec<_>>();
        let animation=animation::Plan::compile_with_buffers(&input,Path::new("."),None)?;
        let selected=views(&input,&nodes,&animation,0.);
        install(&r,&input,&mut nodes,Path::new("."),&selected)?;
        assert_eq!(nodes[1].camera.as_ref().unwrap().evaluated.0.lock().unwrap().aspects.iter().map(|(aspect,_)|*aspect).collect::<Vec<_>>(),vec![2f32.to_bits()]);
        let installed=r.install(v2::from_test_value(capture.clone())?,Path::new("."))?;
        assert_eq!(installed.camera_programs.len(),2);
        let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        slot.submit_outputs(&installed,&r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
        assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().flatten().all(|v|v.is_finite()));
        assert!(slot.submit_outputs(&installed,&r,1.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()}).is_err(),"a later selected sample must still reject invalid calibration");
        capture["images"]["view"]["render"]["resolution"]=json!([8,8]);
        assert!(r.install(v2::from_test_value(capture)?,Path::new(".")).err().context("selected invalid capture accepted")?.to_string().contains("invalid projective camera calibration"));
        let mut inherited=base.clone();
        inherited["render"]["resolution"]=json!([16,8]);
        inherited["images"]=json!({"view":{"render":{"resolution":[8,8]}}});
        inherited["animation"]=json!([{"tracks":[{"path":["render","camera"],"keys":[[0,"main"],[1,"capture"]]}]}]);
        r.install(v2::from_test_value(inherited)?,Path::new("."))?;
        let mut selected=base;selected["render"]["camera"]=json!("capture");
        assert!(r.install(v2::from_test_value(selected)?,Path::new(".")).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn signed_spectral_samples_preserve_a_white_environment() -> Result<()> {
        let r = crate::test_renderer()?;
        for composite in [false,true] {
            let mut input:crate::Scene=serde_json::from_value(json!({"version":2,
                "render":{"resolution":[8,8]},
                "programs":{
                    "sensor":{"source":"struct Inputs {}; CameraRay camera(CameraContext c,Inputs i){CameraRay r;r.origin=c.origin;r.direction=c.forward;r.t_min=.1;r.t_max=1000.;return r;} float3 sensor(SensorContext c,Inputs i){return float3(c.wavelength>0.?1.:0.);}"},
                    "copy":{"source":"struct Inputs {Image source;}; float4 composite(CompositeContext c,Inputs i){return compositor_load(i.source,int2(c.uv*c.size));}"}},
                "materials":{"sensor":{"program":"sensor"},"copy":{"program":"copy","inputs":{"source":{"channel":"color"}}}},
                "nodes":[{"type":"camera","material":"sensor"},
                    {"type":"light","shape":"environment","energy":1,"color":[1,1,1]}]}))?;
            if composite {*input.compositor_mut()=serde_json::from_value(json!(["copy"]))?;}
            input.render.settings=serde_json::from_value(json!({"integrator":"path","samples":4096,"denoiser":null}))?;
            let installed=r.install(input,Path::new("."))?;
            assert!(installed.camera_programs[0].spectral);
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            slot.submit_outputs(&installed,&r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            let pixels=slot.collect(&r)?.linear_rgba.context("white spectral readback")?;
            let mean:Vec<_>=(0..3).map(|c|pixels.iter().map(|p|p[c]).sum::<f32>()/pixels.len() as f32).collect();
            for c in 0..3 {assert!((mean[c]-1.).abs()<0.01,"signed spectral white, composite={composite}: {mean:?}");}
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn spectral_emitter_surface_and_medium_receive_path_wavelength() -> Result<()> {
        let r = crate::test_renderer()?;
        // A zero/unset context makes these black. No camera/sensor program is
        // present: the authored output dependency must promote RGB to spectra.
        let sources=[
            ("struct Inputs {}; Closure surface(ShadingContext c,Inputs i) {return unlit(float3(c.wavelength>=380. && c.wavelength<=780. ? 1.:0.));}",false),
            ("struct Inputs {}; Closure surface(ShadingContext c,Inputs i) {return unlit(float3(c.wavelength>=380. && c.wavelength<=780. ? 1.:0.));}",false),
            ("struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;m.emission=float3(c.wavelength>=380. && c.wavelength<=780. ? 1.:0.);return m;}",true),
        ];
        for (source,medium) in sources {
            let mean=render(&r,scene(source,medium),true,false)?;
            // Select the same spectral path for a wavelength-independent
            // reference through its sensor, without assuming a white calibration.
            let mut reference=scene(&source.replace("c.wavelength>=380. && c.wavelength<=780. ? 1.:0.","1."),medium);
            reference["programs"]["sensor"]=json!({"source":"struct Inputs {}; CameraRay camera(CameraContext c,Inputs i) {CameraRay r;r.origin=c.origin+(c.uv.x-.5)*c.right+(.5-c.uv.y)*c.up;r.direction=c.forward;r.t_min=.01;r.t_max=1000.;return r;} float3 sensor(SensorContext c,Inputs i) {return float3(c.wavelength>=380. && c.wavelength<=780. ? 1.:0.);}"});
            reference["materials"]["sensor"]=json!({"program":"sensor"});
            reference["nodes"][0]["material"]=json!("sensor");
            let reference=render(&r,reference,false,false)?;
            eprintln!("spectral medium={medium}: {mean:?}, reference {reference:?}");
            for c in 0..3 {assert!(mean[c]>0.1 && (mean[c]-reference[c]).abs()<0.02*reference[c],"missing scene wavelength: {mean:?}, reference {reference:?}");}
        }
        // A short-wave emitter must also retain its spectral shape rather than
        // receiving one fixed representative wavelength for the whole image.
        let blue="struct Inputs {}; Closure surface(ShadingContext c,Inputs i) {return unlit(float3(c.wavelength<500. ? 1.:0.));}";
        let mean=render(&r,scene(blue,false),true,false)?;
        assert!(mean[2]>0.3 && mean[2]>3.*mean[0] && mean[2]>3.*mean[1],"spectral emission lost its band: {mean:?}");
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn unused_emitter_wavelength_keeps_rgb_in_raster_and_path() -> Result<()> {
        let r = crate::test_renderer()?;
        // Both surface and source are constant RGB even though the source reads
        // wavelength. This also compiles emission dispatch outside path kernels.
        let source="struct Inputs {}; Closure surface(ShadingContext c,Inputs i) {float unused=c.wavelength;return unlit(float3(.25));}";
        for path in [false,true] {
            let mean=render(&r,scene(source,false),false,path)?;
            for channel in mean {assert!((channel-0.25).abs()<0.001,"RGB radiance changed: {mean:?}");}
        }
        Ok(())
    }
}

#[cfg(test)]
mod calibration_validation_tests {
    use super::*;
    #[test]
    fn invalid_projective_camera_calibration_is_refused() -> Result<()> {
        let program=std::sync::Arc::new(vork_slang::Compiler::discover().compile_camera(&programs::abi(),include_str!("../../../libraries/std/camera.slang"),Path::new("std/camera.slang"),"invalidcalibration")?);
        for (field,value) in [("clip_end",0.),("clip_start",-1.)] {
            let inputs=BTreeMap::from([(field.to_owned(),serde_json::json!(value))]);
            let words=program.inputs.pack(&inputs,|_,_|unreachable!())?;
            let state=State {program:program.clone(),words,offset:0,focus:-1.,time:0.,evaluated:Default::default()};
            assert!(state.calibration(1.).is_err(),"invalid {field} must fail camera calibration");
        }
        Ok(())
    }
}
