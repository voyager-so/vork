//! Shared build-time and runtime camera dispatch generation.
use std::path::PathBuf;

#[derive(Clone, Default)]
pub(crate) struct Native {
    pub modules: Vec<(String,String)>,
    pub include: Vec<PathBuf>,
    pub standard_camera: bool,
}

pub(crate) fn generate<'a>(abi: &vork_slang::Abi, instances: impl IntoIterator<Item=(u32,&'a vork_slang::CameraProgram)>) -> Native {
    use std::fmt::Write;
    let mut native=Native {modules:abi.native_modules(),include:Vec::new(),standard_camera:false};
    let mut imports="import vork;\n".to_owned();
    let mut cameras=String::new();let mut sensors=String::new();
    for (id,program) in instances {
        let name=format!("camera_program_{}",program.camera_entry.as_ref().or(program.sensor_entry.as_ref()).unwrap());
        if !native.modules.iter().any(|(module,_)|module==&name) {
            writeln!(imports,"import {name};").unwrap();
            let mut source=program.native.clone();
            for dependency in &program.dependencies {
                if let Ok(bytes)=std::fs::read(dependency) {writeln!(source,"// dependency {}",blake3::hash(&bytes)).unwrap();}
            }
            native.modules.push((name,source));
            native.include.extend(program.include.clone());
        }
        if let Some(entry)=&program.camera_entry {
            let entry=entry.split("_0_").next().unwrap();
            writeln!(cameras,"case {id}u: {{float4 r[3];{entry}(ctx,base,r);ray.origin=r[0].xyz;ray.wavelength_scale=r[0].w;ray.direction=r[1].xyz;ray.t_min=r[1].w;ray.weight=r[2].xyz;ray.t_max=r[2].w;break;}}").unwrap();
        }
        if let Some(entry)=&program.sensor_entry {
            let entry=entry.split("_0_").next().unwrap();
            writeln!(sensors,"case {id}u: {{float4 r[1];{entry}(ctx,base,r);return r[0].xyz;}}").unwrap();
        }
    }
    let contexts=include_str!("../../vork-slang/src/camera_context.slang")
        .replace("CameraContext","PathCameraContext").replace("CameraRay","PathCameraRay").replace("SensorContext","PathSensorContext");
    native.modules.push(("camera_dispatch".into(),format!("{imports}\n{contexts}\npublic PathCameraRay camera_dispatch(uint id,uint base,PathCameraContext p) {{CameraContext ctx=CameraContext(p.film_size,p.uv,p.lens_sample,p.time,p.focus_distance,p.origin,p.right,p.up,p.forward);PathCameraRay ray;ray.origin=ctx.origin;ray.direction=ctx.forward;switch(id) {{{cameras}}}return ray;}}\npublic float3 sensor_dispatch(uint id,uint base,PathSensorContext p) {{SensorContext ctx=SensorContext(p.path_length,p.wavelength,p.uv,p.time);switch(id) {{{sensors}}}return float3(1.);}}")));
    native
}
