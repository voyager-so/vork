//! Retained environment evaluation into the existing HDR transport path.
use super::*;
use vork_scene::v2;
use vork_slang::{Compiler, EnvironmentProgram};
#[path = "environment_program_images.rs"]
mod images;

#[cfg(test)]
pub(crate) struct Baked {
    pub image: image::Rgb32FImage,
    pub camera: image::Rgb32FImage,
}
pub(crate) struct Source {
    pub texture:wgpu::Texture,
    pub identity:blake3::Hash,
    pub primary:Option<Primary>,
}
#[derive(Clone)]
pub(crate) struct Primary {
    pub spatial: bool,
    pub uses_time: bool,
    pub modules: Vec<(String,String)>,
    pub include: Vec<PathBuf>,
    pub words: Vec<u32>,
    pub layout: vork_slang::InputLayout,
    pub material: String,
    pub image_data: Option<wgpu::Buffer>,
    /// One exact camera value when the compiler proves a static constant.
    pub constant_camera: Option<wgpu::Buffer>,
}
pub(crate) struct Program {
    key: blake3::Hash,
    size: [u32;2],
    animated: bool,
    uses_time: bool,
    resources: Option<Resources>,
    inputs: Vec<u32>,
    animation: Option<programs::InputAnimation>,
    images: Option<images::Images>,
}
struct Resources {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    time: wgpu::Buffer,
    input: wgpu::Buffer,
    output: wgpu::Texture,
    image_sizes: Vec<[u32;2]>,
    image_data: wgpu::Buffer,
}

impl Primary {
    /// The program as a native module that compose and path kernels link for camera rays.
    fn new(abi: &vork_slang::Abi, program: &EnvironmentProgram, material: &str, words: Vec<u32>, spatial: bool) -> Result<Self> {
        let decode = program.inputs.slang_decode();
        let mut modules = abi.native_modules();
        let mut text = program.slang.clone();
        for dependency in &program.dependencies {
            use std::fmt::Write;
            writeln!(text,"// dependency {}",blake3::hash(&std::fs::read(dependency)?))?;
        }
        let origin = if spatial { "origin" } else { "float3(0.)" };
        // The input word base is a frame value: kernels stay shared across scenes.
        // Check the authored source before node tint or spectral/RGB conversion.
        // Every direct camera consumer (including bent escapes) links this wrapper.
        text.push_str(&format!("\nimport material_bindings;\nimport vork_constants;\npublic float3 primary_environment(uint base,float3 origin,float3 direction,float time) {{{decode}let source=environment(EnvironmentContext({origin},direction,time,true),inputs);if(!all(isfinite(source)) || any(source<float3(0.))) {{InterlockedMax(shader_errors[0],REG_ENVIRONMENT_INVALID);return float3(0.);}}return source;}}"));
        modules.push(("camera_environment".into(),text));
        Ok(Self {spatial,uses_time:program.animated,modules,include:program.include.clone(),words,layout:program.inputs.clone(),material:material.into(),image_data:None,constant_camera:None})
    }
}

pub(crate) fn install_source(
    r: &Renderer,
    scene: &v2::Scene,
    name: &str,
    base: &Path,
) -> Result<(Source, Option<Program>)> {
    install_source_in(r, scene, name, base, &Default::default())
}
/// [`install_source`] within a document open, reading media through the sources that open read.
pub(crate) fn install_source_in(
    r: &Renderer,
    scene: &v2::Scene,
    name: &str,
    base: &Path,
    opened: &crate::footage::Opened,
) -> Result<(Source, Option<Program>)> {
    let material = &scene.materials[name];
    let (source, origin) = if let Some(source) = vork_slang::module_source(&material.program) {
        (
            source.to_owned(),
            PathBuf::from(format!("{}.slang", material.program)),
        )
    } else {
        programs::program_source(
            base,
            &material.program,
            scene
                .programs
                .get(&material.program)
                .context("unknown environment program")?,
        )?
    };
    let values = material
        .inputs
        .iter()
        .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let installed = bake_animated(r, &source, &origin, &values, Some((scene, base, name)), opened)?;
    Ok(installed)
}

/// The Slang compile cache key of an environment program.
pub(crate) fn program_key(abi: &vork_slang::Abi, source: &str, origin: &Path) -> Result<blake3::Hash> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"environment-images-v7");
    hash.update(abi.cache_identity()?.as_bytes());
    hash.update(source.as_bytes());
    hash.update(origin.to_string_lossy().as_bytes());
    Ok(hash.finalize())
}

pub(crate) fn compiled(r: &Renderer, source: &str, origin: &Path) -> Result<EnvironmentProgram> {
    let abi = programs::abi();
    let key = program_key(&abi, source, origin)?;
    let tag = key.to_hex();
    programs::cached(r, key, &Compiler::discover(),
        &abi, &source, &origin,
        |compiler| compiler.compile_environment(&abi, source, origin, &tag[..12]))
}

/// Compile (cached) and evaluate an environment program with these input values.
fn bake_animated(
    r: &Renderer,
    source: &str,
    origin: &Path,
    values: &std::collections::BTreeMap<String, serde_json::Value>,
    animation_source: Option<(&v2::Scene, &Path, &str)>,
    opened: &crate::footage::Opened,
) -> Result<(Source, Option<Program>)> {
    let abi = programs::abi();
    let key = program_key(&abi, source, origin)?;
    let program = compiled(r, source, origin)?;
    if !program.warnings.trim().is_empty() {
        eprintln!("{}", program.warnings.trim());
    }
    let (images, mut words) = if let Some((scene,base,_))=animation_source {
        let (images,words)=images::Images::new(r,scene,base,&program.inputs,values,opened)?;
        (Some(images),words)
    } else {
        (None,program.inputs.pack(values, |name,_|bail!("environment input {name} requires a scene"))?)
    };
    let animation = animation_source.map(|(scene, base, name)|
        programs::InputAnimation::new_in(scene, base, name, &program.inputs, &mut words, opened)).transpose()?.flatten();
    let spatial_fallback=program.spatial && animation_source.is_some_and(|(scene,_,_)|scene.delivery);
    if spatial_fallback { crate::warnings::dropped("spatial_environment", serde_json::json!({"program":origin,"inputs":values}), format!("warning: environment {}: ignored spatial origin dependence; evaluating the directional environment at the origin", origin.display())); }
    if program.spatial && !spatial_fallback {
        ensure!(!program.inputs.fields.iter().any(|f|f.ty==vork_slang::Type::Image),"spatial environments do not yet support image inputs");
        let (scene,_,material)=animation_source.context("spatial environment requires a scene")?;
        fn admitted(nodes: &[v2::Node]) -> bool {
            nodes.iter().all(|node| matches!(node.kind,v2::NodeKind::Node {..}|v2::NodeKind::Camera {..}|v2::NodeKind::Light {..}) && admitted(&node.children))
        }
        ensure!(admitted(&scene.nodes),"origin-dependent environments currently support primary rays only: scene geometry is not admitted");
        let settings=environment_node::Settings::from_scene(scene)?;
        ensure!(settings.medium.is_none(),"spatial environments do not support media");
        fn frame(nodes:&[v2::Node], chain:&mut Vec<String>) -> Result<bool> {
            for node in nodes {
                chain.push(node.name.clone());
                let found=matches!(node.kind,v2::NodeKind::Light {shape:v2::LightShape::Environment,..}) || frame(&node.children,chain)?;
                if found {
                    ensure!(node.transform.translation==[0.;3] && node.transform.scale==[1.;3] && node.transform.rotation.unwrap_or([0.,0.,0.,1.])==[0.,0.,0.,1.] && node.target.is_none(),"spatial environment requires an identity environment frame and ancestors");
                    return Ok(true);
                }
                chain.pop();
            }
            Ok(false)
        }
        let mut chain=Vec::new();frame(&scene.nodes,&mut chain)?;
        for layer in &scene.animation {
            for track in layer.tracks.iter().flatten() {
                let p=&track.path;
                ensure!(!(p.len()>=3 && p[0]=="nodes" && chain.contains(&p[1]) && matches!(p[2].as_str(),"translation"|"rotation"|"scale"|"target")),"spatial environment frame cannot have transform tracks");
            }
            ensure!(layer.clip.as_ref().is_none_or(|clip|!chain.contains(&clip.target)),"spatial environment frame cannot have animation clips");
        }
        return Ok((Source {texture:source_texture(r,[1,1]),identity:key,primary:Some(Primary::new(&abi,&program,material,words,true)?)},None));
    }
    let image_words=images.as_ref().map(|i|i.words(&vork_time::Request::Numeric(0.))).transpose()?.unwrap_or_else(||vec![0]);
    let image_capacity=images.as_ref().map_or(image_words.len(),|images|images.capacity(&image_words));
    crate::gpu_memory::storage_binding(&r.device.limits(),"environment image inputs",(image_capacity*4) as u64,"reduce environment image dimensions or count")?;
    // Camera bindings must retain authored blend baselines, before the lighting
    // bake evaluates its independent animation plan at time zero.
    let mut primary = animation_source.map(|(_,_,material)| {
        let words = images.as_ref().map_or_else(|| words.clone(), |images| images.primary_words(&program.inputs, &words, &image_words));
        Primary::new(&abi,&program,material,words,false)
    }).transpose()?;
    if let Some(animation) = &animation { animation.words(0., &mut words); }
    // Preserve lighting quadrature independently of the camera's exact program
    // evaluation. Installation releases bakes no longer read by transport.
    let size=images.as_ref().and_then(|i|i.size(&image_words)).map_or(constants::ENVIRONMENT_PROGRAM_SIZE,
        |[w,h]|[w.max(constants::ENVIRONMENT_PROGRAM_SIZE[0]),h.max(constants::ENVIRONMENT_PROGRAM_SIZE[1])]);
    let [w,h]=size;
    let mut shader = program.wgsl.clone();
    shader.push_str(&format!("\n@group(0) @binding(0) var<storage,read> words:array<u32>;\n@group(0) @binding(1) var pixels:texture_storage_2d_array<rgba32uint,write>;\n@group(0) @binding(2) var<uniform> environment_time:vec4<f32>;\nfn {}(i:u32)->u32 {{return words[i];}}\n@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{if gid.x>={w}u || gid.y>=u32(environment_time.z) {{return;}}let theta=3.141592653589793*(f32(gid.y)+environment_time.y+.5)/{h}.;let phi=6.283185307179586*((f32(gid.x)+.5)/{w}.-.5);let direction=vec3(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));var result:array<vec4<f32>,2>;{}(vec3<f32>(0.),direction,environment_time.x,0u,&result);textureStore(pixels,vec2<i32>(gid.xy),0,bitcast<vec4<u32>>(result[1]));textureStore(pixels,vec2<i32>(gid.xy),1,bitcast<vec4<u32>>(result[0]));}}\n",vork_slang::MATERIAL_WORD,program.entry));
    if let Some(images)=&images {shader.push_str(&images.shader());}
    let mut hash = blake3::Hasher::new();
    hash.update(bytemuck::cast_slice(&image_words));
    if let Some(images)=&images {hash.update(images.identity().as_bytes());}
    hash.update(shader.as_bytes());
    hash.update(bytemuck::cast_slice(&words));
    if let Some(animation) = &animation {
        hash.update(animation.identity().as_bytes());
    }
    let mut installed = Program {
        key: hash.finalize(),
        size,
        animated: program.animated || animation.is_some() || images.as_ref().is_some_and(images::Images::animated),
        uses_time: program.animated,
        resources: None,
        inputs: words.clone(),
        animation,
        images,
    };
    let d = &r.device;
    let module = crate::timing::shader_module(&d, wgpu::ShaderModuleDescriptor {
        label: Some("environment program"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    // Constant programs may optimize away all input reads. Keep the two bindings
    // explicit so zero-input and reflected-input modules share the bake ABI.
    let mut entries=vec![
            pipelines::buffer_entry(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            ),
            wgpu::BindGroupLayoutEntry {binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::StorageTexture {access:wgpu::StorageTextureAccess::WriteOnly,format:wgpu::TextureFormat::Rgba32Uint,view_dimension:wgpu::TextureViewDimension::D2Array},count:None},
            pipelines::buffer_entry(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            ),
        ];
    entries.push(pipelines::buffer_entry(3,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}));
    let bindings=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("environment program bake"),entries:&entries});
    let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("environment program bake"),
        bind_group_layouts: &[Some(&bindings)],
        immediate_size: 0,
    });
    let pipeline = r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("environment program bake"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    });
    let input = init(
        d,
        "environment inputs",
        bytemuck::cast_slice(if words.is_empty() { &[0] } else { &words }),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let output=source_texture(r,[w,h]);
    let time = init(
        d,
        "environment time",
        &[0; 16],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let image_data=crate::gpu_memory::buffer(d,&wgpu::BufferDescriptor {label:Some("environment image float bits"),size:(image_capacity*4) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST,mapped_at_creation:true});
    image_data.get_mapped_range_mut(..).slice(..image_words.len()*4).copy_from_slice(bytemuck::cast_slice(&image_words));
    image_data.unmap();
    let image_sizes=installed.images.as_ref().map(|i|i.sizes(&image_words)).unwrap_or_default();
    if let Some(primary)=&mut primary {primary.image_data=Some(image_data.clone());}
    let output_view=output.create_view(&wgpu::TextureViewDescriptor {array_layer_count:Some(2),..Default::default()});
    let entries=[entry(0,input.as_entire_binding()),entry(1,wgpu::BindingResource::TextureView(&output_view)),entry(2,time.as_entire_binding()),entry(3,image_data.as_entire_binding())];
    let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("environment program inputs"),layout:&pipeline.get_bind_group_layout(0),entries:&entries});
    installed.resources = Some(Resources {
        pipeline,
        bind,
        time,
        input,
        output,
        image_sizes,
        image_data,
    });
    let mut source=installed.evaluate_source(r,0.,0.)?;
    if primary.is_some() && !installed.animated && program.constant_camera(installed.images.as_ref().is_none_or(images::Images::uniform))? {
        if let Some(primary)=&mut primary {
            let camera=crate::gpu_memory::buffer(d,&wgpu::BufferDescriptor {
                label:Some("constant camera environment"),size:16,
                usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
            });
            let mut encoder=d.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(source.texture.as_image_copy(),wgpu::TexelCopyBufferInfo {
                buffer:&camera,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:None,rows_per_image:None},
            },wgpu::Extent3d {width:1,height:1,depth_or_array_layers:1});
            crate::command_finish::Segments::submit(&r.queue,encoder);
            primary.constant_camera=Some(camera);
        }
    }
    source.primary=primary;
    Ok((source,installed.animated.then_some(installed)))
}

pub(crate) fn source_texture(r:&Renderer,size:[u32;2])->wgpu::Texture {source_texture_layers(r,size,2)}
pub(crate) fn lighting_texture(r:&Renderer,size:[u32;2])->wgpu::Texture {source_texture_layers(r,size,1)}
fn source_texture_layers(r:&Renderer,[width,height]:[u32;2],layers:u32)->wgpu::Texture {
    crate::gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
        label:Some("evaluated environment camera and lighting float bits"),
        size:wgpu::Extent3d {width,height,depth_or_array_layers:layers},mip_level_count:1,sample_count:1,
        dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba32Uint,
        usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::STORAGE_BINDING|wgpu::TextureUsages::COPY_SRC|wgpu::TextureUsages::COPY_DST,view_formats:&[],
    })
}

impl Program {
    pub(super) fn has_media(&self)->bool {self.images.as_ref().is_some_and(|images|images.animated())}
    pub(super) fn uses_time(&self)->bool {self.uses_time}
    /// Identity of evaluated inputs, rather than the animation's clock. STEP
    /// holds and clamped tracks share the same bake and upload history.
    pub(crate) fn sample_key(&self, request: impl Into<vork_time::Request>, media: impl Into<vork_time::Request>) -> Result<Vec<u8>> {
        let (request,media)=(request.into(),media.into());
        let time=request.numeric()?;
        let mut words = self.inputs.clone();
        if let Some(animation) = &self.animation { animation.words_request(&request,time, &mut words)?; }
        let mut key = bytemuck::cast_slice(&words).to_vec();
        if self.uses_time { key.extend_from_slice(&time.to_bits().to_le_bytes()); }
        if let Some(images)=&self.images {key.extend(images.sample_key(&media)?);}
        Ok(key)
    }
    pub(crate) fn evaluate_source(&self,r:&Renderer,request:impl Into<vork_time::Request>,media:impl Into<vork_time::Request>)->Result<Source> {
        let (request,media)=(request.into(),media.into());
        let time=request.numeric()?;
        let resources=self.resources.as_ref().context("environment evaluation resources")?;
        if let Some(images)=&self.images {if images.animated() {images.upload(r,&resources.image_data,&resources.image_sizes,&images.words(&media)?)?;}}
        if let Some(animation)=&self.animation {
            let mut words=self.inputs.clone();animation.words_request(&request,time,&mut words)?;
            r.queue.write_buffer(&resources.input,0,bytemuck::cast_slice(&words));
        }
        r.queue.write_buffer(&resources.time,0,bytemuck::cast_slice(&[time,0.,self.size[1] as f32,0.]));
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&resources.pipeline);pass.set_bind_group(0,&resources.bind,&[]);pass.dispatch_workgroups(self.size[0].div_ceil(8),self.size[1].div_ceil(8),1);}
        crate::command_finish::Segments::submit(&r.queue,encoder);
        let mut hash=blake3::Hasher::new();hash.update(self.key.as_bytes());hash.update(&self.sample_key(&request,&media)?);
        Ok(Source {texture:resources.output.clone(),identity:hash.finalize(),primary:None})
    }
}

// CPU pixels are an explicit test oracle, never a lighting-install dependency.
#[cfg(test)]
impl Source {
    pub(crate) fn read(&self,r:&Renderer)->Result<Baked> {
        let w=self.texture.width();let h=self.texture.height();let stride=(w*16).div_ceil(256)*256;
        let size=stride as u64*h as u64*2;
        let read=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("environment test readback"),size,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(self.texture.as_image_copy(),wgpu::TexelCopyBufferInfo {buffer:&read,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(stride),rows_per_image:Some(h)}},wgpu::Extent3d {depth_or_array_layers:2,..self.texture.size()});
        r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);
        let layers=mapped.chunks_exact((stride*h) as usize).map(|layer| {
            let values:Vec<f32>=layer.chunks_exact(stride as usize).flat_map(|row|row[..(w*16) as usize].chunks_exact(16).flat_map(|p|p[..12].chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())))).collect();
            image::Rgb32FImage::from_raw(w,h,values).unwrap()
        }).collect::<Vec<_>>();
        Ok(Baked {camera:layers[0].clone(),image:layers[1].clone()})
    }
}
#[cfg(test)]
pub(crate) fn install(r:&Renderer,scene:&v2::Scene,name:&str,base:&Path)->Result<(Baked,Option<Program>)> {
    let(source,program)=install_source(r,scene,name,base)?;Ok((source.read(r)?,program))
}
#[cfg(test)]
impl Program {
    pub(crate) fn evaluate(&self,r:&Renderer,time:f32)->Result<Baked> {self.evaluate_source(r,time,time)?.read(r)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn preprocessed_vfr_environment_renders_at_different_scene_fps() -> Result<()> {
        let r=Renderer::new()?;
        let root=std::env::temp_dir().join(format!("vork-vfr-render-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let presentation=[0f32,0.04,0.16,0.20,0.32,0.36,0.52,0.60,0.72,0.84,0.92,1.12];
        let mut concat=String::from("ffconcat version 1.0\n");
        for index in 0..presentation.len() {
            let mut image=image::RgbImage::from_pixel(64,32,image::Rgb([3*(index as u8+1);3]));
            for y in 14..17 {for x in 13+index as u32%5..16+index as u32%5 {image.put_pixel(x,y,image::Rgb([255,153,51]));}}
            image.save(root.join(format!("frame-{index}.png")))?;
            concat.push_str(&format!("file 'frame-{index}.png'\n"));
            if index+1<presentation.len() {concat.push_str(&format!("duration {}\n",presentation[index+1]-presentation[index]));}
        }
        std::fs::write(root.join("frames.ffconcat"),concat)?;
        let encoded=std::process::Command::new("ffmpeg").current_dir(&root)
            .args(["-v","error","-y","-f","concat","-safe","0","-i","frames.ffconcat","-fps_mode","vfr","-c:v","ffv1","-pix_fmt","bgr0","source.mkv"]).output()?;
        anyhow::ensure!(encoded.status.success(),"encode VFR fixture: {}",String::from_utf8_lossy(&encoded.stderr));
        let movie=root.join("source.mkv");
        let mut source=crate::footage::Source::open(&movie)?;
        let times=source.timeline(vork_time::Time::integer(30))?;
        assert_eq!(times.len(),presentation.len());
        assert!(times.windows(3).any(|t|((t[2].as_f64()-t[1].as_f64())-(t[1].as_f64()-t[0].as_f64())).abs()>0.001));
        let frames:Vec<_>=times.iter().map(|time|source.frame(time,vork_time::Time::integer(30),u32::MAX).map(|frame|frame.into_rgb32f())).collect::<Result<_>>()?;
        let argument=serde_json::json!({"file":movie,"output":root.join("weight.exr"),"fps":30,"threshold":0.25,"color_space":"Linear Rec.709"}).to_string();
        let fragment=vork_std::environment::prepare_frames(&argument,source.files(),&times,|time| {
            Ok(image::DynamicImage::ImageRgb32F(frames[times.iter().position(|t|t==time).unwrap()].clone()))
        },|_|None,|_,_|{})?;
        let mut value:serde_json::Value=serde_json::from_str(&fragment)?;
        value["version"]=2.into();
        value["render"]=serde_json::json!({"fps":24,"resolution":[16,16],"background":"environment"});
        value["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"camera"}));
        // Expose lighting radiance on primary escapes as well, so the rendered
        // residual checks both sequence clocks rather than only the backdrop.
        value["programs"]=serde_json::json!({"residual":{"source":"struct Inputs {Image image;Image lighting_weight;}; float3 environment(EnvironmentContext c,Inputs i){float3 d=normalize(c.direction);float2 uv=float2(atan2(d.z,d.x)/6.283185307179586+.5,acos(clamp(d.y,-1.,1.))/3.141592653589793);return sample(i.image,uv,c).rgb*sample(i.lighting_weight,uv,c).r;}"}});
        value["materials"]["environment_radiance"]["program"]="residual".into();
        let installed=r.install(v2::from_test_value(value.clone())?,&root)?;
        let render=|scene:&InstalledScene,time:f32|->Result<Vec<[f32;4]>> {
            let mut slot=scene.create_frame_slot(&r,ReadbackPrecision::Float)?;
            slot.submit_outputs(scene,&r,time,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            Ok(slot.collect(&r)?.linear_rgba.unwrap())
        };
        let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
        for index in [10usize,0,3,7,times.len()-1,2,10,0] {
            // The same program over this index's static source and weight frames:
            // both sequence clocks must select them.
            let mut reference=value.clone();reference["animation"]=serde_json::json!([]);
            for image in ["environment_image","environment_lighting_weight"] {
                let file=reference["images"][image]["file"].as_str().context("sequence file")?.replace("%06d",&format!("{index:06}"));
                reference["images"][image]["file"]=serde_json::json!(file);
            }
            let reference=r.install(v2::from_test_value(reference)?,&root)?;
            let expected=render(&reference,0.)?;
            slot.submit_outputs(&installed,&r,&times[index],ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            let actual=slot.collect(&r)?.linear_rgba.unwrap();
            for (a,b) in actual.iter().zip(expected) {for c in 0..3 {assert!((a[c]-b[c]).abs()<0.005,"frame {index}: {a:?} != {b:?}");}}
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn held_environment_inputs_share_identity_without_quantizing_changes() -> Result<()> {
        let r=Renderer::new()?;
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[],
            "render":{"duration":2},
            "programs":{"world":{"source":"struct Inputs {float gain=1.;}; float3 environment(EnvironmentContext ctx,Inputs i){return float3(i.gain);}"}},
            "materials":{"world":{"program":"world"}},
            "animation":[{"tracks":[{"path":["materials","world","inputs","gain"],"interpolation":"STEP","keys":[[0,1],[1,1.000001]]}]}]}))?;
        let (first,program)=install(&r,&scene,"world",Path::new("."))?;
        let program=program.context("tracked environment")?;
        assert_eq!(program.sample_key(0.,0.)?,program.sample_key(0.9,0.9)?);
        assert_ne!(program.sample_key(0.9,0.9)?,program.sample_key(1.,1.)?);
        assert_eq!(program.evaluate(&r,0.9)?.image,first.image);
        assert_ne!(program.evaluate(&r,1.)?.image,first.image);
        // A clock-dependent program must still distinguish these sample times.
        let mut timed=program;
        timed.uses_time=true;
        assert_ne!(timed.sample_key(0.,0.)?,timed.sample_key(0.9,0.9)?);
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn environment_images_do_not_pad_the_source_storage() -> Result<()> {
        let r=Renderer::new()?;
        let root=std::env::temp_dir().join(format!("vork-environment-extents-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        image::Rgb32FImage::from_pixel(32,16,image::Rgb([2.,4.,8.])).save(root.join("full.exr"))?;
        image::Rgb32FImage::from_pixel(4,2,image::Rgb([0.25;3])).save(root.join("weight.exr"))?;
        let scene:v2::Scene=serde_json::from_value(serde_json::json!({"version":2,
            "images":{"full":{"file":"full.exr","color_space":"Linear Rec.709"},"weight":{"file":"weight.exr","color_space":"Non-Color"}},
            "materials":{"world":{"program":"std/environment_image","inputs":{"image":{"image":"full"},"lighting_weight":{"image":"weight"}}}},
            "nodes":[]}))?;
        let (source,_)=install_source(&r,&scene,"world",&root)?;
        assert_eq!(source.texture.depth_or_array_layers(),2);
        assert_eq!(source.primary.as_ref().unwrap().image_data.as_ref().unwrap().size(),(2+2*10)*4);
        assert_eq!(source.primary.as_ref().unwrap().constant_camera.as_ref().unwrap().size(),16);
        let baked=source.read(&r)?;
        for (camera,lighting) in baked.camera.pixels().zip(baked.image.pixels()) {
            assert_eq!(camera.0,[2.,4.,8.]);
            assert_eq!(lighting.0,[0.5,1.,2.]);
        }
        // These scenes need the camera inputs, not a retained camera bake.
        for path in [false,true] {
            let mut scene=scene.clone();
            if path {scene.render.settings.integrator=vork_scene::Integrator::Path(Default::default());}
            scene.nodes=serde_json::from_value(serde_json::json!([{"type":"camera"},{"type":"light","shape":"environment","material":"world"}]))?;
            let settings=environment_node::Settings::from_scene(&scene)?;
            let installed=environment::Environment::new(&r,settings,&scene,&root)?;
            assert_eq!([installed.source.texture().width(),installed.source.texture().height()],[1,1]);
            assert_eq!(installed.source.texture().depth_or_array_layers(),1);
            assert!(!installed.lighting_is_black);
        }
        // A visible camera source can have exactly black lighting. Tiny positive
        // lighting must not be discarded by a moments/energy threshold.
        for weight in [0.,1e-30] {
            image::Rgb32FImage::from_pixel(4,2,image::Rgb([weight;3])).save(root.join("weight.exr"))?;
            let mut scene=scene.clone();
            scene.render.settings.integrator=vork_scene::Integrator::Path(Default::default());
            scene.nodes=serde_json::from_value(serde_json::json!([{"type":"camera"},{"type":"light","shape":"environment","material":"world"}]))?;
            let settings=environment_node::Settings::from_scene(&scene)?;
            let installed=environment::Environment::new(&r,settings,&scene,&root)?;
            assert_eq!(installed.lighting_is_black,weight==0.);
            assert!(installed.primary.as_ref().unwrap().constant_camera.is_some());
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn image_inputs_keep_camera_radiance_and_residual_lighting() -> Result<()> {
        let r=Renderer::new()?;
        let root=std::env::temp_dir().join(format!("vork-image-environment-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let full=image::Rgb32FImage::from_fn(16,8,|x,y|image::Rgb([x as f32+1.,y as f32+1.,100000.]));
        full.save(root.join("full.exr"))?;
        image::Rgb32FImage::from_pixel(16,8,image::Rgb([0.25;3])).save(root.join("weight.exr"))?;
        let scene:v2::Scene=serde_json::from_value(serde_json::json!({"version":2,
            "images":{"full":{"file":"full.exr","color_space":"Linear Rec.709","wrap_t":"clamp_to_edge"},"weight":{"file":"weight.exr","color_space":"Non-Color","wrap_t":"clamp_to_edge"}},
            "materials":{"world":{"program":"std/environment_image","inputs":{"image":{"image":"full"},"lighting_weight":{"image":"weight"}}}},
            "nodes":[]}))?;
        for _ in 0..2 {
            let (baked,_)=install(&r,&scene,"world",&root)?;
            // A small image cannot lower the lighting grid below the program floor.
            assert_eq!(<[u32;2]>::from(baked.camera.dimensions()),constants::ENVIRONMENT_PROGRAM_SIZE);
            let step=[baked.camera.width()/full.width(),baked.camera.height()/full.height()];
            for (camera,lighting) in baked.camera.pixels().zip(baked.image.pixels()) {
                for k in 0..3 {assert!((lighting[k]-camera[k]*0.25).abs()<=camera[k].abs()*1e-6,"lighting: {lighting:?} camera {camera:?}");}
            }
            // Grid texels beside each image texel centre reconstruct that texel; the
            // last column looks inward rather than across the longitude seam.
            for (x,y,expected) in full.enumerate_pixels() {
                let gx=x*step[0]+step[0]/2-u32::from(x+1==full.width());
                let camera=baked.camera.get_pixel(gx,y*step[1]+step[1]/2);
                for k in 0..3 {assert!((camera[k]-expected[k]).abs()<0.01,"camera: {camera:?} expected {expected:?}");}
            }
        }
        let mut moving=serde_json::to_value(&scene)?;
        let source=vork_scene::libraries::module_source("std/environment_image").unwrap()
            .replace("atan2(d.z,d.x)","(atan2(d.z,d.x)+ctx.time)");
        moving["programs"]=serde_json::json!({"moving":{"source":source}});
        moving["materials"]["world"]["program"]="moving".into();
        let moving=serde_json::from_value(moving)?;
        let (first,program)=install(&r,&moving,"world",&root)?;
        let program=program.context("animated image coordinates retain evaluation")?;
        assert_ne!(first.image,program.evaluate(&r,std::f32::consts::TAU/16.)?.image);
        assert_eq!(first.image,program.evaluate(&r,0.)?.image);
        std::fs::remove_dir_all(root)?;Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn schwarzschild_shader_matches_analytic_deflection_and_photon_sphere() -> Result<()> {
        // Execute the example's actual authored segment program.
        let mut source=include_str!("../../../examples/black-hole/propagation.slang").to_owned();
        source.push_str(r#"
float3 environment(EnvironmentContext ctx,Inputs inputs) {
    uint id=uint(ctx.origin.x);
    if(id==0u) {
        // Circular optical rays satisfy 1+r*d(log n)/dr=0. Convert the
        // isotropic-coordinate root to Schwarzschild's areal radius.
        var lo=.7;var hi=4.;
        for(uint j=0u;j<32u;j++) {
            let r=.5*(lo+hi);
            if(1.+r*gradient(float3(r,0.,0.)).x>0.) hi=r;else lo=r;
        }
        let r=.5*(lo+hi);
        return float3(r*pow(1.+.5/r,2.),0.,0.);
    }
    let b=id==1u ? 50. : id==2u ? 100. : 3.*sqrt(3.)*(id==3u ? .999 : 1.001);
    var p=float3(-1100.,b,0.);var d=float3(1.,0.,0.);var previous=0.;
    for(uint n=0u;n<8192u;n++) {
        let step=propagation(PropagationContext(p,d,0.,previous),inputs);
        if(step.absorbed) return float3(0.,1.,float(n));
        p+=step.direction*step.distance;d=step.direction;previous=step.distance;
        if(length(p)>1000. && dot(p,d)>0.) return float3(atan2(-d.y,d.x),0.,float(n));
    }
    return float3(0.,2.,0.);
}
"#);
        let program=Compiler::discover().compile_environment(&programs::abi(),&source,Path::new("geodesic-check.slang"),"geodesiccheck")?;
        let shader=format!("{}\nfn {}(i:u32)->u32 {{return 0u;}}\n@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>>;\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{var value:array<vec4<f32>,2>;{}(vec3<f32>(f32(gid.x),0.,0.),vec3<f32>(0.,0.,-1.),0.,0u,&value);result[gid.x]=value[0];}}",program.wgsl,vork_slang::MATERIAL_WORD,program.entry);
        let r=Renderer::new()?;let d=&r.device;
        let module=crate::timing::shader_module(d,wgpu::ShaderModuleDescriptor {label:Some("Schwarzschild numerical check"),source:wgpu::ShaderSource::Wgsl(shader.into())});
        let pipeline=r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:None,layout:None,module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:r.cache.as_ref()});
        let output=init(d,"geodesic results",&[0;80],wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let read=init(d,"geodesic readback",&[0;80],wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST);
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding())]});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(5,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&read,0,80);
        crate::command_finish::Segments::submit(&r.queue,encoder);
        let(tx,rx)=std::sync::mpsc::channel();read.map_async(wgpu::MapMode::Read,..,move |v|{let _=tx.send(v);});r.wait(None)?;rx.recv()??;
        let mapped=read.get_mapped_range(..);let values:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        assert!((values[0][0]-3.).abs()<1e-4,"photon sphere: {:?}",values[0]);
        for (index,b) in [(1,50f32),(2,100.)] {
            // The authored optical field tapers over radii 500..1000. Integrate
            // its first-order bending correction independently of the segment
            // solver; the infinite Schwarzschild series alone is no longer its
            // boundary condition. Keep the existing weak-field error bound.
            let impact=f64::from(b);
            let lo=(500_f64.powi(2)-impact*impact).sqrt();
            let hi=(1000_f64.powi(2)-impact*impact).sqrt();
            let dx=(hi-lo)/4096.;
            let shell=(0..=4096).map(|j| {
                let x=lo+f64::from(j)*dx;let r=x.hypot(impact);let t=(r-500.)/500.;
                let taper=1.-t*t*(3.-2.*t);let derivative=-6.*t*(1.-t)/500.;
                let weight=if j==0 || j==4096 {1.} else if j%2==0 {2.} else {4.};
                weight*((taper-1.)/r.powi(3)-derivative/r.powi(2))
            }).sum::<f64>()*dx/3.;
            let correction=4.*impact*shell-4./impact*(1.-hi/1000.);
            let expected=4./b+15.*std::f32::consts::PI/(4.*b*b)+correction as f32;
            // Third and higher weak-field terms plus bounded f32 accumulation.
            assert!((values[index][0]-expected).abs()<80./b.powi(3)+3e-5,"b={b}: {:?}, expected {expected}",values[index]);
            assert_eq!(values[index][1],0.);
        }
        assert_eq!(values[3][1],1.,"below critical impact parameter must capture");
        assert_eq!(values[4][1],0.,"above critical impact parameter must escape");
        drop(mapped);read.unmap();
        Ok(())
    }
    fn spatial_scene() -> serde_json::Value {
        serde_json::json!({"version":2,"composition":{"space":"3d","view":{"program":"std/standard"}},"render":{"resolution":[8,4],"background":"environment","settings":{"samples":1,"denoiser":null}},
            "programs":{"world":{"source":"struct Inputs {float gain=1.;}; float3 environment(EnvironmentContext ctx,Inputs i){return float3((ctx.origin.x+3.)*.1,ctx.time*.2,.1)*i.gain;}"}},
            "materials":{"world":{"program":"world"}},
            "nodes":[{"type":"light","name":"world","shape":"environment","material":"world"},
                {"type":"camera","name":"camera","translation":[0,0,10],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":4}}}]})
    }
    /// A narrow camera-only cap lies between every midpoint in the lighting bake.
    /// Actual camera evaluations must validate the source before any RGB transform.
    #[test]
    #[ignore = "requires pinned Slang and a real GPU"]
    fn camera_source_validation_covers_directions_between_bake_samples() -> Result<()> {
        let r=Renderer::new()?;
        for integrator in ["raster","path"] {
            for mode in 0..4 {
                let value=["negative","NaN","infinity","valid"][mode as usize];
                let mut input=spatial_scene();
                input["render"]["settings"]=serde_json::json!({"integrator":integrator,"samples":1,"denoiser":null});
                input["programs"]["world"]["source"]=serde_json::json!("struct Inputs {uint mode=0;};float3 environment(EnvironmentContext c,Inputs i) {let v=i.mode==0u ? -1. : (i.mode==1u ? asfloat(0x7fc00000u) : (i.mode==2u ? asfloat(0x7f800000u) : 2.));return float3(c.camera && c.direction.z < -0.999999 ? v : 1.);}");
                input["materials"]["world"]["inputs"]["mode"]=mode.into();
                let installed=r.install(vork_scene::v2::from_test_value(input)?,Path::new("."))?;
                let mut frame=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
                frame.submit(&installed,&r,0.)?;
                let result=frame.collect(&r);
                if mode==3 {
                    assert!(result?.linear_rgba.unwrap().iter().all(|p|p[..3].iter().all(|v|v.is_finite() && *v>0.)),"valid source must render");
                } else {
                    let error=result.err().context("invalid camera source must refuse the exposure")?;
                    assert!(error.to_string().contains("environment radiance must be finite and nonnegative"),"{integrator} {value}: {error:#}");
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn delivery_spatial_environment_warns_and_renders_approximation() -> Result<()> {
        let r=Renderer::new()?;
        let scene=vork_scene::v2::from_test_value(spatial_scene())?;
        let collector=crate::warnings::Collector::default();
        let _scope=collector.enter();
        let installed=r.install_with_options(scene,Path::new("."),crate::InstallOptions {test_participating_media:false,..Default::default()})?;
        assert!(!installed.environment.primary.as_ref().unwrap().spatial);
        assert!(collector.snapshot().iter().any(|w|w.code=="spatial_environment"));
        let mut frame=installed.create_frame_slot(&r,Default::default())?;
        frame.submit(&installed,&r,0.)?;
        let pixels=frame.collect(&r)?.rgba;
        assert!(pixels.chunks_exact(4).all(|p|p[0]>100 && p[2]>50),"directional radiance must survive admission");
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn spatial_primary_uses_orthographic_origins_and_shutter_time() -> Result<()> {
        let r=Renderer::new()?;
        let input=spatial_scene();
        let installed=r.install(vork_scene::v2::from_test_value(input.clone())?,Path::new("."))?;
        assert!(installed.environment.primary.is_some());
        let mut frame=installed.create_frame_slot(&r,Default::default())?;
        frame.submit(&installed,&r,0.)?;let first=frame.collect(&r)?.rgba;
        assert!(first[4*7]>first[0]+40,"orthographic directions agree, but ray origins must differ");
        frame.submit(&installed,&r,1.)?;let second=frame.collect(&r)?.rgba;
        assert!(second[1]>first[1]+40,"environment sees the requested sample time");
        // Reusing one renderer must not reuse another source's primary pipeline.
        let mut changed=input;changed["programs"]["world"]["source"]=serde_json::json!("struct Inputs {};float3 environment(EnvironmentContext c,Inputs i){return float3(.1,(c.origin.x+3.)*.1,.1);}");
        let installed=r.install(vork_scene::v2::from_test_value(changed)?,Path::new("."))?;
        let mut frame=installed.create_frame_slot(&r,Default::default())?;
        frame.submit(&installed,&r,0.)?;let changed=frame.collect(&r)?.rgba;
        assert!(changed[4*7+1]>changed[1]+40);
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn spatial_primary_averages_aperture_radiance_not_mean_origin() -> Result<()> {
        let r=Renderer::new()?;
        let mut value=spatial_scene();
        value["render"]["resolution"]=serde_json::json!([1,1]);
        value["render"]["settings"]["samples"]=64.into();

        value["programs"]["world"]["source"]=serde_json::json!("struct Inputs {};float3 environment(EnvironmentContext c,Inputs i){return float3(.01+10.*dot(c.origin.xy,c.origin.xy));}");
        value["nodes"][1]=serde_json::json!({"type":"camera","name":"camera","translation":[0,0,10],"material":{"program":"std/camera","inputs":{"lens":50}}});
        let render=|value:serde_json::Value|->Result<Vec<u8>> {
            let scene=r.install(vork_scene::v2::from_test_value(value)?,Path::new("."))?;
            let mut slot=scene.create_frame_slot(&r,Default::default())?;
            slot.submit(&scene,&r,0.)?;Ok(slot.collect(&r)?.rgba)
        };
        let pinhole=render(value.clone())?;
 value["nodes"][1]["material"]["inputs"].as_object_mut().unwrap().extend(serde_json::json!({"focus_distance":10,"aperture_fstop":0.1,"use_dof":true}).as_object().unwrap().clone());
        let aperture=render(value)?;
        assert!(aperture[0]>pinhole[0]+40,"nonlinear aperture radiance must be averaged after evaluation: pinhole={pinhole:?}, aperture={aperture:?}");
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn spatial_path_origin_precedes_camera_clip_plane() -> Result<()> {
        let r=Renderer::new()?;
        let mut value=spatial_scene();
        value["render"]["settings"]["integrator"]="path".into();
        value["programs"]["world"]["source"]=serde_json::json!("struct Inputs {};float3 environment(EnvironmentContext c,Inputs i){return float3(c.origin.z*.01);}");
        let mut previous:Option<Vec<[f32;4]>>=None;
        for near in [0.1,4.5] {
            value["nodes"][1]["material"]["inputs"]["clip_start"]=serde_json::json!(near);
            let scene=r.install(vork_scene::v2::from_test_value(value.clone())?,Path::new("."))?;
            let mut frame=scene.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            frame.submit(&scene,&r,0.)?;let output=frame.collect(&r)?;let pixels=output.linear_rgba.context("linear readback")?;
            assert!(output.rgba[0]>70,"observer remains at z=10");
            if let Some(previous)=&previous {crate::test_linear::assert_linear_eq(&pixels,previous);}
            previous=Some(pixels);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang and Vulkan"]
    fn spatial_primary_refuses_unsupported_receivers_and_frames() -> Result<()> {
        let r=Renderer::new()?;
        for patch in [serde_json::json!({"type":"mesh","geometry":"unused"}),serde_json::json!({"type":"asset","file":"unused.glb"})] {
            let mut value=spatial_scene();value["nodes"].as_array_mut().unwrap().push(patch);
            let scene=vork_scene::v2::from_test_value(value)?;
            assert!(install(&r,&scene,"world",Path::new(".")).err().context("must refuse spatial lighting")?.to_string().contains("primary rays only"));
        }
        let mut value=spatial_scene();value["nodes"][0]["translation"]=serde_json::json!([1,0,0]);
        let scene=vork_scene::v2::from_test_value(value)?;
        assert!(install(&r,&scene,"world",Path::new(".")).err().context("must refuse spatial frame")?.to_string().contains("identity"));
        Ok(())
    }
}
