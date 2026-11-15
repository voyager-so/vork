use super::*;
/// Where a graph's entries read what is not the graph's own: a world's channels, AOVs and installed images,
/// or a canvas's image files. A canvas has no world channels; its scene refuses them
/// (`vork_scene` `validate_canvas`).
#[derive(Clone, Copy)]
pub(super) enum Sources<'a> {
    World(&'a InstalledScene, &'a crate::view::Targets),
    /// Each image input's texture, and whether its colour is associated.
    Canvas(&'a std::collections::BTreeMap<String, (wgpu::TextureView, bool)>),
}
impl<'a> Sources<'a> {
    fn world(self, channel: &str) -> Result<(&'a InstalledScene, &'a crate::view::Targets)> {
        match self {
            Self::World(scene, targets) => Ok((scene, targets)),
            Self::Canvas(_) => anyhow::bail!("compositor channel {channel} is a 3D world's; a 2D canvas has color"),
        }
    }
    /// An image input's texture and whether its colour is associated. File and video textures carry straight
    /// alpha; render images and OpenEXR files hold associated RGBA.
    fn image(self, name: &str) -> Result<(&'a wgpu::TextureView, bool)> {
        match self {
            Self::Canvas(images) => images.get(name).map(|(view, associated)| (view, *associated)).context("uninstalled compositor image"),
            Self::World(s, _) => {
                let view = if let Some(view) = s.file_images.get(name) { view } else {
                    &s.graph.views.entries.iter().find_map(|entry| match &entry.kind {
                        crate::views::ViewKind::Image { name: image, render } if image == name => Some(render.as_ref()), _ => None,
                    }).context("uninstalled compositor image")?.output
                };
                Ok((view, !s.file_images.contains_key(name) || associated_file(&s.scene.images[name])))
            }
        }
    }
}
/// Whether an image file's pixels are stored associated: OpenEXR's are.
pub(super) fn associated_file(image: &vork_scene::v2::Image) -> bool {
    image.file.as_ref().is_some_and(|path| Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")))
}
fn buffer<'a>(
    layout: &mut Vec<wgpu::BindGroupLayoutEntry>,
    bindings: &mut Vec<wgpu::BindGroupEntry<'a>>,
    at: u32,
    value: &'a wgpu::Buffer,
    kind: wgpu::BufferBindingType,
) {
    layout.push(pipelines::buffer_entry(
        at,
        wgpu::ShaderStages::COMPUTE,
        kind,
    ));
    bindings.push(entry(at, value.as_entire_binding()));
}
#[allow(clippy::too_many_arguments)]
pub(super) fn pass(
    sources: Sources<'_>,
    r: &Renderer,
    uniform: &wgpu::Buffer,
    program: &Program,
    words: &wgpu::Buffer,
    source: &wgpu::Buffer,
    full: [u32; 2],
    size: [u32; 2],
    source_size: [u32; 2],
    accumulated: Normalization,
    entry_name: &str,
    output: &wgpu::Buffer,
    nodes: &std::collections::BTreeMap<String, (wgpu::Buffer, [u32; 2])>,
    original: &wgpu::Buffer,
    original_normalization: Normalization,
) -> Result<graph::Dispatch> {
    use wgpu::BufferBindingType::{Storage, Uniform as UniformBinding};
    let [w, h] = size;
    let extent = size;
    let group = [constants::IMAGE_WORKGROUP_SIZE; 2];
    let frame_words = std::mem::size_of::<Uniform>() / 16;
    let word = |at: usize| at / 16;
    let temporal = word(std::mem::offset_of!(Uniform, temporal));
    let scene = word(std::mem::offset_of!(Uniform, scene));
    let normalization = match accumulated {
        Normalization::Frame => format!("frame.words[{temporal}].y"),
        Normalization::None => "1.".into(),
    };
    let original_normalization = match original_normalization {
        Normalization::Frame => format!("frame.words[{temporal}].y"),
        Normalization::None => "1.".into(),
    };
    // Image sizes are runtime words, so one shader serves every resolution:
    // [0] dispatch xy, extent zw; [1] full xy, source zw;
    // [1+slot] the named input image extent xy.
    let mut sizes = vec![[0u32; 4]; 2 + program.channels.len()];
    sizes[0] = [w, h, extent[0], extent[1]];
    sizes[1] = [full[0], full[1], source_size[0], source_size[1]];
    for (index, channel) in program.channels.iter().enumerate() {
        if !program.reads(index as u32+1) {continue;}
        if let Some(name) = channel.strip_prefix("step:") {
            let (_, extent) = nodes.get(name).with_context(|| format!("unknown earlier compositor step {name}"))?;
            sizes[2 + index] = [extent[0], extent[1], 0, 0];
        }
    }
    let size_words = sizes.len();
    let sizes = init(
        &r.device,
        "compositor image sizes",
        bytemuck::cast_slice(&sizes),
        wgpu::BufferUsages::UNIFORM,
    );
    let mut layout = Vec::new();
    let mut bindings = Vec::new();
    buffer(
        &mut layout,
        &mut bindings,
        0,
        words,
        Storage { read_only: true },
    );
    buffer(&mut layout, &mut bindings, 1, source, Storage { read_only: true });
    buffer(&mut layout, &mut bindings, 3, uniform, UniformBinding);
    buffer(&mut layout, &mut bindings, 2, output, Storage { read_only: false });
    let destination = "@group(0) @binding(2) var<storage,read_write> destination:array<vec4<f32>>;";
    let store = "destination[gid.y*image_sizes[0].x+gid.x]=result[0];";
    let mut declarations = String::new();
    let mut loads = String::new();
    let mut extents = String::new();
    let mut next_binding = 4u32;
    let mut bound = std::collections::BTreeMap::<&str, String>::new();
    let colour = "return composite_load(p);";
    for (index, channel) in program.channels.iter().enumerate() {
        let slot = index as u32 + 1;
        if !program.reads(slot) {continue;}
        let key = if channel.starts_with("aov:") {
            "aov:"
        } else if matches!(channel.as_str(), "motion") {
            "motion"
        } else {
            channel.as_str()
        };
        let existing = bound.get(key).cloned();
        let new = existing.is_none();
        let binding = next_binding;
        if new {
            next_binding += 1;
        }
        let name = existing.unwrap_or_else(|| format!("channel_{slot}"));
        let (declaration, load, channel_size) = if channel=="color" {
            if new {buffer(&mut layout,&mut bindings,binding,original,Storage {read_only:true});}
            (format!("var<storage,read> {name}:array<vec4<f32>>;"),
                format!("return {name}[u32(q.y)*image_sizes[1].x+u32(q.x)]/{original_normalization};"),
                "image_sizes[1].xy".to_owned())
        } else if channel=="shadow_catcher" {
            let (s, targets) = sources.world(channel)?;
            if new {
                buffer(&mut layout,&mut bindings,binding,&targets.catcher,Storage {read_only:true});
            }
            let load=if s.has_catchers {
                format!("let p=u32(q.y)*image_sizes[1].x+u32(q.x);let a={name}[p*3u];let b={name}[p*3u+1u];let valid=b.xyz!=vec3(0.);let ratio=select(vec3(1.),a.xyz/select(vec3(1.),b.xyz,valid),valid);return vec4(mix(vec3(1.),ratio,a.w/max(b.w,1.)),1.);")
            } else {"return vec4(1.);".into()};
            (format!("var<storage,read> {name}:array<vec4<f32>>;"),load,"image_sizes[1].xy".to_owned())
        } else if let Some(node) = channel.strip_prefix("step:") {
            let (value, _) = nodes.get(node).context("unknown earlier compositor step")?;
            buffer(&mut layout, &mut bindings, binding, value, Storage { read_only: true });
            let extent = format!("image_sizes[{}].xy", 1 + slot);
            (format!("var<storage,read> {name}:array<vec4<f32>>;"),
                format!("let extent={extent};let at=vec2<u32>(clamp(p,vec2(0),vec2<i32>(extent)-1));return {name}[at.y*extent.x+at.x];"), extent)
        } else if let Some(image) = channel.strip_prefix("image:") {
            let (view, associated) = sources.image(image)?;
            if new {
                let mut item=pipelines::texture_entry(binding,wgpu::TextureSampleType::Float {filterable:false},false);
                item.visibility=wgpu::ShaderStages::COMPUTE;
                layout.push(item);bindings.push(entry(binding,wgpu::BindingResource::TextureView(view)));
            }
            // Associate each straight-alpha texel before interpolation, so hidden RGB cannot bleed into edges.
            let result=if associated {"value"} else {"vec4(value.rgb*value.a,value.a)"};
            (format!("var {name}:texture_2d<f32>;"),format!("let at=clamp(p,vec2(0),vec2<i32>(textureDimensions({name}))-1);let value=textureLoad({name},at,0);return {result};"),format!("textureDimensions({name})"))
        } else if channel.starts_with("aov:") {
            let (s, targets) = sources.world(channel)?;
            if new {
                buffer(
                    &mut layout,
                    &mut bindings,
                    binding,
                    &targets.aov_values,
                    Storage { read_only: true },
                );
            }
            let aov = s
                .render
                .aovs
                .iter()
                .position(|a| format!("aov:{}", a.name) == *channel)
                .context("unknown compositor AOV")?;
            (
                format!("var<storage,read> {name}:array<vec4<f32>>;"),
                format!(
                    "return {name}[(u32(q.y)*image_sizes[1].x+u32(q.x))*{}u+{aov}u];",
                    s.render.aovs.len()
                ),
                "image_sizes[1].xy".to_owned(),
            )
        } else {
            let (_, targets) = sources.world(channel)?;
            let view = match channel.as_str() {
                "depth" => &targets.post_depth,
                "view_normal" => &targets.post_normals,
                "object_index" => &targets.post_ids,
                _ => &targets.post_motion,
            };
            let mut texture = pipelines::texture_entry(
                binding,
                if channel == "object_index" {
                    wgpu::TextureSampleType::Uint
                } else {
                    wgpu::TextureSampleType::Float { filterable: false }
                },
                false,
            );
            if new {
                texture.visibility = wgpu::ShaderStages::COMPUTE;
                layout.push(texture);
                bindings.push(entry(binding, wgpu::BindingResource::TextureView(view)));
            }
            let load = match channel.as_str() {
                "depth" => {
                    let params = word(std::mem::offset_of!(Uniform, params));
                    let flags = word(std::mem::offset_of!(Uniform, flags));
                    let camera = word(std::mem::offset_of!(Uniform, camera_program));
                    format!(
                        "let z=textureLoad({name},q,0).x;let n=frame.words[{params}].x;let f=frame.words[{params}].y;let projected=select(n*f/(f-z*(f-n)),mix(n,f,z),frame.words[{flags}].x>.5);let metric=select(projected,z/(1.-z),(bitcast<u32>(frame.words[{camera}].z)&2u)!=0u);return vec4(select(metric,{}.,z>=1.),0.,0.,1.);",
                        vork_slang::registry::EXR_BACKGROUND_DEPTH
                    )
                }
                "view_normal" => {
                    let view = word(std::mem::offset_of!(Uniform, view));
                    format!(
                        "let n=textureLoad({name},q,0).xyz;let v=mat4x4<f32>(frame.words[{view}],frame.words[{}],frame.words[{}],frame.words[{}]);let normal=(v*vec4(n,0.)).xyz;return vec4(normal/max(length(normal),1e-20),1.);",
                        view + 1,
                        view + 2,
                        view + 3
                    )
                }
                "object_index" => format!(
                    "return vec4(f32(textureLoad({name},q,0).x&{}u),0.,0.,1.);",
                    vork_slang::registry::INDIRECT_ID_MASK
                ),
                _ => format!(
                    "return vec4(textureLoad({name},q,0).xy*vec2<f32>(image_sizes[1].xy),0.,1.);"
                ),
            };
            (
                format!(
                    "var {name}:texture_2d<{}>;",
                    if channel == "object_index" {
                        "u32"
                    } else {
                        "f32"
                    }
                ),
                load,
                "image_sizes[1].xy".to_owned(),
            )
        };
        if new {
            declarations.push_str(&format!("@group(0) @binding({binding}) {declaration}\n"));
            bound.insert(key, name);
        }
        loads.push_str(&format!("case {slot}u:{{{load}}}"));
        extents.push_str(&format!("case {slot}u:{{return vec2<f32>({channel_size});}}"));
    }
    buffer(&mut layout, &mut bindings, next_binding, &sizes, UniformBinding);
    declarations.push_str(&format!("@group(0) @binding({next_binding}) var<uniform> image_sizes:array<vec4<u32>,{size_words}>;\n"));
    let view = word(std::mem::offset_of!(Uniform, view));
    let projection = word(std::mem::offset_of!(Uniform, composite_projection));
    let optics = word(std::mem::offset_of!(Uniform, composite_optics));
    let sensor = word(std::mem::offset_of!(Uniform, composite_sensor));
    let aperture = word(std::mem::offset_of!(Uniform, aperture));
    let clip = word(std::mem::offset_of!(Uniform, params));
    let shutter = word(std::mem::offset_of!(Uniform, composite_shutter));
    let step = word(std::mem::offset_of!(Uniform, composite_step));
    let display = word(std::mem::offset_of!(Uniform, display));
    let context = format!(
        "fn vork_compositor_context(i:u32)->vec4<f32>{{if i<4u{{return frame.words[{view}u+i];}}if i<8u{{return frame.words[{projection}u+i-4u];}}switch i{{case 8u:{{return frame.words[{optics}];}}case 9u:{{return frame.words[{sensor}];}}case 10u:{{return frame.words[{aperture}];}}case 11u:{{return frame.words[{clip}];}}case 12u:{{return vec4(frame.words[{scene}].y,frame.words[{step}].z,0.,0.);}}case 13u:{{return frame.words[{shutter}];}}case 14u:{{return frame.words[{step}];}}default:{{return vec4(vec2<f32>(image_sizes[1].xy),frame.words[{display}].yz);}}}}}}"
    );
    let call = format!(
            "let sizes=image_sizes[0];if gid.x>=sizes.x||gid.y>=sizes.y{{return;}}var result:array<vec4<f32>,1>;{entry_name}((vec2<f32>(gid.xy)+.5)/vec2<f32>(sizes.xy),vec2<f32>(sizes.zw),frame.words[{scene}].x,0u,&result);{store}"
        );
    // Resource identities are fixed by the installed plan. Specialize those
    // words only; numeric defaults and values still use the author input ABI.
    let mut handle_cases = String::new();
    let mut masks = std::collections::BTreeMap::<usize, u32>::new();
    for (index, field) in program.compiled.inputs.fields.iter().enumerate() {
        if let Some(slot) = program.slots.get(&field.name) {
            handle_cases.push_str(&format!("case {}u:{{return {slot}u;}}", field.offset));
            *masks.entry(index / 32).or_default() |= 1u32 << (index % 32);
        }
    }
    for (index, mask) in masks {
        handle_cases.push_str(&format!("case {index}u:{{return words[i]|{mask}u;}}"));
    }
    let material = format!(
        "fn vork_material_word(i:u32)->u32{{switch i{{{handle_cases}default:{{return words[i];}}}}}}"
    );
    let input_words=program.words.size()/4;
    let color_read = format!("return source[q.y*image_sizes[1].z+q.x]/{normalization};");
    let code = if entry_name == "vork_copy" {
        "fn vork_copy(uv:vec2<f32>,size:vec2<f32>,time:f32,base:u32,r:ptr<function,array<vec4<f32>,1>>) { (*r)[0]=vork_sample_image(0u,uv,vec2(0.),vec2(0.)); }"
    } else { &program.compiled.wgsl };
    let shader = format!(
        "struct CompositeFrame{{words:array<vec4<f32>,{frame_words}>}};\n{}\n@group(0) @binding(0) var<storage,read> words:array<u32,{input_words}>;@group(0) @binding(1) var<storage,read> source:array<vec4<f32>>;@group(0) @binding(3) var<uniform> frame:CompositeFrame;{destination}{declarations}{material}fn composite_load(p:vec2<i32>)->vec4<f32>{{let q=vec2<u32>(clamp(p,vec2(0),vec2<i32>(image_sizes[1].zw)-1));{color_read}}}fn composite_channel_load(slot:u32,p:vec2<i32>)->vec4<f32>{{let q=clamp(p,vec2(0),vec2<i32>(image_sizes[1].xy)-1);switch slot{{case 0u:{{{colour}}}{loads}default:{{return vec4(0.);}}}}}}fn composite_extent(slot:u32)->vec2<f32>{{switch slot{{case 0u:{{return vec2<f32>(image_sizes[1].zw);}}{extents}default:{{return vec2(1.);}}}}}}fn vork_compositor_size(slot:u32)->vec2<f32>{{return composite_extent(slot);}}fn vork_compositor_load(slot:u32,p:vec2<i32>)->vec4<f32>{{return composite_channel_load(slot,p);}}fn vork_sample_image(slot:u32,uv:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec4<f32>{{let p=uv*composite_extent(slot)-.5;let q=vec2<i32>(floor(p));let f=fract(p);return mix(mix(composite_channel_load(slot,q),composite_channel_load(slot,q+vec2(1,0)),f.x),mix(composite_channel_load(slot,q+vec2(0,1)),composite_channel_load(slot,q+vec2(1,1)),f.x),f.y);}}{context}@compute @workgroup_size({},{})fn main(@builtin(global_invocation_id)gid:vec3<u32>,@builtin(workgroup_id)group:vec3<u32>,@builtin(local_invocation_id)local:vec3<u32>){{{call}}}",
        code,
        group[0],
        group[1]
    );
    if r.adapter_info.backend==wgpu::Backend::Metal {
        crate::pipelines::check_stage_limits("compositor image pass",&[&layout],wgpu::ShaderStages::COMPUTE)
            .context("reduce compositor pass image and buffer inputs")?;
    }
    let limits = r.device.limits();
    let buffers = layout
        .iter()
        .filter(|e| {
            matches!(
                e.ty,
                wgpu::BindingType::Buffer {
                    ty: Storage { .. },
                    ..
                }
            )
        })
        .count() as u32;
    let textures = layout
        .iter()
        .filter(|e| matches!(e.ty, wgpu::BindingType::Texture { .. }))
        .count() as u32;
    let storage_textures=layout.iter().filter(|e|matches!(e.ty,wgpu::BindingType::StorageTexture {..})).count() as u32;
    ensure!(
        storage_textures<=limits.max_storage_textures_per_shader_stage && buffers <= limits.max_storage_buffers_per_shader_stage
            && textures <= limits.max_sampled_textures_per_shader_stage,
        crate::gpu_memory::device_limit("compositor pass", format_args!("{storage_textures} storage textures, {buffers} storage buffers, {textures} sampled textures"), format_args!("{} storage textures, {} storage buffers, {} sampled textures",limits.max_storage_textures_per_shader_stage,limits.max_storage_buffers_per_shader_stage,limits.max_sampled_textures_per_shader_stage), "reduce compositor pass image and buffer inputs")
    );
    ensure!(
        group[0] > 0
            && group[1] > 0
            && group[0] <= limits.max_compute_workgroup_size_x
            && group[1] <= limits.max_compute_workgroup_size_y
            && u64::from(group[0]) * u64::from(group[1])
                <= u64::from(limits.max_compute_invocations_per_workgroup),
        crate::gpu_memory::device_limit("compositor workgroup", format_args!("{}x{} threads ({} invocations)",group[0],group[1],u64::from(group[0])*u64::from(group[1])), format_args!("{}x{} threads, {} invocations; each axis must be positive",limits.max_compute_workgroup_size_x,limits.max_compute_workgroup_size_y,limits.max_compute_invocations_per_workgroup), "set the compositor program workgroup to positive dimensions within these limits")
    );
    let groups = [w.div_ceil(group[0]), h.div_ceil(group[1]), 1];
    ensure!(
        groups
            .iter()
            .all(|n| *n <= limits.max_compute_workgroups_per_dimension),
        crate::gpu_memory::device_limit("compositor dispatch", format_args!("{groups:?} workgroups at {w}x{h}"), format_args!("{} workgroups per axis",limits.max_compute_workgroups_per_dimension), "reduce compositor image resolution or increase the program workgroup within its limits")
    );
    let d = &r.device;
    let module=if r.check_only {
        crate::native::module(d,"check compositor image pass",None,Some(""),(group[0],group[1],1))
    } else if r.adapter_info.backend==wgpu::Backend::Vulkan && !layout.iter().any(|entry|matches!(entry.ty,wgpu::BindingType::Sampler(_))) {
        // Vulkan HAL packs each group's sparse bindings in sorted order. Its
        // automatic shader remapping applies to Naga modules, not passthrough.
        let mut authored:Vec<_>=layout.iter().map(|entry|entry.binding).collect();authored.sort_unstable();
        let slots:Vec<_>=authored.into_iter().enumerate().map(|(physical,authored)|(0,authored,physical as u32)).collect();
        let mut hash=blake3::Hasher::new();hash.update(b"native-image-hlsl-9");let compiler=Compiler::discover();
        hash.update(programs::admit(&|key|r.disk_cache.path("slang-input",1,key),&compiler)?.version().as_bytes());hash.update(&serde_json::to_vec(&slots)?);hash.update(shader.as_bytes());
        let code:Vec<u8>=programs::cached_generated(r,hash.finalize(),||compiler.compile_image_spirv(&shader,&slots))?;
        crate::native::module(d,"compositor image pass",Some(&code),None,(group[0],group[1],1))
    } else {
        crate::timing::shader_module(d,wgpu::ShaderModuleDescriptor {label:Some("compositor image pass"),source:wgpu::ShaderSource::Wgsl(shader.into())})
    };
    let bindings_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compositor image pass"),
        entries: &layout,
    });
    let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compositor image pass"),
        bind_group_layouts: &[Some(&bindings_layout)],
        immediate_size: 0,
    });
    let pipeline = r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compositor image pass"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    });
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compositor image pass"),
        layout: &bindings_layout,
        entries: &bindings,
    });
    Ok(graph::Dispatch::new(&pipeline, bind, groups))
}
