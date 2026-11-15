//! Ordinary image bindings for standalone environment evaluation. File/media
//! decoding and clocks are shared with the other program stages.
use super::*;
use vork_scene::v2;

pub(super) struct Images {
    sources: Vec<Source>,
    animation: crate::animation::Plan,
    fps: vork_time::Time,
    max_side: u32,
}
struct Source {
    name: String,
    definition: v2::Image,
    /// `definition.is_srgb()`, read once when the image is loaded: it is the same for every texel and frame.
    srgb: bool,
    pixels: Vec<u32>,
    media: Option<std::sync::Mutex<crate::footage::Source>>,
    identity: blake3::Hash,
}
impl Images {
    pub(super) fn new(r: &Renderer, scene: &v2::Scene, base: &Path, layout: &vork_slang::InputLayout,
        values: &std::collections::BTreeMap<String, serde_json::Value>, opened: &crate::footage::Opened) -> Result<(Self, Vec<u32>)> {
        let mut sources=Vec::new();
        let words=layout.pack(values, |field,value| {
            let name=value.get("image").and_then(serde_json::Value::as_str).with_context(||format!("environment input {field}: expected {{image: name}}"))?;
            let definition=scene.images.get(name).with_context(||format!("unknown image {name}"))?.clone();
            let srgb=definition.is_srgb();
            let path=base.join(definition.file.as_ref().context("environment programs cannot sample rendered images")?);
            let media=crate::footage::is_media(&path).then(||opened.source(&path).map(|mut source| {source.prepare_gpu(r);source})).transpose()?.map(std::sync::Mutex::new);
            let bytes = if media.is_none() {
                match std::fs::read(&path) {
                    Ok(bytes) => Some(bytes),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound
                        && !path.extension().is_some_and(|e|e.eq_ignore_ascii_case("ies")) => None,
                    Err(error) => return Err(error).with_context(||format!("read environment image {}; restore the file or correct its path",path.display())),
                }
            } else {None};
            let identity=if let Some(media)=&media {media.lock().unwrap().identity()?}
                else {bytes.as_ref().map_or_else(||blake3::hash(b"missing-image-magenta"),|bytes|blake3::hash(bytes))};
            let pixels=if media.is_none() {
                encode(&definition,srgb,match &bytes {
                    // An environment is light: `decode_light`.
                    Some(bytes) => crate::texture_source::decode_light(&path,bytes,r.device.limits().max_texture_dimension_2d,definition.color_space,crate::texture_source::upload(srgb))
                        .map_err(|error|crate::texture_source::decode_context(crate::texture_source::for_image(error,name),"environment image",&path))?,
                    None => crate::texture_source::missing_image(&path),
                })
            } else {Vec::new()};
            let slot=sources.len() as u32;
            sources.push(Source {name:name.into(),definition,srgb,pixels,media,identity});
            Ok(slot)
        })?;
        let images=Self {sources,animation:crate::animation::Plan::compile_in(scene,base,None,opened)?,fps:scene.render.fps,max_side:r.device.limits().max_texture_dimension_2d};
        Ok((images,words))
    }
    pub(super) fn sizes(&self,data:&[u32])->Vec<[u32;2]> {
        (0..self.sources.len()).map(|k| {let at=data[k] as usize;[data[at],data[at+1]]}).collect()
    }
    /// Static planes are exact-bit deduplicated. Footage reserves its native
    /// four-plane bound so any later frame can update the same buffer.
    pub(super) fn capacity(&self,data:&[u32])->usize {
        self.sources.len().max(1)+self.sources.iter().enumerate().map(|(k,source)| {
            if source.media.is_some() {let at=data[k] as usize;10+data[at] as usize*data[at+1] as usize*4} else {source.pixels.len()}
        }).sum::<usize>()
    }
    pub(super) fn upload(&self,r:&Renderer,buffer:&wgpu::Buffer,sizes:&[[u32;2]],data:&[u32])->Result<()> {
        ensure!(sizes==self.sizes(data),"environment footage dimensions must remain constant");
        ensure!((data.len()*4) as u64<=buffer.size(),"environment image frame exceeds its installed storage");
        r.queue.write_buffer(buffer,0,bytemuck::cast_slice(data));
        Ok(())
    }
    /// Native image handles address descriptors in the ordinary material word
    /// table. Their texels remain float32 bits in the shared packed image buffer.
    pub(super) fn primary_words(&self,layout:&vork_slang::InputLayout,inputs:&[u32],images:&[u32])->Vec<u32> {
        let mut words=inputs.to_vec();
        for (n,field) in layout.fields.iter().enumerate() {
            if field.ty!=vork_slang::Type::Image || inputs[n/32]&(1<<(n%32))==0 {continue;}
            let slot=inputs[field.offset] as usize;
            let at=images[slot] as usize;
            // Native samplers distinguish retained float32 environment inputs
            // from ordinary texture/sampler descriptors, in every context.
            words[field.offset]=0x80000000 | words.len() as u32;
            words.extend_from_slice(&images[at..at+5]);
            words.push(slot as u32);
        }
        words
    }
    pub(super) fn shader(&self)->String {
        use std::fmt::Write;
        let mut shader=String::from("@group(0) @binding(3) var<storage,read> environment_images:array<u32>;\n");let mut loads=String::new();let mut sizes=String::new();let mut modes=String::new();
        let wrap=|w|match w {v2::Wrap::Repeat=>0,v2::Wrap::ClampToEdge=>1,v2::Wrap::MirroredRepeat=>2};
        for (k,source) in self.sources.iter().enumerate() {
            // An image's extent is read from its descriptor (words 0 and 1), not spelled here,
            // so images of another size use the same text.
            write!(loads,"case {k}u: {{let base=environment_images[{k}u];let pixel=u32(p.y)*environment_images[base]+u32(p.x);var bits:vec4<u32>;for(var c=0u;c<4u;c++) {{let descriptor=environment_images[base+6u+c];if (environment_images[base+5u]&(1u<<c))!=0u {{bits[c]=descriptor;}} else {{bits[c]=environment_images[base+descriptor+pixel];}}}}return bitcast<vec4<f32>>(bits);}}").unwrap();
            write!(sizes,"case {k}u: {{let base=environment_images[{k}u];return vec2<i32>(i32(environment_images[base]),i32(environment_images[base+1u]));}}").unwrap();
            write!(modes,"case {k}u: {{return vec3<u32>({}u,{}u,{}u);}}",wrap(source.definition.wrap_s),wrap(source.definition.wrap_t),u32::from(matches!(source.definition.mag_filter,v2::MagFilter::Nearest))).unwrap();
        }
        writeln!(shader,"fn environment_image_extent(slot:u32)->vec2<i32> {{switch slot {{{sizes} default: {{return vec2(1);}}}}}}\nfn environment_image_modes(slot:u32)->vec3<u32> {{switch slot {{{modes} default: {{return vec3(0u);}}}}}}\nfn environment_image_texel(slot:u32,p:vec2<i32>)->vec4<f32> {{switch slot {{{loads} default: {{return vec4(1.);}}}}}}").unwrap();
        shader.push_str(&sampling_shader());shader
    }
    pub(super) fn identity(&self)->blake3::Hash {let mut hash=blake3::Hasher::new();for source in &self.sources {hash.update(source.identity.as_bytes());hash.update(&serde_json::to_vec(&source.definition.time).expect("serializable image clock"));}hash.update(self.animation.input_identity().as_bytes());hash.finalize()}
    pub(super) fn size(&self,words:&[u32])->Option<[u32;2]> {
        (!self.sources.is_empty()).then(||self.sources.iter().enumerate().fold([1,1],|size,(k,_)| {let at=words[k] as usize;[size[0].max(words[at]),size[1].max(words[at+1])]}))
    }
    pub(super) fn animated(&self)->bool {self.sources.iter().any(|s|s.media.is_some())}
    pub(super) fn uniform(&self)->bool {
        self.sources.iter().all(|s|s.media.is_none() && s.pixels[5]==15)
    }
    pub(super) fn sample_key(&self,request:&vork_time::Request)->Result<Vec<u8>> {
        let time=request.numeric()?;
        let mut result=Vec::new();
        let mut samples=self.animation.samples();self.animation.sample_into(time,&mut samples);
        for s in &self.sources {if let Some(media)=&s.media {
            let time=request.exact()?;
            let base=s.definition.time.as_ref().map(|t|vork_time::exact::Germ::point(t.exact())).unwrap_or_else(||time.clone());
            let clock=self.animation.image_time_exact(&samples,&time,&s.name,base)?;
            result.extend_from_slice(&(media.lock().unwrap().frame_index_exact(&clock,self.fps)? as u64).to_le_bytes());
        }}
        Ok(result)
    }
    pub(super) fn words(&self,request:&vork_time::Request)->Result<Vec<u32>> {
        let time=request.numeric()?;
        let mut words=vec![0;self.sources.len().max(1)];
        let mut samples=self.animation.samples();self.animation.sample_into(time,&mut samples);
        for (k,s) in self.sources.iter().enumerate() {
            words[k]=words.len() as u32;
            if let Some(media)=&s.media {
                let time=request.exact()?;
            let base=s.definition.time.as_ref().map(|t|vork_time::exact::Germ::point(t.exact())).unwrap_or_else(||time.clone());
            let clock=self.animation.image_time_exact(&samples,&time,&s.name,base)?;
                words.extend(encode(&s.definition,s.srgb,media.lock().unwrap().frame_color_exact(&clock,self.fps,self.max_side,s.definition.color_space,crate::texture_source::upload(s.srgb),true)
                    .map_err(|error|crate::texture_source::for_image(error,&s.name))?));
            } else {words.extend_from_slice(&s.pixels);}
        }
        Ok(words)
    }
}
// width, height, wrap S/T, filter, constant-component mask, four plane
// descriptors. A descriptor is either the exact constant bits or a relative
// word offset. Equal nonconstant components share one plane, without quantization.
// `srgb` is the image's `is_srgb()`, decided once per image by the caller, never per texel.
fn encode(definition:&v2::Image,srgb:bool,image:image::DynamicImage)->Vec<u32> {
    let image=image.into_rgba32f();
    let wrap=|w| match w {v2::Wrap::Repeat=>0,v2::Wrap::ClampToEdge=>1,v2::Wrap::MirroredRepeat=>2};
    let planes:[Vec<u32>;4]=std::array::from_fn(|c|image.pixels().map(|p| {
        let v=p.0[c];if c<3 && srgb {crate::texture_source::decode_srgb(v)} else {v}
    }).map(f32::to_bits).collect());
    let mut words=vec![image.width(),image.height(),wrap(definition.wrap_s),wrap(definition.wrap_t),u32::from(matches!(definition.mag_filter,v2::MagFilter::Nearest)),0,0,0,0,0];
    for c in 0..4 {
        if planes[c].iter().all(|v|*v==planes[c][0]) {
            words[5]|=1<<c;words[6+c]=planes[c][0];
        } else if let Some(prior)=(0..c).find(|prior|planes[*prior]==planes[c]) {
            words[6+c]=words[6+prior];
        } else {
            words[6+c]=words.len() as u32;words.extend_from_slice(&planes[c]);
        }
    }
    words
}

#[cfg(test)]
#[test]
fn packed_environment_images_preserve_component_bits() -> Result<()> {
    let definition:v2::Image=serde_json::from_value(serde_json::json!({"file":"unused.exr","color_space":"Non-Color"}))?;
    let image=image::Rgba32FImage::from_fn(3,2,|x,y| {let n=(y*3+x) as f32;image::Rgba([n,n,100000.,if n==0. {-0.} else {n/7.}])});
    let words=encode(&definition,definition.is_srgb(),image::DynamicImage::ImageRgba32F(image.clone()));
    assert_eq!(words.len(),10+2*6);
    assert_eq!(words[6],words[7]);
    assert_eq!(words[5],1<<2);
    for (index,pixel) in image.pixels().enumerate() {for c in 0..4 {
        let descriptor=words[6+c];let bits=if words[5]&(1<<c)!=0 {descriptor} else {words[descriptor as usize+index]};
        assert_eq!(bits,pixel.0[c].to_bits());
    }}
    Ok(())
}

#[cfg(test)]
#[test]
fn missing_environment_image_is_magenta_and_warned() -> Result<()> {
    let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
    let scene = v2::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}],
        "images":{"plate":{"file":"missing-refuseearly-environment.exr"}}}"#)?;
    let layout = vork_slang::InputLayout {words:2, fields:vec![vork_slang::InputField {
        name:"map".into(), ty:vork_slang::Type::Image, offset:1, enumeration:None, range:None,
    }]};
    let values = [("map".into(), serde_json::json!({"image":"plate"}))].into_iter().collect();
    let collector = crate::warnings::Collector::default();
    let _scope = collector.enter();
    let (images, _) = Images::new(&renderer, &scene, Path::new("."), &layout, &values, &Default::default())?;
    let words = images.words(&vork_time::Request::Numeric(0.))?;
    let start = words[0] as usize;
    assert_eq!(&words[start..start+2], [1, 1]);
    assert_eq!(words[start+5],15);
    assert_eq!(&words[start+6..start+10], [1f32.to_bits(), 0, 1f32.to_bits(), 1f32.to_bits()]);
    assert_eq!(collector.snapshot()[0].code, "missing_image");
    Ok(())
}
fn sampling_shader()->String {
    format!(r#"
fn environment_address(p:i32,n:i32,mode:u32)->i32 {{
    if mode==1u {{return clamp(p,0,n-1);}}
    if mode==2u {{let q=((p%(2*n))+2*n)%(2*n);return min(q,2*n-1-q);}}
    return ((p%n)+n)%n;
}}
fn environment_image_load(slot:u32,p:vec2<i32>)->vec4<f32> {{
    let size=environment_image_extent(slot);let mode=environment_image_modes(slot);
    let q=vec2(environment_address(p.x,size.x,mode.x),environment_address(p.y,size.y,mode.y));
    return environment_image_texel(slot,q);
}}
fn {sample}(slot:u32,uv:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec4<f32> {{
    if slot==4294967295u {{return vec4(1.);}}
    let base=environment_images[slot];
    if environment_images[base+5u]==15u {{return environment_image_texel(slot,vec2(0));}}
    let size=vec2<f32>(environment_image_extent(slot));
    // Bound before integer conversion, including very large authored UVs.
    let mode=environment_image_modes(slot);
    var bounded=uv;
    for(var k=0u;k<2u;k++) {{if mode[k]==1u {{bounded[k]=clamp(uv[k],0.,1.);}} else {{let period=select(1.,2.,mode[k]==2u);bounded[k]=uv[k]-floor(uv[k]/period)*period;}}}}
    let p=bounded*size-.5;let q=vec2<i32>(floor(p));let f=p-floor(p);
    if mode.z!=0u {{return environment_image_load(slot,vec2<i32>(floor(p+.5)));}}
    let a=environment_image_load(slot,q);let b=environment_image_load(slot,q+vec2(1,0));
    let c=environment_image_load(slot,q+vec2(0,1));let d=environment_image_load(slot,q+vec2(1,1));
    return (a+(b-a)*f.x)+((c+(d-c)*f.x)-(a+(b-a)*f.x))*f.y;
}}
"#,sample=vork_slang::SAMPLE_IMAGE)
}
