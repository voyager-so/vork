//! Display programs execute once per output pixel. Scene-linear outputs bypass this stage.
use super::*;
use std::fmt::Write;
use vork_scene::v2;
use vork_slang::{Compiler, DisplayProgram};

/// A compiled display program with its packed inputs and image data, ready to evaluate.
pub(crate) struct Prepared {
    /// The view's compiled program. `None` runs none: the picture's colour is already code values for the
    /// destination (`identity`), and the stage only finishes it.
    program: Option<DisplayProgram>,
    color_space: u32,
    domain: u32,
    /// The `DisplayTarget` this stage encodes for, named to the view as `ctx.display`: what the view's code
    /// values must be for, and what its XYZ is converted to.
    destination: u32,
    /// The WGSL of this destination's result-to-destination step (`result_to_destination`).
    encode: String,
    /// The destination's peak luminance and reference white in cd/m², as the view is told them.
    peak: f32,
    white: f32,
    /// The largest code value the destination shows.
    ceiling: f32,
    /// Whether the stage reads a canvas's float picture as a texture and writes a texture, instead of reading
    /// and writing a frame's planes.
    picture: bool,
    /// Whether the result is delivered floating point, unquantized, for the composition that reads this
    /// picture: a world's frame keeps it in its staging planes for its canvas (`InstallOptions::view_picture`);
    /// a picture stage writes an RGBA16F destination for the world that reads the canvas.
    floating: bool,
    words: Vec<u32>,
    animation: Option<programs::InputAnimation>,
    sources: Vec<ImageSource>,
}
/// One `Image` input as linear RGBA32F texels.
struct ImageData {
    width: u32,
    height: u32,
    texels: Vec<f32>,
}
enum ImageSource {
    /// A bundled program's library data (OpenEXR bytes).
    Library(&'static [u8]),
    /// A scene `images` entry with a file, or a library's file beside the executable (`image` is None).
    File {
        /// The scene image's name, for diagnostics.
        image: Option<String>,
        path: PathBuf,
        bytes: Vec<u8>,
        srgb: bool,
        color_space: Option<v2::ColorSpace>,
    },
}

/// A number as a WGSL float literal.
fn wgsl(value: f32) -> String {
    let text = format!("{value:?}");
    // 100.0 is written 100., as the stage always wrote it; an exponent's digits stay.
    if text.contains('e') { text } else { text.trim_end_matches('0').to_owned() }
}
/// SMPTE ST 2084: cd/m² as a PQ code value.
fn pq(nits: f32) -> f32 {
    let l = (nits.max(0.) / 10000.).powf(2610. / 16384.);
    ((3424. / 4096. + 2413. / 128. * l) / (1. + 2392. / 128. * l)).powf(2523. / 32.)
}
/// The engine's one result-to-destination step for a view's XYZ, as the WGSL statements that define `encoded`
/// from `result` (absolute XYZ D65 in cd/m², or code values when `result.w` is positive): the destination's
/// primaries, then its transfer. An SDR display shows 100 cd/m² at code value one. A Rec.2100 display takes
/// Rec.2020 light limited to its peak: as ST 2084 code values, or as HLG through the BT.2100 inverse OOTF,
/// with the system gamma for that peak and OpenColorIO's bounded surround gain, and the OETF.
fn result_to_destination(target: crate::output_device::Target) -> String {
    const REC2020: &str = "clamp(vec3(dot(result.rgb,vec3(1.7166512,-.3556708,-.2533663)),dot(result.rgb,vec3(-.6666844,1.6164812,.0157685)),dot(result.rgb,vec3(.0176399,-.0427706,.9421031))),vec3(0.),vec3(PEAK))";
    match target.display {
        v2::OutputDisplay::Rec2100Pq => format!("let nits={};
    let l=pow(nits/10000.,vec3(.1593017578125));
    let encoded=select(pow((.8359375+18.8515625*l)/(1.+18.6875*l),vec3(78.84375)),result.rgb,result.w>0.);", REC2020.replace("PEAK", &wgsl(target.peak))),
        v2::OutputDisplay::Rec2100Hlg => {
            let gamma = crate::output_device::hlg_gamma(target.peak);
            format!("let light={}/{peak};
    let scene=light*pow(max(dot(light,vec3(.2627,.6780,.0593)),{floor}),{exponent});
    let encoded=select(select(.17883277*log(max(12.*scene-.28466892,vec3(.000001)))+.55991073,sqrt(3.*scene),scene<=vec3(1./12.)),result.rgb,result.w>0.);",
                REC2020.replace("PEAK", &wgsl(target.peak)), peak = wgsl(target.peak),
                floor = wgsl(crate::output_device::hlg_floor(target.peak)), exponent = wgsl((1. - gamma) / gamma))
        }
        v2::OutputDisplay::Srgb => "let c=result.rgb/100.;
    let linear=vec3(dot(c,vec3(3.24096994,-1.53738318,-.49861076)),dot(c,vec3(-.96924364,1.8759675,.04155506)),dot(c,vec3(.05563008,-.20397696,1.05697151)));
    let encoded=select(select(12.92*linear,1.055*pow(max(linear,vec3(0.)),vec3(1./2.4))-.055,linear>vec3(.0031308)),result.rgb,result.w>0.);".to_owned(),
        // Display P3: P3 primaries at D65 with the sRGB transfer.
        v2::OutputDisplay::DisplayP3 => "let c=result.rgb/100.;
    let linear=vec3(dot(c,vec3(2.49349691,-.93138362,-.40271078)),dot(c,vec3(-.82948897,1.76266406,.02362469)),dot(c,vec3(.03584583,-.07617239,.95688452)));
    let encoded=select(select(12.92*linear,1.055*pow(max(linear,vec3(0.)),vec3(1./2.4))-.055,linear>vec3(.0031308)),result.rgb,result.w>0.);".to_owned(),
    }
}

/// The cache identity of a view program's compiled form.
pub(crate) fn program_key(abi: &vork_slang::Abi, source: &str, origin: &Path) -> Result<blake3::Hash> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"view-stage-context-v4");
    hash.update(abi.cache_identity()?.as_bytes());
    hash.update(source.as_bytes());
    hash.update(origin.to_string_lossy().as_bytes());
    Ok(hash.finalize())
}

/// `target` is the display this stage encodes for: the render call's `output.display` and its luminances at
/// the root. A world inside a canvas renders for sRGB, the encoding of every picture a canvas paints.
/// `floating`: the result is also delivered unquantized (`Prepared::floating`).
pub(crate) fn prepare(
    r: &Renderer,
    scene: &v2::Scene,
    base: &Path,
    target: crate::output_device::Target,
    opened: &crate::footage::Opened,
    floating: bool,
) -> Result<Prepared> {
    let (material, display) = scene.view_material()?;
    let material = material.as_deref();
    let name = &display.program;
    let (source, origin) = match vork_slang::module_source(name) {
        Some(source) => (source.to_owned(), PathBuf::from(format!("{name}.slang"))),
        None => programs::program_source(
            base,
            name,
            scene
                .programs
                .get(name)
                .with_context(|| format!("composition.view: unknown program {name}"))?,
        )?,
    };
    let abi = programs::abi();
    let key = program_key(&abi, &source, &origin)?;
    let tag = key.to_hex();
    let program = programs::cached(
        r,
        key,
        &Compiler::discover(),
        &abi, &source, &origin,
        |compiler| compiler.compile_display(&abi, &source, &origin, &tag[..12]),
    )
    .with_context(|| format!("composition.view: program {name}"))?;
    if !program.warnings.trim().is_empty() {
        eprintln!("{}", program.warnings.trim());
    }
    let mut values = display
        .inputs
        .iter()
        .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    // A bundled program's unbound image inputs read its library data.
    let mut defaults = std::collections::BTreeMap::new();
    for field in program
        .inputs
        .fields
        .iter()
        .filter(|f| f.ty == vork_slang::Type::Image)
    {
        if values.contains_key(&field.name) {
            continue;
        }
        if let Some(data) = vork_scene::libraries::program_image(name, &field.name, target.display) {
            // Data that stands in for another input's library data yields to an author's binding of that
            // input, and is the rendering only at the luminances it was baked for.
            if data.instead_of.is_some_and(|input| display.inputs.contains_key(input)) || !data.baked_for(target.peak, target.white) {
                continue;
            }
            defaults.insert(field.name.clone(), data.image);
            values.insert(field.name.clone(), serde_json::Value::Null);
        }
    }
    let mut sources = Vec::new();
    let mut words = program.inputs.pack(&values, |input, value| {
        let source = if let Some(image) = defaults.get(input) {
            let executable = match image {
                vork_scene::libraries::ProgramImage::File(_) => Some(std::env::current_exe()?.canonicalize()?),
                _ => None,
            };
            library_image(*image, executable.as_deref())?
        } else {
            let v2::Input::Texture(binding) = serde_json::from_value::<v2::Input>(value.clone())
                .with_context(|| format!("input {input}: expected {{\"image\": name}}"))? else {
                bail!("input {input}: expected {{\"image\": name}}, got {value}");
            };
            let image = scene.images.get(&binding.image).with_context(|| format!("input {input}: {}",
                v2::diagnostics::unknown("image", &binding.image, scene.images.keys().map(String::as_str))))?;
            let file = image.file.as_ref().with_context(|| format!("input {input}: image {} is a render image; display programs read file images", binding.image))?;
            let path = base.join(file);
            let bytes = std::fs::read(&path).with_context(|| format!("read image {}", path.display()))?;
            ImageSource::File { image: Some(binding.image.clone()), path, bytes, srgb: image.is_srgb(), color_space: image.color_space }
        };
        sources.push(source);
        Ok(sources.len() as u32 - 1)
    }).with_context(|| format!("composition.view: program {name}"))?;
    let animation = match material {
        Some(name) => programs::InputAnimation::with_discrete(scene, base, name, &program.inputs, &mut words, true, opened)?,
        None => None,
    };
    Ok(Prepared::new(scene, target, floating, Some(program), words, animation, sources))
}
impl Prepared {
    /// The stage's fixed facts: the composition's encoding and reference domain, and the display it encodes for.
    fn new(scene: &v2::Scene, target: crate::output_device::Target, floating: bool, program: Option<DisplayProgram>, words: Vec<u32>,
        animation: Option<programs::InputAnimation>, sources: Vec<ImageSource>) -> Self {
        Self {
            color_space: match scene.composition.as_ref().map(v2::Composition::working_space).unwrap_or(v2::WorkingColorSpace::LinearRec709) {
                v2::WorkingColorSpace::LinearRec709 => 0, v2::WorkingColorSpace::LinearRec2020 => 1,
                v2::WorkingColorSpace::ACEScg => 2, v2::WorkingColorSpace::Srgb => 3, v2::WorkingColorSpace::DisplayP3 => 4,
            },
            domain: u32::from(scene.is_canvas()),
            destination: match target.display {
                v2::OutputDisplay::Srgb => 0, v2::OutputDisplay::DisplayP3 => 1,
                v2::OutputDisplay::Rec2100Pq => 2, v2::OutputDisplay::Rec2100Hlg => 3,
            },
            encode: result_to_destination(target),
            peak: target.peak,
            white: target.white,
            // Code values stay within the display's signal range, whichever form the view returned: the peak's
            // code for PQ, one otherwise. XYZ is limited to the peak itself before it is encoded.
            ceiling: if target.display == v2::OutputDisplay::Rec2100Pq { pq(target.peak) } else { 1. },
            picture: false,
            floating,
            animation,
            program,
            words,
            sources,
        }
    }
    /// Whether a canvas's default view changes nothing for `target`: `std/standard` at its defaults is the
    /// identity on the canvas's own encoding, so a canvas that authors no view and is written for the display
    /// whose code values its encoding is (sRGB's for an sRGB canvas, Display P3's for a Display P3 canvas)
    /// runs no program. A canvas that blends in linear light is no display's code values: its view always runs.
    pub(crate) fn identity(scene: &v2::Scene, target: crate::output_device::Target) -> bool {
        scene.composition.as_ref().is_some_and(|composition| composition.space == v2::Space::Canvas && composition.view.is_none()
            && composition.working_space().display() == Some(target.display))
    }
}

/// Follow the companion-tool layout: resolve data beside the canonical executable.
fn library_image(image: vork_scene::libraries::ProgramImage, executable: Option<&Path>) -> Result<ImageSource> {
    match image {
        vork_scene::libraries::ProgramImage::Embedded(bytes) => Ok(ImageSource::Library(bytes)),
        vork_scene::libraries::ProgramImage::File(relative) => {
            let path = executable.context("locate executable for library image")?
                .parent().context("locate executable directory for library image")?.join(relative);
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read library image {}", path.display()))?;
            Ok(ImageSource::File { image: None, path, bytes, srgb: false, color_space: Some(v2::ColorSpace::NonColor) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LIBRARIES` is a const, so this crate has its own copy of the list. Each
    /// embedded table must still be the one in vork-scene: same address, and the
    /// bytes of the file in the repository.
    #[test]
    fn embedded_library_tables_exist_once() -> Result<()> {
        let mut tables = 0;
        for data in vork_scene::libraries::LIBRARIES.iter().flat_map(|library| library.images) {
            let vork_scene::libraries::ProgramImage::Embedded(here) = data.image else { continue };
            let display = data.displays.first().copied().unwrap_or(v2::OutputDisplay::Srgb);
            let owner = vork_scene::libraries::program_image(data.program, data.input, display).context("library table")?;
            let vork_scene::libraries::ProgramImage::Embedded(there) = owner.image else { panic!("embedded in one crate only") };
            assert!(std::ptr::eq(here.as_ptr(), there.as_ptr()), "{}.{} is embedded more than once", data.program, data.input);
            assert_eq!(here.len(), there.len());
            tables += 1;
        }
        assert_eq!(tables, 3, "the three AgX tables");
        let file = |name: &str| std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../libraries/std").join(name));
        for (input, display, name) in [("lut", v2::OutputDisplay::Srgb, "agx.exr"),
            ("destination_lut", v2::OutputDisplay::DisplayP3, "agx-p3.exr"),
            ("destination_lut", v2::OutputDisplay::Rec2100Pq, "agx-pq.exr"),
            ("destination_lut", v2::OutputDisplay::Rec2100Hlg, "agx-pq.exr")] {
            let vork_scene::libraries::ProgramImage::Embedded(bytes) = vork_scene::libraries::program_image("std/agx", input, display).unwrap().image else { panic!("{name} is embedded") };
            assert_eq!(bytes, file(name)?.as_slice(), "{name}");
        }
        Ok(())
    }

    #[test]
    fn filmic_uses_companion_file_and_reports_expected_path() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-filmic-data-{}", std::process::id()));
        let executable = root.join("vork");
        let path = root.join("libraries/blender/filmic.exr");
        let image = vork_scene::libraries::program_image("blender/filmic", "lut", v2::OutputDisplay::Srgb).unwrap().image;
        let error = library_image(image, Some(&executable)).err().unwrap();
        assert!(error.to_string().contains(&path.display().to_string()), "{error:#}");
        std::fs::create_dir_all(path.parent().unwrap())?;
        let original = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../libraries/blender/filmic.exr"))?;
        std::fs::write(&path, &original)?;
        let loaded = library_image(image, Some(&executable))?;
        match &loaded {
            ImageSource::File { bytes, srgb, .. } => { assert_eq!(bytes, &original); assert!(!srgb); }
            _ => panic!("Filmic must load from disk"),
        }
        let decoded = loaded.decode(16384)?;
        assert!(decoded.width > 0 && decoded.height > 0);
        assert!(matches!(library_image(vork_scene::libraries::program_image("std/agx", "lut", v2::OutputDisplay::Srgb).unwrap().image, None)?, ImageSource::Library(_)));
        assert!(vork_scene::libraries::program_image("std/openpbr", "lut", v2::OutputDisplay::Srgb).is_none());
        // std/agx has its own bake for Display P3, and none in that input for sRGB.
        assert!(vork_scene::libraries::program_image("std/agx", "destination_lut", v2::OutputDisplay::Srgb).is_none());
        let p3 = vork_scene::libraries::program_image("std/agx", "destination_lut", v2::OutputDisplay::DisplayP3).unwrap();
        assert_eq!(p3.instead_of, Some("lut"));
        let cube = library_image(p3.image, None)?.decode(16384)?;
        assert_eq!((cube.width, cube.height), (57 * 57, 57));
        // Both Rec.2100 displays read the one HDR cube, baked for a 1000 cd/m² display.
        for display in [v2::OutputDisplay::Rec2100Pq, v2::OutputDisplay::Rec2100Hlg] {
            let hdr = vork_scene::libraries::program_image("std/agx", "destination_lut", display).unwrap();
            assert!(hdr.baked_for(1000., 203.) && hdr.baked_for(1000., 100.) && !hdr.baked_for(600., 203.));
            let cube = library_image(hdr.image, None)?.decode(16384)?;
            assert_eq!((cube.width, cube.height), (57 * 57, 57));
        }
        // Number literals keep the sRGB stage's text: 100 cd/m² is written as before.
        assert_eq!([wgsl(100.), wgsl(1.), wgsl(0.5), wgsl(203.), wgsl(1e-10)], ["100.", "1.", "0.5", "203.", "1e-10"]);
        assert!((pq(100.) - 0.5080784).abs() < 1e-5 && (pq(1000.) - 0.7518271).abs() < 1e-5 && (pq(10000.) - 1.).abs() < 1e-6);
        let srgb = result_to_destination(crate::output_device::Target::default());
        assert!(srgb.starts_with("let c=result.rgb/100.;") && srgb.contains("3.24096994"));
        let hlg = result_to_destination(crate::output_device::Target::for_display(v2::OutputDisplay::Rec2100Hlg));
        assert!(hlg.contains("vec3(1000.)") && hlg.contains("-0.16666"), "{hlg}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}

impl ImageSource {
    /// Decode once at installation; images remain resident for per-pixel evaluation.
    fn decode(&self, max_side: u32) -> Result<ImageData> {
        let (decoded, srgb) = match self {
            Self::Library(bytes) => (
                crate::texture_source::decode_format(image::ImageFormat::OpenExr, bytes, max_side)?,
                false,
            ),
            Self::File { image, path, bytes, srgb, color_space } => (
                crate::texture_source::decode_color(path, bytes, max_side, *color_space, crate::texture_source::upload(*srgb))
                    .map_err(|error| crate::texture_source::for_image(error, image.as_deref().unwrap_or_default()))
                    .with_context(|| format!("decode image {}", path.display()))?,
                *srgb,
            ),
        };
        let rgba = decoded.to_rgba32f();
        let (width, height) = rgba.dimensions();
        let mut texels = rgba.into_raw();
        if srgb {
            for texel in texels.chunks_exact_mut(4) {
                for c in &mut texel[..3] {
                    *c = texture_source::decode_srgb(*c);
                }
            }
        }
        Ok(ImageData {
            width,
            height,
            texels,
        })
    }
}

pub(crate) struct Stage {
    pipeline: wgpu::ComputePipeline,
    words: wgpu::Buffer,
    images: Vec<wgpu::TextureView>,
    input_words: Vec<u32>,
    animation: Option<programs::InputAnimation>,
}
impl Prepared {
    pub(crate) fn install(self, r: &Renderer) -> Result<Stage> {
        let d = &r.device;
        let decoded = self
            .sources
            .iter()
            .map(|source| source.decode(d.limits().max_texture_dimension_2d))
            .collect::<Result<Vec<_>>>()
            .context("composition.view: display program image")?;
        let mut shader = self.program.as_ref().map(|program| program.wgsl.clone()).unwrap_or_default();
        shader.push_str(include_str!(concat!(env!("OUT_DIR"), "/constants.wgsl")));
        let mut loads = String::new();
        let mut extents = String::new();
        for k in 0..decoded.len() {
            writeln!(
                shader,
                "@group(0) @binding({}) var display_image_{k}:texture_2d<f32>;",
                k + 4
            )?;
            write!(
                loads,
                "case {k}u: {{let last=vec2<i32>(textureDimensions(display_image_{k}))-1;return textureLoad(display_image_{k},clamp(p,vec2(0),last),0);}}"
            )?;
            write!(
                extents,
                "case {k}u: {{return vec2<f32>(textureDimensions(display_image_{k}));}}"
            )?;
        }
        // A floating picture stage writes the picture its reader samples; every other stage writes the frame.
        let output_format = if self.picture && self.floating { wgpu::TextureFormat::Rgba16Float } else { r.output_format() };
        let format = match output_format {
            wgpu::TextureFormat::Rgba16Float => "rgba16float",
            wgpu::TextureFormat::Bgra8Unorm => "bgra8unorm",
            _ => "rgba8unorm",
        };
        let post = std::mem::offset_of!(Uniform, post) / 16;
        let ceiling = wgsl(self.ceiling);
        // The view runs once for the pixel. With no program the picture's colour is the result as it is: code
        // values for this stage's destination.
        let run = match &self.program {
            Some(program) => format!("{entry}(value.rgb,gid.xy,bitcast<u32>(settings.x),display_frame.words[{post}].y,{color_space}u,{domain}u,{destination}u,{peak},{white},0u,&result);",
                entry = program.entry, color_space = self.color_space, domain = self.domain, destination = self.destination, peak = wgsl(self.peak), white = wgsl(self.white)),
            None => format!("result[0]=vec4(value.rgb,f32(1u+{}u));", self.destination),
        };
        // Where the stage's input comes from and where its result goes. A frame keeps its float planes, its
        // invalid-pixel marks and its packed readback planes in one buffer. A canvas's float picture is a
        // texture of associated colour: the view reads and returns straight colour, and the result goes to a
        // texture, either the final 8-bit frame (clamped and rounded here, once) or, floating, the picture a
        // world reads (the view's code values for sRGB, associated again, saturating at RGBA16F's 65504).
        // A picture stage adds no dither: a canvas's own colours are exact codes, and a world's picture in it
        // arrives carrying the world's dither.
        let (source, value, deliver) = if self.picture {
            ("@group(0) @binding(1) var display_picture:texture_2d<f32>;".to_owned(),
                "let texel=textureLoad(display_picture,vec2<i32>(gid.xy),0);let value=vec4(select(vec3(0.),texel.rgb/max(texel.a,1e-30),texel.a>0.),texel.a);".to_owned(),
                if self.floating {
                    "let extended=encode_view(result[0],gid.xy,8.,0.);\n    textureStore(display_output,vec2<i32>(gid.xy),clamp(vec4(extended*value.a,value.a),vec4(-65504.),vec4(65504.)));".to_owned()
                } else {
                    format!("let mapped=clamp(encode_view(result[0],gid.xy,8.,0.),vec3(0.),vec3({ceiling}));\n    textureStore(display_output,vec2<i32>(gid.xy),round(clamp(vec4(mapped,value.a),vec4(0.),vec4(1.))*255.)/255.);")
                })
        } else {
            // A world drawn for a canvas also keeps its view's result unquantized and unclamped: code values
            // for sRGB, with the output dither the 8-bit frame is quantized with, in the world's
            // own pixels. The dither is the world's: the canvas that composes the picture only rounds, once, and
            // adds no noise to its own content. The values replace the pixel's consumed staging input in place,
            // as the 16-bit colour plane does (a frame asks for one or the other). A discarded pixel carries
            // the opaque diagnostic marker there, as the integer output does.
            let floating = if self.floating {
                format!("\n    let extended=select(encode_view(result[0],gid.xy,8.,1.-display_frame.words[{post}].z),marker.rgb,invalid);\n    display_pixels[base*count+i]=bitcast<u32>(extended.r);display_pixels[(base+1u)*count+i]=bitcast<u32>(extended.g);display_pixels[(base+2u)*count+i]=bitcast<u32>(extended.b);\n    if invalid {{display_pixels[(base+3u)*count+i]=bitcast<u32>(marker.a);}}")
            } else { String::new() };
            ("@group(0) @binding(1) var<storage,read_write> display_pixels:array<u32>;".to_owned(),
                "let value=vec4(bitcast<f32>(display_pixels[base*count+i]),bitcast<f32>(display_pixels[(base+1u)*count+i]),bitcast<f32>(display_pixels[(base+2u)*count+i]),bitcast<f32>(display_pixels[(base+3u)*count+i]));".to_owned(),
                format!(r#"var mapped=clamp(encode_view(result[0],gid.xy,8.,1.-display_frame.words[{post}].z),vec3(0.),vec3({ceiling}));
    // Quantize once: WGSL round uses ties-to-even, matching packed PNG readback.
    // Texture stores may otherwise use a different hardware UNORM rounding rule.
    let bitmap=arrayLength(&display_pixels)-(count+31u)/32u;
    let invalid=(display_pixels[bitmap+i/32u]&(1u<<(i%32u)))!=0u;
    let marker=unpack4x8unorm(REG_DIAGNOSTIC_MAGENTA_RGBA);
    let quantized=select(round(clamp(vec4(mapped,value.a),vec4(0.),vec4(1.))*255.)/255.,marker,invalid);
    if((flags&REG_OUTPUT_GPU_COLOR)!=0u) {{textureStore(display_output,vec2<i32>(gid.xy),quantized);}}
    if((flags&REG_OUTPUT_COLOR_PLANE)!=0u) {{display_pixels[i]=pack4x8unorm(quantized);}}
    // 16-bit PNG colour replaces this pixel's consumed staging input in place. The view ran once above: only
    // the encode's dither depends on the depth.
    if((flags&REG_OUTPUT_COLOR_PLANE16)!=0u) {{
        mapped=clamp(encode_view(result[0],gid.xy,16.,1.-display_frame.words[{post}].z),vec3(0.),vec3({ceiling}));
        let color=select(vec4(mapped,value.a),marker,invalid);
        display_pixels[base*count+i]=pack2x16unorm(color.rg);display_pixels[(base+1u)*count+i]=pack2x16unorm(color.ba);}}{floating}"#))
        };
        writeln!(
            shader,
            r#"
struct DisplayFrame {{words:array<vec4<f32>,{frame_words}>}};
@group(0) @binding(0) var<storage,read> display_words:array<u32>;
{source}
@group(0) @binding(2) var<uniform> display_frame:DisplayFrame;
@group(0) @binding(3) var<storage,read_write> display_errors:array<atomic<u32>>;
@group(1) @binding(0) var display_output:texture_storage_2d<{format},write>;
fn {word}(i:u32)->u32 {{return display_words[i];}}
fn display_load(slot:u32,p:vec2<i32>)->vec4<f32> {{switch slot {{{loads}default: {{return vec4(1.);}}}}}}
fn display_extent(slot:u32)->vec2<f32> {{switch slot {{{extents}default: {{return vec2(1.);}}}}}}
fn vork_view_load(slot:u32,p:vec2<i32>)->vec4<f32> {{return display_load(slot,p);}}
fn {sample}(slot:u32,uv:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec4<f32> {{
    let p=uv*display_extent(slot)-.5;let q=vec2<i32>(floor(p));let f=p-floor(p);
    return mix(mix(display_load(slot,q),display_load(slot,q+vec2(1,0)),f.x),mix(display_load(slot,q+vec2(0,1)),display_load(slot,q+vec2(1,1)),f.x),f.y);
}}
// The one result-to-destination step, then output dither. A view's result is a value and its form (w): code
// values for this destination are the result as they are; absolute XYZ D65 in cd/m² is converted to the
// destination's primaries and transfer.
fn encode_view(result:vec4<f32>,pixel:vec2<u32>,depth:f32,dither:f32)->vec3<f32> {{
    {encode}
    var seed=pixel.x+pixel.y*0x9e3779b9u;
    seed=(seed^(seed>>16u))*0x45d9f3bu;seed=(seed^(seed>>16u))*0x45d9f3bu;seed=seed^(seed>>16u);
    let u=f32(seed)*2.3283064365386963e-10*2.-1.;
    return encoded+sign(u)*(1.-sqrt(1.-abs(u)))*.0033*(255./(exp2(depth)-1.))*dither;
}}
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {{
    let temporal=display_frame.words[{temporal}];
    if(temporal.x+1.<temporal.y) {{return;}}
    let flags=u32(display_frame.words[{projection}].w);
    if((flags&(REG_OUTPUT_GPU_COLOR|REG_OUTPUT_COLOR_PLANE))==0u) {{return;}}
    let settings=display_frame.words[{display}];let size=vec2<u32>(settings.yz);
    if any(gid.xy>=size) {{return;}}
    let count=size.x*size.y;let i=gid.y*size.x+gid.x;let base=u32(settings.w);
    {value}
    var result:array<vec4<f32>,1>;
    {run}
    // Code values name the display they are for (w is one plus its index): any display but this stage's
    // destination is refused, never reinterpreted. Either form must be finite.
    if result[0].w>0. && result[0].w!=f32(1u+{destination}u) {{atomicMax(&display_errors[0],REG_VIEW_DESTINATION);result[0]=vec4(0.);}}
    if !all(abs(result[0].rgb)<=vec3(3.402823466e38)) {{atomicMax(&display_errors[0],REG_DISPLAY_NONFINITE);result[0]=vec4(0.);}}
    {deliver}
}}
"#,
            destination = self.destination,
            encode = self.encode,
            frame_words = std::mem::size_of::<Uniform>() / 16,
            projection = std::mem::offset_of!(Uniform, projection) / 16,
            temporal = std::mem::offset_of!(Uniform, temporal) / 16,
            display = std::mem::offset_of!(Uniform, display) / 16,
            word = vork_slang::MATERIAL_WORD,
            sample = vork_slang::SAMPLE_IMAGE,
        )?;
        // The output consumes one texture slot in the same compute stage, and so does a picture stage's source,
        // which is also one of the stage's sampled textures: what is left is the view's image inputs.
        let source_textures=u32::from(self.picture);
        let texture_limit=d.limits().max_sampled_textures_per_shader_stage.saturating_sub(source_textures).min(
            if r.adapter_info.backend==wgpu::Backend::Metal {pipelines::METAL_STAGE_LIMITS.textures.saturating_sub(1+source_textures)} else {u32::MAX});
        anyhow::ensure!(decoded.len() as u32<=texture_limit,
            "display program has {} image inputs; limit is {texture_limit}; reduce display image inputs",decoded.len());
        let module = crate::timing::shader_module(
            d,
            wgpu::ShaderModuleDescriptor {
                label: Some("per-pixel display program"),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            },
        );
        let mut entries = vec![
            pipelines::buffer_entry(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: true },
            ),
            if self.picture {
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }
            } else {
                pipelines::buffer_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                )
            },
            pipelines::buffer_entry(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Uniform,
            ),
            pipelines::buffer_entry(
                3,
                wgpu::ShaderStages::COMPUTE,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
        ];
        entries.extend(
            (0..decoded.len() as u32).map(|k| wgpu::BindGroupLayoutEntry {
                binding: k + 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }),
        );
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display program"),
            entries: &entries,
        });
        let output = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display output"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: output_format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display program"),
            bind_group_layouts: &[Some(&layout), Some(&output)],
            immediate_size: 0,
        });
        let pipeline = r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("display program"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: r.cache.as_ref(),
        });
        let words = init(
            d,
            "display program inputs",
            bytemuck::cast_slice(if self.words.is_empty() {
                &[0]
            } else {
                &self.words
            }),
            wgpu::BufferUsages::STORAGE,
        );
        let images = decoded
            .iter()
            .map(|image| {
                let texture = crate::gpu_memory::texture(
                    d,
                    &wgpu::TextureDescriptor {
                        label: Some("display program image"),
                        size: wgpu::Extent3d {
                            width: image.width,
                            height: image.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                );
                r.queue.write_texture(
                    texture.as_image_copy(),
                    bytemuck::cast_slice(&image.texels),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(image.width * 16),
                        rows_per_image: Some(image.height),
                    },
                    texture.size(),
                );
                texture.create_view(&Default::default())
            })
            .collect();
        Ok(Stage {
            pipeline,
            words,
            images,
            input_words: self.words,
            animation: self.animation,
        })
    }
}
impl Stage {
    pub(crate) fn frame_inputs(&self, r: &Renderer) -> Option<wgpu::Buffer> {
        self.animation.as_ref()?;
        Some(init(&r.device, "animated display inputs", bytemuck::cast_slice(&self.input_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST))
    }
    pub(crate) fn upload(&self, buffer: &wgpu::Buffer, time: f32, changes: &mut crate::values::ChangeTracker) {
        let mut words = self.input_words.clone();
        self.animation.as_ref().unwrap().words(time, &mut words);
        changes.upload(buffer, 0, bytemuck::cast_slice(&words));
    }
    pub(crate) fn dispatch(
        &self,
        r: &Renderer,
        uniform: &wgpu::Buffer,
        words: Option<&wgpu::Buffer>,
        source: wgpu::BindingResource<'_>,
        errors: &wgpu::Buffer,
        size: [u32; 2],
    ) -> graph::Dispatch {
        let mut entries = vec![
            entry(0, words.unwrap_or(&self.words).as_entire_binding()),
            entry(1, source),
            entry(2, uniform.as_entire_binding()),
            entry(3, errors.as_entire_binding()),
        ];
        entries.extend(
            self.images
                .iter()
                .enumerate()
                .map(|(i, view)| entry(i as u32 + 4, wgpu::BindingResource::TextureView(view))),
        );
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display program"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        graph::Dispatch::new(
            &self.pipeline,
            bind,
            [size[0].div_ceil(8), size[1].div_ceil(8), 1],
        )
    }
}

/// What a composition's picture refuses with, by the code recorded for it: the frame's shader diagnostics and
/// a picture step's own error word say the same thing. A picture is not finite because its view's result is
/// not, or because its image graph's is: a world's view reads its graph's result, and a canvas's graph checks
/// its own (`compositor::CanvasGraph`).
pub(crate) const NONFINITE: &str = "composition.compositor or composition.view returned a non-finite colour";
pub(crate) const OTHER_DISPLAY: &str = "composition.view: view returned code values for another display than its destination; return view_from_codes(codes, ctx.display) or view_from_xyz";
/// Refuse a picture step's recorded error word (zero when its result was usable everywhere).
pub(crate) fn refuse(code: u32) -> Result<()> {
    ensure!(code != vork_slang::registry::VIEW_DESTINATION, "{OTHER_DISPLAY}");
    ensure!(code == 0, "{NONFINITE}");
    Ok(())
}
/// The error word of one step over a canvas's picture (its view stage, its image graph): where the step
/// records a result it cannot use when the picture is not drawn for a frame that collects its own diagnostics.
pub(crate) struct ErrorWord {
    errors: wgpu::Buffer,
    readback: wgpu::Buffer,
}
impl ErrorWord {
    pub(crate) fn new(r: &Renderer, label: &str) -> Self {
        let errors = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some(label), size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
            label: Some(label), size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        Self { errors, readback }
    }
    /// The word (four bytes): cleared by the step before it runs, copied by a frame that reads it with its
    /// other results, in the same commands or before the step next clears it.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer { &self.errors }
    /// Wait for the step's last run and refuse an unusable result: for a picture with no collection point of its own.
    pub(crate) fn check(&self, r: &Renderer) -> Result<()> {
        let mut encoder = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas picture check") });
        encoder.copy_buffer_to_buffer(&self.errors, 0, &self.readback, 0, 4);
        let submission = r.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |result| { let _ = sender.send(result); });
        r.wait(Some(submission))?;
        receiver.recv().context("canvas picture error readback")??;
        let code = { let mapped = self.readback.get_mapped_range(..); u32::from_le_bytes(mapped[..4].try_into().expect("four bytes")) };
        self.readback.unmap();
        refuse(code)
    }
}

/// A canvas's view over its drawn picture: the same view stage, reading the picture Moda drew as a texture
/// (associated RGBA16F; or the 8-bit picture of a canvas encoded for another display than its own, where the
/// view is not the identity) and writing a texture. The view receives straight colour in the encoding the
/// canvas blends in (`ctx.color_space`) and
/// the display reference domain, the render seed, the picture's local time and its animated inputs, as a
/// world's view does. A root canvas's stage writes its 8-bit frame, clamped and rounded once; `floating`, it
/// writes the RGBA16F picture a world reads. Neither adds dither: dither is a world's, and a world's picture
/// in the canvas carries it. A canvas whose view is the identity (`Prepared::identity`) compiles and runs no
/// program: the stage finishes the picture as it is, with the one clamp and rounding every view's result takes.
pub(crate) struct PictureStage {
    stage: Stage,
    floating: bool,
    /// A result that cannot be used is zeroed and recorded here, unless the picture is drawn for a frame that
    /// collects its own diagnostics (`encode`).
    errors: ErrorWord,
    size: [u32; 2],
    seed: u32,
}
impl PictureStage {
    /// `target` is the display the stage encodes for: a render's `output.display`, or sRGB for a reader
    /// (`floating`: a world that reads the canvas, an EXR).
    pub(crate) fn new(r: &Renderer, scene: &v2::Scene, base: &Path, target: crate::output_device::Target, floating: bool) -> Result<Self> {
        let mut prepared = if Prepared::identity(scene, target) { Prepared::new(scene, target, floating, None, Vec::new(), None, Vec::new()) }
            else { prepare(r, scene, base, target, &crate::footage::Opened::default(), floating)? };
        prepared.picture = true;
        let stage = prepared.install(r)?;
        Ok(Self { stage, floating, errors: ErrorWord::new(r, "canvas view errors"), size: scene.render.resolution, seed: scene.render.settings.seed })
    }
    /// The format `encode` writes: the picture a world reads, or the renderer's 8-bit frame.
    pub(crate) fn format(&self, r: &Renderer) -> wgpu::TextureFormat {
        if self.floating { wgpu::TextureFormat::Rgba16Float } else { r.output_format() }
    }
    /// Record the stage over `source` (the canvas's float picture) at the picture's local instant `request`,
    /// writing `destination`. The view's animated inputs are sampled at the exact request, as the canvas's own
    /// tracks are, so a key falls on the same side of a boundary for both. `errors` is where an unusable result is recorded: the shader diagnostics of the frame the
    /// picture is drawn for, which that frame refuses where it is collected; without them this stage's own
    /// word, cleared here and read by `errors` or `check`.
    pub(crate) fn encode(&self, r: &Renderer, encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture, destination: &wgpu::Texture,
        request: &vork_time::Request, errors: Option<&wgpu::Buffer>) -> Result<()> {
        let time = request.numeric()?;
        // Moda's picture: float, or 8-bit when the stage encodes an 8-bit canvas for another display than its own.
        ensure!(matches!(source.format(), wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba8Unorm), "a canvas view reads the canvas's drawn picture");
        ensure!(destination.format() == self.format(r) && destination.usage().contains(wgpu::TextureUsages::STORAGE_BINDING),
            "the canvas frame cannot be written on this device");
        if errors.is_none() { encoder.clear_buffer(self.errors.buffer(), 0, None); }
        let errors = errors.unwrap_or(self.errors.buffer());
        let mut frame = Uniform::zeroed();
        // One finished sample, GPU colour only.
        frame.temporal = [0., 1., 0., 0.];
        frame.projection[3] = constants::OUTPUT_GPU_COLOR as f32;
        frame.display = [f32::from_bits(self.seed), self.size[0] as f32, self.size[1] as f32, 0.];
        frame.post[1] = time;
        let uniform = init(&r.device, "canvas view context", bytemuck::bytes_of(&frame), wgpu::BufferUsages::UNIFORM);
        let inputs = self.stage.animation.as_ref().map(|animation| -> Result<_> {
            let mut words = self.stage.input_words.clone();
            animation.words_request(request, time, &mut words)?;
            Ok(init(&r.device, "canvas animated view inputs", bytemuck::cast_slice(&words), wgpu::BufferUsages::STORAGE))
        }).transpose()?;
        let output = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas view output"), layout: &self.stage.pipeline.get_bind_group_layout(1),
            entries: &[entry(0, wgpu::BindingResource::TextureView(&destination.create_view(&Default::default())))],
        });
        let picture = source.create_view(&Default::default());
        self.stage.dispatch(r, &uniform, inputs.as_ref(), wgpu::BindingResource::TextureView(&picture), errors, self.size)
            .with_material(&output).encode_packet(encoder, false);
        Ok(())
    }
    /// This stage's own error word (four bytes) after the last `encode` without a frame's diagnostics. Copy it
    /// in the same commands, before the next `encode` clears it, and `refuse` it where the frame is collected.
    pub(crate) fn errors(&self) -> &wgpu::Buffer { self.errors.buffer() }
    /// Wait for the last `encode` and refuse an unusable result: for a picture with no collection point of its own.
    pub(crate) fn check(&self, r: &Renderer) -> Result<()> { self.errors.check(r) }
}
