use crate::{FrameOutput, FrameReadback, ReadbackOutputs};
use anyhow::{Result, anyhow};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
};


/// Blender ImageFormatSettings.color_depth for PNG colour: 8 or 16 bits per component.
/// Both carry the same exposure and display transform; control channels keep their own depths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PngDepth {
    #[default]
    Eight,
    Sixteen,
}
impl std::str::FromStr for PngDepth {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "8" => Ok(Self::Eight),
            "16" => Ok(Self::Sixteen),
            _ => Err(vork_scene::diagnostics::unknown("png_depth", value, ["8", "16"])),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Output {
    Aov(String),
    Color,
    Depth,
    ObjectIndex,
    ViewNormal,
    Motion,
    Camera,
}
impl std::str::FromStr for Output {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if let Some(name)=value.strip_prefix("aov:") {
            // AOV layers share the EXR namespace with built-in channels.
            return if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')) {Err("AOV output requires a name using ASCII letters, digits, _ or -".into())}
            else if vork_scene::v2::OUTPUT_CHANNELS.contains(&name) {Err(format!("AOV {name:?} collides with a built-in channel and EXR layer; rename the AOV"))}
            else {Ok(Self::Aov(name.to_owned()))};
        }
        Self::builtin(value).ok_or_else(|| {
            ::vork_scene::v2::diagnostics::unknown("output channel", value, vork_scene::v2::OUTPUT_CHANNELS.iter().copied())
        })
    }
}
impl Output {
    /// A built-in channel by its token; vork_scene::v2::OUTPUT_CHANNELS lists the tokens.
    fn builtin(name: &str) -> Option<Self> {
        Some(match name {
            "color" => Self::Color,
            "depth" => Self::Depth,
            "object_index" => Self::ObjectIndex,
            "view_normal" => Self::ViewNormal,
            "motion" => Self::Motion,
            "camera" => Self::Camera,
            "shadow_catcher" => Self::Aov("shadow_catcher".into()),
            _ => return None,
        })
    }
    /// Every built-in channel, the default selection before declared AOVs.
    pub fn builtins() -> impl Iterator<Item = Self> {
        vork_scene::v2::OUTPUT_CHANNELS.iter().map(|name| Self::builtin(name).expect("every listed channel token parses"))
    }
}
#[derive(Clone)]
pub struct Outputs(pub Vec<Output>);
impl Outputs {
    pub fn new(outputs: Vec<Output>) -> Self {
        let mut unique = Vec::new();
        for output in outputs {
            if !unique.contains(&output) {
                unique.push(output);
            }
        }
        Self(unique)
    }
    pub fn has(&self, output: Output) -> bool {
        self.0.contains(&output)
    }
    pub fn readback(&self, packed: bool, float: bool, png_depth: PngDepth) -> ReadbackOutputs {
        let color = packed && self.has(Output::Color);
        ReadbackOutputs {
            aovs: self.0.iter().any(|o|matches!(o,Output::Aov(_))),
            color: color && png_depth == PngDepth::Eight,
            color16: color && png_depth == PngDepth::Sixteen,
            depth_ids: (packed && self.has(Output::Depth)) || self.has(Output::ObjectIndex),
            normals: packed && self.has(Output::ViewNormal),
            motion: packed && self.has(Output::Motion),
            float: crate::FloatOutputs {
                color: float && self.has(Output::Color),
                depth: float && self.has(Output::Depth),
                normals: float && self.has(Output::ViewNormal),
                motion: float && self.has(Output::Motion),
            },
        }
    }
}
enum Input {
    Canvas { frame: usize, rgba: Arc<Vec<u8>>, lease: Arc<crate::ring::Lease> },
    CanvasFloat { frame: usize, rgba: Arc<Vec<[f32; 4]>>, lease: Arc<crate::ring::Lease> },
    Frame {
        frame: usize,
        pixels: FrameReadback,
    },
    Unpacked {
        frame: usize,
        pixels: Result<Arc<FrameOutput>>,
        lease: Option<Arc<crate::ring::Lease>>,
    },
    /// A refused frame: its output orders complete with no files.
    Skip {
        frame: usize,
    },
    Finish,
}
enum Job {
    Canvas { order: usize, frame: usize, rgba: Arc<Vec<u8>>, lease: Arc<crate::ring::Lease> },
    CanvasFloat { order: usize, frame: usize, rgba: Arc<Vec<[f32; 4]>>, lease: Arc<crate::ring::Lease> },
    Unpack {
        frame: usize,
        pixels: FrameReadback,
    },
    Plane {
        order: usize,
        frame: usize,
        outputs: Outputs,
        exr: bool,
        pixels: Arc<FrameOutput>,
        lease: Option<Arc<crate::ring::Lease>>,
    },
}
type Files = Vec<(String, Vec<u8>)>;
/// Whether a writer's destination is one still: a `.png` path. Anything else is a directory of frames.
fn one_still(out: &std::path::Path) -> bool {
    !crate::shot::directory_marked(out) && out.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))
}
/// Every PNG vork writes uses the fast setting: canvas or world, still or sequence frame, colour or control
/// channel. One policy, no split by kind: the smaller file the default setting gives costs far more time than
/// it saves bytes (about 1.8 s on a 4K still). Lossless either way.
const COMPRESSION: png::Compression = png::Compression::Fast;
fn encode_canvas(w: u32, h: u32, rgba: &[u8], display: vork_scene::v2::OutputDisplay) -> Result<Files> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(COMPRESSION);
        // The picture names its display, as a world's does.
        let cicp = cicp(display);
        if cicp.is_none() { encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual); }
        let mut writer = encoder.write_header()?;
        if let Some(cicp) = cicp { writer.write_chunk(png::chunk::ChunkType(*b"cICP"), &cicp)?; }
        writer.write_image_data(rgba)?;
    }
    Ok(vec![(".png".to_owned(), bytes)])
}
/// The cICP chunk (PNG Third Edition) naming a colour picture's display: H.273 primaries and transfer, RGB,
/// full range. An sRGB picture carries the sRGB chunk instead.
fn cicp(display: vork_scene::v2::OutputDisplay) -> Option<[u8; 4]> {
    use vork_scene::v2::OutputDisplay::*;
    match display {
        Srgb => None,
        DisplayP3 => Some([12, 13, 0, 1]),
        Rec2100Pq => Some([9, 16, 0, 1]),
        Rec2100Hlg => Some([9, 18, 0, 1]),
    }
}
fn encode(w: u32, h: u32, p: &FrameOutput, outputs: &Outputs, png_depth: PngDepth, display: vork_scene::v2::OutputDisplay) -> Result<Files> {
    use png::{
        BitDepth::{Eight, Sixteen},
        ColorType::{Grayscale, Rgb, Rgba},
    };
    let mut files = Vec::new();
    let mut png = |suffix: &str, color, depth, data: &[u8]| -> Result<()> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, w, h);
            encoder.set_color(color);
            encoder.set_depth(depth);
            encoder.set_compression(COMPRESSION);
            // The colour picture names its display; control channels are data.
            let cicp = if suffix == ".png" { cicp(display) } else { None };
            if suffix == ".png" && cicp.is_none() {encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);}
            let mut writer = encoder.write_header()?;
            if let Some(cicp) = cicp { writer.write_chunk(png::chunk::ChunkType(*b"cICP"), &cicp)?; }
            writer.write_image_data(data)?;
        }
        files.push((suffix.to_owned(), bytes));
        Ok(())
    };
    if outputs.has(Output::Color) {
        match png_depth {
            PngDepth::Eight => png(".png", Rgba, Eight, &p.rgba)?,
            PngDepth::Sixteen => {
                let words: Vec<_> = p.rgba16.iter().map(|v| v.map(u16::to_be)).collect();
                png(".png", Rgba, Sixteen, bytemuck::cast_slice(&words))?
            }
        }
    }
    // Keep one output item per pixel. Flattening byte arrays through nested
    // iterators left a scalar Iterator::next state machine on this hot path;
    // collecting fixed-size records lets LLVM vectorize the endian conversion.
    // PNG stores 16-bit components big-endian on every host.
    if outputs.has(Output::Depth) {
        let words: Vec<_> = p.depth_u16.iter().map(|v| v.to_be()).collect();
        png(".depth.png", Grayscale, Sixteen, bytemuck::cast_slice(&words))?;
    }
    if outputs.has(Output::ObjectIndex) {
        let words: Vec<_> = p.ids.iter().map(|v| v.to_be()).collect();
        png(".object_index.png", Grayscale, Sixteen, bytemuck::cast_slice(&words))?;
    }
    if outputs.has(Output::ViewNormal) {
        png(".view_normal.png", Rgb, Eight, &p.normals)?;
    }
    if outputs.has(Output::Motion) {
        let words: Vec<_> = p.motion.iter().map(|v| [v[0].to_be(), v[1].to_be(), 0u16]).collect();
        png(".motion.png", Rgb, Sixteen, bytemuck::cast_slice(&words))?;
    }
    for output in &outputs.0 {
        if let Output::Aov(name)=output {
            let aov=p.aovs.get(name).ok_or_else(||anyhow!("missing AOV {name}"))?;
            let components=if aov.kind==vork_scene::v2::AovType::Value {1} else {3};
            let bytes:Vec<_>=aov.pixels.iter().flat_map(|v|v[..components].iter().flat_map(|c|((c.clamp(0.,1.)*65535.).round() as u16).to_be_bytes())).collect();
            png(&if name=="shadow_catcher" {".shadow_catcher.png".into()} else {format!(".aov.{name}.png")},if components==1 {Grayscale} else {Rgb},Sixteen,&bytes)?;
        }
    }
    Ok(files)
}
fn float_channels<'a, const N: usize>(
    names: [&'a str; N],
    values: &Option<Vec<[f32; N]>>,
) -> Result<Vec<(&'a str, Vec<f32>)>> {
    let values = values
        .as_ref()
        .ok_or_else(|| anyhow!("EXR channels require float readback"))?;
    Ok(names
        .into_iter()
        .enumerate()
        .map(|(c, name)| (name, values.iter().map(|v| v[c]).collect()))
        .collect())
}

/// OpenEXR stores linear HDR and unquantized control signals; RGBA is premultiplied.
#[cfg(test)]
fn encode_exr(w: u32, h: u32, p: &FrameOutput, outputs: &Outputs, codec: crate::ExrCodec) -> Result<Files> {
    encode_exr_clock(w,h,p,outputs,codec,None,Default::default())
}
/// A picture's associated RGBA as the default layer's R, G, B and A channels, its colour in the storage space.
fn beauty_channels(pixels: &[[f32; 4]], color: crate::output_device::RawColor) -> Vec<(&'static str, Vec<f32>)> {
    let mut channels: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(pixels.len()));
    for &pixel in pixels {
        for (channel, value) in channels.iter_mut().zip(color.store_associated(pixel)) { channel.push(value); }
    }
    ["R", "G", "B", "A"].into_iter().zip(channels).collect()
}
/// A canvas's EXR frame: its picture after its view (associated RGBA, sRGB code values), through the
/// writer a world's frame takes.
fn encode_exr_canvas(w: u32, h: u32, rgba: &[[f32; 4]], codec: crate::ExrCodec, fps: (i32, u32), color: crate::output_device::RawColor) -> Result<Files> {
    anyhow::ensure!(rgba.len() == (w * h) as usize, "EXR channel size mismatch");
    let channels = beauty_channels(rgba, color).into_iter().map(|(channel, values)|
        exr::prelude::AnyChannel::new(exr::meta::attribute::Text::from_slice_unchecked(channel.as_bytes()), exr::prelude::FlatSamples::F32(values))).collect();
    write_exr(w, h, channels, codec, Some(fps), color)
}
/// `color`: the storage space of the file's colour (beauty and colour AOVs), which the file names in its
/// chromaticities and `vork:color_space`. Alpha, depth, indices, normals, motion and value AOVs are not colour.
fn encode_exr_clock(w: u32, h: u32, p: &FrameOutput, outputs: &Outputs, codec: crate::ExrCodec, fps: Option<(i32,u32)>, color: crate::output_device::RawColor) -> Result<Files> {
    use exr::prelude::*;
    let mut all_channels = Vec::new();
    let count = (w * h) as usize;
    for output in &outputs.0 {
        // Plain OpenEXR: beauty R,G,B,A and depth Z in the default layer; every other
        // channel is <name>.<channel>.
        let (name, channels): (&str, Vec<(&str, Vec<f32>)>) = match output {
            Output::Color => {
                let pixels = p.linear_rgba.as_ref().ok_or_else(|| anyhow!("EXR channels require float readback"))?;
                ("", beauty_channels(pixels, color))
            },
            Output::Depth => (
                "",
                vec![(
                    "Z",
                    p.depth_f32
                        .as_ref()
                        .ok_or_else(|| anyhow!("EXR depth requires float readback"))?
                        .clone(),
                )],
            ),
            Output::ObjectIndex => ("object_index", vec![("X", p.ids.iter().map(|v| *v as f32).collect())]),
            Output::ViewNormal => ("view_normal", float_channels(["X", "Y", "Z"], &p.normals_f32)?),
            Output::Motion => ("motion", float_channels(["X", "Y"], &p.motion_f32)?),
            Output::Camera => continue,
            Output::Aov(name) => {
                let aov=p.aovs.get(name).ok_or_else(||anyhow!("missing AOV {name}"))?;
                let labels:&[&str]=if aov.kind==vork_scene::v2::AovType::Value {&["X"]} else {&["R","G","B"]};
                // shadow_catcher is a ratio of radiances, not a radiance: it stays in the working space.
                let stored=aov.kind==vork_scene::v2::AovType::Color && name!="shadow_catcher";
                (name.as_str(),labels.iter().enumerate().map(|(i,label)|(*label,aov.pixels.iter().map(|v|if stored {color.store([v[0],v[1],v[2]])[i]} else {v[i]}).collect())).collect())
            },
        };
        anyhow::ensure!(
            channels.iter().all(|(_, v)| v.len() == count),
            "EXR channel size mismatch"
        );
        all_channels.extend(channels.into_iter().map(|(channel, values)| {
            AnyChannel::new(
                exr::meta::attribute::Text::from_slice_unchecked(if name.is_empty() { channel.to_owned() } else { format!("{name}.{channel}") }.as_bytes()),
                FlatSamples::F32(values),
            )
        }));
    }
    write_exr(w, h, all_channels, codec, fps, color)
}
/// The one EXR file writer: the channels as one layer, with the names of the file's colour.
fn write_exr(w: u32, h: u32, all_channels: Vec<exr::prelude::AnyChannel<exr::prelude::FlatSamples>>, codec: crate::ExrCodec, fps: Option<(i32,u32)>,
    color: crate::output_device::RawColor) -> Result<Files> {
    use exr::prelude::*;
    let mut files = Vec::new();
    if !all_channels.is_empty() {
        let mut image = Image::from_layer(Layer::new(
            (w as usize, h as usize),
            LayerAttributes {frames_per_second:fps,..Default::default()},
            codec.encoding(),
            AnyChannels::sort(all_channels.into()),
        ));
        // The file says what its colour is: the storage space's primaries and white, its name, and the
        // picture's reference domain: scene (a world's raw picture bypasses the view) or display (a canvas's
        // picture after its view).
        let [red, green, blue, white] = color.storage.chromaticities().map(|[x, y]| Vec2(x, y));
        image.attributes.chromaticities = Some(exr::meta::attribute::Chromaticities { red, green, blue, white });
        image.layer_data.attributes.other.insert(Text::from("vork:color_space"), AttributeValue::Text(Text::from(color.storage.name())));
        image.layer_data.attributes.other.insert(Text::from("vork:reference_domain"), AttributeValue::Text(Text::from(color.reference_domain())));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write().non_parallel().to_buffered(&mut bytes)?;
        files.push((".exr".into(), bytes.into_inner()));
    }
    Ok(files)
}

/// One core-sized pool unpacks readbacks and compresses individual planes.
/// Ring leases bound mapped pixels, unpacked pixels, and ordered results.
pub struct FrameOutputWriter {
    jobs: mpsc::Sender<Input>,
    coordinator: thread::JoinHandle<Result<()>>,
    workers: Vec<thread::JoinHandle<()>>,
    writer: thread::JoinHandle<Result<()>>,
    pub worker_count: usize,
}
impl FrameOutputWriter {
    pub(crate) fn canvas(out: PathBuf, w: u32, h: u32, pending_frames: usize, first_frame: usize, display: vork_scene::v2::OutputDisplay) -> Self {
        Self::new(out, w, h, Outputs(vec![Output::Color]), false, crate::ExrCodec::default(),
            PngDepth::Eight, pending_frames, first_frame, (1, 1), display, Default::default())
    }
    pub(crate) fn push_canvas(&self, frame: usize, rgba: Arc<Vec<u8>>, lease: Arc<crate::ring::Lease>) -> Result<()> {
        self.jobs.send(Input::Canvas { frame, rgba, lease })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    /// A canvas's EXR destination (`x.%06d.exr`): one file per frame, its colour stored as `raw` says.
    pub(crate) fn canvas_exr(out: PathBuf, w: u32, h: u32, exr_codec: crate::ExrCodec, pending_frames: usize, first_frame: usize, fps: (i32, u32),
        raw: crate::output_device::RawColor) -> Self {
        Self::new(out, w, h, Outputs(vec![Output::Color]), true, exr_codec, PngDepth::Eight, pending_frames, first_frame, fps,
            vork_scene::v2::OutputDisplay::Srgb, raw)
    }
    /// One frame of a canvas's picture after its view: associated RGBA, sRGB code values.
    pub(crate) fn push_canvas_float(&self, frame: usize, rgba: Arc<Vec<[f32; 4]>>, lease: Arc<crate::ring::Lease>) -> Result<()> {
        self.jobs.send(Input::CanvasFloat { frame, rgba, lease })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    pub fn new(
        out: PathBuf,
        w: u32,
        h: u32,
        outputs: Outputs,
        exr: bool,
        exr_codec: crate::ExrCodec,
        png_depth: PngDepth,
        pending_frames: usize,
        // Absolute index of the first pushed frame; output order counts from it.
        first_frame: usize,
        fps: (i32,u32),
        // The display colour PNGs are encoded for, which they name.
        display: vork_scene::v2::OutputDisplay,
        // The storage space of an EXR's colour, which it names.
        raw: crate::output_device::RawColor,
    ) -> Self {
        let plane_count = if exr { 1 } else { outputs.0.len() };
        let worker_count = thread::available_parallelism()
            .map_or(1, usize::from)
            .min(pending_frames.max(1) * plane_count.max(1));

        let (requests, input) = mpsc::channel::<Input>();
        let (jobs, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let (done, results) = mpsc::channel();
        let workers = (0..worker_count)
            .map(|_| {
                let rx = rx.clone();
                let done = done.clone();
                let requests = requests.clone();
                thread::spawn(move || {
                    loop {
                        let Ok(job) = rx.lock().unwrap().recv() else {
                            break;
                        };
                        match job {
                            Job::Canvas { order, frame, rgba, lease } => {
                                let started = std::time::Instant::now();
                                let files = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode_canvas(w, h, &rgba, display)))
                                    .unwrap_or_else(|_| Err(anyhow!("frame compression worker panicked")));
                                crate::timing::encoded(started);
                                if done.send((order, frame, files, Some(lease))).is_err() { break; }
                            }
                            Job::CanvasFloat { order, frame, rgba, lease } => {
                                let started = std::time::Instant::now();
                                let files = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode_exr_canvas(w, h, &rgba, exr_codec, fps, raw)))
                                    .unwrap_or_else(|_| Err(anyhow!("frame compression worker panicked")));
                                crate::timing::encoded(started);
                                if done.send((order, frame, files, Some(lease))).is_err() { break; }
                            }
                            Job::Unpack { frame, pixels } => {
                                let lease = pixels.credit.clone();
                                let started = std::time::Instant::now();
                                let pixels =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        pixels.unpack()
                                    }))
                                    .map_err(|_| anyhow!("readback unpack worker panicked"))
                                    .and_then(|result| result).map(Arc::new);
                                crate::timing::encoded(started);
                                if requests
                                    .send(Input::Unpacked {
                                        frame,
                                        pixels,
                                        lease,
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Job::Plane {
                                order,
                                frame,
                                outputs,
                                exr,
                                pixels,
                                lease,
                            } => {
                                let started = std::time::Instant::now();
                                let files =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        if exr {
                                            encode_exr_clock(w, h, &pixels, &outputs, exr_codec,Some(fps),raw)
                                        } else {
                                            encode(w, h, &pixels, &outputs, png_depth, display)
                                        }
                                    }))
                                    .unwrap_or_else(|_| {
                                        Err(anyhow!("frame compression worker panicked"))
                                    });
                                crate::timing::encoded(started);
                                if done.send((order, frame, files, lease)).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                })
            })
            .collect();
        // Only this coordinator produces pool jobs. Explicit Finish lets it
        // close the pool after all unpack jobs have expanded into plane jobs.
        let coordinator = thread::spawn(move || {
            let mut unpacking = 0;
            let mut finished = false;
            for request in input {
                match request {
                    Input::Canvas { frame, rgba, lease } => {
                        jobs.send(Job::Canvas { order: frame - first_frame, frame, rgba, lease })?;
                    }
                    Input::CanvasFloat { frame, rgba, lease } => {
                        jobs.send(Job::CanvasFloat { order: frame - first_frame, frame, rgba, lease })?;
                    }
                    Input::Frame { frame, pixels } => {
                        unpacking += 1;
                        jobs.send(Job::Unpack { frame, pixels })?;
                    }
                    Input::Unpacked {
                        frame,
                        pixels,
                        lease,
                    } => {
                        unpacking -= 1;
                        let groups = if exr {
                            vec![outputs.clone()]
                        } else {
                            outputs.0.iter().map(|o| Outputs(vec![o.clone()])).collect()
                        };
                        for (i, outputs) in groups.into_iter().enumerate() {
                            let order = (frame - first_frame) * plane_count + i;
                            match &pixels {
                                Ok(pixels) => jobs.send(Job::Plane {
                                    order,
                                    frame,
                                    outputs,
                                    exr,
                                    pixels: pixels.clone(),
                                    lease: lease.clone(),
                                })?,
                                Err(error) => done.send((
                                    order,
                                    frame,
                                    Err(anyhow!(error.to_string())),
                                    lease.clone(),
                                ))?,
                            }
                        }
                    }
                    Input::Skip { frame } => {
                        for i in 0..plane_count {
                            done.send(((frame - first_frame) * plane_count + i, frame, Ok(Files::new()), None))?;
                        }
                    }
                    Input::Finish => finished = true,
                }
                if finished && unpacking == 0 {
                    break;
                }
            }
            Ok(())
        });
        let writer = thread::spawn(move || {
            let mut pending = BTreeMap::new();
            let mut next = 0;
            let mut error = None;
            for (order, frame, files, lease) in results {
                pending.insert(order, (frame, files, lease));
                while let Some((frame, files, _lease)) = pending.remove(&next) {
                    let started = std::time::Instant::now();
                    let written = files.and_then(|files| {
                        for (suffix, bytes) in files {
                            let path = if exr {
                                let stem = out.file_stem().unwrap().to_string_lossy();
                                let stem = stem.replace("%06d", &format!("{frame:06}"));
                                out.with_file_name(format!("{stem}{suffix}"))
                            } else if one_still(&out) {
                                // A one-still PNG destination: colour at the path, other channels beside it.
                                if suffix == ".png" {
                                    out.clone()
                                } else {
                                    let stem = out.file_stem().unwrap().to_string_lossy();
                                    out.with_file_name(format!("{stem}{suffix}"))
                                }
                            } else {
                                out.join(format!("{frame:06}{suffix}"))
                            };
                            std::fs::write(path, bytes)?;
                        }
                        Ok(())
                    });
                    crate::timing::encoded(started);
                    if let Err(e) = written {
                        if error.is_none() {
                            error = Some(e);
                        }
                    }
                    next += 1;
                }
            }
            if !pending.is_empty() {
                return Err(anyhow!("frame output worker left a gap in output order"));
            }
            error.map_or(Ok(()), Err)
        });
        Self {
            jobs: requests,
            coordinator,
            workers,
            writer,
            worker_count,
        }
    }
    pub fn push(&self, frame: usize, pixels: FrameReadback) -> Result<()> {
        self.jobs
            .send(Input::Frame { frame, pixels })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    /// Write nothing for a refused frame, keeping later frames in order.
    pub fn skip(&self, frame: usize) -> Result<()> {
        self.jobs
            .send(Input::Skip { frame })
            .map_err(|_| anyhow!("frame output workers stopped"))
    }
    pub fn finish(self) -> Result<()> {
        let _ = self.jobs.send(Input::Finish);
        drop(self.jobs);
        let coordinated = self
            .coordinator
            .join()
            .map_err(|_| anyhow!("frame output coordinator panicked"));
        let mut panicked = false;
        for worker in self.workers {
            panicked |= worker.join().is_err();
        }
        let result = self
            .writer
            .join()
            .map_err(|_| anyhow!("frame output writer panicked"))?;
        coordinated??;
        if panicked {
            return Err(anyhow!("frame output worker panicked"));
        }
        result
    }
}

#[cfg(test)]
mod exr_tests {
    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn rendered_png16_preserves_precision_and_scales_dither() -> Result<()> {
        let r = crate::test_renderer()?;
        assert!(!crate::InstallOptions::default().readback.color16);
        let readback = ReadbackOutputs { color: true, color16: true, ..Default::default() };
        let scene = serde_json::from_value(serde_json::json!({
            "version": 2, "nodes": [{"type": "camera"}],
            "composition":{"space":"3d","view":{"program": "display"}},"render": {"resolution": [16,16], "background": [0.25,0.25,0.25],
                "settings": {"samples": 1}},
            "programs": {"display": {"source": "import std.view; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(ctx.color,ctx,1.);}"}}
        }))?;
        let installed = r.install_with_options(scene, std::path::Path::new("."), crate::InstallOptions {
            readback, gpu_color: false, ..Default::default()
        })?;
        let mut frame = installed.create_frame_slot(&r, crate::ReadbackPrecision::Packed)?;
        frame.submit_outputs(&installed, &r, 0., readback)?;
        let pixels = frame.collect(&r)?;
        assert_eq!(pixels.rgba16.len(), 256);
        for (low, high) in pixels.rgba.chunks_exact(4).zip(&pixels.rgba16) {
            for c in 0..4 {
                assert!((f32::from(low[c])/255. - f32::from(high[c])/65535.).abs() <= 1./255.);
            }
            // A constant 0.25 input must not retain 8-bit dither in RGBA16.
            assert!((f32::from(high[0])/65535. - 0.25).abs() <= 1.5/65535.);
        }
        let files = encode(16,16,&pixels,&Outputs::new(vec![Output::Color]),PngDepth::Sixteen,Default::default())?;
        let path = std::env::temp_dir().join(format!("vork-png16-{}.png",std::process::id()));
        std::fs::write(&path,&files[0].1)?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info()?;
        let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut decoded)?;
        assert_eq!(info.bit_depth,png::BitDepth::Sixteen);
        let expected: Vec<u8> = pixels.rgba16.iter().flatten().flat_map(|v|v.to_be_bytes()).collect();
        assert_eq!(&decoded[..info.buffer_size()],expected.as_slice());
        Ok(())
    }
    #[test]
    fn every_png_uses_the_fast_setting_and_decodes_to_its_pixels() -> Result<()> {
        assert!(matches!(COMPRESSION, png::Compression::Fast));
        let rgba: Vec<u8> = (0..64 * 64 * 4).map(|i| (i * 7 % 251) as u8).collect();
        let mut reader = png::Decoder::new(std::io::Cursor::new(encode_canvas(64, 64, &rgba, Default::default())?.remove(0).1)).read_info()?;
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels)?;
        assert_eq!(&pixels[..info.buffer_size()], rgba.as_slice());
        Ok(())
    }
    #[test]
    fn aov_channels_cannot_shadow_built_in_layers() {
        assert!("aov:heat".parse::<Output>().is_ok());
        assert!("aov:color".parse::<Output>().unwrap_err().contains("collides"));
        assert!("aov:object_index".parse::<Output>().unwrap_err().contains("collides"));
        assert_eq!(Output::builtins().count(), vork_scene::v2::OUTPUT_CHANNELS.len());
        assert!("ZIP".parse::<crate::ExrCodec>().is_err());
        assert_eq!(("8".parse(), "16".parse()), (Ok(PngDepth::Eight), Ok(PngDepth::Sixteen)));
        assert!("32".parse::<PngDepth>().is_err());
    }
    #[test]
    fn channel_grammar_is_namespaced_and_builtin_defaults_are_stable() {
        // Unimplemented selections are unknown channels, not reserved promises.
        for token in ["aov:", "aov:a:b", "aov:a.b", "aov:../file", "unknown:x", "cryptomatte_object"] {
            assert!(token.parse::<Output>().is_err(), "{token}");
        }
        for token in ["cryptomatte:object", "cryptomatte:material", "cryptomatte:asset", "lightgroup:key"] {
            assert!(token.parse::<Output>().unwrap_err().starts_with("unknown output channel"), "{token}");
        }
        let defaults: Vec<Output> = Output::builtins().collect();
        assert_eq!(defaults, vec![Output::Color, Output::Depth, Output::ObjectIndex, Output::ViewNormal, Output::Motion, Output::Camera, Output::Aov("shadow_catcher".into())]);
    }
    #[test]
    fn multilayer_and_png_control_units_round_trip() -> Result<()> {
        use exr::prelude::*;
        let mut pixels = FrameOutput {
            aovs: [("heat".into(), crate::frame::AovPixels {
                kind: vork_scene::v2::AovType::Value,
                pixels: vec![[0.25, 0., 0., 0.], [0.75, 0., 0., 0.]],
            })].into(),
            rgba: vec![128, 128, 255, 128, 0, 0, 0, 0],
            rgba16: vec![[0x0102, 0x8000, 0xffff, 0x7fff], [0, 0, 0, 0]],
            linear_rgba: Some(vec![[0.09, 0.2158605 * 0.5, 2., 0.5], [0., 0., 0., 0.]]),
            depth_f32: Some(vec![1.2345679, 1e10]),
            normals_f32: Some(vec![[0.1234567, 0.2345678, 0.964], [0., 0., 0.]]),
            motion_f32: Some(vec![[0., 0.], [0., 0.]]),
            depth_u16: vec![100, 0],
            ids: vec![65535, 0],
            normals: vec![255, 128, 128, 0, 0, 0],
            motion: vec![[32768, 32768], [32768, 32768]],
            camera: serde_json::json!({}),
            exposure: None,
            timing: crate::FrameTiming {
                cpu_prepare_ms: 0.,
                cpu_submit_ms: 0.,
                cpu_finish_ms: 0.,
                denoise_wait_ms: 0.,
                denoise_pack_ms: 0.,
                denoise_filter_ms: 0.,
                denoise_upload_ms: 0.,
                gpu_ms: 0.,
                nonfinite: None,
                dropped_light_samples: 0,
                dropped_path_samples: 0,
                invalid_pixels: Default::default(),
                terminated_medium_paths: 0,
                discarded_medium_radiance: 0.,
                gpu_elapsed_ms: 0.,
                timing_source: "unmeasured",
                gpu_spans_ms: vec![],
                sample_count: 1,
                variants_pending: false,
                path_kernel: None,
            },
        };
        // Raw storage converts colour only: beauty and colour AOVs. Alpha, indices, value AOVs and the catcher
        // ratio keep their values, and the file names its primaries, white point, space and reference domain.
        // The second beauty pixel is light without coverage (a bloom halo on transparent film): it keeps its energy.
        {
            use vork_scene::v2::{AovType, WorkingColorSpace as S};
            let world: vork_scene::v2::Scene = serde_json::from_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}]}))?;
            let mut raw = pixels.clone();
            raw.linear_rgba = Some(vec![[2., -0.25, 0.5, 0.5], [0.5, 0.25, 0.125, 0.]]);
            let colour = [[2., -0.25, 0.5, 1.], [0.5, 0.25, 0.125, 0.]];
            for name in ["tint", "shadow_catcher"] {
                raw.aovs.insert(name.into(), crate::frame::AovPixels { kind: AovType::Color, pixels: colour.to_vec() });
            }
            let outputs = Outputs(vec![Output::Color, Output::ObjectIndex, Output::Aov("heat".into()), Output::Aov("tint".into()), Output::Aov("shadow_catcher".into())]);
            let mut default_bytes = None;
            for (storage, expected) in [
                (None, [2., -0.25, 0.5]),
                (Some(S::LinearRec709), [2., -0.25, 0.5]),
                (Some(S::LinearRec2020), [1.19414353, -0.08600947, 0.45857722]),
                (Some(S::ACEScg), [1.16500378, -0.08197488, 0.44874606]),
            ] {
                let destination = vork_scene::v2::OutputDestination { color_space: storage, .."raw.%06d.exr".into() };
                let color = crate::output_device::RawColor::of(&world, &destination);
                let storage = storage.unwrap_or(S::LinearRec709);
                assert_eq!(color.storage, storage);
                let encoded = encode_exr_clock(1, 2, &raw, &outputs, crate::ExrCodec::default(), Some((24, 1)), color)?;
                let decoded = read().no_deep_data().largest_resolution_level().all_channels().all_layers().all_attributes()
                    .from_buffered(std::io::Cursor::new(&encoded[0].1))?;
                let layer = &decoded.layer_data[0];
                let channel = |name: &str| layer.channel_data.list.iter().find(|c| c.name.to_string() == name).unwrap().sample_data.values_as_f32().collect::<Vec<_>>();
                let halo = color.store([0.5, 0.25, 0.125]);
                for (i, name) in ["R", "G", "B"].into_iter().enumerate() {
                    for prefix in ["", "tint."] {
                        let values = channel(&format!("{prefix}{name}"));
                        assert!((values[0] - expected[i]).abs() < 2e-6 && values[1] == halo[i], "{storage:?} {prefix}{name}: {values:?}");
                    }
                    assert_eq!(channel(&format!("shadow_catcher.{name}")), vec![colour[0][i], colour[1][i]], "the catcher ratio is not converted");
                }
                assert_eq!(channel("A"), vec![0.5, 0.]);
                assert_eq!(channel("object_index.X"), vec![65535., 0.]);
                assert_eq!(channel("heat.X"), vec![0.25, 0.75]);
                let [red, green, blue, white] = storage.chromaticities().map(|[x, y]| Vec2(x, y));
                assert_eq!(decoded.attributes.chromaticities, Some(exr::meta::attribute::Chromaticities { red, green, blue, white }));
                assert_eq!(layer.attributes.other[&Text::from("vork:color_space")], AttributeValue::Text(Text::from(storage.name())));
                assert_eq!(layer.attributes.other[&Text::from("vork:reference_domain")], AttributeValue::Text(Text::from("scene")));
                assert_eq!(layer.attributes.frames_per_second, Some((24, 1)));
                // The working space stores the samples as they are, named or omitted: the same file.
                if storage == S::LinearRec709 {
                    assert_eq!([channel("R")[0], channel("G")[0], channel("B")[0]].map(f32::to_bits), [2f32, -0.25, 0.5].map(f32::to_bits));
                    assert_eq!(default_bytes.get_or_insert_with(|| encoded[0].1.clone()), &encoded[0].1);
                }
            }
        }
        // A canvas's EXR: its picture after its view, associated sRGB code values, through the
        // same writer. It is display-referred; omitted storage keeps the code values as they are, and a linear
        // space takes display-linear light, converted on unassociated colour.
        {
            use vork_scene::v2::WorkingColorSpace as S;
            let canvas: vork_scene::v2::Scene = serde_json::from_value(serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[]}))?;
            let picture = [[0.25, 0.25, 0.25, 0.5], [1.5, -0.25, 0.5, 1.]];
            for (storage, expected) in [(None, picture), (Some(S::Srgb), picture),
                (Some(S::LinearRec709), [[0.21404114 * 0.5, 0.21404114 * 0.5, 0.21404114 * 0.5, 0.5], [2.5371552, -0.019349845, 0.21404114, 1.]])] {
                let destination = vork_scene::v2::OutputDestination { color_space: storage, .."frames.%06d.exr".into() };
                let color = crate::output_device::RawColor::of(&canvas, &destination);
                let encoded = encode_exr_canvas(1, 2, &picture, crate::ExrCodec::default(), (24, 1), color)?;
                let decoded = read().no_deep_data().largest_resolution_level().all_channels().all_layers().all_attributes()
                    .from_buffered(std::io::Cursor::new(&encoded[0].1))?;
                let layer = &decoded.layer_data[0];
                let channel = |name: &str| layer.channel_data.list.iter().find(|c| c.name.to_string() == name).unwrap().sample_data.values_as_f32().collect::<Vec<_>>();
                for (i, name) in ["R", "G", "B", "A"].into_iter().enumerate() {
                    let values = channel(name);
                    for pixel in 0..2 {
                        if storage == Some(S::LinearRec709) { assert!((values[pixel] - expected[pixel][i]).abs() < 2e-5, "{name}: {values:?}"); }
                        else { assert_eq!(values[pixel].to_bits(), expected[pixel][i].to_bits(), "{name}: {values:?}"); }
                    }
                }
                let storage = storage.unwrap_or(S::Srgb);
                let [red, green, blue, white] = storage.chromaticities().map(|[x, y]| Vec2(x, y));
                assert_eq!(decoded.attributes.chromaticities, Some(exr::meta::attribute::Chromaticities { red, green, blue, white }));
                assert_eq!(layer.attributes.other[&Text::from("vork:color_space")], AttributeValue::Text(Text::from(storage.name())));
                assert_eq!(layer.attributes.other[&Text::from("vork:reference_domain")], AttributeValue::Text(Text::from("display")));
                assert_eq!(layer.attributes.frames_per_second, Some((24, 1)));
            }
        }
        // The public PNG protocol uses network byte order, including the zero
        // blue component of the RGB16 motion image. Decode the actual files so
        // this also checks their PNG type, depth and dimensions.
        let pngs = encode(1, 2, &pixels, &Outputs(vec![Output::Depth, Output::ObjectIndex, Output::Motion]), PngDepth::Eight, Default::default())?;
        for (suffix, color, expected) in [
            (".depth.png", png::ColorType::Grayscale, vec![0, 100, 0, 0]),
            (".object_index.png", png::ColorType::Grayscale, vec![255, 255, 0, 0]),
            (".motion.png", png::ColorType::Rgb, vec![128, 0, 128, 0, 0, 0, 128, 0, 128, 0, 0, 0]),
        ] {
            let bytes = &pngs.iter().find(|(name, _)| name == suffix).unwrap().1;
            let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info()?;
            assert!(reader.info().srgb.is_none(), "control channels are data");
            let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut decoded)?;
            assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (1, 2, color, png::BitDepth::Sixteen));
            assert_eq!(&decoded[..info.buffer_size()], &expected, "{suffix}");
        }
        // Colour follows the selected depth; 16-bit components are big-endian RGBA16.
        for (depth, bit_depth, expected) in [
            (PngDepth::Eight, png::BitDepth::Eight, vec![128, 128, 255, 128, 0, 0, 0, 0]),
            (PngDepth::Sixteen, png::BitDepth::Sixteen, vec![1, 2, 128, 0, 255, 255, 127, 255, 0, 0, 0, 0, 0, 0, 0, 0]),
        ] {
            let files = encode(1, 2, &pixels, &Outputs(vec![Output::Color]), depth, Default::default())?;
            assert_eq!(files.len(), 1);
            assert_eq!(files[0].0, ".png");
            let mut reader = png::Decoder::new(std::io::Cursor::new(&files[0].1)).read_info()?;
            assert_eq!(reader.info().srgb,Some(png::SrgbRenderingIntent::Perceptual));
            let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut decoded)?;
            assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (1, 2, png::ColorType::Rgba, bit_depth));
            assert_eq!(&decoded[..info.buffer_size()], &expected, "{depth:?}");
        }
        // A picture for another display names it in cICP and carries no sRGB chunk; its pixels are the same
        // plane, and control channels beside it stay untagged data.
        let chunks = |bytes: &[u8]| {
            let (mut chunks, mut offset) = (Vec::new(), 8);
            while offset < bytes.len() {
                let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                chunks.push((bytes[offset + 4..offset + 8].to_vec(), bytes[offset + 8..offset + 8 + length].to_vec()));
                offset += 12 + length;
            }
            chunks
        };
        let srgb = encode(1, 2, &pixels, &Outputs(vec![Output::Color]), PngDepth::Eight, Default::default())?;
        assert!(chunks(&srgb[0].1).iter().all(|(name, _)| name != b"cICP"));
        let wide = encode(1, 2, &pixels, &Outputs(vec![Output::Color, Output::Depth]), PngDepth::Eight, vork_scene::v2::OutputDisplay::DisplayP3)?;
        let tagged = chunks(&wide[0].1);
        assert_eq!(tagged.iter().find(|(name, _)| name == b"cICP").map(|(_, data)| data.as_slice()), Some(&[12, 13, 0, 1][..]));
        assert!(tagged.iter().all(|(name, _)| !matches!(name.as_slice(), b"sRGB" | b"gAMA" | b"cHRM" | b"iCCP")));
        let position = |name: &[u8]| tagged.iter().position(|(chunk, _)| chunk == name).unwrap();
        assert!(position(b"cICP") < position(b"IDAT"));
        assert!(chunks(&wide[1].1).iter().all(|(name, _)| !matches!(name.as_slice(), b"cICP" | b"sRGB")));
        assert_eq!(png::Decoder::new(std::io::Cursor::new(&wide[0].1)).read_info()?.info().srgb, None);
        for (display, expected) in [(vork_scene::v2::OutputDisplay::Rec2100Pq, [9, 16, 0, 1]), (vork_scene::v2::OutputDisplay::Rec2100Hlg, [9, 18, 0, 1])] {
            assert_eq!(cicp(display), Some(expected));
        }
        // A canvas's picture names its display the same way.
        let canvas = encode_canvas(1, 1, &[1, 2, 3, 255], vork_scene::v2::OutputDisplay::DisplayP3)?;
        let tagged = chunks(&canvas[0].1);
        assert_eq!(tagged.iter().find(|(name, _)| name == b"cICP").map(|(_, data)| data.as_slice()), Some(&[12, 13, 0, 1][..]));
        assert!(tagged.iter().all(|(name, _)| name != b"sRGB"));
        assert!(chunks(&encode_canvas(1, 1, &[1, 2, 3, 255], Default::default())?[0].1).iter().any(|(name, _)| name == b"sRGB"));
        let files = encode_exr(
            1,
            2,
            &pixels,
            &Outputs(vec![
                Output::Color,
                Output::Depth,
                Output::ObjectIndex,
                Output::ViewNormal,
                Output::Motion,
                Output::Camera,
                Output::Aov("heat".into()),
            ]),
            crate::ExrCodec::default(),
        )?;
        assert_eq!(files.len(), 1);
        let image = read()
            .no_deep_data()
            .largest_resolution_level()
            .all_channels()
            .all_layers()
            .all_attributes()
            .from_buffered(std::io::Cursor::new(&files[0].1))?;
        assert_eq!(image.layer_data[0].encoding.compression, Compression::ZIP16);
        assert_eq!(image.layer_data.len(), 1);
        assert_eq!(image.layer_data[0].channel_data.list.len(), 12);
        let sample = |channel: &str| {
            image.layer_data[0]
                .channel_data
                .list
                .iter()
                .find(|c| c.name.to_string() == channel)
                .unwrap()
                .sample_data
                .values_as_f32()
                .collect::<Vec<_>>()
        };
        assert_eq!(sample("Z"), [1.2345679, 1e10]);
        assert_eq!(sample("object_index.X"), [65535., 0.]);
        assert_eq!(sample("heat.X"), [0.25, 0.75]);
        assert_eq!(sample("R"), [0.09, 0.]);
        assert_eq!(sample("G"), [0.2158605 * 0.5, 0.]);
        assert_eq!(sample("B"), [2., 0.]);
        assert_eq!(sample("A"), [0.5, 0.]);
        assert_eq!(sample("view_normal.X"), [0.1234567, 0.]);
        assert_eq!(sample("motion.X"), [0., 0.]);
        assert_eq!(sample("motion.Y"), [0., 0.]);
        // Beauty and depth sit in the default layer; every other layer is a
        // built-in channel name, reserved against AOV names at scene validation.
        for channel in &image.layer_data[0].channel_data.list {
            let name = channel.name.to_string();
            match name.split_once('.') {
                None => assert!(["R", "G", "B", "A", "Z"].contains(&name.as_str()), "{name}"),
                Some((layer, _)) => assert!(layer == "heat" || vork_scene::v2::OUTPUT_CHANNELS.contains(&layer), "{name}"),
            }
        }
        for name in ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"] {
            let codec: crate::ExrCodec = name.parse().unwrap();
            let files = encode_exr(1, 2, &pixels, &Outputs(vec![Output::Color, Output::Depth, Output::ObjectIndex]), codec)?;
            let image = read().no_deep_data().largest_resolution_level().all_channels()
                .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&files[0].1))?;
            assert_eq!(std::mem::discriminant(&image.layer_data[0].encoding.compression),
                std::mem::discriminant(&codec.encoding().compression), "{name}");
            let ids = image.layer_data[0].channel_data.list.iter()
                .find(|c| c.name.to_string() == "object_index.X").unwrap()
                .sample_data.values_as_f32().collect::<Vec<_>>();
            assert_eq!(ids, [65535., 0.], "{name} must retain exact object IDs");
        }
        // Declared lists beyond the former shader budget keep their names and
        // values through EXR and explicit --channels selections.
        for count in [0, 1, 3, 8, 32] {
            pixels.aovs = (0..count).map(|i| (format!("a{i}"), crate::frame::AovPixels {
                kind: if i % 2 == 0 {vork_scene::v2::AovType::Value} else {vork_scene::v2::AovType::Color},
                pixels: vec![[(i+1) as f32 / 64., 0.5, 0.75, 1.]; 2],
            })).collect();
            let outputs = Outputs::new(std::iter::once(Output::Color).chain((0..count).map(|i| Output::Aov(format!("a{i}")))).collect());
            let files = encode_exr(1, 2, &pixels, &outputs, crate::ExrCodec::default())?;
            let image = read().no_deep_data().largest_resolution_level().all_channels()
                .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&files[0].1))?;
            let channels = &image.layer_data[0].channel_data.list;
            assert_eq!(channels.len(), 4 + (0..count).map(|i| if i%2==0 {1} else {3}).sum::<usize>());
            for i in 0..count {
                for (component, expected) in if i%2==0 {vec![("X", (i+1) as f32/64.)]} else {vec![("R", (i+1) as f32/64.), ("G", 0.5), ("B", 0.75)]} {
                    let name = format!("a{i}.{component}");
                    let channel = channels.iter().find(|c| c.name.to_string()==name).unwrap();
                    assert_eq!(channel.sample_data.values_as_f32().collect::<Vec<_>>(), vec![expected;2]);
                }
            }
            if count>0 {
                let selection: Output = format!("aov:a{}",count-1).parse().unwrap();
                let selected = Outputs::new(vec![selection]);
                let png = encode(1,2,&pixels,&selected,PngDepth::Eight,Default::default())?;
                assert_eq!(png.len(),1);
                assert_eq!(png[0].0,format!(".aov.a{}.png",count-1));
                let files = encode_exr(1,2,&pixels,&selected,crate::ExrCodec::default())?;
                let image = read().no_deep_data().largest_resolution_level().all_channels()
                    .all_layers().all_attributes().from_buffered(std::io::Cursor::new(&files[0].1))?;
                assert!(image.layer_data[0].channel_data.list.iter().all(|c| c.name.to_string().starts_with(&format!("a{}.",count-1))));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod canvas_writer_tests {
    use super::*;

    fn read(path: &std::path::Path, expected: &[u8]) {
        let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()))
            .read_info().unwrap();
        assert_eq!(decoder.info().srgb, Some(png::SrgbRenderingIntent::Perceptual));
        let mut pixels = vec![0; decoder.output_buffer_size().unwrap()];
        let info = decoder.next_frame(&mut pixels).unwrap();
        assert_eq!(info.color_type, png::ColorType::Rgba);
        assert_eq!(&pixels[..info.buffer_size()], expected);
    }

    #[test]
    fn canvas_writer_preserves_pixels_indices_and_shared_credits() {
        let root = std::env::temp_dir().join(format!("vork-canvas-writer-{}", std::process::id()));
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::create_dir_all(root.join("b")).unwrap();
        let writers: Vec<_> = ["a", "b"].into_iter().map(|p| FrameOutputWriter::canvas(root.join(p), 2, 1, 2, 7, Default::default())).collect();
        let credits: [Arc<crate::ring::Credit>; 2] = Default::default();
        let pixels = |frame: usize| Arc::new(vec![frame as u8, 34, 56, 78, 90, 123, 210, 255]);
        // More frames than credits exercises backpressure and Finish accounting for ready inputs.
        for frame in 7..13 {
            let lease = Arc::new(credits[frame % 2].acquire());
            let rgba = pixels(frame);
            for writer in &writers { writer.push_canvas(frame, rgba.clone(), lease.clone()).unwrap(); }
        }
        for writer in writers { writer.finish().unwrap(); }
        for destination in ["a", "b"] {
            for frame in 7..13 { read(&root.join(destination).join(format!("{frame:06}.png")), &pixels(frame)); }
        }
        let still = FrameOutputWriter::canvas(root.join("still.png"), 2, 1, 1, 42, Default::default());
        still.push_canvas(42, pixels(42), Arc::new(credits[0].acquire())).unwrap();
        still.finish().unwrap();
        read(&root.join("still.png"), &pixels(42));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn canvas_writer_failure_releases_shared_credit() {
        let root = std::env::temp_dir().join(format!("vork-canvas-writer-failure-{}", std::process::id()));
        std::fs::create_dir_all(root.join("blocked.png")).unwrap();
        let credit: Arc<crate::ring::Credit> = Default::default();
        let writer = FrameOutputWriter::canvas(root.join("blocked.png"), 1, 1, 1, 0, Default::default());
        writer.push_canvas(0, Arc::new(vec![1, 2, 3, 255]), Arc::new(credit.acquire())).unwrap();
        assert!(writer.finish().is_err());
        assert_eq!(Arc::strong_count(&credit), 1, "all writer leases were released");
        let writer = FrameOutputWriter::canvas(root.join("invalid.png"), 2, 2, 1, 0, Default::default());
        writer.push_canvas(0, Arc::new(vec![1]), Arc::new(credit.acquire())).unwrap();
        assert!(writer.finish().is_err());
        assert_eq!(Arc::strong_count(&credit), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
