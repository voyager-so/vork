//! Ordered image graphs. Each entry can retain its result for later image inputs.
use super::*;
use vork_slang::{Compiler, CompositeProgram};
mod images;
mod canvas;
pub(crate) use canvas::CanvasGraph;
#[derive(Clone, Copy)]
pub(crate) enum Normalization {
    None,
    Frame,
}
impl Normalization {
    fn first(self, first: bool) -> Self {
        if first { self } else { Self::None }
    }
}
pub(crate) struct Program {
    compiled: CompositeProgram,
    words: wgpu::Buffer,
    input_words: Vec<u32>,
    animation: Option<programs::InputAnimation>,
    channels: Vec<String>,
    slots: std::collections::BTreeMap<String, u32>,
}
/// Scene producers reached by a requested final image or control plane.
#[derive(Clone, Copy, Default)]
pub(crate) struct Dependencies {
    pub color: bool,
    pub beauty: bool,
    pub depth: bool,
    pub ids: bool,
    pub normals: bool,
    pub motion: bool,
    pub aovs: bool,
    pub aov_output: bool,
    pub catcher: bool,
    pub images: bool,
}
impl Dependencies {
    pub fn controls(self) -> bool {self.depth || self.ids || self.normals || self.motion}
    pub fn resources(self) -> Vec<&'static str> {
        let mut reads=Vec::new();
        for (enabled,name) in [(self.beauty,"linear sample accumulation"),(self.depth,"resolved depth"),(self.ids,"resolved IDs"),(self.normals,"resolved normals"),(self.motion,"resolved motion"),(self.aovs,"AOV resolved"),(self.catcher,"catcher radiance sums"),(self.images,"render images")] {
            if enabled {reads.push(name);}
        }
        reads
    }
}
impl Program {
    pub(crate) fn reads_input(&self,name:&str)->bool {self.slots.get(name).is_some_and(|&slot|self.reads(slot))}
    fn reads(&self, slot:u32) -> bool {self.compiled.image_reads.as_ref().is_none_or(|reads|reads.contains(&slot))}
}
/// Walk the ordered image DAG backwards, following previous and named edges.
/// A replacement root kills its previous image, including unread earlier entries.
pub(crate) fn dependencies(programs:&std::collections::HashMap<String,Program>, names:&[vork_scene::v2::CompositeEntry]) -> (Dependencies,Vec<bool>) {
    let mut result=Dependencies {color:true,..Default::default()};
    let mut live=vec![false;names.len()];
    let mut previous=true;
    let mut nodes=std::collections::HashSet::new();
    for (i,entry) in names.iter().enumerate().rev() {
        if !previous && !entry.name().is_some_and(|n|nodes.contains(n)) {continue;}
        live[i]=true;
        let program=&programs[entry.material()];
        previous=program.reads(0);
        for (i,channel) in program.channels.iter().enumerate() {
            if !program.reads(i as u32+1) {continue;}
            match channel.as_str() {
                "color"=>result.beauty=true,
                "depth"=>result.depth=true,
                "object_index"=>result.ids=true,
                "view_normal"=>result.normals=true,
                "motion"=>result.motion=true,
                "shadow_catcher"=>result.catcher=true,
                c if c.starts_with("aov:")=>result.aovs=true,
                c if c.starts_with("step:")=>{nodes.insert(c[5..].to_owned());},
                c if c.starts_with("image:")=>result.images=true,
                _=>(),
            }
        }
    }
    result.beauty |= previous;
    (result,live)
}

pub(crate) fn image_reads(programs:&std::collections::HashMap<String,Program>, names:&[vork_scene::v2::CompositeEntry]) -> std::collections::BTreeSet<String> {
    let (_,live)=dependencies(programs,names);
    names.iter().zip(live).filter(|(_,live)|*live).flat_map(|(entry,_)| {
        let program=&programs[entry.material()];
        program.channels.iter().enumerate().filter(|(i,_)|program.reads(*i as u32+1))
            .filter_map(|(_,name)|name.strip_prefix("image:").map(str::to_owned))
    }).collect()
}
/// Close the cross-view DAG as well as each view's compositor DAG. Unread
/// render-image bindings never install captures or reserve their scene work.
pub(crate) fn image_plan(scene:&Scene,programs:&std::collections::HashMap<String,Program>,mut main:Dependencies) -> Result<(Dependencies,Vec<String>)> {
    fn reads(scene:&Scene,programs:&std::collections::HashMap<String,Program>,names:&[vork_scene::v2::CompositeEntry],root:Option<&str>,d:Dependencies)->std::collections::BTreeSet<String> {
        let mut reads=if d.color {image_reads(programs,names)} else {Default::default()};
        // Transport can read off-subtree surfaces through seed publications.
        // Keep their bound captures whenever a view consumes scene radiance.
        if d.beauty || d.controls() || d.aovs || d.catcher {
            reads.extend(scene.image_reads(if d.beauty || d.aovs || d.catcher {None} else {root}));
        }
        reads
    }
    let mut needed=reads(scene,programs,scene.compositor(),scene.render.root.as_deref(),main);
    let mut visited=std::collections::BTreeSet::new();
    while let Some(name)=needed.iter().find(|name|!visited.contains(*name)).cloned() {
        visited.insert(name.clone());
        let Some((image,render))=scene.images.get(&name).and_then(|i|i.render.as_ref().map(|render|(i,render))) else {continue;};
        let (mut d,_)=dependencies(programs,image.compositor());
        let input_reads=reads(scene,programs,image.compositor(),render.root.as_deref(),d);
        d.images=input_reads.iter().any(|name|scene.images.get(name).is_some_and(|i|i.render.is_some()));
        needed.extend(input_reads);
        main.color |= d.color;main.beauty |= d.beauty;main.depth |= d.depth;main.ids |= d.ids;
        main.normals |= d.normals;main.motion |= d.motion;main.aovs |= d.aovs;main.catcher |= d.catcher;
    }
    let order:Vec<_>=scene.image_order(None)?.into_iter().filter(|name|needed.contains(name)).collect();
    main.images=!order.is_empty();
    Ok((main,order))
}

/// Animated words belong to a recorded view/ring slot; the common change tracker
/// copies each sample immediately before the work that consumes it.
pub(crate) struct FrameInputs {
    buffers: std::collections::HashMap<String, wgpu::Buffer>,
}
impl FrameInputs {
    pub(crate) fn new(programs: &std::collections::HashMap<String, Program>, r: &Renderer, names: &[vork_scene::v2::CompositeEntry]) -> Self {
        let buffers = names.iter().filter_map(|name| {
            let program = &programs[name.material()];
            program.animation.as_ref()?;
            Some((name.material().clone(), init(&r.device, "animated compositor inputs",
                bytemuck::cast_slice(&program.input_words),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)))
        }).collect();
        Self { buffers }
    }
    /// A world's inputs at a sample's time.
    pub(crate) fn upload(&self, s: &InstalledScene, time: f32, changes: &mut crate::values::ChangeTracker) -> Result<()> {
        self.publish(&s.compositor, changes, |animation, words| { animation.words(time, words); Ok(()) })
    }
    /// Each animated program's words as `sample` evaluates them, published through the change tracker: a
    /// world's at a sample's time, a canvas's at its frame's exact instant.
    fn publish(&self, programs: &std::collections::HashMap<String, Program>, changes: &mut crate::values::ChangeTracker,
        mut sample: impl FnMut(&programs::InputAnimation, &mut [u32]) -> Result<()>) -> Result<()> {
        for (name, buffer) in &self.buffers {
            let program = &programs[name];
            let mut words = program.input_words.clone();
            sample(program.animation.as_ref().expect("an animated program"), &mut words)?;
            changes.upload(buffer, 0, bytemuck::cast_slice(&words));
        }
        Ok(())
    }
}
/// Enable producers independently of the user's file/readback channels.
pub(crate) fn channel_flags(s: &InstalledScene, names: &[vork_scene::v2::CompositeEntry]) -> u32 {
    use vork_slang::registry::{COMPOSITOR_IDS, COMPOSITOR_MOTION};
    let (_,live)=dependencies(&s.compositor,names);
    names.iter().zip(live).filter(|(_,live)|*live)
        .flat_map(|(name,_)| {
            let program=&s.compositor[name.material()];
            program.channels.iter().enumerate().filter(|(i,_)|program.reads(*i as u32+1)).map(|(_,channel)|channel)
        })
        .fold(0, |flags, channel| {
            flags
                | match channel.as_str() {
                    "object_index" => COMPOSITOR_IDS,
                    "motion" => COMPOSITOR_MOTION,
                    _ => 0,
                }
        })
}
/// A graph's entries as (authored path, material) instances.
fn instances<'a>(instances: &mut Vec<(String, &'a String)>, path: String, chain: &'a [vork_scene::v2::CompositeEntry]) {
    for (index, name) in chain.iter().enumerate() {
        instances.push((format!("{path}[{index}]"), name.material()));
    }
}
/// The programs of a world's graphs: its own and each rendered image's.
pub(crate) fn install(
    r: &Renderer,
    scene: &Scene,
    base: &Path,
    opened: &crate::footage::Opened,
) -> Result<std::collections::HashMap<String, Program>> {
    let mut all = Vec::new();
    // Authored paths precede resolved chains so automatic lens/motion entries
    // cannot shift the index reported for an authored per-step instance.
    instances(&mut all, "composition.compositor".into(), scene.compositor());
    for (name, image) in &scene.images {
        if image.render.is_some() {
            instances(&mut all, format!("images.{name}.composition.compositor"), image.compositor());
        }
    }
    install_instances(r, scene, base, opened, all)
}
/// The programs of a composition's own graph alone: a canvas's, which renders no images.
fn install_own(
    r: &Renderer,
    scene: &Scene,
    base: &Path,
    opened: &crate::footage::Opened,
) -> Result<std::collections::HashMap<String, Program>> {
    let mut own = Vec::new();
    instances(&mut own, "composition.compositor".into(), scene.compositor());
    install_instances(r, scene, base, opened, own)
}
fn install_instances(
    r: &Renderer,
    scene: &Scene,
    base: &Path,
    opened: &crate::footage::Opened,
    instances: Vec<(String, &String)>,
) -> Result<std::collections::HashMap<String, Program>> {
    let mut result = std::collections::HashMap::new();
    for (path, name) in instances {
        if result.contains_key(name) {
            continue;
        }
        let material = scene
            .materials
            .get(name)
            .context("unknown compositor material")?;
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
                    .context("unknown compositor program")?,
            )?
        };
        let mut inputs = std::collections::BTreeMap::new();
        inputs.extend(
            material
                .inputs
                .iter()
                .map(|(k, v)| Ok((k.clone(), serde_json::to_value(v)?)))
                .collect::<Result<std::collections::BTreeMap<_, _>>>()?,
        );
        let abi = programs::abi();
        let mut hash = blake3::Hasher::new();
        // Explicit beauty bindings have their own slot; zero is implicit previous.
        hash.update(b"composite-graph-4");
        // Compilation needs slot aliases, not the names bound to those slots.
        // Rebinding the same program to another graph image reuses its code.
        let mut resources = std::collections::BTreeMap::new();
        let mut aliases = std::collections::BTreeMap::new();
        for (name, value) in inputs.iter().filter(|(_, value)| value.is_object()) {
            use vork_scene::v2::Input;
            let identity = match serde_json::from_value::<Input>(value.clone())? {
                Input::Texture(v) => format!("image:{}", v.image),
                Input::Step(v) => format!("step:{}", v.step),
                Input::Aov(v) => format!("aov:{}", v.aov),
                Input::Channel(v) => v.channel.token().to_owned(),
                _ => anyhow::bail!("{path}: input {name} must bind an image"),
            };
            let next = resources.len() as u32 + 1;
            let slot = *resources.entry(identity).or_insert(next);
            aliases.insert(name, slot);
        }
        hash.update(&serde_json::to_vec(&aliases)?);
        hash.update(abi.cache_identity()?.as_bytes());
        hash.update(source.as_bytes());
        hash.update(origin.to_string_lossy().as_bytes());
        let key = hash.finalize();
        let tag = key.to_hex();
        let compiled = programs::cached(
            r,
            key,
            &Compiler::discover(),
            &abi, &source, &origin,
            |compiler| {
                compiler.compile_composite_bound(
                    &abi,
                    &source,
                    &origin,
                    &tag[..12],
                    &inputs,
                )
            },
        ).with_context(|| format!("{path}: material {name}, program {} (materials.{name}.inputs)", material.program))?;
        let mut channels = Vec::new();
        let mut slots = std::collections::BTreeMap::new();
        let mut words = compiled.inputs.pack(&inputs, |name, value| {
            // Internal image keys: a kernel channel token, "image:<name>" or "aov:<name>"; graph images use "step:<name>".
            let channel = match serde_json::from_value::<vork_scene::v2::Input>(value.clone()) {
                Ok(vork_scene::v2::Input::Texture(t)) if scene.images.contains_key(&t.image) => format!("image:{}", t.image),
                Ok(vork_scene::v2::Input::Step(n)) => format!("step:{}", n.step),
                Ok(vork_scene::v2::Input::Channel(c)) => c.channel.token().to_owned(),
                Ok(vork_scene::v2::Input::Aov(a)) if scene.render.aovs.iter().any(|d| d.name == a.aov) => format!("aov:{}", a.aov),
                _ => anyhow::bail!("compositor input {name}: expected {{\"step\": earlier name}}, {{\"image\": images name}}, {{\"channel\": ...}}, {{\"aov\": render.aovs name}}; got {value}"),
            };
            let channel = channel.as_str();
            let slot = if let Some(index) = channels.iter().position(|c| c == channel) {
                index as u32 + 1
            } else {
                channels.push(channel.to_owned());
                channels.len() as u32
            };
            slots.insert(name.to_owned(), slot);
            Ok(slot)
        }).with_context(|| format!("{path}: material {name}, program {} (materials.{name}.inputs)", material.program))?;
        let animation = programs::InputAnimation::new_in(scene, base, name, &compiled.inputs, &mut words, opened)?;
        let input_words = words.clone();
        let words = init(
            &r.device,
            "compositor inputs",
            bytemuck::cast_slice(if words.is_empty() { &[0] } else { &words }),
            wgpu::BufferUsages::STORAGE,
        );
        result.insert(
            name.clone(),
            Program {
                compiled,
                words,
                input_words,
                animation,
                channels,
                slots,
            },
        );
    }
    Ok(result)
}

pub(crate) fn record(
    s: &InstalledScene,
    r: &Renderer,
    uniform: &wgpu::Buffer,
    targets: &crate::view::Targets,
    source: &wgpu::Buffer,
    names: &[vork_scene::v2::CompositeEntry],
    inputs: &FrameInputs,
    size: [u32; 2],
    accumulated: Normalization,
    recorder: &mut graph::Recorder,
    stage: graph::Pass,
    beauty: bool,
) -> Result<wgpu::Buffer> {
    let mut source = source.clone();
    let mut dispatches = Vec::new();
    let mut accumulated=accumulated;
    if stage==graph::Pass::Compositor && s.has_catchers && beauty {
        let (finished,dispatch)=crate::catcher::finish(s,r,uniform,targets,&source,size,accumulated)?;
        source=finished;dispatches.push(dispatch);accumulated=Normalization::None;
    }
    let (finished, images) = record_images(images::Sources::World(s, targets), &s.compositor, r, uniform, &source, names,
        &inputs.buffers, size, accumulated, stage, |label, bytes, usage| s.transients.buffer(&r.device, label, bytes, usage))?;
    source = finished;
    dispatches.extend(images);
    if !dispatches.is_empty() {
        recorder.compositor(stage, dispatches, Vec::new(), None);
    }
    Ok(source)
}

/// Record a graph's entries over `source` (associated RGBA, row-major, `size`): one dispatch per live entry,
/// and a last one that resamples to `size` when the last entry works at another resolution. A world's and a
/// canvas's graph are recorded here; `sources` is where the entries' world channels and image inputs come
/// from, and `allocate` makes each entry's image.
#[allow(clippy::too_many_arguments)]
fn record_images(
    sources: images::Sources<'_>,
    programs: &std::collections::HashMap<String, Program>,
    r: &Renderer,
    uniform: &wgpu::Buffer,
    source: &wgpu::Buffer,
    names: &[vork_scene::v2::CompositeEntry],
    inputs: &std::collections::HashMap<String, wgpu::Buffer>,
    size: [u32; 2],
    accumulated: Normalization,
    stage: graph::Pass,
    mut allocate: impl FnMut(&str, u64, wgpu::BufferUsages) -> wgpu::Buffer,
) -> Result<(wgpu::Buffer, Vec<graph::Dispatch>)> {
    let mut source = source.clone();
    let mut dispatches = Vec::new();
    let original = source.clone();
    let original_normalization = accumulated;
    let mut previous_size = size;
    let mut nodes = std::collections::BTreeMap::new();
    let (_, live) = dependencies(programs, names);
    for (index, entry) in names.iter().enumerate() {
        if !live[index] {continue;}
        let name = entry.material();
        let program = &programs[name];
        let words = inputs.get(name).unwrap_or(&program.words);
        let extent = entry.extent(size)?;
        let bytes = u64::from(extent[0]).checked_mul(u64::from(extent[1]))
            .and_then(|pixels| pixels.checked_mul(constants::ACCUMULATION_PIXEL_BYTES))
            .context("compositor image size overflows")?;
        let limits = r.device.limits();
        ensure!(bytes <= u64::from(limits.max_storage_buffer_binding_size) && bytes <= limits.max_buffer_size,
            crate::gpu_memory::device_limit(format_args!("compositor entry {index}"), format_args!("{bytes} bytes"), format_args!("storage binding={} bytes, buffer={} bytes",limits.max_storage_buffer_binding_size,limits.max_buffer_size), "reduce compositor image resolution"));
        let output = allocate(&format!("{stage:?} image {index}"),
            bytes,
            wgpu::BufferUsages::STORAGE);
        // Size queries use authored metadata even when the previous producer is dead.
        previous_size=if index==0 {size} else {names[index-1].extent(size)?};
        dispatches.push(images::pass(sources, r, uniform, program, words, &source,
            size, extent, previous_size, accumulated.first(index == 0), &program.compiled.entry, &output, &nodes, &original, original_normalization)?);
        source = output;
        previous_size = extent;
        if let Some(name) = entry.name() { nodes.insert(name.to_owned(), (source.clone(), extent)); }
    }
    if previous_size != size {
        let program = &programs[names.last().unwrap().material()];
        let output = allocate("compositor output", u64::from(size[0]) * u64::from(size[1]) * constants::ACCUMULATION_PIXEL_BYTES, wgpu::BufferUsages::STORAGE);
        dispatches.push(images::pass(sources, r, uniform, program, &program.words, &source,
            size, size, previous_size, Normalization::None, "vork_copy", &output, &nodes, &original, original_normalization)?);
        source = output;
    }
    Ok((source, dispatches))
}
