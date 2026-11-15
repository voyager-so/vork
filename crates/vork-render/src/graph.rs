//! Install-time frame graph. Passes declare the named resources they read and
//! write, in execution order; `Graph::new` drops passes the plan does not run or
//! whose writes reach no sink (the view colour, and CPU readback when the install
//! reads pixels back), and rejects a read of a produced resource before its writer. The compiled graph is
//! a flat step list with one run bit per step: a frame sets the bits from its
//! values and walks the list. No names, hashing or allocation exist per frame.
//!
//! Transients are backed once per installed scene and shared by every frame
//! slot: one queue executes slot submissions in order, and `Transients::submit`
//! keeps each frame's samples contiguous. Within a frame, one backing serves
//! several names whose step intervals are disjoint and whose descriptors are
//! identical (wgpu cannot place two resources in one allocation).
//!
//! Each view records its steps once, when it is created: pass modules build
//! their bind groups and dispatches and hand compute work to a `Recorder`;
//! `Graph::record` orders it with the raster and frame passes.
//!
//! To add a pass: add a `Pass` variant, declare it in `declare` at its place in
//! the order with its reads, writes, install condition and run bit, then record
//! its compute work with `Recorder::compute` (post passes; also list it in
//! `Pass::is_compute`), or encode it in `encode_view` (raster) or
//! `FrameSlot::encode_sample` (frame passes).
use crate::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub(crate) enum Pass {
    Deform,
    /// Acceleration-structure rebuilds after deformation, before any ray reads.
    Rays,
    /// The irradiance volume update, when its lighting inputs changed.
    Probes,
    Fibre,
    /// Every render image view, in install DAG order.
    Images,
    /// Same-frame seed publications, without retained lighting.
    ImageInputs,
    Depth,
    /// Decoupled visibility passes and their slot merges (passes/decoupled.rs).
    Visibility,
    Clusters,
    Forward,
    /// Decoupled slot shading into the resolved colour.
    Shade,
    /// Screen refraction: the shade's opaque image pyramid, then the transparent
    /// composite over the resolved colour (passes/decoupled.rs).
    Refraction,
    SubsurfaceDiffuse,
    SubsurfaceCompose,
    TransparentControls,
    Controls,
    PathEnvironment,
    Path,
    Reflections,
    Ao,
    PostSource,
    Denoise,
    Compositor,
    Resolve,
    AovResolve,
    Readback,
}
impl Pass {
    /// Post passes whose work is prebuilt compute dispatches.
    fn is_compute(self) -> bool {
        matches!(
            self,
            Pass::SubsurfaceDiffuse | Pass::Controls
                | Pass::PathEnvironment
                | Pass::Path
                | Pass::Reflections
                | Pass::Ao
                | Pass::PostSource
                | Pass::Compositor
                | Pass::Resolve
        )
    }
    /// View passes run once per view; frame passes once per sample of the frame.
    pub(crate) fn per_view(self) -> bool {
        !matches!(
            self,
            Pass::Deform | Pass::Rays | Pass::Probes | Pass::Fibre | Pass::Images | Pass::ImageInputs | Pass::Readback
        )
    }
}

/// Per-frame run condition of a compiled step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Run {
    Always,
    /// The view produces colour (not a nominal controls-only sample).
    Color,
    /// Colour post processing after the last shutter sample.
    FinalColor,
    /// The probe volume's lighting inputs changed.
    Probes,
    Fibre,
    /// First sample of a frame whose images are captured.
    Images,
    /// The sample's pixels are read back.
    Pixels,
}
/// Values of the run conditions for one sample of one view.
#[derive(Clone, Copy)]
pub(crate) struct RunBits {
    pub color: bool,
    pub final_sample: bool,
    pub probes: bool,
    pub fibre: bool,
    pub images: bool,
    pub pixels: bool,
}
impl RunBits {
    pub(crate) fn runs(self, run: Run) -> bool {
        match run {
            Run::Always => true,
            Run::Color => self.color,
            Run::FinalColor => self.color && self.final_sample,
            Run::Probes => self.probes,
            Run::Fibre => self.fibre,
            Run::Images => self.images,
            Run::Pixels => self.pixels,
        }
    }
}

/// Resources that outlive one sample: read before their writer is valid.
/// Frame scope spans the shutter samples of one output frame; persistent
/// resources span frames and are rewritten only when their inputs change.
const RETAINED: &[&str] = &[
    "catcher radiance sums",
    "time step accumulation",
    "time step albedo",
    "time step normal",
    "linear sample accumulation",
    "denoising guides",
    "filtered denoising guides",
    // AOV planes share the packed readback buffer and sum across samples.
    "packed PNG pixels",
    "path adaptive moments",

    "render images",
    "ray structures",
    "mesh source areas",
    "light probe volume",
    "fibre transport",
];
fn retained(name: &str) -> bool {
    RETAINED.contains(&name)
}

pub(crate) struct Declared {
    pass: Pass,
    reads: Vec<&'static str>,
    writes: &'static [&'static str],
    run: Run,
    enabled: bool,
}
impl Declared {
    pub(crate) fn reads(&mut self, names: &'static [&'static str]) -> &mut Self {
        self.reads = names.to_vec();
        self
    }
    pub(crate) fn writes(&mut self, names: &'static [&'static str]) -> &mut Self {
        self.writes = names;
        self
    }
    pub(crate) fn run(&mut self, run: Run) -> &mut Self {
        self.run = run;
        self
    }
    /// Install-time condition: the scene plan needs this pass.
    pub(crate) fn when(&mut self, enabled: bool) -> &mut Self {
        self.enabled = enabled;
        self
    }
}
#[derive(Default)]
pub(crate) struct Builder(Vec<Declared>);
impl Builder {
    pub(crate) fn pass(&mut self, pass: Pass) -> &mut Declared {
        self.0.push(Declared {
            pass,
            reads: Vec::new(),
            writes: &[],
            run: Run::Always,
            enabled: true,
        });
        self.0.last_mut().unwrap()
    }
}

/// Resources outside the frame: what the install's consumers read.
const VIEW_COLOUR: &str = "view colour";
const CPU_READBACK: &str = "CPU readback";

const MSAA: &[&str] = &[
    "depth 4x",
    "ID 4x",
    "screen motion 4x",
    "world normals 4x",
    "ambient 4x",
];

fn declare(b: &mut Builder, plan: &crate::plan::ScenePlan) {
    use Pass::*;
    let e = &plan.effects;
    b.pass(Deform)
        .when(plan.deform)
        .writes(&["deformed vertices"]);
    b.pass(Rays)
        .when(plan.rays)
        .reads(&["deformed vertices", "ray structures"])
        .writes(&["ray structures", "mesh source areas"]);
    if e.image_feedback {
        b.pass(ImageInputs).run(Run::Images)
            .reads(&["deformed vertices", "ray structures", "mesh source areas"])
            .writes(&["render image inputs"]);
    }
    b.pass(Probes)
        .when(e.probes)
        .run(Run::Probes)
        .reads(&["deformed vertices", "ray structures", "mesh source areas", "light probe volume"])
        .writes(&["light probe volume"]);
    if e.image_feedback && e.probes {
        b.0.last_mut().unwrap().reads(&["deformed vertices", "ray structures", "mesh source areas", "light probe volume", "render image inputs"]);
    }
    b.pass(Fibre).when(e.fibre).run(Run::Fibre).reads(&["deformed vertices"]).writes(&["fibre transport"]);
    if e.image_feedback && e.fibre {
        b.0.last_mut().unwrap().reads(&["deformed vertices", "render image inputs"]);
    }
    // Captures consume shared scene producers even when the main compositor
    // replaces camera beauty. Their inputs must survive main-view pruning.
    b.pass(Images)
        .run(Run::Images)
        .reads(&["deformed vertices", "ray structures", "mesh source areas", "light probe volume", "fibre transport"])
        .writes(&["render images"]);
    b.pass(Depth)
        .reads(if e.trace {&["deformed vertices","ray structures"]} else {&["deformed vertices"]})
        .writes(&["depth 4x", "camera eligibility"]);
    // Visibility draws the depth pass's camera-culled draw commands.
    b.pass(Visibility)
        .when(plan.decoupled)
        .run(Run::Color)
        .reads(if e.trace {&["deformed vertices","camera eligibility","ray structures"]} else {&["deformed vertices","camera eligibility"]})
        .writes(&["visibility keys", "visibility depth", "visibility slots"]);
    b.pass(Clusters)
        .when(plan.cluster)
        .run(Run::Color)
        .writes(&["cluster light lists"]);
    // Controls: unlit surface responses, no lights or shadows.
    b.pass(Forward)
        .reads(if e.trace {&["deformed vertices", "depth 4x", "camera eligibility", "render images","ray structures"]} else {&["deformed vertices", "depth 4x", "camera eligibility", "render images"]})
        .writes(&MSAA[1..]);
    b.pass(TransparentControls)
        .when(plan.transparent)
        .reads(if e.trace {&["deformed vertices", "depth 4x","ray structures"]} else {&["deformed vertices", "depth 4x"]})
        .writes(&MSAA[1..]);
    b.pass(Controls).when(e.standalone_controls(None)).reads(MSAA).writes(&[
        "denoising guides",
        "resolved depth",
        "resolved IDs",
        "resolved normals",
        "resolved ambient",
        "resolved motion",
    ]);
    // The shade blends transparent slots over the opaque ones. Its ambient is the
    // opaque surfaces' as far as they are transmitted, replacing the controls'
    // nearest-surface (transparent) ambient.
    const SHADE_READS: &[&str] = &["render images", "visibility slots", "visibility keys", "cluster light lists", "ray structures", "light probe volume", "fibre transport", "resolved IDs"];
    b.pass(Shade)
        .when(plan.decoupled)
        .run(Run::Color)
        .reads(&SHADE_READS[..SHADE_READS.len()-usize::from(!e.indirect)])
        .writes(&[
            "resolved IDs",
            "HDR resolved",
            "resolved ambient",
            "decoupled AO receivers",
            "decoupled AO normals",
            "reflection records",
            "reflection guides",
            // Budgeted packets trace before reusing camera records. Reserve
            // their output from Shade even when this view takes the full path.
            "full-resolution reflection estimates",
            "AOV resolved",
            "refraction pyramid",
            "refraction receivers",
            "subsurface records",
        ]);
    // Burley diffusion filters the shade's capture; the compose replaces the
    // captured slot's diffusion in the resolved colour and ambient.
    b.pass(SubsurfaceDiffuse)
        .when(e.subsurface)
        .run(Run::Color)
        .reads(&["visibility slots", "subsurface records"])
        .writes(if plan.screen_refraction {&["subsurface correction", "subsurface ambient correction", "subsurface receiver correction", "subsurface receiver ambient correction"]} else {&["subsurface correction", "subsurface ambient correction"]});
    b.pass(SubsurfaceCompose)
        .when(e.subsurface)
        .run(Run::Color)
        .reads(if plan.screen_refraction {&["subsurface correction", "subsurface ambient correction", "HDR resolved", "resolved ambient", "subsurface receiver correction", "subsurface receiver ambient correction", "refraction pyramid"]} else {&["subsurface correction", "subsurface ambient correction", "HDR resolved", "resolved ambient"]})
        .writes(if plan.screen_refraction {&["HDR resolved", "resolved ambient", "refraction pyramid"]} else {&["HDR resolved", "resolved ambient"]});
    b.pass(Refraction)
        .when(plan.screen_refraction)
        .run(Run::Color)
        .reads(&["visibility slots", "cluster light lists", "ray structures", "light probe volume", "refraction pyramid", "refraction receivers", "HDR resolved", "AOV resolved"])
        .writes(&["refraction pyramid", "HDR resolved", "AOV resolved", "reflection records", "reflection guides"]);
    b.pass(PathEnvironment)
        .when(e.path)
        .run(if e.implicit {Run::Always} else {Run::Color})
        .reads(&["render images", "path adaptive moments"])
        // Each active sample's camera environment, left in its output pixel for
        // transport to read and replace.
        .writes(&["HDR resolved"]);
    b.pass(Path)
        .when(e.path)
        .run(if e.implicit {Run::Always} else {Run::Color})
        .reads(&["ray structures", "mesh source areas", "render images", "path adaptive moments", "denoising guides", "HDR resolved"])
        .writes(if e.implicit {&["HDR resolved", "path adaptive moments", "AOV resolved", "denoising guides", "catcher radiance sums", "packed PNG pixels", "resolved depth", "resolved normals", "resolved IDs"]} else {&["HDR resolved", "path adaptive moments", "AOV resolved", "denoising guides", "catcher radiance sums", "packed PNG pixels"]});
    b.pass(AovResolve).when(plan.aovs).run(Run::Color).reads(&["AOV resolved"]).writes(&["packed PNG pixels"]);
    b.pass(Reflections)
        .when(e.reflections)
        .run(Run::Color)
        .reads(&[
            "visibility slots",
            "reflection records",
            "reflection guides",
            "resolved depth",
            "resolved normals",
            "decoupled AO receivers",
            "decoupled AO normals",
            "resolved ambient",
            "HDR resolved",
            "ray structures",
            "light probe volume",
        ])
        .writes(&["full-resolution reflection estimates"]);
    b.pass(Ao)
        .when(e.indirect)
        .run(Run::Color)
        .reads(&[
            "resolved depth",
            "resolved normals",
            "resolved ambient",
            "HDR resolved",
            "decoupled AO receivers",
            "decoupled AO normals",
            "ray structures",
            "light probe volume",
        ])
        .writes(&[
            "AO selected receivers",
            "AO filtered visibility depth normal",
            "AO per-sample visibility depth normal",
            "hemisphere incident radiance",
            "hemisphere per-sample incident radiance",
        ]);
    b.pass(PostSource)
        .when(e.compositor || e.denoise)
        .run(Run::Color)
        .reads(&[
            "cluster light lists",

            "HDR resolved",
            "resolved depth",
            "resolved IDs",
            "resolved normals",
            "resolved ambient",
            "resolved motion",
            "full-resolution reflection estimates",
            "reflection guides",
            "hemisphere incident radiance",
            "hemisphere per-sample incident radiance",
            "AO selected receivers",
            "AO filtered visibility depth normal",
            "AO per-sample visibility depth normal",
            "linear sample accumulation",
        ])
        .writes(&[
            "linear sample accumulation",
        ]);
    b.pass(Denoise)
        .when(e.denoise)
        .run(Run::FinalColor)
        .reads(&["linear sample accumulation", "resolved normals", "denoising guides"])
        .writes(&["linear sample accumulation"]);
    b.pass(Compositor)
        .when(e.compositor)
        .run(Run::FinalColor)
        .reads(&["linear sample accumulation", "resolved depth", "resolved IDs", "resolved normals", "resolved motion", "AOV resolved", "catcher radiance sums"])
        .writes(&["compositor colour"]);
    b.pass(Resolve)
        .when(!e.fuse_controls())
        .reads(&[
            "cluster light lists",

            "HDR resolved",
            "decoupled AO receivers",
            "decoupled AO normals",
            "resolved depth",
            "resolved IDs",
            "resolved normals",
            "resolved ambient",
            "resolved motion",
            "full-resolution reflection estimates",
            "reflection guides",
            "hemisphere incident radiance",
            "hemisphere per-sample incident radiance",
            "AO selected receivers",
            "AO filtered visibility depth normal",
            "AO per-sample visibility depth normal",
            "linear sample accumulation",
            "compositor colour",
        ])
        .writes(&[
            "linear sample accumulation",
            "packed PNG pixels",
            VIEW_COLOUR,
        ]);
    b.pass(Resolve)
        .when(e.fuse_controls())
        .reads(&["cluster light lists","HDR resolved", "depth 4x", "ID 4x", "world normals 4x", "ambient 4x", "screen motion 4x", "linear sample accumulation"])
        .writes(&["linear sample accumulation", "packed PNG pixels", VIEW_COLOUR]);
    b.pass(Readback)
        .run(Run::Pixels)
        .reads(&["packed PNG pixels"])
        .writes(&[CPU_READBACK]);
}

#[derive(Clone, Copy, Debug)]
struct Scheduled {
    pass: Pass,
    run: Run,
}
/// The compiled pass order of one installed scene, and each resource's
/// interval over it.
pub(crate) struct Graph {
    pub(crate) views: crate::views::ViewTable,
    order: Vec<Scheduled>,
    intervals: HashMap<&'static str, (u16, u16)>,
    /// Resources a compiled pass or a sink reads.
    #[cfg(test)]
    read: HashSet<&'static str>,
}
impl Graph {
    pub(crate) fn uses(&self,name:&str)->bool {self.intervals.contains_key(name)}
    pub(crate) fn runs(&self, pass: Pass) -> bool {self.order.iter().any(|s|s.pass==pass)}
    pub(crate) fn draws(&self, kind:crate::bundles::Kind) -> bool {
        use crate::bundles::Kind;
        self.runs(match kind {Kind::Depth=>Pass::Depth,Kind::Controls=>Pass::Forward,
            Kind::TransparentControls=>Pass::TransparentControls,
            Kind::Visibility(_)|Kind::TransparentFragments(_)=>Pass::Visibility,
            Kind::Fibre(_,_)=>Pass::Fibre})
    }
    /// The compiled passes in queue order, once each.
    pub(crate) fn passes(&self) -> Vec<Pass> {
        let mut passes = Vec::new();
        for scheduled in &self.order {
            if !passes.contains(&scheduled.pass) {passes.push(scheduled.pass);}
        }
        passes
    }
    /// `readback`: the install's consumers read pixels back to the CPU.
    pub(crate) fn new(plan: &crate::plan::ScenePlan, readback: bool) -> Result<Self> {
        Self::for_outputs(plan, readback, plan.effects.outputs)
    }
    pub(crate) fn for_outputs(plan:&crate::plan::ScenePlan, readback:bool, outputs:Option<crate::compositor::Dependencies>)->Result<Self> {
        let mut b = Builder::default();
        declare(&mut b, plan);
        let view=outputs.is_none_or(|o|o.color || o.controls());
        let sinks: &[_] = match (view,readback) {
            (true,true)=>&[VIEW_COLOUR,CPU_READBACK],
            (true,false)=>&[VIEW_COLOUR],
            (false,true)=>&[CPU_READBACK],
            (false,false)=>&[],
        };
        if let Some(outputs)=outputs {
            for d in &mut b.0 {
                match d.pass {
                    Pass::Compositor => {d.reads=outputs.resources();d.enabled &= outputs.color;},
                    Pass::PostSource => {
                        d.enabled &= outputs.beauty || outputs.catcher;
                        if !plan.effects.indirect && !plan.effects.reflections {
                            d.reads=vec!["HDR resolved","linear sample accumulation"];
                        }
                    },
                    Pass::Denoise => {
                        d.enabled &= outputs.beauty || outputs.catcher;
                        if plan.effects.path {d.reads.retain(|name|*name!="resolved normals");}
                    },
                    Pass::Controls => d.enabled &= plan.effects.standalone_controls(Some(outputs)),
                    Pass::Shade => d.enabled &= outputs.beauty || outputs.aovs || outputs.catcher,
                    // The camera environment pass exists only to feed transport.
                    Pass::Path | Pass::PathEnvironment => {
                        d.enabled &= outputs.beauty || outputs.aovs || outputs.catcher || (plan.effects.implicit && outputs.controls());
                        if !plan.effects.denoise {d.reads.retain(|name|*name!="denoising guides");}
                    },
                    Pass::Resolve if !outputs.color && !outputs.controls() => d.enabled=false,
                    Pass::Resolve if plan.effects.fuse_controls() && outputs.color => {
                        d.reads=vec!["HDR resolved","linear sample accumulation"];
                        if outputs.controls() {d.reads.extend(MSAA);}
                    },
                    Pass::Resolve if plan.effects.compositor || !outputs.color || (!plan.effects.indirect && !plan.effects.reflections) => {
                        d.reads=if outputs.color {if plan.effects.compositor {vec!["compositor colour"]} else {vec!["HDR resolved","linear sample accumulation"]}} else if plan.effects.fuse_controls() && outputs.controls() {MSAA.to_vec()} else {Vec::new()};
                        for (needed,name) in [(outputs.controls(),"resolved depth"),(outputs.controls(),"resolved IDs"),(outputs.normals,"resolved normals"),(outputs.motion,"resolved motion")] {
                            if needed && !plan.effects.fuse_controls() {d.reads.push(name);}
                        }
                    },
                    Pass::AovResolve => d.enabled &= outputs.aov_output,
                    _=>(),
                }
            }
        }
        Self::compile(b, sinks)
    }
    fn compile(b: Builder, sinks: &[&'static str]) -> Result<Self> {
        // A pass runs if a running pass or a sink reads one of its writes. Order
        // does not decide liveness, so a misordered writer is an error below.
        let mut declared: Vec<_> = b.0.into_iter().filter(|d| d.enabled).collect();
        let mut live = vec![false; declared.len()];
        let mut read: HashSet<_> = sinks.iter().copied().collect();
        while let Some(i) = (0..declared.len())
            .find(|&i| !live[i] && declared[i].writes.iter().any(|name| read.contains(name)))
        {
            live[i] = true;
            read.extend(declared[i].reads.iter().copied());
        }
        let mut live = live.into_iter();
        declared.retain(|_| live.next().unwrap());
        // Outputs of dropped passes stay bound as unread placeholders.
        let produced: HashSet<_> = declared
            .iter()
            .flat_map(|d| d.writes.iter().copied())
            .collect();
        let mut written = HashSet::new();
        let mut intervals = HashMap::<&'static str, (u16, u16)>::new();
        for (at, d) in declared.iter().enumerate() {
            let at = at as u16;
            for &name in d.reads.iter().chain(d.writes) {
                intervals
                    .entry(name)
                    .and_modify(|i| i.1 = at)
                    .or_insert((at, at));
            }
            for name in &d.reads {
                ensure!(
                    !produced.contains(name) || written.contains(name) || retained(name),
                    "frame graph: {:?} reads {name} before its writer",
                    d.pass
                );
            }
            written.extend(d.writes.iter().copied());
        }
        for name in RETAINED.iter() {
            intervals.insert(name, (0, u16::MAX));
        }
        Ok(Self {
            views: Default::default(),
            intervals,
            #[cfg(test)]
            read,
            order: declared
                .iter()
                .map(|d| Scheduled {
                    pass: d.pass,
                    run: d.run,
                })
                .collect(),
        })
    }
    /// Some compiled pass or sink reads `name`; an unread output needs no backing.
    #[cfg(test)]
    pub(crate) fn reads(&self, name: &str) -> bool {
        self.read.contains(name)
    }
    /// One view's step list: its recorded compute work and the raster and
    /// frame passes, in graph order. Compute passes the view recorded nothing
    /// for drop out. Image views run no frame passes.
    pub(crate) fn record(&self, frame_passes: bool, recorder: Recorder) -> Result<Vec<Step>> {
        let mut recorded=HashMap::<Pass,Vec<Compute>>::new();
        for (pass,node) in recorder.0 {recorded.entry(pass).or_default().push(node);}
        let mut steps = Vec::new();
        for &Scheduled { pass, run } in &self.order {
            if !frame_passes && !pass.per_view() {
                continue;
            }
            let nodes=recorded.remove(&pass).unwrap_or_default();
            if nodes.is_empty() {
                if !pass.is_compute() { steps.push(Step {pass,run,work:Work::Encode}); }
            } else {
                let spans:Vec<_>=nodes.iter().map(|node|node.span).collect();
                for (index,mut node) in nodes.into_iter().enumerate() {
                    // Expanded nodes with the same span share one interval.
                    // Distinct contiguous stages of a pass own separate pairs.
                    node.begin=index==0 || spans[index-1]!=node.span;
                    node.end=index+1==spans.len() || spans[index+1]!=node.span;
                    steps.push(Step {pass,run,work:Work::Compute(node)});
                }
            }
        }
        // Producers may record shared bindings before liveness is known; only
        // the compiled dependency closure becomes executable work.
        Ok(steps)
    }
}

/// Fully bound compute work. Optional pipelines are resolved while recording;
/// replay cannot request a pipeline that was not installed.
#[derive(Clone)]
pub(crate) struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    variant: Option<crate::pipelines::Drawn<wgpu::ComputePipeline>>,
    bind: wgpu::BindGroup,
    groups: [u32; 3],
    dynamic_offset: Option<u32>,
    color_only: bool,
    /// Group 1 material tables, or screen reflection field storage.
    material: Option<wgpu::BindGroup>,
    /// Ray geometry and reflection record arrays as group 3.
    hits: Option<wgpu::BindGroup>,
    /// Group 2 read tables: irradiance and shared ray-visibility tables.
    read_table: Option<wgpu::BindGroup>,
}
impl Dispatch {
    pub(crate) fn encode_packet(&self,encoder:&mut wgpu::CommandEncoder,specialized:bool) {
        let mut pass=encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(self.variant.as_ref().map_or(&self.pipeline,|p|p.current(specialized)));
        pass.set_bind_group(0,&self.bind,self.dynamic_offset.as_slice());
        if let Some(bind)=&self.material {pass.set_bind_group(1,bind,&[]);}
        if let Some(bind)=&self.read_table {pass.set_bind_group(2,bind,&[]);}
        if let Some(bind)=&self.hits {pass.set_bind_group(3,bind,&[]);}
        let [x,y,z]=self.groups;pass.dispatch_workgroups(x,y,z);
    }

    pub(crate) fn new(
        pipeline: &wgpu::ComputePipeline,
        bind: wgpu::BindGroup,
        groups: [u32; 3],
    ) -> Self {
        Self {
            pipeline: pipeline.clone(),
            variant: None,
            bind,
            groups,
            dynamic_offset: None,
            color_only: true,
            material: None,
            hits: None,
            read_table: None,
        }
    }
    pub(crate) fn with_pipeline(mut self, pipeline: crate::pipelines::Drawn<wgpu::ComputePipeline>) -> Self {
        self.variant = Some(pipeline);
        self
    }

    pub(crate) fn with_offset(mut self, offset: u32) -> Self {
        self.dynamic_offset = Some(offset);
        self
    }
    pub(crate) fn with_material(mut self, material: &wgpu::BindGroup) -> Self {
        self.material = Some(material.clone());
        self
    }
    pub(crate) fn with_hits(mut self, hits: wgpu::BindGroup) -> Self {
        self.hits = Some(hits);
        self
    }
    pub(crate) fn with_read_table(mut self, table: Option<wgpu::BindGroup>) -> Self {
        self.read_table = table;
        self
    }
    /// The dispatch also writes the control targets, not only colour.
    pub(crate) fn all_targets(mut self) -> Self {
        self.color_only = false;
        self
    }
}
/// The prebuilt compute work of one step. Clears and begin timing belong to the
/// first encoded range; end timing belongs to the last.
pub(crate) struct Compute {
    label: &'static str,
    span: &'static str,
    /// Binds the view's colour output as group 1.
    output: bool,
    dispatches: Vec<Dispatch>,
    /// Maximum dispatches per submission segment; zero keeps the step together.
    separate: usize,
    /// Measured milliseconds per workgroup row for adaptive path bands.
    time_budget: Option<std::sync::Mutex<f64>>,
    clear: Vec<wgpu::Buffer>,
    marker: Option<crate::timing::Marker>,
    begin: bool,
    end: bool,
}
impl Compute {
    fn encode_time_budgeted(
        &self, encoder: &mut wgpu::CommandEncoder, f: &crate::view::ViewState,
        r: &crate::Renderer, segments: &mut crate::command_finish::Segments,
    ) -> Result<()> {
        let mut row_ms=self.time_budget.as_ref().unwrap().lock().unwrap();
        let at=timing::query(self.span);
        let queries=r.device.create_query_set(&wgpu::QuerySetDescriptor {
            label:Some("path dispatch budget"),ty:wgpu::QueryType::Timestamp,count:at+2,
        });
        let resolve=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
            label:Some("path dispatch budget resolve"),size:256,
            usage:wgpu::BufferUsages::QUERY_RESOLVE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
        });
        let readback=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
            label:Some("path dispatch budget readback"),size:16,
            usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,
        });
        let marker=r.timestamp_marker.get_or_init(||timing::Marker::new(r));
        marker.encode(encoder,&f.queries,at);
        let mut begin=0;
        let mut previous=if *row_ms>0. {self.dispatches.len()} else {1};
        while begin<self.dispatches.len() {
            let count=budget_rows(*row_ms,self.dispatches.len()-begin,previous);
            let end=begin+count;
            self.encode_range::<true>(encoder,&queries,&f.output_bind,f.color_enabled,
                !f.variants_pending(),begin..end,true);
            encoder.resolve_query_set(&queries,at..at+2,&resolve,0);
            encoder.copy_buffer_to_buffer(&resolve,0,&readback,0,16);
            let (submission,_)=segments.interop(r,encoder,"after timed path band");
            let (tx,rx)=std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read,..,move |value|{let _=tx.send(value);});
            r.wait(submission)?;
            rx.recv()??;
            let mapped=readback.get_mapped_range(..);
            let ticks:&[u64]=bytemuck::cast_slice(&mapped);
            let ms=timing::elapsed_ms(ticks[0],ticks[1],r.queue.get_timestamp_period())
                .filter(|ms|ms.is_finite() && *ms>0.)
                .ok_or_else(||anyhow::anyhow!("invalid GPU timestamps for path dispatch budget"))?;
            drop(mapped);readback.unmap();
            // Keep the high-water cost: a cheap sky band must not enlarge the
            // following interior band. A new view starts with a cold pilot.
            *row_ms=(*row_ms).max(ms/count as f64);
            if std::env::var_os("VORK_PATH_DISPATCH_TIMES").is_some() {
                eprintln!("path dispatch: sample={} rows={}..{} gpu_ms={ms:.6} budget_ms={}",
                    f.sample_index,begin*8,(end*8).min(f.internal[1] as usize),constants::SAMPLE_GROUP_GPU_MS);
            }
            previous=count;
            begin=end;
        }
        marker.encode(encoder,&f.queries,at+1);
        Ok(())
    }
    fn encode<const SEGMENT:bool>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
        output: &wgpu::BindGroup,
        color_enabled: bool,
        specialized: bool,
        range: std::ops::Range<usize>,
    ) {
        self.encode_range::<SEGMENT>(encoder,queries,output,color_enabled,specialized,range,false);
    }
    fn encode_range<const SEGMENT:bool>(
        &self, encoder: &mut wgpu::CommandEncoder, queries: &wgpu::QuerySet,
        output: &wgpu::BindGroup, color_enabled: bool, specialized: bool,
        range: std::ops::Range<usize>, merge_rows: bool,
    ) {
        let first = range.start == 0;
        let begin = SEGMENT || (self.begin && first);
        let end = SEGMENT || (self.end && range.end == self.dispatches.len());
        let separate_begin=begin && !self.clear.is_empty();
        if separate_begin {self.marker.as_ref().expect("installed compositor timing marker").encode(encoder,queries,timing::query(self.span));}
        if first { for buffer in &self.clear { encoder.clear_buffer(buffer,0,None); } }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: ((begin && !separate_begin) || end).then_some(wgpu::ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: (begin && !separate_begin).then_some(timing::query(self.span)),
                end_of_pass_write_index: end.then_some(timing::query(self.span) + 1),
            }),
        });
        let merged_rows = if merge_rows {self.dispatches[range.clone()].iter().map(|d|d.groups[1]).sum::<u32>()} else {0};
        let dispatches = if merge_rows { &self.dispatches[range.start..range.start+1] } else { &self.dispatches[range] };
        for dispatch in dispatches {
            if dispatch.color_only && !color_enabled {
                continue;
            }
            pass.set_pipeline(dispatch.variant.as_ref().map_or(&dispatch.pipeline, |pipeline|pipeline.current(specialized)));
            if self.output {
                pass.set_bind_group(1, output, &[]);
            }
            pass.set_bind_group(0, &dispatch.bind, dispatch.dynamic_offset.as_slice());
            if let Some(material) = &dispatch.material {
                pass.set_bind_group(1, material, &[]);
            }
            if let Some(table) = &dispatch.read_table {
                pass.set_bind_group(2, table, &[]);
            }
            if let Some(hits) = &dispatch.hits {
                pass.set_bind_group(3, hits, &[]);
            }

            let [x, y, z] = dispatch.groups;
            pass.dispatch_workgroups(x, if merge_rows {merged_rows} else {y}, z);
        }
    }
}

/// A 500 ms target with 2x headroom, shared by all backends. Unknown/cold
/// shaders start at one workgroup row; growth is limited to fourfold so a
/// cheap pilot cannot immediately launch the whole image.
fn budget_rows(row_ms:f64,remaining:usize,previous:usize)->usize {
    let measured=if row_ms.is_finite() && row_ms>0. {
        (constants::SAMPLE_GROUP_GPU_MS/row_ms).floor().max(1.) as usize
    } else {1};
    measured.min(previous.saturating_mul(4)).min(remaining).max(1)
}
pub(crate) enum Work {
    /// Encoded by its pass module over the view's or slot's prebuilt objects.
    Encode,
    Compute(Compute),
}
/// One recorded step of a view: built once when the view is created.
pub(crate) struct Step {
    pub pass: Pass,
    pub run: Run,
    pub work: Work,
}
impl Step {
    pub(crate) fn needs_interop(&self)->bool {
        matches!(&self.work,Work::Compute(work) if work.time_budget.is_some())
    }
    pub(crate) fn separates(&self) -> bool {
        matches!(&self.work, Work::Compute(work) if work.separate>0)
    }
    /// Only queries written by this recorded step may be resolved. Call after
    /// checking the step's run bit, using the same view as the encoder.
    pub(crate) fn timestamp_mask(&self, view: &crate::view::ViewState) -> u64 {
        if self.pass==Pass::Reflections && view.decoupled.as_ref().is_some_and(|v|v.packets) {return 0;}
        let mask = |name| crate::timing::mask(name);
        match &self.work {
            Work::Compute(work) => mask(work.span),
            Work::Encode => match self.pass {
                Pass::Deform => mask("deform"),
                Pass::Rays => mask("acceleration structures"),
                Pass::Probes => mask("probes"),
                Pass::Fibre => mask("fibre"),
                Pass::Images => mask("images"),
                Pass::ImageInputs => mask("image inputs"),
                Pass::Depth => mask("depth"),
                Pass::Visibility if view.decoupled.as_ref().is_some_and(|v|v.packets) => 0,
                Pass::Visibility => mask("visibility"),
                // Banded shading interleaves its stages: one "shade" span covers them.
                Pass::Shade if crate::passes::decoupled::fused_active(view) => mask("shade"),
                Pass::Shade if view.decoupled.as_ref().is_some_and(|v| v.packets || v.bands > 1) => mask("shade"),
                Pass::Shade => mask("shade") | mask("material compose") | if view.decoupled.as_ref().is_some_and(crate::passes::decoupled::classified) {mask("material classify")} else {0},
                Pass::Refraction => mask("refraction"),
                Pass::Clusters if view.lighting.clusters.is_some() => mask("clusters"),
                Pass::Forward => mask("forward"),
                Pass::SubsurfaceCompose if view.subsurface.is_some() => mask("subsurface_compose"),
                // The controls pass finishes the transparency interval; path
                // controls also start it because there is no raster colour pass.
                Pass::TransparentControls if !view.transparent_order.is_empty() => mask("transparency"),
                _ => 0,
            },
        }
    }
}
/// Collects the compute work pass modules build while a view is created.
#[derive(Default)]
pub(crate) struct Recorder(Vec<(Pass, Compute)>);
impl Recorder {
    pub(crate) fn compute(
        &mut self,
        pass: Pass,
        label: &'static str,
        span: &'static str,
        output: bool,
        dispatches: Vec<Dispatch>,
    ) {
        debug_assert!(pass.is_compute() && (matches!(pass, Pass::Compositor | Pass::Ao) || self.0.iter().all(|(p, _)| *p != pass)));
        self.0.push((
            pass,
            Compute {
                label,
                span,
                output,
                separate: 0,
                time_budget: None,
                dispatches,
                clear: Vec::new(),
                marker: None,
                begin: true,
                end: true,
            },
        ));
    }
    pub(crate) fn separate_last(&mut self) {self.separate_last_every(1);}
    pub(crate) fn time_budget_last(&mut self) {
        self.0.last_mut().expect("recorded compute step").1.time_budget=Some(std::sync::Mutex::new(0.));
    }
    pub(crate) fn separate_last_every(&mut self,dispatches:usize) {
        assert!(dispatches>0);
        self.0.last_mut().expect("recorded compute step").1.separate = dispatches;
    }
    pub(crate) fn compositor(&mut self, stage: Pass, dispatches:Vec<Dispatch>, clear:Vec<wgpu::Buffer>, marker:Option<crate::timing::Marker>) {
        self.compute(stage,"compositor image pass","compositor",false,dispatches);
        self.0.last_mut().unwrap().1.clear=clear;
        self.0.last_mut().unwrap().1.marker=marker;
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Texture {
        size: wgpu::Extent3d,
        mips: u32,
        samples: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    Buffer {
        size: u64,
        usage: wgpu::BufferUsages,
    },
}
#[derive(Clone, PartialEq)]
enum Backing {
    Texture(wgpu::Texture),
    Buffer(wgpu::Buffer),
}
impl Backing {
    fn destroy(&self) {
        match self {
            Backing::Texture(texture) => texture.destroy(),
            Backing::Buffer(buffer) => buffer.destroy(),
        }
    }
}
/// Destroy and drop the backings `outgrown` selects.
fn destroy_where(physical: &mut Vec<Physical>, outgrown: impl Fn(&Physical) -> bool) {
    physical.retain(|p| {
        let destroy = outgrown(p);
        if destroy { p.backing.destroy(); }
        !destroy
    });
}
struct Physical {
    key: Key,
    busy: Vec<(u16, u16)>,
    /// Names it held in the replaced generation (Transients::begin_generation).
    replaced: Vec<String>,
    backing: Backing,
    bytes: u64,
}
#[derive(Default)]
pub(crate) struct State {
    named: HashMap<(String, Key), Backing>,
    physical: Vec<Physical>,
    requested: u64,
}
/// A publication receipt checked at the existing output fence.
pub(crate) enum PublicationDiagnostic {
    /// The producing frame's shader diagnostic word. A reservation retry there
    /// is a redo of every queued frame, not a failed publication.
    Shader,
    Code(&'static str),
    ComputedGeometry,
    PointDisplacement,
    SolverStatus(String),
}
pub(crate) type PublicationChecks = std::sync::Arc<Mutex<Vec<(wgpu::SubmissionIndex, Vec<(PublicationDiagnostic, wgpu::Buffer)>)>>>;
/// Transient and frame-scoped backings shared by all frame slots and views of
/// one installed scene. Used only while creating slots.
pub(crate) struct Transients {
    intervals: HashMap<&'static str, (u16, u16)>,
    state: crate::residency::Cache<Mutex<State>>,
    /// Submission-ordered input history for the shared published render images.
    /// Held while one frame submits all its samples.
    pub(crate) submit: Mutex<crate::values::ChangeTracker>,
    // An aborted exposure may have submitted prefixes and discarded suffixes.
    // This belongs to the installation, not evictable storage or a frame slot.
    pub(crate) recording_error: std::sync::Arc<std::sync::OnceLock<String>>,
    pub(crate) publication_checks: PublicationChecks,
}
impl Transients {
    pub(crate) fn new(graph: &Graph, ledger: &std::sync::Arc<crate::residency::Ledger>) -> Self {
        Self {
            intervals: graph.intervals.clone(),
            state: crate::residency::Cache::new(ledger),
            submit: Mutex::default(),
            recording_error: Default::default(),
            publication_checks: Default::default(),
        }
    }
    pub(crate) fn check_recording(&self) -> Result<()> {
        if let Some(error) = self.recording_error.get() {
            bail!("frame recording or publication previously failed; reinstall the scene before submitting again: {error}");
        }
        Ok(())
    }
    /// Called under `submit` for the entire exposure, through its final submission.
    /// Retained keys are speculative until all recording succeeds. Prefixes may
    /// already be queued, so neither resetting keys nor replacing one slot can
    /// recover shared captures, lighting, field publication and simulation state.
    pub(crate) fn record(&self, encode: impl FnOnce() -> Result<()>) -> Result<()> {
        self.check_recording()?;
        encode().map_err(|error| {
            self.recording_error.get_or_init(|| format!("{error:#}"));
            error.context("frame recording aborted; reinstall the scene before submitting again")
        })
    }
    pub(crate) fn retain(&self) -> std::sync::Arc<Mutex<State>> {
        self.state.get_or_try_init(|| {
            *self.submit.lock().unwrap() = Default::default();
            Ok((Mutex::new(State::default()), 0))
        }).expect("transient storage")
    }
    fn get(&self, label: &str, key: Key, bytes: u64, create: impl FnOnce() -> Backing) -> Backing {
        let retained = self.retain();
        let mut state = retained.lock().unwrap();
        state.requested += bytes;
        if let Some(backing) = state.named.get(&(label.to_owned(), key.clone())) {
            return backing.clone();
        }
        // Undeclared names keep a whole-frame interval: shared, never aliased.
        let interval = self.intervals.get(label).copied().unwrap_or((0, u16::MAX));
        let disjoint = |&(a, b): &(u16, u16)| b < interval.0 || interval.1 < a;
        destroy_where(&mut state.physical, |p| p.key != key && p.busy.is_empty() && p.replaced.iter().any(|name| name == label));
        let found = state
            .physical
            .iter()
            .position(|p| p.key == key && p.busy.iter().all(disjoint));
        let i = found.unwrap_or_else(|| {
            state.physical.push(Physical {
                key: key.clone(),
                busy: Vec::new(),
                replaced: Vec::new(),
                backing: create(),
                bytes,
            });
            state.physical.len() - 1
        });
        state.physical[i].busy.push(interval);
        let backing = state.physical[i].backing.clone();
        state.named.insert((label.to_owned(), key), backing.clone());
        self.state.set_bytes(state.physical.iter().map(|p|p.bytes).sum());
        backing
    }
    pub(crate) fn texture(
        &self,
        d: &wgpu::Device,
        desc: &wgpu::TextureDescriptor,
    ) -> wgpu::Texture {
        let key = Key::Texture {
            size: desc.size,
            mips: desc.mip_level_count,
            samples: desc.sample_count,
            format: desc.format,
            usage: desc.usage,
        };
        let bytes = crate::gpu_memory::texture_bytes(desc);
        match self.get(desc.label.unwrap_or(""), key, bytes, || {
            Backing::Texture(crate::gpu_memory::texture(&d, desc))
        }) {
            Backing::Texture(t) => t,
            Backing::Buffer(_) => unreachable!(),
        }
    }
    /// One-level 2D target; the transient form of `crate::texture`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn target(
        &self,
        d: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
        samples: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        let live=self.intervals.contains_key(label) || !matches!(label,
            "depth 4x"|"ID 4x"|"screen motion 4x"|"world normals 4x"|"ambient 4x"|
            "HDR resolved"|"resolved depth"|"resolved IDs"|"resolved normals"|"resolved ambient"|"resolved motion");
        let (w,h)=if live {(w,h)} else {(1,1)};
        self.texture(
            d,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: extent(w, h),
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            },
        )
    }
    pub(crate) fn buffer(
        &self,
        d: &wgpu::Device,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let key = Key::Buffer { size, usage };
        match self.get(label, key, size, || {
            Backing::Buffer(crate::gpu_memory::buffer(&d, &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            }))
        }) {
            Backing::Buffer(b) => b,
            Backing::Texture(_) => unreachable!(),
        }
    }
    /// Start a generation of frame slots that replaces every existing one
    /// (InstalledScene::redo_frame). The replaced slots must never submit again,
    /// even if a replacement fails: the new generation reuses their backings
    /// where sizes match and destroys each outgrown one before allocating its
    /// replacement, so a redo never holds both sizes of pooled storage.
    pub(crate) fn begin_generation(&self) {
        let retained = self.retain();
        let mut state = retained.lock().unwrap();
        state.requested = 0;
        for physical in &mut state.physical {
            physical.busy.clear();
            physical.replaced.clear();
        }
        for ((label, _), backing) in std::mem::take(&mut state.named) {
            if let Some(physical) = state.physical.iter_mut().find(|p| p.backing == backing) { physical.replaced.push(label); }
        }
    }
    /// Destroy the backings the current generation did not request (no busy
    /// interval): storage of names the replaced slots used and the new ones do not.
    pub(crate) fn release_previous_generations(&self) {
        let retained = self.retain();
        let mut state = retained.lock().unwrap();
        destroy_where(&mut state.physical, |p| p.busy.is_empty());
        self.state.set_bytes(state.physical.iter().map(|p|p.bytes).sum());
    }
    /// Bytes every slot and view requested (separate allocations before the
    /// graph) and bytes the graph allocated.
    pub(crate) fn bytes(&self) -> (u64, u64) {
        let retained = self.retain();
        let state = retained.lock().unwrap();
        (
            state.requested,
            state.physical.iter().map(|p| p.bytes).sum::<u64>(),
        )
    }
}

/// Encode one view step over the view's prebuilt resources.
pub(crate) fn encode_view(
    encoder: &mut wgpu::CommandEncoder,
    f: &crate::view::ViewState,
    step: &Step,
    r: &crate::Renderer,
    segments: &mut crate::command_finish::Segments,
) -> Result<()> {
    if step.pass==Pass::Reflections && f.decoupled.as_ref().is_some_and(|v|v.packets) {return Ok(());}
    let split = step.separates();
    if split { crate::command_finish::submission_boundary(r, encoder, segments); }
    match (&step.work, step.pass) {
        #[cfg(test)]
        (Work::Compute(work), Pass::Path) if r.test_gpu_segment_times.is_some() => {
            // The frame still resolves its ordinary span. Keep those endpoints
            // initialized while individual dispatches use their own query sets.
            let marker=r.timestamp_marker.get_or_init(||timing::Marker::new(r));
            marker.encode(encoder,&f.queries,timing::query(work.span));
            for index in 0..work.dispatches.len() {
                let at=timing::query(work.span);
                let queries=r.device.create_query_set(&wgpu::QuerySetDescriptor {label:Some("medium segment qualification"),ty:wgpu::QueryType::Timestamp,count:at+2});
                let resolve=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("medium segment query resolve"),size:256,usage:wgpu::BufferUsages::QUERY_RESOLVE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
                let readback=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("medium segment query readback"),size:16,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
                work.encode::<true>(encoder,&queries,&f.output_bind,f.color_enabled,!f.variants_pending(),index..index+1);
                encoder.resolve_query_set(&queries,at..at+2,&resolve,0);
                encoder.copy_buffer_to_buffer(&resolve,0,&readback,0,16);
                let (submission,_)=segments.interop(r,encoder,"after medium segment qualification");
                let (tx,rx)=std::sync::mpsc::channel();
                readback.map_async(wgpu::MapMode::Read,..,move |value|{let _=tx.send(value);});
                r.wait(submission)?;
                rx.recv()??;
                let mapped=readback.get_mapped_range(..);let ticks:&[u64]=bytemuck::cast_slice(&mapped);
                let ms=timing::elapsed_ms(ticks[0],ticks[1],r.queue.get_timestamp_period())
                    .filter(|ms|ms.is_finite() && *ms>=0.)
                    .context("invalid GPU timestamps for medium segment; rerun on a device with working timestamp queries")?;
                let source="gpu_timestamp";
                drop(mapped);readback.unmap();
                r.test_gpu_segment_times.as_ref().unwrap().lock().unwrap().push((f.sample_index,index,ms,source));
            }
            marker.encode(encoder,&f.queries,timing::query(work.span)+1);
        }
        (Work::Compute(work), Pass::Path) if work.time_budget.is_some() => {
            work.encode_time_budgeted(encoder,f,r,segments)?;
        }
        (Work::Compute(work), _) if split && !work.dispatches.is_empty() => {
            for begin in (0..work.dispatches.len()).step_by(work.separate) {
                if begin>0 {crate::command_finish::submission_boundary(r,encoder,segments);}
                work.encode::<false>(encoder,&f.queries,&f.output_bind,f.color_enabled,!f.variants_pending(),begin..(begin+work.separate).min(work.dispatches.len()));
            }
        }
        (Work::Compute(work), _) => work.encode::<false>(encoder, &f.queries, &f.output_bind,
            f.color_enabled, !f.variants_pending(), 0..work.dispatches.len()),
        (Work::Encode, Pass::Depth) => passes::depth::encode(encoder, f, r),
        (Work::Encode, Pass::Clusters) => passes::clusters::encode(encoder, f),
        (Work::Encode, Pass::Forward) => passes::forward::encode(encoder, f, r),
        (Work::Encode, Pass::Visibility) => passes::decoupled::visibility(encoder, f, r, segments)?,
        (Work::Encode, Pass::Shade) => passes::decoupled::shade(encoder, f, r, segments)?,
        (Work::Encode, Pass::Refraction) => passes::decoupled::refraction(encoder, f, r)?,
        (Work::Encode, Pass::AovResolve) => {if let Some(resolve)=&f.aov_resolve {resolve.encode(encoder);}},

        (Work::Encode, Pass::SubsurfaceCompose) => passes::subsurface::compose(encoder, f),
        (Work::Encode, Pass::TransparentControls) => {
            passes::transparent::encode_controls(encoder, f)
        }
        (Work::Encode, pass) => unreachable!("{pass:?} is not a view raster pass"),
    }
    if split { crate::command_finish::submission_boundary(r, encoder, segments); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispatch_budget_cold_growth_headroom_and_slow_rows() {
        assert_eq!(budget_rows(0.,135,1),1);
        assert_eq!(budget_rows(f64::NAN,135,1),1);
        assert_eq!(budget_rows(1.,135,1),4);
        assert_eq!(budget_rows(1.,135,4),16);
        assert_eq!(budget_rows(10.,135,64),25);
        assert_eq!(budget_rows(10.,3,64),3);
        assert_eq!(budget_rows(1000.,135,64),1);
        for previous in [1,4,16,64,256] {
            for cost in [0.1,1.,3.,20.,250.] {
                let rows=budget_rows(cost,135,previous);
                assert!(rows>=1 && rows<=135 && rows<=previous*4);
                assert!(rows as f64*cost<=constants::SAMPLE_GROUP_GPU_MS);
            }
        }
    }
    #[test]
    fn contiguous_compute_stages_own_complete_timestamp_intervals() -> Result<()> {
        let mut builder=Builder::default();
        builder.pass(Pass::Ao).writes(&[VIEW_COLOUR]);
        let graph=Graph::compile(builder,&[VIEW_COLOUR])?;
        let mut recorder=Recorder::default();
        for span in ["gi demand","gi pilots","gi pilots","gi beauty","gi beauty","gi reconstruction","gi reconstruction"] {
            recorder.compute(Pass::Ao,"GI stage",span,false,Vec::new());
        }
        let recorded=graph.record(true,recorder)?;
        let actual:Vec<_>=recorded.iter().map(|step| {
            let Work::Compute(work)=&step.work else {panic!("compute stage")};
            (work.span,work.begin,work.end)
        }).collect();
        assert_eq!(actual,vec![
            ("gi demand",true,true),("gi pilots",true,false),("gi pilots",false,true),
            ("gi beauty",true,false),("gi beauty",false,true),
            ("gi reconstruction",true,false),("gi reconstruction",false,true),
        ]);
        Ok(())
    }

    #[test]
    fn recording_failure_survives_transient_eviction() -> Result<()> {
        let ledger = std::sync::Arc::new(crate::residency::Ledger::default());
        let transients = Transients {
            intervals: Default::default(),
            state: crate::residency::Cache::new(&ledger),
            submit: Default::default(),
            recording_error: Default::default(),
            publication_checks: Default::default(),
        };
        transients.record(|| Ok(()))?;
        let error = transients.record(|| {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing footage").into())
        }).unwrap_err();
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
        // Reacquiring dropped storage resets publication history, but cannot
        // make partially submitted simulation or slot-private TLAS state valid.
        drop(transients.retain());
        transients.state.set_bytes(1);
        assert_eq!(ledger.evict(1), 1);
        drop(transients.retain());
        let retry = transients.record(|| panic!("failed installation admitted recording")).unwrap_err();
        assert!(retry.to_string().contains("reinstall"));
        assert!(retry.to_string().contains("missing footage"));
        Ok(())
    }
    #[test]
    fn asynchronous_publication_failure_invalidates_all_slots() -> Result<()> {
        let ledger = std::sync::Arc::new(crate::residency::Ledger::default());
        let transients = Transients {
            intervals: Default::default(), state: crate::residency::Cache::new(&ledger),
            submit: Default::default(), recording_error: Default::default(),
            publication_checks: Default::default(),
        };
        // A slot retains this handle even after its transient storage is evicted.
        let slot = transients.recording_error.clone();
        slot.set("medium work budget exhausted in render-image capture".into()).unwrap();
        let error = transients.record(|| panic!("invalid publication was reused")).unwrap_err();
        assert!(error.to_string().contains("reinstall"));
        assert!(error.to_string().contains("render-image capture"));
        Ok(())
    }
    #[test]
    fn standalone_controls_pipeline_covers_every_scheduled_output() {
        use crate::{compositor::Dependencies, plan::{EffectPlan, ScenePlan}};
        // Include composited subsurface/previs (Monster), explicit controls,
        // image-only roots, path and implicit ownership, and capture subsets.
        for flags in 0..256 {
            // Spatial diffusion is a raster provider; path owns its own transport.
            if flags & 1 != 0 && flags & 128 != 0 { continue; }
            let effects = EffectPlan {
                path: flags & 1 != 0, implicit: flags & 2 != 0,
                compositor: flags & 4 != 0, indirect: flags & 8 != 0,
                reflections: flags & 16 != 0, denoise: flags & 32 != 0,
                force_standalone_controls: flags & 64 != 0, subsurface: flags & 128 != 0,
                ..Default::default()
            };
            for bits in 0..64 {
                let outputs = Dependencies {
                    color: true, beauty: bits & 1 != 0, depth: bits & 2 != 0,
                    normals: bits & 4 != 0, images: bits & 8 != 0,
                    aovs: bits & 16 != 0, aov_output: bits & 16 != 0, catcher: bits & 32 != 0,
                    ..Default::default()
                };
                // Installation drops beauty-only providers for control/image roots.
                let effects = if !outputs.beauty && !outputs.catcher {
                    EffectPlan { subsurface: false, indirect: false, reflections: false,
                        denoise: false, ..effects }
                } else { effects };
                let plan = ScenePlan {
                    effects: EffectPlan { outputs: Some(outputs), ..effects },
                    decoupled: !effects.path, exposure_samples: 1, visibility_samples: 1,
                    screen_refraction: false, draws: Vec::new(), casters: Default::default(),
                    transparent: false, deform: false, rays: effects.path, cluster: false,
                    samples: 1, aovs: outputs.aovs,
                };
                let graph = Graph::new(&plan, true)
                    .unwrap_or_else(|error| panic!("effects={flags}, outputs={bits}: {error}"));
                assert!(!graph.runs(Pass::Controls) || effects.standalone_controls(Some(outputs)),
                    "scheduled controls without installation: effects={flags}, outputs={bits}");
                if flags == 132 && bits == 1 {
                    assert!(graph.runs(Pass::Controls), "composited subsurface beauty needs standalone controls");
                    assert!(effects.standalone_controls(Some(outputs)));
                }
            }
        }
    }

    #[test]
    fn decoupled_shading_follows_culling_and_controls() {
        let mut plan = crate::plan::ScenePlan {
            decoupled: true, exposure_samples: 1, visibility_samples: 1, screen_refraction: false,
            effects: Default::default(),
            draws: Vec::new(), casters: Default::default(), transparent: false,
            deform: false, rays: false, cluster: false,
            samples: 1, aovs: false,
        };
        let graph = Graph::new(&plan, true).unwrap();
        let index = |graph: &Graph, pass| graph.order.iter().position(|p| p.pass == pass).unwrap();
        assert!(graph.reads("visibility slots"));
        assert!(index(&graph, Pass::Depth) < index(&graph, Pass::Visibility));
        assert!(index(&graph, Pass::Visibility) < index(&graph, Pass::Shade));
        assert!(index(&graph, Pass::Forward) < index(&graph, Pass::Shade));
        assert!(index(&graph, Pass::Shade) < index(&graph, Pass::Resolve));
        plan.aovs = true;
        let graph = Graph::new(&plan, true).unwrap();
        assert!(index(&graph, Pass::Shade) < index(&graph, Pass::AovResolve));
        plan.screen_refraction = true;
        let graph = Graph::new(&plan, true).unwrap();
        assert!(index(&graph, Pass::Shade) < index(&graph, Pass::Refraction));
        assert!(index(&graph, Pass::Refraction) < index(&graph, Pass::AovResolve));
        assert!(index(&graph, Pass::Refraction) < index(&graph, Pass::Resolve));
        assert!(graph.reads("refraction pyramid") && graph.reads("refraction receivers"));
        plan.screen_refraction = false;
        plan.effects.indirect = true;
        let graph = Graph::new(&plan, true).unwrap();
        assert!(graph.reads("visibility slots"));
        assert!(index(&graph, Pass::Depth) < index(&graph, Pass::Visibility));
        assert!(index(&graph, Pass::Visibility) < index(&graph, Pass::Shade));
        assert!(index(&graph, Pass::Controls) < index(&graph, Pass::Shade));
        assert!(index(&graph, Pass::Shade) < index(&graph, Pass::Ao));
        assert!(index(&graph, Pass::Ao) < index(&graph, Pass::Resolve));
        assert!(graph.reads("decoupled AO receivers"));
        plan.effects.indirect = false;
        plan.effects.reflections = true;
        let graph = Graph::new(&plan, true).unwrap();
        assert!(graph.reads("visibility slots"));
        assert!(index(&graph, Pass::Depth) < index(&graph, Pass::Visibility));
        assert!(index(&graph, Pass::Visibility) < index(&graph, Pass::Shade));
        assert!(index(&graph, Pass::Controls) < index(&graph, Pass::Shade));
        assert!(index(&graph, Pass::Shade) < index(&graph, Pass::Reflections));
        assert!(index(&graph, Pass::Reflections) < index(&graph, Pass::Resolve));
        assert!(graph.reads("reflection guides"));
        assert!(graph.reads("reflection records"));
    }
    #[test]
    fn output_dependencies_prune_scene_producers_and_keep_requested_controls() {
        let mut plan=crate::plan::ScenePlan {
            effects:crate::plan::EffectPlan {compositor:true,outputs:Some(crate::compositor::Dependencies {color:true,..Default::default()}),..Default::default()},
            decoupled:true,exposure_samples:1,visibility_samples:1,screen_refraction:false,
            draws:Vec::new(),casters:Default::default(),transparent:false,
            deform:false,rays:false,cluster:false,samples:1,aovs:false,
        };
        let graph=Graph::new(&plan,true).unwrap();
        assert_eq!(graph.passes(),vec![Pass::Compositor,Pass::Resolve,Pass::Readback]);
        plan.effects.outputs.as_mut().unwrap().depth=true;
        let graph=Graph::new(&plan,true).unwrap();
        assert!(graph.runs(Pass::Depth) && graph.runs(Pass::Controls));
        assert!(!graph.runs(Pass::Visibility) && !graph.runs(Pass::Shade) && !graph.runs(Pass::PostSource));
        plan.effects.outputs.as_mut().unwrap().beauty=true;
        let graph=Graph::new(&plan,true).unwrap();
        assert!(graph.runs(Pass::Visibility) && graph.runs(Pass::Shade) && graph.runs(Pass::PostSource));
        plan.effects.outputs.as_mut().unwrap().depth=false;
        let graph=Graph::new(&plan,true).unwrap();
        assert!(!graph.runs(Pass::Forward));
        assert!(graph.runs(Pass::Images),"material shading can consume captures without a controls pass");
        plan.effects.path=true;plan.effects.compositor=false;plan.decoupled=false;
        let graph=Graph::new(&plan,true).unwrap();
        assert!(graph.runs(Pass::Path));
        assert!(!graph.runs(Pass::Depth) && !graph.runs(Pass::Forward) && !graph.runs(Pass::Controls));
        plan.effects.denoise=true;
        let graph=Graph::new(&plan,true).unwrap();
        assert!(graph.runs(Pass::Denoise));
        assert!(!graph.runs(Pass::Depth) && !graph.runs(Pass::Forward));
        plan.effects.denoise=false;
        plan.effects.outputs=Some(crate::compositor::Dependencies {depth:true,..Default::default()});
        let graph=Graph::new(&plan,true).unwrap();
        assert!(graph.runs(Pass::Depth) && graph.runs(Pass::Resolve));
        assert!(!graph.runs(Pass::Path));
        plan.effects.outputs=Some(crate::compositor::Dependencies {aovs:true,aov_output:true,..Default::default()});
        plan.aovs=true;
        let graph=Graph::new(&plan,true).unwrap();
        assert!(graph.runs(Pass::Path) && graph.runs(Pass::AovResolve));
        assert!(!graph.runs(Pass::Resolve) && !graph.runs(Pass::Forward));
        plan.effects.outputs=Some(Default::default());
        assert!(Graph::new(&plan,false).unwrap().passes().is_empty());
    }

    #[test]
    fn capture_only_output_retains_shared_scene_producers() {
        for path in [false, true] {
          for lighting in [false, true] {
            for feedback in [false, true] {
                let mut plan=crate::plan::ScenePlan {
                    effects:crate::plan::EffectPlan {
                        path, compositor:true, probes:lighting, fibre:lighting, image_feedback:feedback && lighting,
                        outputs:Some(crate::compositor::Dependencies {color:true,images:true,..Default::default()}),
                        ..Default::default()
                    },
                    decoupled:!path, exposure_samples:1, visibility_samples:1, screen_refraction:false,
                    draws:Vec::new(), casters:Default::default(), transparent:false,
                    deform:true, rays:true, cluster:false, samples:1, aovs:false,
                };
                let graph=Graph::new(&plan,true).unwrap();
                let passes=graph.passes();
                let index=|pass|passes.iter().position(|p|*p==pass).expect("capture producer retained");
                assert!(index(Pass::Deform)<index(Pass::Rays));
                assert!(index(Pass::Rays)<index(Pass::Images));
                if lighting {
                    assert!(index(Pass::Probes)<index(Pass::Images));
                    assert!(index(Pass::Fibre)<index(Pass::Images));
                    if feedback {assert!(index(Pass::Rays)<index(Pass::ImageInputs));}
                }
                assert!(index(Pass::Images)<index(Pass::Compositor));
                assert!(!graph.runs(Pass::Path) && !graph.runs(Pass::Shade) && !graph.runs(Pass::Depth),
                    "capture dependencies must not revive unused main-view shading");
                // Removing the capture consumer still prunes all shared work.
                plan.effects.outputs.as_mut().unwrap().images=false;
                assert_eq!(Graph::new(&plan,true).unwrap().passes(),
                    vec![Pass::Compositor,Pass::Resolve,Pass::Readback]);
            }
          }
        }
    }

    #[test]
    fn rejects_read_before_writer() {
        let mut b = Builder::default();
        b.pass(Pass::Forward)
            .reads(&["depth 4x"])
            .writes(&[VIEW_COLOUR]);
        b.pass(Pass::Depth).writes(&["depth 4x"]);
        assert!(Graph::compile(b, &[VIEW_COLOUR]).is_err());
    }
    #[test]
    fn physical_fibre_transport_is_a_live_lighting_dependency() {
        let plan = crate::plan::ScenePlan {
            decoupled: true, exposure_samples: 1, visibility_samples: 1, screen_refraction: false,
            effects: crate::plan::EffectPlan { fibre: true, ..Default::default() },
            draws: Vec::new(), casters: Default::default(), transparent: true,
            deform: false, rays: false, cluster: false,
            samples: 1, aovs: false,
        };
        let graph = Graph::new(&plan, true).unwrap();
        let fibre = graph.order.iter().position(|p| p.pass == Pass::Fibre)
            .expect("lighting must retain its transport producer");
        let shade = graph.order.iter().position(|p| p.pass == Pass::Shade).unwrap();
        assert!(fibre < shade);
    }

    #[test]
    fn probe_volume_update_exists_only_when_enabled_and_precedes_colour() {
        for probes in [false, true] {
          for image_feedback in [false, true] {
            let plan = crate::plan::ScenePlan {
            decoupled: true, exposure_samples: 1, visibility_samples: 1, screen_refraction: false,
                effects: crate::plan::EffectPlan { probes, image_feedback: image_feedback && probes, ..Default::default() },
                draws: Vec::new(), casters: Default::default(),
                transparent: false, deform: false,
                rays: probes, cluster: true, samples: 2, aovs: false,
            };
            let graph = Graph::new(&plan, true).unwrap();
            let index = |p| graph.order.iter().position(|s| s.pass == p);
            assert_eq!(index(Pass::Probes).is_some(), probes);
            if let Some(update) = index(Pass::Probes) {
                assert!(index(Pass::Rays).unwrap() < update && update < index(Pass::Shade).unwrap());
                assert_eq!(graph.order[update].run, Run::Probes);
                if image_feedback {
                    let seed = index(Pass::ImageInputs).expect("same-frame image producer");
                    assert!(seed < update);
                    assert!(update < index(Pass::Images).unwrap());
                } else {
                    assert!(index(Pass::ImageInputs).is_none());
                }
            }
          }
        }
    }
    #[test]
    fn clustered_lists_precede_raster_consumers() {
        for path in [false, true] {
            for cluster in [false, true] {
                for screen_refraction in [false, true] {
                    if path && screen_refraction {continue;}
                    let plan = crate::plan::ScenePlan {
                        decoupled: !path, exposure_samples: 1, visibility_samples: 1, screen_refraction,
                        effects: crate::plan::EffectPlan { path, ..Default::default() },
                        draws: Vec::new(), casters: Default::default(),
                        transparent: screen_refraction, deform: false,
                        rays: path, cluster, samples: 2, aovs: false,
                    };
                    let graph = Graph::new(&plan, true).unwrap();
                    let index = |p| graph.order.iter().position(|s| s.pass == p);
                    assert_eq!(index(Pass::Clusters).is_some(), cluster);
                    if let Some(clusters) = index(Pass::Clusters) {
                        assert!(clusters < index(if path { Pass::Path } else { Pass::Shade }).unwrap());
                        if screen_refraction {assert!(clusters < index(Pass::Refraction).unwrap());}
                        let step = &graph.order[clusters];
                        for _repeat in [false, true] {
                            let bits = RunBits { color: true, final_sample: false,
                                probes: false, fibre: false, images: false, pixels: false };
                            assert!(bits.runs(step.run), "each colour sample needs current camera clusters");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn retained_reads_precede_writers_and_dropped_outputs_are_placeholders() {
        let mut b = Builder::default();
        b.pass(Pass::Compositor).when(false).writes(&[]);
        b.pass(Pass::Resolve)
            .reads(&["linear sample accumulation"])
            .writes(&["linear sample accumulation", VIEW_COLOUR]);
        let graph = Graph::compile(b, &[VIEW_COLOUR]).unwrap();
        assert_eq!(graph.order.len(), 1);
        assert_eq!(graph.order[0].pass, Pass::Resolve);
    }
    #[test]
    fn denoiser_retains_controls_before_sample_accumulation() {
        let plan = crate::plan::ScenePlan {
            decoupled: false, exposure_samples: 1, visibility_samples: 1, screen_refraction: false,
            effects: crate::plan::EffectPlan {
                denoise: true,
                ..Default::default()
            },
            draws: Vec::new(),
            casters: Default::default(),
            transparent: false,
            deform: false,
            rays: false,
            cluster: false,
            samples: 4,
            aovs: false,
        };
        let graph = Graph::new(&plan, true).unwrap();
        let index = |p| graph.order.iter().position(|s| s.pass == p).unwrap();
        assert!(index(Pass::Controls) < index(Pass::PostSource));
        assert!(index(Pass::PostSource) < index(Pass::Denoise));
        assert!(index(Pass::Denoise) < index(Pass::Resolve));
    }
    #[test]
    fn query_controls_retain_acceleration_without_beauty() {
        for trace in [false,true] {
            let plan=crate::plan::ScenePlan {
                decoupled:false,exposure_samples:1,visibility_samples:1,screen_refraction:false,
                effects:crate::plan::EffectPlan {path:true,trace,..Default::default()},
                draws:Vec::new(),casters:Default::default(),transparent:false,
                deform:false,rays:true,cluster:false,samples:1,aovs:false,
            };
            let outputs=crate::compositor::Dependencies {normals:true,..Default::default()};
            let graph=Graph::for_outputs(&plan,true,Some(outputs)).unwrap();
            assert!(!graph.runs(Pass::Path));
            let mut declared=Builder::default();declare(&mut declared,&plan);
            for pass in [Pass::Depth,Pass::Forward,Pass::TransparentControls] {
                let node=declared.0.iter().find(|node|node.pass==pass).unwrap();
                assert_eq!(node.reads.contains(&"ray structures"),trace);
            }
            if trace {
                let index=|pass|graph.order.iter().position(|step|step.pass==pass).unwrap();
                assert!(index(Pass::Rays)<index(Pass::Depth));
                assert!(index(Pass::Rays)<index(Pass::Forward));
            }
        }
    }
    #[test]
    fn passes_prune_by_sinks() {
        let build = || {
            let mut b = Builder::default();
            b.pass(Pass::Rays).writes(&["ray structures"]);
            b.pass(Pass::Forward)
                .reads(&["ray structures"])
                .writes(&["HDR resolved"]);
            // Written but read by nothing that reaches a sink.
            b.pass(Pass::Ao)
                .writes(&["AO filtered visibility depth normal"]);
            b.pass(Pass::Resolve)
                .reads(&["HDR resolved"])
                .writes(&["packed PNG pixels", VIEW_COLOUR]);
            b.pass(Pass::Readback)
                .reads(&["packed PNG pixels"])
                .writes(&[CPU_READBACK]);
            b
        };
        let passes = |g: &Graph| g.order.iter().map(|s| s.pass).collect::<Vec<_>>();
        let all = Graph::compile(build(), &[VIEW_COLOUR, CPU_READBACK]).unwrap();
        use Pass::*;
        assert_eq!(passes(&all), [Rays, Forward, Resolve, Readback]);
        assert!(all.reads("packed PNG pixels"));
        let colour = Graph::compile(build(), &[VIEW_COLOUR]).unwrap();
        assert_eq!(passes(&colour), [Rays, Forward, Resolve]);
        assert!(!colour.reads("packed PNG pixels"));
    }
}
