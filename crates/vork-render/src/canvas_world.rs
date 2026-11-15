//! Pictures crossing between a 2D canvas and a 3D world (END-STATE-API §7), on the GPU: neither side's
//! pixels pass through the CPU.
//!
//! - A world inside a canvas (`composition: {space: "3d"}` on a canvas node) is its own scene installed on the
//!   canvas's device. Each frame the 3D renderer draws it at the world node's local time; its viewed picture,
//!   floating point (the view's code values for sRGB, with the world's output dither, unquantized),
//!   becomes a texture of Moda's pixel pool, registered as a GPU capture (`moda_engine::external_texture`), and
//!   the world node is a rectangle the size of the canvas's box filled with that image. The crossing converts
//!   it to the encoding the canvas blends in, as Moda's engine converts a decoded image it uploads.
//! - A canvas inside a world (`composition: {space: "2d"}` on a node of a 3D world) is seen through a picture of
//!   it: an image whose `render.root` names the node. Each frame Moda's engine draws the canvas at the node's local
//!   time and its picture after its view, sRGB code values whatever encoding the canvas blends in
//!   (`canvas::Pictures::reader_stage`), is written into the image's texture, scene-linear with associated
//!   alpha as a render image holds it, which materials then sample.
//!
//! Pictures cross between compositions as sRGB code values, floating point where they hold more than eight
//! bits: a composition's working encoding is its own, and its view is the one thing that reads it.
use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Which way a picture crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Crossing {
    /// A world's viewed picture (straight alpha, sRGB code values) into the canvas: the encoding the canvas
    /// blends in, associated alpha, the form of Moda's image paints.
    ToCanvas,
    /// An 8-bit canvas's frame (associated alpha, sRGB-encoded) into the world: scene-linear associated alpha.
    ToWorld,
    /// A canvas's float picture (a float canvas's, or any canvas's through its reader's view stage) into the
    /// world: as `ToWorld`, keeping values beyond the display range.
    ToWorldFloat,
}

/// The crossing shader. `{canvas}` is the straight colour a picture enters a canvas with, an expression of the
/// sRGB code values `c.rgb`: those values themselves for an sRGB canvas, else `to_canvas_encoding(c.rgb)`, a
/// function declared before the shader (`Convert::into_canvas`).
const SHADER: &str = "@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn to_canvas(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(source, vec2<i32>(at.xy), 0);
    return vec4<f32>({canvas} * c.a, c.a);
}
@fragment fn to_world(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(source, vec2<i32>(at.xy), 0);
    if (c.a <= 0.0) { return vec4<f32>(0.0); }
    let e = min(c.rgb / c.a, vec3<f32>(1.0));
    let linear = select(pow((e + 0.055) / 1.055, vec3<f32>(2.4)), e / 12.92, e <= vec3<f32>(0.04045));
    return vec4<f32>(linear * c.a, c.a);
}
@fragment fn to_world_float(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(source, vec2<i32>(at.xy), 0);
    if (c.a <= 0.0) { return vec4<f32>(0.0); }
    // Code values outside [0, 1] decode by the extended transfer: linear below zero, the power curve above one.
    let e = c.rgb / c.a;
    let linear = select(pow(max((e + 0.055) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4)), e / 12.92, e <= vec3<f32>(0.04045));
    // RGBA16F holds magnitudes up to 65504: a brighter component saturates there instead of becoming infinite.
    return vec4<f32>(clamp(linear * c.a, vec3<f32>(-65504.0), vec3<f32>(65504.0)), c.a);
}";

/// The crossing passes: one full-frame draw from a source texture into a target of the same size.
#[derive(Default)]
pub(crate) struct Convert {
    pipelines: RefCell<HashMap<(Crossing, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    /// The WGSL function `to_canvas_encoding`, sRGB code values as the encoding of the canvas the pictures
    /// enter; `None` for an sRGB canvas, which takes them as they are.
    canvas: Option<String>,
}
impl Convert {
    /// The crossings of a canvas that blends in `working`: a world's picture, sRGB code values like every
    /// picture that crosses, enters the canvas in that encoding, so Moda filters and blends it there (the
    /// conversion Moda's engine gives a decoded image when it uploads it; `canvas_draw::working_color`).
    pub(crate) fn into_canvas(working: vork_scene::v2::WorkingColorSpace) -> Self {
        Self { pipelines: Default::default(), canvas: crate::canvas_draw::working_color(working).map(|working| working.from_srgb.wgsl("to_canvas_encoding")) }
    }
    pub(crate) fn encode(&self, renderer: &crate::Renderer, encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture, target: &wgpu::TextureView,
        format: wgpu::TextureFormat, crossing: Crossing) {
        let device = &renderer.device;
        let mut pipelines = self.pipelines.borrow_mut();
        let pipeline = pipelines.entry((crossing, format)).or_insert_with(|| {
            let source = match &self.canvas {
                Some(function) => format!("{function}{}", SHADER.replace("{canvas}", "to_canvas_encoding(c.rgb)")),
                None => SHADER.replace("{canvas}", "c.rgb"),
            };
            let module = crate::timing::shader_module(device, wgpu::ShaderModuleDescriptor { label: Some("picture crossing"), source: wgpu::ShaderSource::Wgsl(source.into()) });
            // Through the persistent pipeline cache; the compiler chooses `cache`.
            renderer.compiler.render_cached(&wgpu::RenderPipelineDescriptor {
                label: Some("picture crossing"),
                layout: None,
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(match crossing { Crossing::ToCanvas => "to_canvas", Crossing::ToWorld => "to_world", Crossing::ToWorldFloat => "to_world_float" }),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        });
        let view = source.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("picture crossing"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("picture crossing"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A world inside a canvas: its scene installed on the canvas's device, and the frame slot it draws through.
pub(crate) struct World {
    scene: crate::InstalledScene,
    slot: crate::FrameSlot,
    pub size: [u32; 2],
    /// The viewed picture the canvas reads, and the pass that copies it out of the frame's staging planes.
    picture: wgpu::Texture,
    copy: wgpu::ComputePipeline,
}
impl World {
    pub(crate) fn install(renderer: &crate::Renderer, scene: vork_scene::Scene, base: &Path) -> Result<Self> {
        let size = scene.render.resolution;
        // The world's view keeps its result floating point for the canvas (`display_program`).
        let scene = renderer.install_with_options(scene, base, crate::InstallOptions { readback: Default::default(), view_picture: true, ..Default::default() })?;
        let slot = scene.create_frame_slot(renderer, Default::default())?;
        let picture = crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
            label: Some("floating world view"), size: wgpu::Extent3d {width:size[0],height:size[1],depth_or_array_layers:1},
            mip_level_count:1, sample_count:1, dimension:wgpu::TextureDimension::D2, format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, view_formats:&[],
        });
        let module = crate::timing::shader_module(&renderer.device, wgpu::ShaderModuleDescriptor {
            label:Some("floating world view"), source:wgpu::ShaderSource::Wgsl(r#"
@group(0) @binding(0) var<storage,read> pixels:array<u32>;
@group(0) @binding(1) var picture_output:texture_storage_2d<rgba16float,write>;
@group(0) @binding(2) var<uniform> info:vec4<u32>;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) p:vec3<u32>) {
    if(any(p.xy>=info.yz)){return;} let count=info.y*info.z;let i=p.y*info.y+p.x;
    // RGBA16F holds magnitudes up to 65504: a brighter component saturates there instead of becoming infinite.
    textureStore(picture_output,p.xy,clamp(vec4(bitcast<f32>(pixels[info.x*count+i]),bitcast<f32>(pixels[(info.x+1u)*count+i]),bitcast<f32>(pixels[(info.x+2u)*count+i]),bitcast<f32>(pixels[(info.x+3u)*count+i])),vec4(-65504.),vec4(65504.)));
}"#.into()),
        });
        // Through the persistent pipeline cache; the compiler chooses `cache`.
        let copy = renderer.compiler.compute_cached(&wgpu::ComputePipelineDescriptor {
            label:Some("floating world view"), layout:None,module:&module,entry_point:Some("main"),compilation_options:Default::default(),cache:None,
        });
        Ok(Self { scene, slot, size, picture, copy })
    }
    /// The world's viewed picture at `request`, its node's local time: straight alpha, the view's code values
    /// for sRGB with the world's output dither, unquantized and unclamped (RGBA16F), left on the GPU.
    pub(crate) fn draw(&mut self, renderer: &crate::Renderer, request: &vork_time::Request) -> Result<&wgpu::Texture> {
        loop {
            let drawn = self.slot.submit_outputs(&self.scene, renderer, request.clone(), crate::ReadbackOutputs::default())
                .and_then(|()| self.slot.collect(renderer).map(|_| ())).map_err(crate::failure::device);
            match drawn {
                Ok(()) => break,
                Err(error) => {
                    let Some(redo) = error.downcast_ref::<crate::FrameRedo>() else { return Err(error) };
                    let demand = redo.0;
                    self.scene.redo_frame(renderer, std::slice::from_mut(&mut self.slot), demand, 0).map_err(crate::failure::device)?;
                }
            }
        }
        let params = crate::init(&renderer.device,"world picture planes",bytemuck::cast_slice(&[
            self.slot.view.post.display_base.expect("world view staging"),self.size[0],self.size[1],0u32]),wgpu::BufferUsages::UNIFORM);
        let target=self.picture.create_view(&Default::default());
        let group=renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("world picture"),layout:&self.copy.get_bind_group_layout(0),entries:&[
            crate::entry(0,self.slot.view.post.pixels.as_entire_binding()),crate::entry(1,wgpu::BindingResource::TextureView(&target)),crate::entry(2,params.as_entire_binding()),
        ]});
        let mut encoder=renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label:Some("world picture")});
        {let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());pass.set_pipeline(&self.copy);pass.set_bind_group(0,&group,&[]);pass.dispatch_workgroups(self.size[0].div_ceil(8),self.size[1].div_ceil(8),1);}
        renderer.queue.submit([encoder.finish()]);
        Ok(&self.picture)
    }
}

/// A canvas seen in a world: the canvas's own document, drawn by Moda's engine on the thread that renders the
/// world. The document holds Moda's text and engine state, which stay on that thread.
#[derive(Clone)]
pub(crate) struct Picture {
    id: u64,
    node: String,
    scene: Arc<vork_scene::Scene>,
    base: Arc<PathBuf>,
    _held: Arc<Held>,
}
/// Releases a picture's thread state when the last copy of the picture goes.
struct Held(u64);
impl Drop for Held {
    fn drop(&mut self) {
        let removed = STATES.try_with(|states| states.borrow_mut().remove(&self.0));
        drop(removed);
    }
}
struct State {
    document: crate::canvas::Document,
    engine: Option<crate::canvas_draw::Engine>,
    convert: Convert,
    worlds: Option<crate::canvas::Pictures>,
}
thread_local! {
    static STATES: RefCell<HashMap<u64, State>> = RefCell::new(HashMap::new());
}
impl Picture {
    /// The picture of canvas node `node`, whose document is `scene`. Its text lays out now, so a canvas that
    /// cannot be drawn is refused at install.
    pub(crate) fn new(renderer: &crate::Renderer, node: &str, scene: vork_scene::Scene, base: &Path) -> Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let picture = Self { id, node: node.to_string(), scene: Arc::new(scene), base: Arc::new(base.to_path_buf()), _held: Arc::new(Held(id)) };
        picture.with_state(|state| {
            state.worlds = Some(crate::canvas::Pictures::new(renderer, &state.document, base, true)?);
            Ok(())
        })?;
        Ok(picture)
    }
    /// The canvas node's name.
    pub(crate) fn node(&self) -> &str { &self.node }
    /// The picture's size in px.
    pub(crate) fn size(&self) -> [u32; 2] { self.scene.render.resolution }
    fn with_state<T>(&self, f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        // Release the registry borrow before entering another composition: nested worlds can
        // install and draw their own canvas pictures on this same thread.
        let state = STATES.with(|states| states.borrow_mut().remove(&self.id));
        let mut state = match state {
            Some(state) => state,
            None => {
                let document = crate::canvas::Document::new(&self.scene, &self.base)?;
                document.install_images()?;
                State { document, engine: None, convert: Convert::default(), worlds: None }
            }
        };
        let result = f(&mut state);
        STATES.with(|states| states.borrow_mut().insert(self.id, state));
        result
    }
    /// Draw the canvas at `request` (its node's local time) and write the frame into `target`, a texture of the
    /// picture's size, as scene-linear associated alpha. Moda submits its drawing before the caller's encoder.
    /// `errors`: the owning frame's shader diagnostics, where this picture's view reports a non-finite result.
    pub(crate) fn encode(&self, renderer: &crate::Renderer, encoder: &mut wgpu::CommandEncoder, request: &vork_time::Request, target: &wgpu::TextureView,
        errors: Option<&wgpu::Buffer>) -> Result<()> {
        self.with_state(|state| {
            if state.engine.is_none() {
                state.engine = Some(crate::canvas_draw::Engine::new_picture(renderer, state.document.float(), state.document.working())?);
            }
            if state.worlds.is_none() {
                state.worlds = Some(crate::canvas::Pictures::new(renderer, &state.document, &self.base, true)?);
            }
            let engine = state.engine.as_ref().expect("made");
            let drawn = state.document.draw(engine, state.worlds.as_ref().expect("installed"), renderer, request, errors)?;
            let crossing = if drawn.format() == wgpu::TextureFormat::Rgba16Float { Crossing::ToWorldFloat } else { Crossing::ToWorld };
            state.convert.encode(renderer, encoder, &drawn, target, wgpu::TextureFormat::Rgba16Float, crossing);
            Ok(())
        })
    }
}
