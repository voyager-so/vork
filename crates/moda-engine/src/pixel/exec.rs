//! The GPU executor for image nodes (motion-model.md §10, stage 5): captures in, pixel nodes run as scheduled
//! passes over domains, images out.
//!
//! The executor is independent of the graph model: a host (`moda-render`'s graph lane) walks the frame's image plan
//! in dependency order, hands the executor the renderer's captures ([`Executor::adopt_capture`]), asks it to run
//! each node ([`Executor::node`]) and drops each image after its last use, so the pool holds only what's live.
//! A node may be several passes (a blur is two, a glow three); temporal nodes (`motion_blur`, `echo`) receive their
//! looks at other times as ordinary inputs, with weights, and fold them four at a time.
//!
//! Lengths in node parameters are frame pixels; the executor multiplies them by the render scale.

use std::collections::BTreeMap;

use super::color::conversion;
use super::image::{ColorTag, Domain, Image, Storage};
use super::kernels;
use super::pass::{Kernel, PassInput, Passes, MAX_INPUTS};
use super::pool::TexturePool;
use super::PixelError;

/// An evaluated node parameter (the host converts its values into these).
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Num(f64),
    Vec2(f64, f64),
    /// Linear-light RGBA, straight alpha (the graph's colour values).
    Color([f32; 4]),
    Bool(bool),
    Str(String),
    /// A flat list of numbers (the `sprites` node's particles, moda-motion `particles::to_floats`).
    Floats(std::sync::Arc<[f32]>),
}

pub type Params = BTreeMap<String, Param>;

/// Parameter access with defaults and type errors that name the node.
pub struct Args<'a> {
    pub node: &'a str,
    pub params: &'a Params,
}

impl Args<'_> {
    pub fn num(&self, k: &str, default: f64) -> Result<f64, PixelError> {
        match self.params.get(k) {
            None => Ok(default),
            Some(Param::Num(v)) if v.is_finite() => Ok(*v),
            Some(p) => Err(self.bad(k, "a finite number", p)),
        }
    }

    /// A number broadcast to both axes, or a vec2.
    pub fn vec2(&self, k: &str, default: (f64, f64)) -> Result<(f64, f64), PixelError> {
        match self.params.get(k) {
            None => Ok(default),
            Some(Param::Num(v)) if v.is_finite() => Ok((*v, *v)),
            Some(Param::Vec2(x, y)) if x.is_finite() && y.is_finite() => Ok((*x, *y)),
            Some(p) => Err(self.bad(k, "a number or a vec2", p)),
        }
    }

    /// A number broadcast to a grey, or a colour's linear RGB (alpha ignored).
    pub fn rgb(&self, k: &str, default: f32) -> Result<[f32; 3], PixelError> {
        match self.params.get(k) {
            None => Ok([default; 3]),
            Some(Param::Num(v)) if v.is_finite() => Ok([*v as f32; 3]),
            Some(Param::Color(c)) => Ok([c[0], c[1], c[2]]),
            Some(p) => Err(self.bad(k, "a number or a colour", p)),
        }
    }

    pub fn color(&self, k: &str) -> Result<Option<[f32; 4]>, PixelError> {
        match self.params.get(k) {
            None => Ok(None),
            Some(Param::Color(c)) => Ok(Some(*c)),
            Some(Param::Num(v)) if v.is_finite() => Ok(Some([*v as f32, *v as f32, *v as f32, 1.0])),
            Some(p) => Err(self.bad(k, "a colour", p)),
        }
    }

    pub fn string<'s>(&'s self, k: &str, default: &'s str) -> Result<&'s str, PixelError> {
        match self.params.get(k) {
            None => Ok(default),
            Some(Param::Str(s)) => Ok(s),
            Some(p) => Err(self.bad(k, "a string", p)),
        }
    }

    /// A bool; 0 / 1 read as false / true.
    pub fn bool(&self, k: &str, default: bool) -> Result<bool, PixelError> {
        match self.params.get(k) {
            None => Ok(default),
            Some(Param::Bool(b)) => Ok(*b),
            Some(Param::Num(v)) if *v == 0.0 || *v == 1.0 => Ok(*v == 1.0),
            Some(p) => Err(self.bad(k, "a bool", p)),
        }
    }

    fn bad(&self, k: &str, want: &str, got: &Param) -> PixelError {
        PixelError::Invalid(format!("`{}.{k}` needs {want}, not {got:?}", self.node))
    }
}

/// Executor counters (`--stats`, the bench).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecStats {
    pub passes: u64,
    pub nodes: u64,
    pub captures: u64,
    pub peak_bytes: u64,
}

/// Runs the pixel graph's nodes on the GPU.
pub struct Executor {
    pub pool: TexturePool,
    pub passes: Passes,
    nodes: std::cell::Cell<u64>,
    captures: std::cell::Cell<u64>,
    /// The largest domain a node may make (the frame grown by half its size on every side): transforms and blurs
    /// past it are cut, so a runaway scale can't allocate a giant texture.
    limit: std::cell::Cell<Domain>,
    frame: std::cell::Cell<(u32, u32)>,
    /// The 3D layers' pipeline, made on the first solid.
    solids: std::cell::OnceCell<crate::solid::SolidPipeline>,
    /// Agent shaders' time budget, ms per 1920×1080 output px.
    shader_budget: std::cell::Cell<f64>,
}

/// Agent shaders' default time budget: 50 ms per 1920×1080 output px (docs/design/effects-code-first.md §4.3).
pub const DEFAULT_SHADER_BUDGET_MS: f64 = 50.0;

/// A convenience for kernels: the input list from images.
fn inputs<'a>(imgs: &[&'a Image]) -> Vec<PassInput<'a>> {
    imgs.iter()
        .map(|i| PassInput {
            texture: &i.color().texture,
            domain: i.domain,
        })
        .collect()
}

impl Executor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, budget_bytes: u64) -> Self {
        Self {
            pool: TexturePool::new(device, budget_bytes),
            passes: Passes::new(device, queue),
            nodes: Default::default(),
            captures: Default::default(),
            limit: std::cell::Cell::new(Domain::EMPTY),
            frame: std::cell::Cell::new((0, 0)),
            solids: std::cell::OnceCell::new(),
            shader_budget: std::cell::Cell::new(super::agent_budget_per_1080p()),
        }
    }

    /// Set the frame this executor works for: node outputs are clipped to it grown by half its size per side.
    pub fn set_frame(&self, w: u32, h: u32) {
        self.frame.set((w, h));
        self.limit.set(Domain::frame(w, h).grow(w / 2 + 1, h / 2 + 1));
    }

    /// The frame's output size ([`Executor::set_frame`]).
    pub fn frame_size(&self) -> (u32, u32) {
        self.frame.get()
    }

    /// Run a WGSL shader node (see [`super::nodes::ShaderCode`]) at `time_s` seconds, within the agent-shader time
    /// budget ([`Executor::set_shader_budget`]).
    pub fn shader(
        &self,
        node: &str,
        code: &super::nodes::ShaderCode,
        params: &Params,
        named: &[(&str, &Image)],
        scale: f64,
        time_s: f64,
    ) -> Result<(Image, super::nodes::ShaderReport), PixelError> {
        self.nodes.set(self.nodes.get() + 1);
        let args = Args { node, params };
        let input = |name: &str| -> Result<&Image, PixelError> {
            named
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, i)| *i)
                .ok_or_else(|| PixelError::Invalid(format!("`{node}` needs `{name}`")))
        };
        super::nodes::shader(self, &args, code, &input, scale, time_s, self.shader_budget.get())
    }

    /// The agent-shader time budget, ms per 1920×1080 output px (default [`super::agent_budget_per_1080p`]).
    pub fn set_shader_budget(&self, per_1080p_ms: f64) {
        self.shader_budget.set(per_1080p_ms);
    }

    pub fn shader_budget(&self) -> f64 {
        self.shader_budget.get()
    }

    /// Run an agent shader module (moda-wgsl) over `out`: tiled, timed against `budget_ms`, non-finite pixels
    /// reported (output px).
    #[allow(clippy::too_many_arguments)]
    pub fn run_agent(
        &self,
        name: &str,
        wgsl: &str,
        ins: &[&Image],
        params: &[[f32; 4]],
        out: Domain,
        tag: ColorTag,
        budget_ms: f64,
    ) -> Result<(Image, Option<(u32, u32)>, f64), PixelError> {
        let out = self.clip(out);
        if out.is_empty() {
            return Ok((self.solid(Domain::new(0, 0, 1, 1), [0.0; 4], tag)?, None, 0.0));
        }
        let run = self.passes.run_agent(&self.pool, name, wgsl, &inputs(ins), params, out, Storage::F16, budget_ms)?;
        Ok((Image::rgba(out, tag, run.texture), run.nan, run.ms))
    }

    /// A magenta checker over `domain` (8 output px squares): what a preview shows for a shader past its budget.
    pub fn magenta(&self, domain: Domain, tag: ColorTag) -> Result<Image, PixelError> {
        let tex = self.passes.run(&self.pool, kernels::magenta(), &[], &[], domain, Storage::F16)?;
        Ok(Image::rgba(domain, tag, tex))
    }

    pub fn stats(&self) -> ExecStats {
        ExecStats {
            passes: self.passes.count(),
            nodes: self.nodes.get(),
            captures: self.captures.get(),
            peak_bytes: self.pool.stats().peak,
        }
    }

    fn clip(&self, d: Domain) -> Domain {
        let l = self.limit.get();
        if l.is_empty() {
            d
        } else {
            d.intersect(&l)
        }
    }

    /// A renderer capture (an `Rgba16Float` texture of the frame's size, premultiplied) as an image covering
    /// `domain`.
    pub fn adopt_capture(&self, texture: wgpu::Texture, domain: Domain, tag: ColorTag) -> Result<Image, PixelError> {
        let t = self.pool.adopt(texture)?;
        if t.size() != (domain.w, domain.h) {
            return Err(PixelError::Invalid("a capture's size must match its domain".into()));
        }
        self.captures.set(self.captures.get() + 1);
        Ok(Image::rgba(domain, tag, t))
    }

    /// One pass producing `rgba`; other groups of the first input pass through,
    /// re-cut to the output domain with their storage and metadata preserved.
    pub fn run(
        &self,
        kernel: Kernel<'_>,
        ins: &[&Image],
        params: &[[f32; 4]],
        out: Domain,
        tag: ColorTag,
    ) -> Result<Image, PixelError> {
        let out = self.clip(out);
        if out.is_empty() {
            return self.solid(Domain::new(0, 0, 1, 1), [0.0; 4], tag);
        }
        let tex = self.passes.run(&self.pool, kernel, &inputs(ins), params, out, Storage::F16)?;
        let mut img = Image::rgba(out, tag, tex);
        if let Some(first) = ins.first() {
            for group in first.groups.iter().filter(|g| g.name != super::RGBA) {
                let mut group = group.clone();
                if first.domain != out {
                    group.texture = self.passes.run(
                        &self.pool, kernels::copy(),
                        &[PassInput { texture: &group.texture, domain: first.domain }],
                        &[], out, group.storage(),
                    )?;
                }
                img = img.with_group(group);
            }
        }
        Ok(img)
    }

    /// A constant image (premultiplied `rgba`).
    pub fn solid(&self, domain: Domain, rgba: [f32; 4], tag: ColorTag) -> Result<Image, PixelError> {
        let tex = self
            .passes
            .run(&self.pool, kernels::solid(), &[], &[rgba], domain, Storage::F16)?;
        Ok(Image::rgba(domain, tag, tex))
    }

    /// `img` re-cut to `domain` (transparent where it had no pixels).
    pub fn crop(&self, img: &Image, domain: Domain) -> Result<Image, PixelError> {
        if img.domain == domain {
            return Ok(img.clone());
        }
        let tex = self
            .passes
            .run(&self.pool, kernels::copy(), &inputs(&[img]), &[], domain, Storage::F16)?;
        Ok(Image::rgba(domain, img.tag(), tex))
    }

    pub fn convert(&self, img: &Image, to: ColorTag) -> Result<Image, PixelError> {
        kernels::convert_image(&self.passes, &self.pool, img, to)
    }

    /// `img` seen through a plane homography (a 2.5D layer): `inverse` maps an output frame pixel `(x, y, 1)` to the
    /// input's pixel (homogeneous, third component 1 / depth), over `out`. Pixels whose ray meets the plane behind
    /// the camera stay transparent.
    pub fn project(&self, img: &Image, inverse: [[f64; 3]; 3], out: Domain) -> Result<Image, PixelError> {
        self.nodes.set(self.nodes.get() + 1);
        let row = |r: [f64; 3]| [r[0] as f32, r[1] as f32, r[2] as f32, 0.0];
        let params = [row(inverse[0]), row(inverse[1]), row(inverse[2])];
        self.run(kernels::project(), &[img], &params, out, img.tag())
    }

    /// Solids (3D layers, `crate::solid`) drawn together over `out`, tagged `srgb`.
    pub fn draw_solids(&self, scene: &crate::solid::SolidScene<'_>, out: Domain) -> Result<Image, PixelError> {
        self.nodes.set(self.nodes.get() + 1);
        let out = self.clip(out);
        if out.is_empty() {
            return self.draw_solids(scene, Domain::new(0, 0, 1, 1));
        }
        if self.solids.get().is_none() {
            let p = crate::solid::SolidPipeline::new(self.passes.device(), self.passes.queue())?;
            let _ = self.solids.set(p);
        }
        let tex = self.solids.get().expect("made above").draw(&self.pool, scene, out)?;
        Ok(Image::rgba(out, ColorTag::Srgb, tex))
    }

    /// Weighted looks of one image (newest first), combined by `mode` (`add`, `over`, `max`), four inputs a pass.
    pub fn accumulate(&self, looks: &[&Image], weights: &[f64], mode: &str) -> Result<Image, PixelError> {
        if looks.is_empty() || looks.len() != weights.len() {
            return Err(PixelError::Invalid("accumulate: one weight per look".into()));
        }
        let m = match mode {
            "add" => 0.0,
            "over" => 1.0,
            "max" => 2.0,
            other => return Err(PixelError::Invalid(format!("unknown accumulate mode `{other}`"))),
        };
        let tag = looks[0].tag();
        let domain = looks.iter().fold(Domain::EMPTY, |d, i| d.union(&i.domain));
        let mut acc: Option<Image> = None;
        let mut k = 0;
        while k < looks.len() {
            let room = if acc.is_some() { MAX_INPUTS - 1 } else { MAX_INPUTS };
            let take = room.min(looks.len() - k);
            let mut ins: Vec<&Image> = Vec::new();
            let mut w = [0.0f32; 4];
            if let Some(a) = &acc {
                ins.push(a);
                w[0] = 1.0;
            }
            for j in 0..take {
                w[ins.len()] = weights[k + j] as f32;
                ins.push(looks[k + j]);
            }
            let n = ins.len() as f32;
            let next = self.run(kernels::accumulate(), &ins, &[[m, n, 0.0, 0.0], w], domain, tag)?;
            acc = Some(next);
            k += take;
        }
        Ok(acc.expect("at least one look"))
    }

    /// Run one pixel node. `inputs` are named as the node's image parameters (`src`, `matte`, `a`, `b`, …; a
    /// temporal node's looks are `src0`, `src1`, …, with `weights`), already in the tag the node works in.
    pub fn node(
        &self,
        node: &str,
        ty: &str,
        params: &Params,
        named: &[(&str, &Image)],
        weights: &[f64],
        scale: f64,
    ) -> Result<Image, PixelError> {
        self.nodes.set(self.nodes.get() + 1);
        let args = Args { node, params };
        let input = |name: &str| -> Result<&Image, PixelError> {
            named
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, i)| *i)
                .ok_or_else(|| PixelError::Invalid(format!("`{node}` needs `{name}`")))
        };
        match ty {
            "convert" => {
                let to = args.string("to", "linear-srgb")?;
                let tag = ColorTag::from_name(to)
                    .ok_or_else(|| PixelError::Invalid(format!("`{node}.to`: no colour tag `{to}`")))?;
                self.convert(input("src")?, tag)
            }
            "channels" => {
                let src = input("src")?;
                let pick = |k: &str, d: &str| -> Result<f32, PixelError> {
                    Ok(match args.string(k, d)? {
                        "r" => 0.0,
                        "g" => 1.0,
                        "b" => 2.0,
                        "a" => 3.0,
                        "0" => 4.0,
                        "1" => 5.0,
                        "luma" => 6.0,
                        other => {
                            return Err(PixelError::Invalid(format!("`{node}.{k}`: no channel `{other}`")))
                        }
                    })
                };
                let p = [pick("r", "r")?, pick("g", "g")?, pick("b", "b")?, pick("a", "a")?];
                self.run(kernels::channels(), &[src], &[p], src.domain, src.tag())
            }
            "motion_blur" | "echo" => {
                let looks: Vec<&Image> = (0..weights.len())
                    .map(|k| input(&format!("src{k}")))
                    .collect::<Result<_, _>>()?;
                let mode = if ty == "motion_blur" { "add" } else { args.string("mode", "over")? };
                self.accumulate(&looks, weights, mode)
            }
            other => super::nodes::run(self, &args, other, &input, scale),
        }
    }

    /// Read an image back over `domain` as 8-bit RGBA: flattened over `background` (premultiplied) when given,
    /// otherwise un-premultiplied (a transparent export). Values are the image's encoded ones (tag them `srgb`
    /// first), rounded to the nearest 8-bit step.
    pub fn read_rgba8(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>) -> anyhow::Result<Vec<u8>> {
        self.read_u8(img, domain, background, background.is_none())
    }

    /// The image over `domain` as premultiplied 8-bit RGBA, as stored (the asset store's image format).
    pub fn read_rgba8_premultiplied(&self, img: &Image, domain: Domain) -> anyhow::Result<Vec<u8>> {
        self.read_u8(img, domain, None, false)
    }

    /// The image as premultiplied 8-bit RGBA in a pool texture over its own domain, quantized on the GPU to the bytes
    /// [`Executor::read_rgba8_premultiplied`] reads back (clamped, rounded half up as `f32::round` does on
    /// non-negative values; the `Rgba8Unorm` target's own conversion would round ties to even). A graph-pixel unit
    /// stays on the GPU this way instead of a readback and an upload.
    pub fn quantize_rgba8_premultiplied(&self, img: &Image) -> Result<super::pool::PoolTexture, PixelError> {
        const BODY: &str = "fn kernel(p: vec2<f32>) -> vec4<f32> {
    let scaled = clamp(here(0u, p), vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0;
    let base = floor(scaled);
    return (base + select(vec4<f32>(0.0), vec4<f32>(1.0), scaled - base >= vec4<f32>(0.5))) / 255.0;
}";
        self.passes.run(&self.pool, Kernel { name: "quantize_rgba8", body: BODY }, &inputs(&[img]), &[], img.domain, Storage::U8)
    }

    /// [`Executor::read_f32`] quantized to 8 bits as [`quantize_rgba`] does, row by row out of the mapped staging
    /// buffer (no `f32` image in between).
    fn read_u8(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>, straight: bool) -> anyhow::Result<Vec<u8>> {
        let read = self.read_u8_start(img, domain, background, straight)?;
        self.read_finish(read)
    }

    /// [`Executor::read_rgba8`] started: the image is flattened and its copy submitted, and the GPU goes on with what
    /// is submitted after it; [`Executor::read_finish`] waits for the copy alone and quantizes it.
    pub fn read_rgba8_start(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>) -> anyhow::Result<PendingRead> {
        self.read_u8_start(img, domain, background, background.is_none())
    }

    fn read_u8_start(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>, straight: bool) -> anyhow::Result<PendingRead> {
        let cut = self.flattened(img, domain, background)?;
        let download = if straight {
            self.passes.download_start(&cut.color().texture)?
        } else {
            // Premultiplied (an opaque frame, flattened): quantized on the GPU to the same bytes, in a pass of its own
            // after the flatten (the f32 values are the ones the CPU would read), so a quarter of the bytes come back
            // (8 MB a 1080p frame, not 33) and the CPU has nothing to convert. Straight colour divides by alpha, which
            // a GPU needn't round as the CPU does: that stays on the CPU (`read_finish`).
            let q = self.quantize_rgba8_premultiplied(&cut).map_err(anyhow::Error::new)?;
            self.passes.download_start(&q)?
        };
        Ok(PendingRead { download, w: domain.w as usize, h: domain.h as usize, straight, deep: false, gpu_quantized: !straight })
    }

    /// As [`Executor::read_rgba8_start`], at 16 bits a channel (16-bit PNG, ProRes from float): the `f32` texels come
    /// back and [`Executor::read_finish`] quantizes them to `u16`s in the machine's byte order.
    pub fn read_rgba16_start(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>) -> anyhow::Result<PendingRead> {
        let cut = self.flattened(img, domain, background)?;
        let download = self.passes.download_start(&cut.color().texture)?;
        Ok(PendingRead { download, w: domain.w as usize, h: domain.h as usize, straight: background.is_none(), deep: true, gpu_quantized: false })
    }

    /// The bytes of a [`Executor::read_rgba8_start`] (RGBA, a byte a channel) or [`Executor::read_rgba16_start`]
    /// (RGBA, a `u16` a channel in the machine's byte order) readback.
    pub fn read_finish(&self, read: PendingRead) -> anyhow::Result<Vec<u8>> {
        let PendingRead { download, w, h, straight, deep, gpu_quantized } = read;
        // Bytes a texel takes in the output.
        let texel = if deep { 8 } else { 4 };
        let mut out = vec![0u8; w * h * texel];
        if gpu_quantized {
            // Already 8-bit (quantized on the GPU in `read_u8_start`).
            self.passes.download_finish(download, |data, stride| {
                for (y, dst) in out.chunks_exact_mut(w * 4).enumerate() {
                    dst.copy_from_slice(&data[y * stride..y * stride + w * 4]);
                }
            })?;
            return Ok(out);
        }
        // Quantized straight out of the mapped `f32` texels into the output, on several threads, a band of rows each
        // (a 1080p frame is 33 MB of f32 texels, a 4K one 133 MB; one thread took ~5 ms of a diorama's ~40 a frame
        // per readback).
        let max = if deep { 65535.0 } else { 255.0 };
        let quantize_rows = |data: &[u8], stride: usize, y0: usize, dst: &mut [u8]| {
            let mut px = [0.0f32; 4];
            for (k, out_row) in dst.chunks_exact_mut(w * texel).enumerate() {
                let row = &data[(y0 + k) * stride..(y0 + k) * stride + w * 16];
                for (src, o) in row.chunks_exact(16).zip(out_row.chunks_exact_mut(texel)) {
                    for (c, b) in px.iter_mut().zip(src.chunks_exact(4)) {
                        *c = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    }
                    let q = quantize_texel(px, straight, max);
                    if deep {
                        for (d, v) in o.chunks_exact_mut(2).zip(q) {
                            d.copy_from_slice(&(v as u16).to_ne_bytes());
                        }
                    } else {
                        for (d, v) in o.iter_mut().zip(q) {
                            *d = v as u8;
                        }
                    }
                }
            }
        };
        self.passes.download_finish(download, |data, stride| {
            let threads = if w * h >= 256 * 256 { READBACK_THREADS } else { 1 };
            let band = h.div_ceil(threads).max(1);
            std::thread::scope(|s| {
                for (b, dst) in out.chunks_mut(band * w * texel).enumerate() {
                    let quantize_rows = &quantize_rows;
                    s.spawn(move || quantize_rows(data, stride, b * band, dst));
                }
            });
        })?;
        Ok(out)
    }

    /// As [`Executor::read_rgba8`], at 16 bits a channel (16-bit PNG, ProRes from float).
    pub fn read_rgba16(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>) -> anyhow::Result<Vec<u16>> {
        let bytes = self.read_finish(self.read_rgba16_start(img, domain, background)?)?;
        Ok(bytes.chunks_exact(2).map(|b| u16::from_ne_bytes([b[0], b[1]])).collect())
    }

    /// The image over `domain` as premultiplied RGBA `f32`, flattened over `background` when given.
    pub fn read_f32(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>) -> anyhow::Result<Vec<f32>> {
        let cut = self.flattened(img, domain, background)?;
        self.passes.download(&cut.color().texture)
    }

    /// The image cut to `domain` as `f32` texels, flattened over `background` when given.
    fn flattened(&self, img: &Image, domain: Domain, background: Option<[f32; 4]>) -> anyhow::Result<Image> {
        Ok(match background {
            Some(bg) => {
                let tex = self.passes.run(
                    &self.pool,
                    kernels::flatten(),
                    &inputs(&[img]),
                    &[bg],
                    domain,
                    Storage::F32,
                )?;
                Image::rgba(domain, img.tag(), tex)
            }
            None => {
                let tex =
                    self.passes
                        .run(&self.pool, kernels::copy(), &inputs(&[img]), &[], domain, Storage::F32)?;
                Image::rgba(domain, img.tag(), tex)
            }
        })
    }
}

/// A readback in flight ([`Executor::read_rgba8_start`], [`Executor::read_rgba16_start`]).
pub struct PendingRead {
    download: super::pass::Download,
    w: usize,
    h: usize,
    straight: bool,
    /// 16 bits a channel (else 8).
    deep: bool,
    /// The texels are 8-bit already (quantized on the GPU), else `f32`.
    gpu_quantized: bool,
}

/// Threads a readback quantizes on ([`Executor::read_rgba8`], [`Executor::read_rgba8_premultiplied`]).
const READBACK_THREADS: usize = 4;

/// Premultiplied float RGBA → integer codes at `max` (255 or 65535), un-premultiplying when `straight`.
pub fn quantize_rgba(px: &[f32], straight: bool, max: f32) -> Vec<u32> {
    let mut out = Vec::with_capacity(px.len());
    for p in px.chunks_exact(4) {
        out.extend(quantize_texel([p[0], p[1], p[2], p[3]], straight, max));
    }
    out
}

/// One texel of [`quantize_rgba`].
#[inline]
pub fn quantize_texel(p: [f32; 4], straight: bool, max: f32) -> [u32; 4] {
    let q = |v: f32| (v.clamp(0.0, 1.0) * max).round() as u32;
    let a = p[3].clamp(0.0, 1.0);
    if straight {
        if a <= 0.0 {
            [0, 0, 0, 0]
        } else {
            [q(p[0] / a), q(p[1] / a), q(p[2] / a), q(a)]
        }
    } else {
        [q(p[0]), q(p[1]), q(p[2]), q(a)]
    }
}

/// A colour's tag-to-tag conversion exists (colour tags only).
pub fn convertible(from: ColorTag, to: ColorTag) -> bool {
    conversion(from, to).is_some()
}

#[cfg(test)]
mod channel_domain_tests {
    use super::*;
    use crate::pixel::{AlphaMode, ChannelGroup};

    #[test]
    #[ignore = "requires a real GPU"]
    fn audit_pass_through_groups_follow_the_output_domain() {
        let gpu = crate::GpuContext::new(crate::BackendChoice::Auto).expect("GPU adapter");
        let exec = Executor::new(&gpu.device, &gpu.queue, 16 * 1024 * 1024);
        let domain = Domain::new(7, 11, 10, 8);
        let mut img = exec.solid(domain, [1.0; 4], ColorTag::Srgb).unwrap();
        let texture = exec.passes.run(&exec.pool, kernels::solid(), &[], &[[0.1234567, 0.0, 0.0, 1.0]], domain, Storage::F32).unwrap();
        img = img.with_group(ChannelGroup {
            name: "depth".into(), channels: vec!["z".into()], tag: ColorTag::Data,
            alpha: AlphaMode::Opaque, texture,
        });
        for out in [domain.grow(2, 3), Domain::new(9, 12, 4, 3), Domain::new(8, 10, 10, 8)] {
            let result = exec.run(kernels::copy(), &[&img], &[], out, img.tag()).unwrap();
            let depth = result.require("depth").unwrap();
            assert_eq!(depth.texture.size(), (out.w, out.h));
            assert_eq!(depth.storage(), Storage::F32);
            assert_eq!(depth.tag, ColorTag::Data);
            assert_eq!(depth.alpha, AlphaMode::Opaque);
            assert_eq!(depth.channels, ["z"]);
            let pixels = exec.passes.download(&depth.texture).unwrap();
            for y in 0..out.h {
                for x in 0..out.w {
                    let expected = if domain.contains(out.x + x as i32, out.y + y as i32) { 0.1234567 } else { 0.0 };
                    assert_eq!(pixels[((y * out.w + x) * 4) as usize], expected, "at {x},{y} in {out:?}");
                }
            }
        }
        let same = exec.run(kernels::copy(), &[&img], &[], domain, img.tag()).unwrap();
        assert!(same.require("depth").unwrap().texture.same(&img.require("depth").unwrap().texture));
    }
}
