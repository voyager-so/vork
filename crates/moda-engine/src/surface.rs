//! Paint-layer surfaces (raster spike, docs/design/raster-layers.md §6.1–6.2).
//!
//! A paint layer hydrates as a rectangle whose one image fill has the src `paint-surface:<w>x<h>:<id>`
//! (`moda_scene::normalize`). That src resolves here to an [`ExternalTextureRef`]-backed pattern texture, so the
//! layer is ONE ordinary pattern draw item whose bytes never depend on the pixels (rule 2): a stroke changes the
//! texture's contents and bumps the layer's **pixel epoch**, never the scene or the asset-store generation, so no
//! plan is rebuilt by painting.
//!
//! The pixels live outside the engine (rule 1). This registry only remembers, per layer:
//! - the GPU textures the pattern caches created for it (one per engine / device), so uploads write straight into
//!   them and every retained bind group and recorded pass keeps sampling the same texture object;
//! - on native builds, the committed tiles as `Arc`s (`retain_cpu`), so a texture created later (a fresh engine,
//!   an export, a re-create) fills itself. The wasm host turns that off: a new texture is marked
//!   [`take_needs_refill`] and the host pulls the tiles from the paint side again.
//!
//! Caches that show pixels listen to [`epoch_total`]: the motion-cache content key and the capture keys fold it.
//!
//! **Mips** (design §6.3). A surface texture has a full mip chain (unless [`set_mips_enabled`]`(false)`, the
//! `?paintMips=off` kill-switch, which gives new textures level 0 only, as the spike did). An upload writes level 0
//! and records the tile's rect ([`take_dirty`]); the pattern cache that owns the texture regenerates, at the start of
//! every engine render (`PatternTextureCache::flush_surface_mips`) and when it creates the texture, only the mip
//! texels under those rects, level by level (level k covers the rect `>> k`), with a small render pass per level:
//! an exact premultiplied 2×2 box filter on the stored bytes (`(a + b + c + d + 2) / 4`), so an incremental regen
//! gives the same bytes as a full one. The texture object and its views never change, so recorded passes keep
//! sampling it. No pixels enter the engine heap.
//!
//! **Budget** (design §6.4). Surface textures are counted with their mips against [`budget_bytes`] (1 GiB by
//! default), separately from other pattern textures; the retained pattern cache evicts least-recently-drawn
//! surfaces past it, never one drawn in the current render. An evicted layer's texture is re-created when it next
//! draws: filled from the retained tiles (native) or blank and named by [`take_needs_refill`] (wasm).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::paint::image_source::{
    DecodedImage, ExternalTextureRef, PatternTexture, PatternTextureState,
};

pub use moda_scene::normalize::{parse_paint_surface_src, PAINT_SURFACE_SRC_PREFIX};

/// Tile side in pixels (straight-alpha encoded-sRGB RGBA8 tiles of 256×256).
pub const TILE: u32 = 256;
/// Bytes of one full tile.
pub const TILE_BYTES: usize = (TILE * TILE * 4) as usize;

/// Default GPU budget for surface textures, mips included (design §6.4).
pub const DEFAULT_BUDGET_BYTES: u64 = 1 << 30;

static MIPS_ON: AtomicBool = AtomicBool::new(true);
static BUDGET: AtomicU64 = AtomicU64::new(DEFAULT_BUDGET_BYTES);

/// The `?paintMips=off` kill-switch: off, surface textures created from now on have level 0 only and are sampled
/// bilinearly from it (the spike's behaviour). Set it before painting; existing textures keep their levels.
pub fn set_mips_enabled(on: bool) {
    MIPS_ON.store(on, Ordering::Relaxed);
}

/// Whether new surface textures get a mip chain.
pub fn mips_enabled() -> bool {
    MIPS_ON.load(Ordering::Relaxed)
}

/// The GPU byte budget for surface textures (mips included) in the retained pattern cache.
pub fn set_budget_bytes(bytes: u64) {
    BUDGET.store(bytes.max(1), Ordering::Relaxed);
}

/// See [`set_budget_bytes`].
pub fn budget_bytes() -> u64 {
    BUDGET.load(Ordering::Relaxed)
}

/// Mip levels a new `w`×`h` surface texture gets: the full chain, or 1 with mips off.
pub fn mip_levels(w: u32, h: u32) -> u32 {
    if mips_enabled() {
        full_mip_levels(w, h)
    } else {
        1
    }
}

/// Levels of a full `w`×`h` mip chain (down to 1×1).
pub fn full_mip_levels(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// GPU bytes of a `w`×`h` RGBA8 texture with `levels` mip levels.
pub fn texture_bytes(w: u32, h: u32, levels: u32) -> u64 {
    (0..levels)
        .map(|k| u64::from((w >> k).max(1)) * u64::from((h >> k).max(1)) * 4)
        .sum()
}

/// The mip-regeneration pipeline for textures of one format (one per format per pattern cache, created with its
/// first texture that has mips).
pub struct MipGen {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
}

/// Level k+1 from level k: an exact premultiplied 2×2 box filter on the stored bytes. Odd sizes drop the last
/// row / column (level k+1 is `max(1, size >> 1)`); reads clamp for the 1-texel levels.
const MIP_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn texel(at: vec2<i32>, last: vec2<i32>) -> vec4<u32> {
  return vec4<u32>(round(clamp(textureLoad(src, min(at, last), 0), vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
  let b = vec2<i32>(floor(pos.xy)) * 2;
  let sum = texel(b, last) + texel(b + vec2<i32>(1, 0), last) + texel(b + vec2<i32>(0, 1), last)
    + texel(b + vec2<i32>(1, 1), last);
  return vec4<f32>((sum + vec4<u32>(2u)) / vec4<u32>(4u)) / 255.0;
}
"#;

/// The box filter of a floating-point texture (an image held in an engine's working colour), for any extent.
/// Level k+1 is `max(1, size >> 1)` texels: each of them covers `size / that` texels of level k per axis, two,
/// or a little more where the size is odd, and is their mean weighted by how much of each it covers, so every
/// texel of level k counts once.
const MIP_FLOAT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let size = vec2<f32>(textureDimensions(src));
  let scale = size / max(floor(size * 0.5), vec2<f32>(1.0));
  // This texel's footprint in level k, [lo, hi) per axis: at most four texels across.
  let lo = floor(pos.xy) * scale;
  let hi = lo + scale;
  let first = vec2<i32>(floor(lo));
  let last = vec2<i32>(size) - vec2<i32>(1);
  var sum = vec4<f32>(0.0);
  for (var j = 0; j < 4; j++) {
    let y = f32(first.y + j);
    let wy = max(min(hi.y, y + 1.0) - max(lo.y, y), 0.0);
    for (var i = 0; i < 4; i++) {
      let x = f32(first.x + i);
      let wx = max(min(hi.x, x + 1.0) - max(lo.x, x), 0.0);
      sum += textureLoad(src, min(first + vec2<i32>(i, j), last), 0) * (wx * wy);
    }
  }
  return sum / (scale.x * scale.y);
}
"#;

impl MipGen {
    /// `format`: the textures' format, `Rgba8Unorm` or `Rgba16Float`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let float = format == wgpu::TextureFormat::Rgba16Float;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paint surface mips"),
            source: wgpu::ShaderSource::Wgsl((if float { MIP_FLOAT_WGSL } else { MIP_WGSL }).into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paint surface mips"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paint surface mips"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("paint surface mips"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        MipGen { pipeline, bgl }
    }
}

/// A texel rect `[x0, x1, y0, y1]` (half-open).
pub type Rect = [u32; 4];

/// Past this many rects at one level, regenerate their bounding rect instead (fewer draws).
const MAX_RECTS_PER_LEVEL: usize = 32;

/// The level-k+1 rect covering everything level-k rect `r` feeds, clipped to the `lw`×`lh` level.
fn halve(r: &Rect, lw: u32, lh: u32) -> Rect {
    [(r[0] >> 1).min(lw), ((r[1] + 1) >> 1).min(lw), (r[2] >> 1).min(lh), ((r[3] + 1) >> 1).min(lh)]
}

/// The GPU side of one surface texture's mip chain: a single-level view per level and a bind group reading each
/// level. Owned by the pattern cache that created the texture (GPU objects stay with their engine; the process-
/// global registry keeps only the dirty rects).
pub struct SurfaceMips {
    views: Vec<wgpu::TextureView>,
    /// `bind_groups[k]` reads level k (the source of level k + 1).
    bind_groups: Vec<wgpu::BindGroup>,
}

impl SurfaceMips {
    pub fn new(device: &wgpu::Device, gen: &MipGen, texture: &wgpu::Texture) -> Self {
        let views: Vec<wgpu::TextureView> = (0..texture.mip_level_count())
            .map(|k| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("paint surface mip view"),
                    base_mip_level: k,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let bind_groups = views[..views.len().saturating_sub(1)]
            .iter()
            .map(|v| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("paint surface mip source"),
                    layout: &gen.bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) }],
                })
            })
            .collect();
        SurfaceMips { views, bind_groups }
    }

    /// Regenerate the mip texels under the level-0 `rects` of `texture`: one encoder, one render pass per level
    /// (level k covers the rects `>> k`), one scissored draw per rect, submitted now — after the uploads' level-0
    /// writes, before any frame that samples them. Returns the CPU time spent, in ms (0 when there was nothing).
    pub fn regen(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gen: &MipGen,
        texture: &wgpu::Texture,
        mut rects: Vec<Rect>,
    ) -> f64 {
        if rects.is_empty() || self.views.len() < 2 {
            return 0.0;
        }
        let t0 = now_ms();
        let (w, h) = (texture.width(), texture.height());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("paint surface mips"),
        });
        for k in 1..self.views.len() as u32 {
            let (lw, lh) = ((w >> k).max(1), (h >> k).max(1));
            let mut next: Vec<Rect> = Vec::with_capacity(rects.len());
            for r in &rects {
                let n = halve(r, lw, lh);
                if n[0] < n[1] && n[2] < n[3] && !next.contains(&n) {
                    next.push(n);
                }
            }
            if next.len() > MAX_RECTS_PER_LEVEL {
                let u = next.iter().fold([u32::MAX, 0, u32::MAX, 0], |a, r| {
                    [a[0].min(r[0]), a[1].max(r[1]), a[2].min(r[2]), a[3].max(r[3])]
                });
                next = vec![u];
            }
            rects = next;
            if rects.is_empty() {
                break;
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paint surface mip level"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[k as usize],
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gen.pipeline);
            pass.set_bind_group(0, &self.bind_groups[k as usize - 1], &[]);
            for r in &rects {
                pass.set_scissor_rect(r[0], r[2], r[1] - r[0], r[3] - r[2]);
                pass.draw(0..3, 0..1);
            }
        }
        crate::scratch::submit(queue, [encoder.finish()]);
        let ms = now_ms() - t0;
        let mut s = lock();
        s.mip_regens += 1;
        s.mip_regen_ms += ms;
        s.mip_regen_last_ms = ms;
        ms
    }
}

struct Attached {
    texture: wgpu::Texture,
    queue: wgpu::Queue,
    token: u64,
    /// The texture has mips: record written rects for the owning cache's regen.
    has_mips: bool,
    /// Level-0 rects written since the owning cache last took them.
    dirty: Vec<Rect>,
    /// GPU bytes, mips included.
    bytes: u64,
}

impl Attached {
    fn mark_dirty(&mut self, r: Rect) {
        if self.has_mips && !self.dirty.contains(&r) {
            self.dirty.push(r);
        }
    }
}

#[derive(Default)]
struct Entry {
    width: u32,
    height: u32,
    /// src key → textures created for it (a size change is a new src, so a new key).
    textures: Vec<(String, Attached)>,
    /// Committed straight tiles (native `retain_cpu` only).
    tiles: BTreeMap<(u32, u32), Arc<Vec<u8>>>,
    epoch: u64,
}

struct State {
    entries: HashMap<String, Entry>,
    epoch_total: u64,
    retain_cpu: bool,
    needs_refill: Vec<String>,
    uploads: u64,
    upload_bytes: u64,
    /// Bumped whenever an attached texture gets a dirty mip rect (caches skip the scan when it hasn't moved).
    dirty_gen: u64,
    mip_regens: u64,
    mip_regen_ms: f64,
    mip_regen_last_ms: f64,
    evictions: u64,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(State {
            entries: HashMap::new(),
            epoch_total: 0,
            retain_cpu: !cfg!(target_arch = "wasm32"),
            needs_refill: Vec::new(),
            uploads: 0,
            upload_bytes: 0,
            dirty_gen: 0,
            mip_regens: 0,
            mip_regen_ms: 0.0,
            mip_regen_last_ms: 0.0,
            evictions: 0,
        })
    })
}

fn lock() -> std::sync::MutexGuard<'static, State> {
    state().lock().unwrap_or_else(|e| e.into_inner())
}

/// Keep committed tiles in the engine process (native: exports and fresh engines fill themselves). The wasm host
/// turns it off so pixels never live in the render engine's heap.
pub fn set_retain_cpu(on: bool) {
    lock().retain_cpu = on;
}

/// The pattern texture a `paint-surface:` src resolves to, or `None` for any other src.
pub fn pattern_texture(src: &str) -> Option<PatternTexture> {
    let (_, w, h) = parse_paint_surface_src(src)?;
    Some(PatternTexture {
        state: PatternTextureState::Loaded,
        key: src.to_string(),
        image: DecodedImage {
            width: w.max(1),
            height: h.max(1),
            rgba_premultiplied: Arc::new(Vec::new()),
            // upload_id 0: the src already names the size, so the texture's identity is stable across strokes.
            straight: false,
            external: Some(ExternalTextureRef {
                key: src.to_string(),
                upload_id: 0,
            }),
        },
        scale_factor: (1.0, 1.0),
    })
}

/// Straight → premultiplied (the pattern pipeline samples premultiplied texels): `(c·a + 127) / 255`.
fn premultiply_into(dst: &mut Vec<u8>, src: &[u8]) {
    dst.clear();
    dst.reserve(src.len());
    for px in src.chunks_exact(4) {
        let a = px[3] as u32;
        dst.push(((px[0] as u32 * a + 127) / 255) as u8);
        dst.push(((px[1] as u32 * a + 127) / 255) as u8);
        dst.push(((px[2] as u32 * a + 127) / 255) as u8);
        dst.push(px[3]);
    }
}

/// Write one tile (straight RGBA, or `None` = transparent) into level 0 of `a`, clipped to the texture, and mark
/// its rect for the mip regen.
fn write_tile(a: &mut Attached, tx: u32, ty: u32, rgba: Option<&[u8]>, scratch: &mut Vec<u8>) -> u64 {
    let (tw, th) = (a.texture.width(), a.texture.height());
    let (x0, y0) = (tx * TILE, ty * TILE);
    if x0 >= tw || y0 >= th {
        return 0;
    }
    let (w, h) = ((tw - x0).min(TILE), (th - y0).min(TILE));
    a.mark_dirty([x0, x0 + w, y0, y0 + h]);
    scratch.clear();
    match rgba {
        Some(src) if src.len() >= TILE_BYTES => {
            let mut row = Vec::new();
            for y in 0..h as usize {
                let off = y * TILE as usize * 4;
                premultiply_into(&mut row, &src[off..off + w as usize * 4]);
                scratch.extend_from_slice(&row);
            }
        }
        _ => scratch.resize((w * h * 4) as usize, 0),
    }
    a.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &a.texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        scratch,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    scratch.len() as u64
}

/// Whether `key` (an external texture key) names a paint surface.
pub fn is_surface_key(key: &str) -> bool {
    key.starts_with(PAINT_SURFACE_SRC_PREFIX)
}

/// A pattern cache created `texture` for the surface src `key`: remember it, and fill it from the committed tiles
/// (native) or mark the layer for a refill from the paint side (wasm). When the texture has mips, the filled rects
/// wait in [`take_dirty`]; the cache regenerates them right away (the texture is created mid-frame, after this
/// render's flush). Returns the token that names the texture.
pub fn attach_texture(key: &str, queue: &wgpu::Queue, texture: &wgpu::Texture) -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let token = NEXT.fetch_add(1, Ordering::Relaxed);
    let Some((id, w, h)) = parse_paint_surface_src(key) else {
        return token;
    };
    let mut s = lock();
    let retain = s.retain_cpu;
    let entry = s.entries.entry(id.to_string()).or_default();
    if entry.width != w || entry.height != h {
        entry.width = w;
        entry.height = h;
        entry.tiles.clear();
    }
    let levels = texture.mip_level_count();
    let mut attached = Attached {
        texture: texture.clone(),
        queue: queue.clone(),
        token,
        has_mips: levels > 1,
        dirty: Vec::new(),
        bytes: texture_bytes(texture.width(), texture.height(), levels),
    };
    let mut scratch = Vec::new();
    let mut bytes = 0;
    for (&(tx, ty), rgba) in &entry.tiles {
        bytes += write_tile(&mut attached, tx, ty, Some(rgba), &mut scratch);
    }
    entry.textures.push((key.to_string(), attached));
    s.upload_bytes += bytes;
    if !retain && !s.needs_refill.iter().any(|n| n == id) {
        s.needs_refill.push(id.to_string());
    }
    token
}

/// A pattern cache dropped `texture` (eviction, clear, a per-render cache ending). `evicted`: dropped for the
/// budget (counted in `evictions`).
pub fn detach_texture(token: u64, evicted: bool) {
    let mut s = lock();
    for entry in s.entries.values_mut() {
        entry.textures.retain(|(_, a)| a.token != token);
    }
    if evicted {
        s.evictions += 1;
    }
}

/// Write tiles of layer `node_id` (pixel size `w`×`h`) into every texture made for it, and bump its pixel epoch.
/// `tiles`: (tx, ty, straight RGBA of 256×256 or `None` = clear). Returns the layer's new epoch.
pub fn upload_tiles(node_id: &str, w: u32, h: u32, tiles: &[(u32, u32, Option<&[u8]>)]) -> u64 {
    let mut s = lock();
    let retain = s.retain_cpu;
    let entry = s.entries.entry(node_id.to_string()).or_default();
    if entry.width != w || entry.height != h {
        entry.width = w;
        entry.height = h;
        entry.tiles.clear();
    }
    let mut scratch = Vec::new();
    let mut bytes = 0;
    let src_key = moda_scene::normalize::paint_surface_src(node_id, w, h);
    let mut marked = false;
    for &(tx, ty, rgba) in tiles {
        for (key, a) in &mut entry.textures {
            if *key == src_key {
                bytes += write_tile(a, tx, ty, rgba, &mut scratch);
                marked |= a.has_mips;
            }
        }
        if retain {
            match rgba {
                Some(px) if px.len() >= TILE_BYTES => {
                    entry.tiles.insert((tx, ty), Arc::new(px[..TILE_BYTES].to_vec()));
                }
                _ => {
                    entry.tiles.remove(&(tx, ty));
                }
            }
        }
    }
    entry.epoch += 1;
    let epoch = entry.epoch;
    s.epoch_total += 1;
    s.uploads += tiles.len() as u64;
    s.upload_bytes += bytes;
    if marked {
        s.dirty_gen += 1;
    }
    epoch
}

/// Milliseconds on a monotonic clock, for the counters (always 0 on wasm, which has no `Instant`; the host times
/// its own frames).
fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1e3
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

/// Changes whenever some texture gets a dirty mip rect: a cache that saw the same value has nothing to regenerate.
pub fn dirty_generation() -> u64 {
    lock().dirty_gen
}

/// The level-0 rects written into texture `token` since the last call (for its cache's mip regen).
pub fn take_dirty(token: u64) -> Vec<Rect> {
    let mut s = lock();
    for entry in s.entries.values_mut() {
        if let Some((_, a)) = entry.textures.iter_mut().find(|(_, a)| a.token == token) {
            return std::mem::take(&mut a.dirty);
        }
    }
    Vec::new()
}

/// Drop everything known about a layer's tiles (its textures stay attached; they keep their contents until the
/// next upload). Used when a headless render installs a document's surfaces.
pub fn clear_tiles(node_id: &str) {
    let mut s = lock();
    if let Some(e) = s.entries.get_mut(node_id) {
        e.tiles.clear();
    }
    s.epoch_total += 1;
}

/// Layers whose texture was (re-)created without tiles to fill it; the host re-sends them.
pub fn take_needs_refill() -> Vec<String> {
    std::mem::take(&mut lock().needs_refill)
}

/// Sum of every layer's pixel-epoch bumps: the part of a cache key that says "pixels changed".
pub fn epoch_total() -> u64 {
    lock().epoch_total
}

/// A layer's pixel epoch (0 when unknown).
pub fn epoch(node_id: &str) -> u64 {
    lock().entries.get(node_id).map_or(0, |e| e.epoch)
}

/// Surface counters (`paint_surface_stats`, `residency_stats`; design §6.4).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceStats {
    /// Paint layers the registry knows.
    pub layers: usize,
    /// Attached surface textures (every engine and device).
    pub textures: usize,
    /// Their GPU bytes, mips included.
    pub texture_bytes: u64,
    /// Committed tiles kept in the engine process (native only).
    pub retained_tiles: usize,
    /// Tile uploads, and the bytes written (fills of new textures included).
    pub uploads: u64,
    pub upload_bytes: u64,
    /// Surface textures dropped for the budget.
    pub evictions: u64,
    /// Layers waiting for the host to re-send their tiles (a texture was created blank).
    pub refining: usize,
    /// Mip regenerations (one per flush or fill that had work) and their total / last CPU encode + submit time.
    pub mip_regens: u64,
    pub mip_regen_ms: f64,
    pub mip_regen_last_ms: f64,
    pub mips_enabled: bool,
    pub budget_bytes: u64,
}

/// See [`SurfaceStats`].
pub fn stats() -> SurfaceStats {
    let s = lock();
    SurfaceStats {
        layers: s.entries.len(),
        textures: s.entries.values().map(|e| e.textures.len()).sum(),
        texture_bytes: s.entries.values().flat_map(|e| e.textures.iter().map(|(_, a)| a.bytes)).sum(),
        retained_tiles: s.entries.values().map(|e| e.tiles.len()).sum(),
        uploads: s.uploads,
        upload_bytes: s.upload_bytes,
        evictions: s.evictions,
        refining: s.needs_refill.len(),
        mip_regens: s.mip_regens,
        mip_regen_ms: s.mip_regen_ms,
        mip_regen_last_ms: s.mip_regen_last_ms,
        mips_enabled: mips_enabled(),
        budget_bytes: budget_bytes(),
    }
}
