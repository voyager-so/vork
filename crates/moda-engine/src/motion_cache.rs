//! motion_cache.rs — pan over cached pixels (docs/roadmap/camera-motion-cache.md).
//!
//! A still camera frame is the *anchor*. The cache holds view-sized tiles on a lattice anchored to it, offset half a
//! view, so the 2 × 2 block of tiles around the anchor is exactly the retained plan's cull window. Each tile is the
//! frame a live camera would present at the anchor moved by whole device pixels (same viewport, same target size, so
//! the retained plan's prepared packet serves it: only the camera uniform changes). While the camera pans at the
//! anchor's zoom, dpr and viewport, with the scene unchanged, a frame whose tiles are all present is a copy of them:
//! the camera delta rounded to whole device pixels, copied with `textureLoad` (no filtering). Tiles fill in when idle
//! around the anchor, and during motion ahead of it (the host asks after each moving frame), so a long pan stays in
//! the cache. Anything else — a zoom, an edit, a resource settling, a tile not there yet — and the frame renders as
//! usual. The host never serves a still frame from the cache: when motion stops it renders exactly, and that frame
//! re-anchors.

use crate::frame::{CameraParams, CameraPresentOverride};

/// What a cached region is valid for: the scene and resources it was rendered from, the render options, and the
/// camera's zoom, dpr, viewport size and clear. Any difference drops the cache.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionKey {
    /// Resident scene generation, render options, resource generations (the host folds them into one value).
    pub content: u64,
    pub zoom: f64,
    pub dpr: f64,
    pub viewport_w: f64,
    pub viewport_h: f64,
    pub width_px: u32,
    pub height_px: u32,
    pub clear_rgb: [f32; 3],
}

impl MotionKey {
    /// The key of a frame of `content` at camera `cam` (the camera position is not part of it).
    pub fn new(content: u64, cam: &CameraParams) -> MotionKey {
        let o = cam.present_override(0.0);
        MotionKey {
            content,
            zoom: cam.zoom,
            dpr: cam.dpr,
            viewport_w: cam.viewport_w,
            viewport_h: cam.viewport_h,
            width_px: o.width_px,
            height_px: o.height_px,
            clear_rgb: cam.clear_rgb,
        }
    }

    /// Device pixels per canvas unit on each axis. The camera projection maps the logical viewport onto the whole
    /// `ceil(viewport × dpr)` target, so this is `zoom × width_px / viewport_w` (not quite `zoom × dpr`).
    fn scale(&self) -> (f64, f64) {
        (
            self.zoom * self.width_px as f64 / self.viewport_w,
            self.zoom * self.height_px as f64 / self.viewport_h,
        )
    }
}

/// Whether a camera plan's pixels move rigidly with the camera, so a frame can be copied from pixels rendered at
/// another camera: no background blur (its pass split is placed in target pixels), no layer blur drawn from a
/// capture, no shader animating with the clock.
pub fn plan_cacheable(plan: &crate::seams::CoveragePlan) -> bool {
    !crate::residency::plan_uses_shader_time(plan)
        && !plan.items.iter().any(|i| {
            matches!(
                i,
                crate::seams::DrawItem::BackdropSplit(_)
                    | crate::seams::DrawItem::LayerBlurComposite { .. }
            )
        })
}

/// Budget for the tiles (view-sized RGBA8 textures). At least 4 must fit, or the cache stays off.
pub const MOTION_CACHE_BUDGET_BYTES: u64 = 160 << 20;
pub const PREVIEW_TILE_SIZE: u32 = 512;

/// Canvas-space slack when asking whether the plan's cull window holds a tile: the tile lattice sits half a view off
/// the anchor in whole device pixels, so an odd-sized view's block of tiles overshoots the window by half a pixel.
const HOLD_SLACK_PX: f64 = 1.0;

/// A view-sized texture of the lattice.
struct Tile {
    texture: wgpu::Texture,
    last_used: u64,
}

pub struct MotionCache {
    preview_tiles: bool,
    generation: u64,
    pending_preview: Option<u64>,
    preview_ticket: u64,
    key: Option<MotionKey>,
    /// The anchor frame's camera. Pixel (0, 0) of the lattice is the anchor frame's top-left pixel.
    anchor: Option<CameraParams>,
    /// The lattice's offset: tile (i, j) covers lattice pixels [i·w − half_w, (i+1)·w − half_w) (w = the frame's
    /// width), and likewise down. The anchor's cull window is then exactly the block of tiles (0..2, 0..2).
    half_w: i64,
    half_h: i64,
    tiles: std::collections::HashMap<(i32, i32), Tile>,
    max_tiles: usize,
    tick: u64,
    pipeline: Option<CompositePipeline>,
    /// Frames served from the cache, anchors taken, tiles rendered (diagnostics).
    pub served: u64,
    pub anchors: u64,
    pub filled: u64,
    /// Anchors dropped before a new still frame replaced them, and why the last one was.
    pub drops: u64,
    /// Fills that laid out part of a tile within the budget and left it for the next fill.
    pub deferred_fills: u64,
    pub last_drop: &'static str,
    /// Fill timings (ms, host clock): the last and slowest plan build for a tile, the last tile render.
    pub last_plan_ms: f64,
    pub max_plan_ms: f64,
    pub last_piece_ms: f64,
}

impl Default for MotionCache {
    fn default() -> Self {
        MotionCache {
            preview_tiles: false,
            generation: 0,
            pending_preview: None,
            preview_ticket: 0,
            key: None,
            anchor: None,
            half_w: 0,
            half_h: 0,
            tiles: Default::default(),
            max_tiles: 0,
            tick: 0,
            pipeline: None,
            served: 0,
            anchors: 0,
            filled: 0,
            drops: 0,
            deferred_fills: 0,
            last_drop: "",
            last_plan_ms: 0.0,
            max_plan_ms: 0.0,
            last_piece_ms: 0.0,
        }
    }
}

/// A tile to render: its lattice coordinate and the present override that renders it. The override is the anchor's
/// camera moved by whole device pixels, at the anchor's viewport and target size, so it is exactly what a live frame
/// at that camera presents (and a live frame's prepared packet serves it).
pub struct PieceJob {
    pub tile: (i32, i32),
    pub present: CameraPresentOverride,
    generation: u64,
}

pub struct PreviewTileTarget {
    texture: wgpu::Texture,
    tile: (i32, i32),
    generation: u64,
    ticket: u64,
}

impl PreviewTileTarget {
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

impl MotionCache {
    pub fn set_preview_tiles_enabled(&mut self, enabled: bool) {
        if self.preview_tiles != enabled {
            self.clear();
            self.preview_tiles = enabled;
        }
    }

    pub fn preview_tiles_enabled(&self) -> bool {
        self.preview_tiles
    }

    fn tile_extent(&self, key: &MotionKey) -> (u32, u32) {
        if self.preview_tiles {
            (PREVIEW_TILE_SIZE, PREVIEW_TILE_SIZE)
        } else {
            (key.width_px, key.height_px)
        }
    }

    /// Forget everything (a scene switch, memory pressure, retained mode off).
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.pending_preview = None;
        self.key = None;
        self.anchor = None;
        for (_, t) in self.tiles.drain() {
            crate::scratch::destroy(&t.texture);
        }
    }

    /// Bytes of GPU memory the tiles hold now.
    pub fn bytes(&self) -> u64 {
        let Some(k) = self.key else { return 0 };
        let (width, height) = self.tile_extent(&k);
        self.tiles.len() as u64 * u64::from(width) * u64::from(height) * 4
    }

    /// Whether the cache holds an anchor for `key`.
    pub fn anchored_for(&self, key: &MotionKey) -> bool {
        self.key.as_ref() == Some(key) && self.anchor.is_some()
    }

    /// The anchor camera.
    pub fn anchor(&self) -> Option<&CameraParams> {
        self.anchor.as_ref()
    }

    /// Start a lattice at a still frame: `key` / `cam` are the frame's. Returns false (and leaves the cache off)
    /// when fewer than 4 tiles fit the budget.
    pub fn begin_anchor(&mut self, key: MotionKey, cam: &CameraParams) -> bool {
        self.clear();
        if self.preview_tiles
            && (key.width_px == 0
                || key.height_px == 0
                || ![key.zoom, key.dpr, key.viewport_w, key.viewport_h]
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                || !cam.camera_x.is_finite()
                || !cam.camera_y.is_finite())
        {
            return false;
        }
        let (width, height) = self.tile_extent(&key);
        let tile_bytes = u64::from(width) * u64::from(height) * 4;
        if tile_bytes == 0 {
            return false;
        }
        let max = (MOTION_CACHE_BUDGET_BYTES / tile_bytes) as usize;
        let minimum = if self.preview_tiles {
            (u64::from(key.width_px - 1) / u64::from(width) + 2)
                * (u64::from(key.height_px - 1) / u64::from(height) + 2)
        } else {
            4
        };
        if (max as u64) < minimum {
            return false;
        }
        self.max_tiles = max;
        self.half_w = if self.preview_tiles {
            0
        } else {
            i64::from(width / 2)
        };
        self.half_h = if self.preview_tiles {
            0
        } else {
            i64::from(height / 2)
        };
        self.key = Some(key);
        let mut anchor = *cam;
        if self.preview_tiles {
            let (scale_x, scale_y) = key.scale();
            anchor.camera_x =
                (cam.camera_x * scale_x / f64::from(width)).floor() * f64::from(width) / scale_x;
            anchor.camera_y =
                (cam.camera_y * scale_y / f64::from(height)).floor() * f64::from(height) / scale_y;
            if !anchor.camera_x.is_finite() || !anchor.camera_y.is_finite() {
                self.clear();
                return false;
            }
        }
        self.anchor = Some(anchor);
        self.anchors += 1;
        true
    }

    /// The camera `cam`'s offset from the anchor in whole device pixels, when `key` matches.
    pub fn offset(&self, key: &MotionKey, cam: &CameraParams) -> Option<(i64, i64)> {
        if self.key.as_ref() != Some(key) {
            return None;
        }
        let anchor = self.anchor.as_ref()?;
        let (sx, sy) = key.scale();
        let dx = ((cam.camera_x - anchor.camera_x) * sx).round();
        let dy = ((cam.camera_y - anchor.camera_y) * sy).round();
        (dx.is_finite() && dy.is_finite() && dx.abs() < 1e9 && dy.abs() < 1e9)
            .then_some((dx as i64, dy as i64))
    }

    /// The tiles a frame at lattice offset `d` overlaps: inclusive ranges of i and j.
    fn tiles_at(&self, d: (i64, i64)) -> ((i32, i32), (i32, i32)) {
        let k = self.key.expect("anchored");
        let (width, height) = self.tile_extent(&k);
        let (w, h) = (i64::from(width), i64::from(height));
        let i0 = floor_div(d.0 + self.half_w, w);
        let i1 = floor_div(d.0 + self.half_w + i64::from(k.width_px) - 1, w);
        let j0 = floor_div(d.1 + self.half_h, h);
        let j1 = floor_div(d.1 + self.half_h + i64::from(k.height_px) - 1, h);
        ((i0 as i32, i1 as i32), (j0 as i32, j1 as i32))
    }

    fn tile_origin(&self, (i, j): (i32, i32)) -> (i64, i64) {
        let k = self.key.expect("anchored");
        let (width, height) = self.tile_extent(&k);
        (
            i64::from(i) * i64::from(width) - self.half_w,
            i64::from(j) * i64::from(height) - self.half_h,
        )
    }

    /// The camera whose frame is tile `t` (the anchor moved by the tile's origin).
    fn tile_camera(&self, t: (i32, i32)) -> CameraParams {
        let (k, a) = (self.key.expect("anchored"), self.anchor.expect("anchored"));
        let (sx, sy) = k.scale();
        let (ox, oy) = self.tile_origin(t);
        let (width, height) = self.tile_extent(&k);
        CameraParams {
            camera_x: a.camera_x + ox as f64 / sx,
            camera_y: a.camera_y + oy as f64 / sy,
            viewport_w: if self.preview_tiles {
                f64::from(width) * k.viewport_w / f64::from(k.width_px)
            } else {
                a.viewport_w
            },
            viewport_h: if self.preview_tiles {
                f64::from(height) * k.viewport_h / f64::from(k.height_px)
            } else {
                a.viewport_h
            },
            ..a
        }
    }

    /// Whether a frame at `cam` can be served from the cache: every tile it overlaps is there.
    pub fn covers(&self, key: &MotionKey, cam: &CameraParams) -> bool {
        let Some(d) = self.offset(key, cam) else {
            return false;
        };
        let ((i0, i1), (j0, j1)) = self.tiles_at(d);
        (j0..=j1).all(|j| (i0..=i1).all(|i| self.tiles.contains_key(&(i, j))))
    }

    /// The next tile to render so frames at `cams` (the frame now, then where the motion is heading) can be served,
    /// in that order; `None` when they all can, or when `key` doesn't match.
    pub fn next_job(&self, key: &MotionKey, cams: &[CameraParams]) -> Option<PieceJob> {
        for cam in cams {
            let Some(d) = self.offset(key, cam) else {
                continue;
            };
            let ((i0, i1), (j0, j1)) = self.tiles_at(d);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if !self.tiles.contains_key(&(i, j)) {
                        return Some(self.job((i, j)));
                    }
                }
            }
        }
        None
    }

    fn job(&self, t: (i32, i32)) -> PieceJob {
        let k = self.key.expect("anchored");
        let mut present = self.tile_camera(t).present_override(0.0);
        // Exactly the frame's target size (`present_override` recomputes it from the same viewport, but keep the
        // lattice arithmetic and the target in agreement by construction).
        let (width, height) = self.tile_extent(&k);
        present.width_px = width;
        present.height_px = height;
        PieceJob {
            tile: t,
            present,
            generation: self.generation,
        }
    }

    /// The camera a plan should be built at so its cull window (the view grown by half a view each side) holds tile
    /// `t` and its neighbour towards `toward` (a lattice offset): the view centred on the tile corner nearest it.
    pub fn plan_camera_for(&self, t: (i32, i32), toward: (i64, i64)) -> CameraParams {
        if self.preview_tiles {
            return self.tile_camera(t);
        }
        let k = self.key.expect("anchored");
        let (w, h) = (i64::from(k.width_px), i64::from(k.height_px));
        let (ox, oy) = self.tile_origin(t);
        let (cx, cy) = (toward.0 + self.half_w, toward.1 + self.half_h); // the frame's centre, lattice px
        let corner_x = if cx >= ox + w / 2 { ox + w } else { ox };
        let corner_y = if cy >= oy + h / 2 { oy + h } else { oy };
        let (k_sx, k_sy) = k.scale();
        let a = self.anchor.expect("anchored");
        CameraParams {
            camera_x: a.camera_x + (corner_x - self.half_w) as f64 / k_sx,
            camera_y: a.camera_y + (corner_y - self.half_h) as f64 / k_sy,
            ..a
        }
    }

    /// Tile `t`'s canvas rect, grown by the hold slack: the plan it renders from must hold every node that meets
    /// it.
    pub fn tile_region(&self, t: (i32, i32)) -> [f64; 4] {
        let k = self.key.expect("anchored");
        let (sx, sy) = k.scale();
        let c = self.tile_camera(t);
        let (width, height) = self.tile_extent(&k);
        let (w, h) = (f64::from(width) / sx, f64::from(height) / sy);
        let margin = if self.preview_tiles {
            -HOLD_SLACK_PX
        } else {
            HOLD_SLACK_PX
        };
        let (mx, my) = (margin / sx, margin / sy);
        [
            c.camera_x + mx,
            c.camera_y + my,
            c.camera_x + w - mx,
            c.camera_y + h - my,
        ]
    }

    pub fn allocate_preview_tile(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        job: &PieceJob,
    ) -> Option<PreviewTileTarget> {
        if !self.preview_tiles
            || self.pending_preview.is_some()
            || self.key.is_none()
            || job.generation != self.generation
            || job.present.width_px != PREVIEW_TILE_SIZE
            || job.present.height_px != PREVIEW_TILE_SIZE
            || PREVIEW_TILE_SIZE > device.limits().max_texture_dimension_2d
            || !matches!(
                format,
                wgpu::TextureFormat::Rgba8Unorm
                    | wgpu::TextureFormat::Bgra8Unorm
                    | wgpu::TextureFormat::Rgba8UnormSrgb
                    | wgpu::TextureFormat::Bgra8UnormSrgb
            )
        {
            return None;
        }
        while self.tiles.len() >= self.max_tiles {
            let oldest = self
                .tiles
                .iter()
                .min_by_key(|(_, tile)| tile.last_used)
                .map(|(address, _)| *address)?;
            if let Some(tile) = self.tiles.remove(&oldest) {
                crate::scratch::destroy(&tile.texture);
            }
        }
        self.preview_ticket = self.preview_ticket.wrapping_add(1);
        self.pending_preview = Some(self.preview_ticket);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview tile"),
            size: wgpu::Extent3d {
                width: PREVIEW_TILE_SIZE,
                height: PREVIEW_TILE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Some(PreviewTileTarget {
            texture,
            tile: job.tile,
            generation: job.generation,
            ticket: self.preview_ticket,
        })
    }

    pub fn commit_preview_tile(&mut self, target: PreviewTileTarget) -> bool {
        if !self.preview_tiles
            || self.key.is_none()
            || target.generation != self.generation
            || self.pending_preview != Some(target.ticket)
        {
            crate::scratch::destroy(&target.texture);
            return false;
        }
        self.pending_preview = None;
        if let Some(previous) = self.tiles.remove(&target.tile) {
            crate::scratch::destroy(&previous.texture);
        }
        while self.tiles.len() >= self.max_tiles {
            let Some(oldest) = self
                .tiles
                .iter()
                .min_by_key(|(_, tile)| tile.last_used)
                .map(|(address, _)| *address)
            else {
                break;
            };
            if let Some(tile) = self.tiles.remove(&oldest) {
                crate::scratch::destroy(&tile.texture);
            }
        }
        self.tick += 1;
        self.tiles.insert(
            target.tile,
            Tile {
                texture: target.texture,
                last_used: self.tick,
            },
        );
        self.filled += 1;
        true
    }

    pub fn discard_preview_tile(&mut self, target: PreviewTileTarget) {
        if self.pending_preview == Some(target.ticket) {
            self.pending_preview = None;
        }
        crate::scratch::destroy(&target.texture);
    }

    /// The texture job `job` renders into. Evicts the least recently used tile past the budget.
    pub fn job_target(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        job: &PieceJob,
    ) -> wgpu::Texture {
        assert!(!self.preview_tiles);
        let k = self.key.expect("anchored");
        while self.tiles.len() >= self.max_tiles {
            let Some(oldest) = self
                .tiles
                .iter()
                .min_by_key(|(_, t)| t.last_used)
                .map(|(c, _)| *c)
            else {
                break;
            };
            if let Some(t) = self.tiles.remove(&oldest) {
                crate::scratch::destroy(&t.texture);
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion cache tile"),
            size: wgpu::Extent3d {
                width: k.width_px,
                height: k.height_px,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.tick += 1;
        self.tiles.insert(
            job.tile,
            Tile {
                texture: texture.clone(),
                last_used: self.tick,
            },
        );
        self.filled += 1;
        texture
    }

    /// Tiles held.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Draw a frame at `cam` into `view` (the presented texture, `format`) from the tiles. False (nothing encoded)
    /// when the cache doesn't cover it.
    pub fn present(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        key: &MotionKey,
        cam: &CameraParams,
    ) -> bool {
        if !self.covers(key, cam) {
            return false;
        }
        let d = self.offset(key, cam).expect("covered");
        if self.pipeline.as_ref().is_none_or(|p| p.format != format) {
            self.pipeline = Some(CompositePipeline::new(device, format));
        }
        let ((i0, i1), (j0, j1)) = self.tiles_at(d);
        let (fw, fh) = (i64::from(key.width_px), i64::from(key.height_px));
        let (tile_width, tile_height) = self.tile_extent(key);
        self.tick += 1;
        let mut draws = Vec::new();
        for j in j0..=j1 {
            for i in i0..=i1 {
                let (ox, oy) = self.tile_origin((i, j));
                // The tile's part of the frame, in frame pixels: frame pixel p is lattice pixel p + d.
                let (px, py) = (ox - d.0, oy - d.1);
                let (rx0, ry0) = (px.max(0), py.max(0));
                let (rx1, ry1) = (
                    (px + i64::from(tile_width)).min(fw),
                    (py + i64::from(tile_height)).min(fh),
                );
                if rx1 <= rx0 || ry1 <= ry0 {
                    continue;
                }
                let tile = self.tiles.get_mut(&(i, j)).expect("covered");
                tile.last_used = self.tick;
                // Frame pixel p reads tile texel p - (px, py).
                let offset: [i32; 4] = [(-px) as i32, (-py) as i32, 0, 0];
                let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("motion cache offset"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: true,
                });
                ubuf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&offset));
                ubuf.unmap();
                let tv = tile.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let pipeline = self.pipeline.as_ref().expect("just built");
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("motion cache tile"),
                    layout: &pipeline.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: ubuf.as_entire_binding(),
                        },
                    ],
                });
                draws.push((
                    bg,
                    [
                        rx0 as u32,
                        ry0 as u32,
                        (rx1 - rx0) as u32,
                        (ry1 - ry0) as u32,
                    ],
                ));
            }
        }
        let pipeline = self.pipeline.as_ref().expect("just built");
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("motion cache composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        for (bg, [x, y, w, h]) in &draws {
            pass.set_scissor_rect(*x, *y, *w, *h);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
        drop(pass);
        self.served += 1;
        true
    }

    /// Count an anchor dropped for `why` (diagnostics).
    pub fn note_deferred_fill(&mut self) {
        self.deferred_fills += 1;
    }

    pub fn note_drop(&mut self, why: &'static str) {
        self.drops += 1;
        self.last_drop = why;
    }

    pub fn stats_json(&self) -> String {
        format!(
            "{{\"anchored\":{},\"tiles\":{},\"maxTiles\":{},\"bytes\":{},\"served\":{},\"anchors\":{},\"filled\":{},\"drops\":{},\"deferredFills\":{},\"lastDrop\":\"{}\",\"lastPlanMs\":{:.1},\"maxPlanMs\":{:.1},\"lastPieceMs\":{:.1}}}",
            self.anchor.is_some(),
            self.tiles.len(),
            self.max_tiles,
            self.bytes(),
            self.served,
            self.anchors,
            self.filled,
            self.drops,
            self.deferred_fills,
            self.last_drop,
            self.last_plan_ms,
            self.max_plan_ms,
            self.last_piece_ms
        )
    }
}

struct CompositePipeline {
    format: wgpu::TextureFormat,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

const COMPOSITE_WGSL: &str = r#"
@group(0) @binding(0) var piece: texture_2d<f32>;
struct Offset { o: vec4<i32> };
@group(0) @binding(1) var<uniform> offset: Offset;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(floor(p.xy)) + offset.o.xy;
    return textureLoad(piece, texel, 0);
}
"#;

impl CompositePipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> CompositePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion cache composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion cache composite"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion cache composite"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("motion cache composite"),
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
        CompositePipeline {
            format,
            bgl,
            pipeline,
        }
    }
}

#[cfg(test)]
#[path = "preview_tiles_tests.rs"]
mod preview_tiles_tests;
