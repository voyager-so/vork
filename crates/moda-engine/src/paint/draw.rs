//! draw.rs — GPU-side pattern texture plumbing (M3b).
//!
//! Mirrors the production texture manager's behavior for pattern fills
//! (`renderers/webgpu/utils/WebGPUTextureManager.ts:107-148` samplers;
//! `WebGPUPathRenderer.createTextureBindGroup` `:2269-2312`):
//! - textures are `rgba8unorm` (never `-srgb`), premultiplied bytes, no mips —
//!   except paint-layer surfaces (`crate::surface`), which have a full mip chain,
//! - in an engine that blends in another encoding than sRGB (`moda_shaders::working_color`) a decoded image's
//!   texture is `rgba16float` in that encoding: each texel is unassociated, converted and associated again when
//!   it is uploaded, so the sampler and the mip chain filter in the encoding the engine blends in,
//! - sampler: linear mag/min; `clamp-to-edge` for `no-repeat` fills,
//!   `repeat` otherwise; surfaces also filter linearly between mips (trilinear
//!   when minified, bilinear from level 0 when magnified),
//! - bind group = { 0: sampler, 1: texture view } at group index 2 of the
//!   pattern pipeline (mask rides group 3),
//! - bind groups are cached per (texture key, sampler kind).

use std::collections::{HashMap, HashSet};

use crate::external_texture::ExternalUploadOutcome;
use crate::paint::image_source::DecodedImage;

/// Per-frame (or per-engine) cache of uploaded pattern textures and their
/// texture bind groups.
pub struct PatternTextureCache {
    /// Trilinear (linear between mip levels): an image drawn smaller than its pixels samples the level whose texels
    /// match the screen's, as a filtered downscale would, in the canvas and the export alike. With one level (a
    /// borrowed capture texture) it is plain bilinear.
    repeat_sampler: wgpu::Sampler,
    clamp_sampler: wgpu::Sampler,
    /// The owning engine's working colour: the encoding decoded images are uploaded in (`level0`). `None` is sRGB,
    /// the bytes as they are.
    working: Option<moda_shaders::WorkingColor>,
    /// The mip-regeneration pipelines for this device, by texture format (each created with the first texture of
    /// its format that has mips).
    mip_gens: HashMap<wgpu::TextureFormat, crate::surface::MipGen>,
    /// Per texture with mips (cache key): its level views and bind groups. Images regenerate the whole chain when
    /// level 0 is written; paint surfaces regenerate under the tiles uploaded since the last render.
    surface_mips: HashMap<String, crate::surface::SurfaceMips>,
    /// `crate::surface::dirty_generation()` when this cache last took its dirty rects.
    surface_dirty_seen: u64,
    /// Bytes of resident surface textures (mips included), budgeted apart from `total_bytes`
    /// (`crate::surface::budget_bytes`).
    surface_bytes: usize,
    /// (texture key, clamp?) → bind group.
    bind_groups: HashMap<(String, bool), wgpu::BindGroup>,
    /// texture key → uploaded texture (kept alive for the bind groups).
    textures: HashMap<String, wgpu::Texture>,
    heap_pixel_allocations: HashMap<String, std::sync::Weak<Vec<u8>>>,
    /// Keys whose texture is a graph-pixel unit's pool texture (`external_texture::CaptureTexture`): the pool owns it,
    /// so dropping the cache's handle must not destroy it.
    borrowed_textures: HashMap<String, String>,
    /// Task #95 R2d (retained mode): per-key (bytes, last-touch tick) for the
    /// MB-budget LRU. Unused (zero-cost) on the per-render path.
    usage: HashMap<String, (usize, u64)>,
    total_bytes: usize,
    tick: u64,
    /// External-texture keys whose copy was deliberately SKIPPED this render
    /// (`ExternalUploadOutcome::Skipped` — the wasm offscreen snapshot guard).
    /// The skipped texture is transparent; it is cached so the current render
    /// pass has a bind group, but it is evicted at the next `begin_frame` so the
    /// following render re-attempts the real upload. Without this, a snapshot
    /// that skips a (panic-prone) live-external copy would durably poison this
    /// (retained, present-shared) cache and blank the image on the live present.
    pending_retry: HashSet<String>,
    /// Paint-surface textures (raster spike): cache key → the surface registry token to detach on drop.
    surface_tokens: HashMap<String, u64>,
}

impl PatternTextureCache {
    /// `working`: the owning engine's working colour (`Engine::new_with_color`).
    pub fn new(device: &wgpu::Device, working: Option<moda_shaders::WorkingColor>) -> Self {
        let mk = |label: &str, mode: wgpu::AddressMode, mipmap_filter: wgpu::MipmapFilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter,
                address_mode_u: mode,
                address_mode_v: mode,
                ..Default::default()
            })
        };
        let linear = wgpu::MipmapFilterMode::Linear;
        PatternTextureCache {
            repeat_sampler: mk("pattern repeat sampler", wgpu::AddressMode::Repeat, linear),
            clamp_sampler: mk("pattern clamp sampler", wgpu::AddressMode::ClampToEdge, linear),
            working,
            mip_gens: HashMap::new(),
            surface_mips: HashMap::new(),
            surface_dirty_seen: 0,
            surface_bytes: 0,
            bind_groups: HashMap::new(),
            textures: HashMap::new(),
            heap_pixel_allocations: HashMap::new(),
            borrowed_textures: HashMap::new(),
            usage: HashMap::new(),
            total_bytes: 0,
            tick: 0,
            pending_retry: HashSet::new(),
            surface_tokens: HashMap::new(),
        }
    }

    /// Task #95 R2d: advance the LRU clock (once per render).
    pub fn begin_frame(&mut self) {
        let expired: Vec<String> = self.borrowed_textures.iter().filter_map(|(key, capture)| {
            crate::external_texture::capture_texture(capture).is_none().then(|| key.clone())
        }).collect();
        for key in expired {
            self.evict(&key, false);
        }
        // Re-attempt any external upload that was SKIPPED last render (offscreen
        // snapshot guard): drop the transparent placeholder so this render
        // re-runs the real copy. Evicting BEFORE the tick advances means a key
        // that is skipped again this render is simply re-marked. This keeps a
        // skipped snapshot from durably poisoning the (present-shared) cache,
        // while a genuinely `Unavailable` key is left cached (not retried) to
        // avoid per-frame texture churn.
        if !self.pending_retry.is_empty() {
            for key in std::mem::take(&mut self.pending_retry) {
                self.evict(&key, false);
            }
        }
        self.tick += 1;
    }

    /// Drop a single key's cached texture + bind groups + LRU bookkeeping. `budget`: dropped to fit a budget
    /// (a surface's registry counts it as an eviction).
    fn evict(&mut self, key: &str, budget: bool) {
        self.heap_pixel_allocations.remove(key);
        let surface = self.surface_tokens.contains_key(key);
        if let Some((bytes, _)) = self.usage.remove(key) {
            if surface {
                self.surface_bytes = self.surface_bytes.saturating_sub(bytes);
            } else {
                self.total_bytes = self.total_bytes.saturating_sub(bytes);
            }
        }
        // At most two bind groups per key (clamp on/off), so two O(1) removes beat
        // an O(N) `retain` scan — which would be O(N²) across a trim sweep.
        let key_string = key.to_string();
        self.bind_groups.remove(&(key_string.clone(), true));
        self.bind_groups.remove(&(key_string, false));
        if let Some(texture) = self.textures.remove(key) {
            if let Some(token) = self.surface_tokens.remove(key) {
                crate::surface::detach_texture(token, budget);
            }
            self.surface_mips.remove(key);
            if self.borrowed_textures.remove(key).is_none() {
                crate::scratch::destroy(&texture);
            }
        }
    }

    /// An image texture's level 0 was just written in full: build its whole mip chain (views and bind groups are
    /// made once per texture and kept with it).
    fn build_mips(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: &str, texture: &wgpu::Texture) {
        let gen = self.mip_gens.entry(texture.format()).or_insert_with(|| crate::surface::MipGen::new(device, texture.format()));
        let mips = crate::surface::SurfaceMips::new(device, gen, texture);
        mips.regen(device, queue, gen, texture, vec![[0, texture.width(), 0, texture.height()]]);
        self.surface_mips.insert(key.to_string(), mips);
    }

    /// Level 0 of the image texture under `key` was rewritten in place (a new video frame, new pixels of the same
    /// size): regenerate its whole chain.
    fn regen_all_mips(&self, device: &wgpu::Device, queue: &wgpu::Queue, key: &str) {
        let (Some(mips), Some(texture)) = (self.surface_mips.get(key), self.textures.get(key)) else {
            return;
        };
        let Some(gen) = self.mip_gens.get(&texture.format()) else { return };
        mips.regen(device, queue, gen, texture, vec![[0, texture.width(), 0, texture.height()]]);
    }

    /// Paint surfaces: regenerate the mips under every tile uploaded into this cache's surface textures since the
    /// last call (one command buffer per texture, submitted now: after the uploads' level-0 writes, before the frame
    /// that samples them). The engine calls it at the start of every render, so the uploads between two frames are
    /// regenerated once, together; it costs a lock and a compare when nothing was uploaded.
    pub fn flush_surface_mips(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.surface_mips.is_empty() {
            return;
        }
        let gen_now = crate::surface::dirty_generation();
        if gen_now == self.surface_dirty_seen {
            return;
        }
        self.surface_dirty_seen = gen_now;
        // A paint surface's texture is 8-bit.
        let Some(gen) = self.mip_gens.get(&wgpu::TextureFormat::Rgba8Unorm) else { return };
        for (key, mips) in &self.surface_mips {
            let (Some(&token), Some(texture)) = (self.surface_tokens.get(key), self.textures.get(key)) else {
                continue;
            };
            let rects = crate::surface::take_dirty(token);
            if !rects.is_empty() {
                mips.regen(device, queue, gen, texture, rects);
            }
        }
    }

    /// Resident texture count and bytes, paint surfaces included (retained mode; `gpu_profile` memory report).
    pub fn resident(&self) -> (usize, usize) {
        (self.textures.len(), self.total_bytes + self.surface_bytes)
    }

    /// Task #95 R2d: evict least-recently-touched textures until under
    /// `budget` bytes. A miss only re-uploads — never correctness-bearing.
    /// Paint surfaces have their own budget (`crate::surface::budget_bytes`,
    /// mips counted): least-recently-drawn first, so layers outside the view go
    /// before the ones that drew last frame, and never one drawn this render. A
    /// surface's texture comes back filled from the retained tiles (native) or
    /// blank and marked for a refill from the host (wasm).
    pub fn trim(&mut self, budget: usize) {
        self.trim_lane(false, budget);
        let surface_budget = usize::try_from(crate::surface::budget_bytes()).unwrap_or(usize::MAX);
        self.trim_lane(true, surface_budget);
    }

    fn trim_lane(&mut self, surfaces: bool, budget: usize) {
        let used = |c: &Self| if surfaces { c.surface_bytes } else { c.total_bytes };
        if used(self) <= budget {
            return;
        }
        let mut keys: Vec<(String, u64)> = self
            .usage
            .iter()
            .filter(|(k, _)| self.surface_tokens.contains_key(*k) == surfaces)
            .map(|(k, (_, t))| (k.clone(), *t))
            .collect();
        keys.sort_by_key(|(_, t)| *t);
        for (key, tick) in keys {
            if used(self) <= budget {
                break;
            }
            if tick == self.tick {
                break; // never evict textures touched this frame
            }
            self.evict(&key, true);
        }
    }

    pub fn clear_gpu_resources(&mut self) {
        self.bind_groups.clear();
        self.heap_pixel_allocations.clear();
        for (key, texture) in self.textures.drain() {
            if let Some(token) = self.surface_tokens.remove(&key) {
                crate::surface::detach_texture(token, false);
            }
            self.surface_mips.remove(&key);
            if self.borrowed_textures.remove(&key).is_none() {
                crate::scratch::destroy(&texture);
            }
        }
        self.usage.clear();
        self.total_bytes = 0;
        self.surface_bytes = 0;
        self.pending_retry.clear();
    }

    /// Look up an already-prepared bind group (no upload). The executor
    /// pre-populates the cache with `bind_group(...)` before the render pass
    /// so pass-lifetime borrows stay immutable.
    /// A view of the texture uploaded under `key` (all its levels), if any.
    pub fn texture_view(&self, key: &str) -> Option<wgpu::TextureView> {
        self.textures.get(key).map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    pub fn get(&self, key: &str, clamp: bool) -> Option<&wgpu::BindGroup> {
        self.bind_groups.get(&(key.to_string(), clamp))
    }

    /// Upload (once) and return the texture bind group for a pattern texture.
    pub fn bind_group(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        key: &str,
        image: &DecodedImage,
        clamp: bool,
    ) -> &wgpu::BindGroup {
        let surface = image
            .external
            .as_ref()
            .is_some_and(|ext| crate::surface::is_surface_key(&ext.key));
        // A graph-pixel unit's pool texture is used as it is (one level); every other texture gets its mip chain.
        let borrowed = image
            .external
            .as_ref()
            .is_some_and(|ext| crate::external_texture::capture_texture(&ext.key).is_some());
        let levels = if surface {
            crate::surface::mip_levels(image.width, image.height)
        } else if borrowed {
            1
        } else {
            crate::surface::full_mip_levels(image.width, image.height)
        };
        // A decoded image in a working colour is half floats: twice the bytes.
        let float_image = image.external.is_none() && self.working.is_some();
        let bytes = crate::surface::texture_bytes(image.width, image.height, levels) as usize * if float_image { 2 } else { 1 };
        self.usage
            .entry(key.to_string())
            .and_modify(|(_, t)| *t = self.tick)
            .or_insert_with(|| (0, self.tick))
            .1 = self.tick;
        if image.external.is_some() && self.heap_pixel_allocations.contains_key(key) {
            self.evict(key, false);
        }
        if image.external.is_none() {
            if let Some(texture) = self.textures.get(key) {
                let oversized = image.width.max(image.height) > device.limits().max_texture_dimension_2d;
                let expected = if oversized { (1, 1) } else { (image.width, image.height) };
                if (texture.width(), texture.height()) != expected || !self.heap_pixel_allocations.contains_key(key) {
                    self.evict(key, false);
                } else if !self.heap_pixel_allocations[key].ptr_eq(&std::sync::Arc::downgrade(&image.rgba_premultiplied)) {
                    if !oversized {
                        let _upload = key.starts_with("video:").then(|| crate::prof::span("video.upload"));
                        let (texels, texel_bytes) = level0(self.working.as_ref(), image);
                        crate::gpu_profile::count_texture_write(texels.len());
                        queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            &texels,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(image.width * texel_bytes),
                                rows_per_image: Some(image.height),
                            },
                            wgpu::Extent3d {
                                width: image.width,
                                height: image.height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                    self.heap_pixel_allocations.insert(key.to_string(), std::sync::Arc::downgrade(&image.rgba_premultiplied));
                    self.regen_all_mips(device, queue, key);
                }
            }
        }
        if !self.textures.contains_key(key) {
            if image.external.is_none() {
                self.heap_pixel_allocations.insert(
                    key.to_string(),
                    std::sync::Arc::downgrade(&image.rgba_premultiplied),
                );
            }
            self.usage.insert(key.to_string(), (bytes, self.tick));
            if surface {
                self.surface_bytes += bytes;
            } else {
                self.total_bytes += bytes;
            }
            let extent = wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            };
            // Defensive belt (WEB-NEXTJS-21G): the sizing policy clamps every
            // decode to the live granted `max_texture_dimension_2d`, so an
            // oversized image should be unreachable — but creating a texture
            // past the device limit is a validation error that poisons every
            // pass binding it (a whole-frame blank), so guard here and render
            // a 1×1 transparent placeholder instead (the same graceful outcome
            // as an unavailable external bitmap).
            let max_dim = device.limits().max_texture_dimension_2d;
            let oversized = image.width.max(image.height) > max_dim;
            if oversized {
                eprintln!(
                    "moda-engine: WARN: pattern texture '{key}' ({}×{}) exceeds the device \
                     max_texture_dimension_2d ({max_dim}); rendering a transparent placeholder",
                    image.width, image.height
                );
            }
            let texture = if oversized {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(key),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            } else if let Some((capture_key, texture)) = image.external.as_ref().and_then(|ext| {
                crate::external_texture::capture_texture(&ext.key).map(|texture| (ext.key.clone(), texture))
            }) {
                // A graph-pixel unit's pool texture, already quantized to the bytes an upload would hold.
                self.borrowed_textures.insert(key.to_string(), capture_key);
                texture
            } else if let Some(ext) = &image.external {
                // Browser-decoded raster (wasm editor lane): the
                // decoded pixels never enter the wasm heap — the installed
                // ExternalImageUploader copies its ImageBitmap straight into
                // this texture. copyExternalImageToTexture's destination is a
                // render pass, so the texture also needs RENDER_ATTACHMENT.
                // Its mip chain is regenerated by render passes into each level (RENDER_ATTACHMENT again).
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(key),
                    size: extent,
                    mip_level_count: levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                // On a miss (no uploader installed, or the ImageBitmap is gone —
                // effectively unreachable: the wasm registry holds bitmaps for
                // the session) we leave the texture UNWRITTEN. wgpu lazily
                // zero-initializes textures, so it samples as transparent
                // instead of garbage — without allocating a decoded-size zero
                // buffer on the very heap this feature keeps pixels off of.
                let outcome = if surface {
                    // A paint layer's surface: filled (now or by later uploads) from the surface registry, which
                    // also keeps its mips current.
                    let token = crate::surface::attach_texture(&ext.key, queue, &texture);
                    self.surface_tokens.insert(key.to_string(), token);
                    if levels > 1 {
                        // Created mid-frame, after this render's flush: build the chain of what the registry just
                        // filled (native) now, before the frame that samples it is submitted.
                        let gen = self.mip_gens.entry(texture.format()).or_insert_with(|| crate::surface::MipGen::new(device, texture.format()));
                        let mips = crate::surface::SurfaceMips::new(device, gen, &texture);
                        mips.regen(device, queue, gen, &texture, crate::surface::take_dirty(token));
                        self.surface_mips.insert(key.to_string(), mips);
                    }
                    ExternalUploadOutcome::Uploaded
                } else {
                    let outcome = crate::external_texture::external_uploader()
                        .map(|u| u.upload(&ext.key, device, queue, &texture, image.width, image.height))
                        .unwrap_or(ExternalUploadOutcome::Unavailable);
                    if outcome == ExternalUploadOutcome::Uploaded && levels > 1 {
                        self.build_mips(device, queue, key, &texture);
                    }
                    outcome
                };
                if outcome == ExternalUploadOutcome::Skipped {
                    // Deliberately skipped this render (wasm offscreen snapshot
                    // guard): cache the transparent texture so this render pass
                    // still has a bind group, but mark the key so `begin_frame`
                    // evicts it next render and the real upload is re-attempted —
                    // it must not durably mask the live present's real pixels.
                    self.pending_retry.insert(key.to_string());
                }
                texture
            } else {
                let _upload = key.starts_with("video:").then(|| crate::prof::span("video.upload"));
                if key.starts_with("video:") && crate::prof::on() {
                    crate::prof::add_work("video.texture_allocations", 0, 1);
                }
                let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
                if levels > 1 {
                    usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
                }
                let (texels, texel_bytes) = level0(self.working.as_ref(), image);
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(key),
                    size: extent,
                    mip_level_count: levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // rgba8unorm, never -srgb (render-semantics §4.8/§7); half floats in a working colour.
                    format: if texel_bytes == 8 { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba8Unorm },
                    usage,
                    view_formats: &[],
                });
                crate::gpu_profile::count_texture_write(texels.len());
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &texels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(image.width * texel_bytes),
                        rows_per_image: Some(image.height),
                    },
                    extent,
                );
                if levels > 1 {
                    self.build_mips(device, queue, key, &texture);
                }
                texture
            };
            self.textures.insert(key.to_string(), texture);
        }

        let cache_key = (key.to_string(), clamp);
        if !self.bind_groups.contains_key(&cache_key) {
            let texture = &self.textures[key];
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = if clamp { &self.clamp_sampler } else { &self.repeat_sampler };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("pattern texture bg: {key}")),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            });
            self.bind_groups.insert(cache_key.clone(), bg);
        }
        &self.bind_groups[&cache_key]
    }
}

/// Level 0 of a decoded image's texture and its bytes per texel. In an sRGB engine, the image's premultiplied
/// sRGB bytes (four): as they are, or associated here when the image holds straight colour. In an engine that
/// blends in `working`, each texel's straight colour converted to that encoding and associated in floating
/// point, as half floats (eight): the texture is then filtered, and its mips are built, in the encoding the
/// engine blends in. A straight image (`DecodedImage::straight`) gives its colour as decoded; a premultiplied
/// one gives what its 8-bit association kept.
fn level0<'a>(working: Option<&moda_shaders::WorkingColor>, image: &'a DecodedImage) -> (std::borrow::Cow<'a, [u8]>, u32) {
    let Some(working) = working else {
        if !image.straight {
            return (std::borrow::Cow::Borrowed(image.rgba_premultiplied.as_slice()), 4);
        }
        let mut texels = image.rgba_premultiplied.as_ref().clone();
        crate::paint::image_source::premultiply_in_place(&mut texels);
        return (std::borrow::Cow::Owned(texels), 4);
    };
    let mut texels = Vec::with_capacity(image.rgba_premultiplied.len() * 2);
    for texel in image.rgba_premultiplied.chunks_exact(4) {
        let alpha = f32::from(texel[3]) / 255.0;
        let colour = if texel[3] == 0 {
            [0.0; 3]
        } else {
            let divisor = if image.straight { 255.0 } else { f32::from(texel[3]) };
            working.from_srgb.apply([texel[0], texel[1], texel[2]].map(|c| f32::from(c) / divisor)).map(|c| c * alpha)
        };
        for value in [colour[0], colour[1], colour[2], alpha] {
            texels.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    (std::borrow::Cow::Owned(texels), 8)
}

impl Drop for PatternTextureCache {
    fn drop(&mut self) {
        self.clear_gpu_resources();
    }
}

/// Build the pattern texture bind group LAYOUT (manifest BGL 15 — "Path
/// Pattern Texture Bind Group Layout": 0 = filtering sampler, 1 = float
/// texture, fragment-only).
pub fn pattern_texture_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Path Pattern Texture Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external_texture::{
        clear_external_uploader, install_external_uploader, lock_external_uploader_for_test,
        ExternalImageUploader,
    };
    use crate::paint::image_source::ExternalTextureRef;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A decoded image's level 0: its bytes as they are in an sRGB engine; in a working colour each texel
    /// unassociated, converted and associated again, as half floats.
    #[test]
    fn level0_holds_a_decoded_image_in_the_working_colour() {
        let image = DecodedImage { width: 2, height: 1, rgba_premultiplied: Arc::new(vec![128, 0, 0, 128, 0, 0, 0, 0]), straight: false, external: None };
        let (bytes, texel_bytes) = level0(None, &image);
        assert_eq!(texel_bytes, 4);
        assert_eq!(bytes.as_ref(), image.rgba_premultiplied.as_slice());
        // Linear light on sRGB's primaries.
        let linear = moda_shaders::WorkingColor {
            from_srgb: moda_shaders::working_color::Conversion { decode: true, matrix: None, encode: false },
            to_srgb: moda_shaders::working_color::Conversion { decode: false, matrix: None, encode: true },
        };
        let halves = |image: &DecodedImage| -> Vec<f32> {
            let (bytes, texel_bytes) = level0(Some(&linear), image);
            assert_eq!(texel_bytes, 8);
            bytes.chunks_exact(2).map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()).collect()
        };
        // Straight red at alpha 128/255, then a texel with no coverage.
        let alpha: f32 = 128.0 / 255.0;
        let texels = halves(&image);
        for (a, b) in texels.iter().zip([alpha, 0.0, 0.0, alpha, 0.0, 0.0, 0.0, 0.0]) {
            assert!((a - b).abs() < 1e-3, "{texels:?}");
        }
        // The transfer takes unassociated colour: code 64 at alpha 128 is straight code 0.5, light 0.214.
        let grey = DecodedImage { width: 1, height: 1, rgba_premultiplied: Arc::new(vec![64, 64, 64, 128]), straight: false, external: None };
        let texels = halves(&grey);
        assert!((texels[0] - 0.21404114 * alpha).abs() < 1e-3 && (texels[3] - alpha).abs() < 1e-3, "{texels:?}");
        // Straight texels keep the colour an 8-bit association rounds away: code 128 at alpha 1/255 is light
        // 0.2158 at that alpha, where its associated byte, 1, would read as white.
        let faint = DecodedImage { width: 1, height: 1, rgba_premultiplied: Arc::new(vec![128, 128, 128, 1]), straight: true, external: None };
        let texels = halves(&faint);
        assert!((texels[0] - 0.21586 / 255.0).abs() < 2e-5 && (texels[3] - 1.0 / 255.0).abs() < 1e-5, "{texels:?}");
        // An sRGB engine associates straight texels into the bytes it always uploaded.
        let straight = DecodedImage { width: 1, height: 1, rgba_premultiplied: Arc::new(vec![200, 100, 50, 128]), straight: true, external: None };
        let (bytes, texel_bytes) = level0(None, &straight);
        assert_eq!((bytes.as_ref(), texel_bytes), (&[100u8, 50, 25, 128][..], 4));
    }

    /// Records every `upload` call and returns a switchable outcome, so a test
    /// can assert exactly WHEN the pattern cache (re-)attempts an external copy.
    struct ProbeUploader {
        calls: AtomicUsize,
        /// 0 = `Uploaded`, 1 = `Skipped`, 2 = `Unavailable`.
        outcome: AtomicU8,
    }

    impl ExternalImageUploader for ProbeUploader {
        fn upload(
            &self,
            _key: &str,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _texture: &wgpu::Texture,
            _width: u32,
            _height: u32,
        ) -> ExternalUploadOutcome {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match self.outcome.load(Ordering::Relaxed) {
                1 => ExternalUploadOutcome::Skipped,
                2 => ExternalUploadOutcome::Unavailable,
                _ => ExternalUploadOutcome::Uploaded,
            }
        }
    }

    /// Clears the process-global uploader slot on drop, so a panicking
    /// assertion can't leak a test uploader into a later test in this
    /// (`--test-threads=1`) GPU binary.
    struct UploaderSlotGuard;
    impl Drop for UploaderSlotGuard {
        fn drop(&mut self) {
            clear_external_uploader();
        }
    }

    fn external_image(asset_key: &str) -> DecodedImage {
        // A heap-free external raster carries no pixels — the uploader fills the
        // texture. The 4×4 dims only drive the (cache-bookkeeping) byte count.
        // `asset_key` is the bare registry key the uploader receives; in
        // production it differs from the COMPOSITE cache key (`pattern_draw_key`
        // embeds state + upload_id), so the tests keep them distinct to pin that
        // the cache (textures, bind groups, `pending_retry`) keys by the CACHE
        // key the pass uses — not the asset key.
        DecodedImage {
            width: 4,
            height: 4,
            rgba_premultiplied: Arc::new(Vec::new()),
            straight: false,
            external: Some(ExternalTextureRef {
                key: asset_key.to_string(),
                upload_id: 1,
            }),
        }
    }

    // These tests share the process-global uploader slot, so each holds the
    // common test lock while it initializes the GPU and exercises the cache.

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn heap_stream_updates_reuse_storage_and_invalidate_resized_storage() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let layout = pattern_texture_bgl(&gpu.device);
        let mut cache = PatternTextureCache::new(&gpu.device, None);
        let mut image = DecodedImage {
            width: 4,
            height: 4,
            rgba_premultiplied: Arc::new(vec![255; 64]),
            straight: false,
            external: None,
        };
        let key = "video:clip|Loaded|4x4";
        cache.bind_group(&gpu.device, &gpu.queue, &layout, key, &image, true);
        let texture = cache.textures[key].clone();
        let first_pixels = Arc::downgrade(&image.rgba_premultiplied);
        image.rgba_premultiplied = Arc::new(vec![0; 64]);
        cache.begin_frame();
        cache.bind_group(&gpu.device, &gpu.queue, &layout, key, &image, false);
        assert_eq!(cache.textures[key], texture);
        // 4×4 with its mip chain (4×4, 2×2, 1×1).
        assert_eq!(cache.total_bytes, 64 + 16 + 4);
        assert!(cache.heap_pixel_allocations[key].ptr_eq(&Arc::downgrade(&image.rgba_premultiplied)));
        assert!(first_pixels.upgrade().is_none());
        cache.bind_group(&gpu.device, &gpu.queue, &layout, key, &image, true);
        assert_eq!(cache.textures[key], texture);
        image.width = 2;
        image.rgba_premultiplied = Arc::new(vec![128; 32]);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, key, &image, true);
        assert_ne!(cache.textures[key], texture);
        // 2×4, 1×2, 1×1.
        assert_eq!(cache.total_bytes, 32 + 8 + 4);
        cache.clear_gpu_resources();
        assert!(cache.heap_pixel_allocations.is_empty());
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn heap_pixel_identity_stays_pinned_until_texture_eviction_without_retaining_pixels() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let image = DecodedImage {
            width: 4,
            height: 4,
            rgba_premultiplied: Arc::new(vec![255; 64]),
            straight: false,
            external: None,
        };
        let layout = pattern_texture_bgl(&gpu.device);
        let mut cache = PatternTextureCache::new(&gpu.device, None);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "heap", &image, true);
        assert_eq!(Arc::weak_count(&image.rgba_premultiplied), 1);
        assert_eq!(Arc::strong_count(&image.rgba_premultiplied), 1);
        cache.evict("heap", false);
        assert_eq!(Arc::weak_count(&image.rgba_premultiplied), 0);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "heap", &image, true);
        assert_eq!(Arc::weak_count(&image.rgba_premultiplied), 1);
        cache.clear_gpu_resources();
        assert_eq!(Arc::weak_count(&image.rgba_premultiplied), 0);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "heap", &image, true);
        let pixels = Arc::downgrade(&image.rgba_premultiplied);
        drop(image);
        assert!(pixels.upgrade().is_none());
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    /// A graph-pixel unit's pool texture bound through the cache survives eviction and a cache clear (the pool owns
    /// it), and leaves the registry when its `CaptureTexture` drops.
    fn native_capture_cache_does_not_destroy_pool_texture() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let exec = crate::pixel::Executor::new(&gpu.device, &gpu.queue, 1 << 20);
        let texture = exec.passes.upload(&exec.pool, 4, 4, crate::pixel::Storage::U8, &[0.5; 64]).unwrap();
        let capture = crate::external_texture::CaptureTexture::new(texture.clone());
        let image = external_image(capture.key());
        let layout = pattern_texture_bgl(&gpu.device);
        let mut cache = PatternTextureCache::new(&gpu.device, None);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "capture", &image, true);
        cache.evict("capture", false);
        exec.passes.download(&texture).expect("eviction must leave pooled capture alive");
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "capture", &image, true);
        cache.clear_gpu_resources();
        exec.passes.download(&texture).expect("cache clear must leave pooled capture alive");
        let key = capture.key().to_string();
        drop(capture);
        assert!(crate::external_texture::capture_texture(&key).is_none());
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn native_capture_cache_expires_only_unregistered_borrowed_entries() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let exec = crate::pixel::Executor::new(&gpu.device, &gpu.queue, 1 << 20);
        let texture = exec.passes.upload(&exec.pool, 4, 4, crate::pixel::Storage::U8, &[0.5; 64]).unwrap();
        let outer = crate::external_texture::CaptureTexture::new(texture.clone());
        let inner = crate::external_texture::CaptureTexture::new(texture.clone());
        let layout = pattern_texture_bgl(&gpu.device);
        let mut cache = PatternTextureCache::new(&gpu.device, None);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "outer", &external_image(outer.key()), true);
        cache.bind_group(&gpu.device, &gpu.queue, &layout, "inner", &external_image(inner.key()), true);
        cache.begin_frame();
        assert!(cache.get("outer", true).is_some());
        assert!(cache.get("inner", true).is_some());
        drop(inner);
        cache.begin_frame();
        assert!(cache.get("outer", true).is_some());
        assert!(cache.get("inner", true).is_none());
        assert_eq!(cache.resident(), (1, 64));
        drop(outer);
        cache.begin_frame();
        assert_eq!(cache.resident(), (0, 0));
        assert!(cache.borrowed_textures.is_empty());
        exec.passes.download(&texture).expect("expiry must leave pooled texture alive");
    }

    /// De-poison contract: a `Skipped` external upload (the wasm offscreen
    /// snapshot guard) still caches a bind group for the current pass, but is
    /// re-attempted on the NEXT render — so it can't durably mask the real pixels
    /// a later live present uploads under the same (present-shared) cache key.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn skipped_external_upload_is_retried_next_render() {
        let _test_lock = lock_external_uploader_for_test();
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let (device, queue) = (&gpu.device, &gpu.queue);
        let layout = pattern_texture_bgl(device);
        let mut cache = PatternTextureCache::new(device, None);

        let probe = Arc::new(ProbeUploader {
            calls: AtomicUsize::new(0),
            outcome: AtomicU8::new(1), // Skipped (snapshot render)
        });
        install_external_uploader(probe.clone());
        let _slot = UploaderSlotGuard;
        // Distinct composite cache key vs bare asset key (see `external_image`).
        let cache_key = "ext-skip|Loaded|ext1";
        let image = external_image("ext-skip-asset");

        // Render 1 (snapshot-style skip): the pass still needs a bind group, so
        // the transparent texture IS cached this render.
        cache.begin_frame();
        cache.bind_group(device, queue, &layout, cache_key, &image, true);
        assert!(
            cache.get(cache_key, true).is_some(),
            "a skipped upload still caches a bind group so the render pass can draw"
        );
        assert_eq!(probe.calls.load(Ordering::Relaxed), 1);

        // Render 2 (live present): begin_frame must evict the skipped placeholder
        // so bind_group re-attempts the real upload — the call count climbs.
        probe.outcome.store(0, Ordering::Relaxed); // Uploaded now
        cache.begin_frame();
        cache.bind_group(device, queue, &layout, cache_key, &image, true);
        assert_eq!(
            probe.calls.load(Ordering::Relaxed),
            2,
            "a skipped external upload must be re-attempted on the next render (de-poison)"
        );

        // Render 3: the previous upload SUCCEEDED, so it stays cached — NOT
        // re-uploaded every render (no churn for healthy textures).
        cache.begin_frame();
        cache.bind_group(device, queue, &layout, cache_key, &image, true);
        assert_eq!(
            probe.calls.load(Ordering::Relaxed),
            2,
            "a successful external upload is cached, not re-uploaded each render"
        );
    }

    /// `Unavailable` (no bitmap / no uploader) is the pre-existing behavior: the
    /// transparent texture is cached as-is and NOT retried every render, so a
    /// genuinely-missing key never churns a fresh texture per frame.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn unavailable_external_upload_is_not_retried() {
        let _test_lock = lock_external_uploader_for_test();
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let (device, queue) = (&gpu.device, &gpu.queue);
        let layout = pattern_texture_bgl(device);
        let mut cache = PatternTextureCache::new(device, None);

        let probe = Arc::new(ProbeUploader {
            calls: AtomicUsize::new(0),
            outcome: AtomicU8::new(2), // Unavailable
        });
        install_external_uploader(probe.clone());
        let _slot = UploaderSlotGuard;
        let cache_key = "ext-missing|Loaded|ext1";
        let image = external_image("ext-missing-asset");

        cache.begin_frame();
        cache.bind_group(device, queue, &layout, cache_key, &image, true);
        assert_eq!(probe.calls.load(Ordering::Relaxed), 1);

        // Next render: nothing was queued for retry, so the cached (transparent)
        // texture is reused — the uploader is NOT called again.
        cache.begin_frame();
        cache.bind_group(device, queue, &layout, cache_key, &image, true);
        assert_eq!(
            probe.calls.load(Ordering::Relaxed),
            1,
            "an Unavailable upload is cached as-is, not re-attempted every render"
        );
    }
}
