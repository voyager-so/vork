//! [`TexturePool`]: the pixel graph's textures, reused across nodes and frames under a byte budget.
//!
//! A texture is handed out as a [`PoolTexture`], a shared handle: images are immutable once made, so a node's
//! output can feed several consumers. When the last handle drops, the texture goes back to the pool's free list
//! for the next request of the same size and storage. The budget caps the bytes *in use*; free textures are
//! destroyed, oldest first, when keeping them would go past it.
//!
//! f16 1080p is 16.6 MB and 4K is 66 MB. Export callers can adjust the allowance to their output and memory policy.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::image::Storage;
use super::PixelError;

/// The default budget: 1 GiB (`MODA_PIXEL_BUDGET_MB` overrides it).
pub const DEFAULT_BUDGET_BYTES: u64 = 1 << 30;

/// Every pool texture can be sampled, drawn into and copied both ways.
const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Bytes held by live handles.
    pub in_use: u64,
    /// In-use plus free bytes (everything allocated on the GPU).
    pub resident: u64,
    /// The largest `in_use` seen since the pool was made (or [`TexturePool::reset_peak`]).
    pub peak: u64,
    pub created: u64,
    pub reused: u64,
    pub budget: u64,
}

struct Free {
    texture: wgpu::Texture,
    w: u32,
    h: u32,
    storage: Storage,
    /// Release order, for oldest-first eviction.
    stamp: u64,
}

struct State {
    free: Vec<Free>,
    stats: PoolStats,
    clock: u64,
}

struct Inner {
    texture: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    storage: Storage,
    pool: Weak<RefCell<State>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let Some(texture) = self.texture.take() else { return };
        let bytes = bytes_for(self.w, self.h, self.storage);
        match self.pool.upgrade() {
            Some(state) => {
                let mut s = state.borrow_mut();
                s.stats.in_use -= bytes;
                s.clock += 1;
                let stamp = s.clock;
                s.free.push(Free {
                    texture,
                    w: self.w,
                    h: self.h,
                    storage: self.storage,
                    stamp,
                });
                trim(&mut s, 0);
            }
            None => texture.destroy(),
        }
    }
}

/// A shared handle to a pooled texture.
#[derive(Clone)]
pub struct PoolTexture(Rc<Inner>);

impl std::fmt::Debug for PoolTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PoolTexture({}×{} {:?})", self.0.w, self.0.h, self.0.storage)
    }
}

impl PoolTexture {
    pub fn texture(&self) -> &wgpu::Texture {
        self.0.texture.as_ref().expect("live")
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.0.view
    }

    pub fn size(&self) -> (u32, u32) {
        (self.0.w, self.0.h)
    }

    pub fn storage(&self) -> Storage {
        self.0.storage
    }

    pub fn bytes(&self) -> u64 {
        bytes_for(self.0.w, self.0.h, self.0.storage)
    }

    /// Two handles to one texture.
    pub fn same(&self, o: &PoolTexture) -> bool {
        Rc::ptr_eq(&self.0, &o.0)
    }
}

fn bytes_for(w: u32, h: u32, storage: Storage) -> u64 {
    w as u64 * h as u64 * storage.bytes_per_pixel()
}

/// Bytes of free textures the pool keeps for reuse, whatever the budget leaves room for. Captures the pool adopts
/// come back to the free list at their own sizes, which few later requests match (a 3D unit's capture is sized to
/// its view, and a diorama frame drew 7 of ~67 MB): kept up to the whole budget they were most of a renderer's
/// footprint (a diorama renderer peaked at 3.5 GB, 2.7 GB this way, at the same speed).
pub const FREE_KEPT_BYTES: u64 = 256 << 20;

/// Destroy free textures, oldest first, until in-use + free + `extra` fits the budget and the free ones fit
/// [`FREE_KEPT_BYTES`].
fn trim(s: &mut State, extra: u64) {
    while (s.stats.resident + extra > s.stats.budget || s.stats.resident - s.stats.in_use > FREE_KEPT_BYTES) && !s.free.is_empty() {
        let oldest = s
            .free
            .iter()
            .enumerate()
            .min_by_key(|(_, f)| f.stamp)
            .map(|(i, _)| i)
            .expect("non-empty");
        let f = s.free.swap_remove(oldest);
        s.stats.resident -= bytes_for(f.w, f.h, f.storage);
        crate::scratch::destroy(&f.texture);
    }
}

pub struct TexturePool {
    device: wgpu::Device,
    state: Rc<RefCell<State>>,
    max_dim: u32,
}

impl TexturePool {
    pub fn new(device: &wgpu::Device, budget_bytes: u64) -> Self {
        Self {
            device: device.clone(),
            max_dim: device.limits().max_texture_dimension_2d,
            state: Rc::new(RefCell::new(State {
                free: Vec::new(),
                stats: PoolStats {
                    budget: budget_bytes,
                    ..Default::default()
                },
                clock: 0,
            })),
        }
    }

    /// The budget from `MODA_PIXEL_BUDGET_MB`, or [`DEFAULT_BUDGET_BYTES`].
    pub fn budget_from_env() -> u64 {
        std::env::var("MODA_PIXEL_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|mb| *mb > 0)
            .map_or(DEFAULT_BUDGET_BYTES, |mb| mb << 20)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn max_dimension(&self) -> u32 {
        self.max_dim
    }

    pub fn stats(&self) -> PoolStats {
        self.state.borrow().stats
    }

    pub fn reset_peak(&self) {
        let mut s = self.state.borrow_mut();
        s.stats.peak = s.stats.in_use;
    }

    pub fn set_budget(&self, bytes: u64) {
        let mut state = self.state.borrow_mut();
        state.stats.budget = bytes;
        trim(&mut state, 0);
    }

    /// Drop every free texture (in-use ones return to an empty free list as they're released).
    pub fn clear_free(&self) {
        let mut s = self.state.borrow_mut();
        for f in std::mem::take(&mut s.free) {
            s.stats.resident -= bytes_for(f.w, f.h, f.storage);
            crate::scratch::destroy(&f.texture);
        }
    }

    /// A `w × h` texture of `storage` (contents undefined: callers draw every texel or clear first).
    pub fn acquire(&self, w: u32, h: u32, storage: Storage) -> Result<PoolTexture, PixelError> {
        if w == 0 || h == 0 {
            return Err(PixelError::Invalid("an image needs a non-empty domain".into()));
        }
        if w > self.max_dim || h > self.max_dim {
            return Err(PixelError::TooLarge {
                w,
                h,
                max: self.max_dim,
            });
        }
        let bytes = bytes_for(w, h, storage);
        let mut s = self.state.borrow_mut();
        let reuse = s
            .free
            .iter()
            .position(|f| f.w == w && f.h == h && f.storage == storage && f.texture.usage().contains(USAGE));
        let texture = match reuse {
            Some(i) => {
                if s.stats.in_use + bytes > s.stats.budget {
                    return Err(PixelError::Budget {
                        needed: bytes,
                        in_use: s.stats.in_use,
                        budget: s.stats.budget,
                    });
                }
                s.stats.reused += 1;
                s.free.swap_remove(i).texture
            }
            None => {
                if s.stats.in_use + bytes > s.stats.budget {
                    return Err(PixelError::Budget {
                        needed: bytes,
                        in_use: s.stats.in_use,
                        budget: s.stats.budget,
                    });
                }
                trim(&mut s, bytes);
                s.stats.created += 1;
                s.stats.resident += bytes;
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("pixel graph image"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: storage.format(),
                    usage: USAGE,
                    view_formats: &[],
                })
            }
        };
        s.stats.in_use += bytes;
        s.stats.peak = s.stats.peak.max(s.stats.in_use);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(PoolTexture(Rc::new(Inner {
            texture: Some(texture),
            view,
            w,
            h,
            storage,
            pool: Rc::downgrade(&self.state),
        })))
    }

    /// Wrap a texture the renderer made (a capture) so it counts against the budget and returns to the free list.
    /// It must be `Rgba16Float` or `Rgba32Float` with at least the pool's usages.
    pub fn adopt(&self, texture: wgpu::Texture) -> Result<PoolTexture, PixelError> {
        let storage = match texture.format() {
            wgpu::TextureFormat::Rgba16Float => Storage::F16,
            wgpu::TextureFormat::Rgba32Float => Storage::F32,
            f => return Err(PixelError::Invalid(format!("the pixel graph can't adopt a {f:?} texture"))),
        };
        if !texture.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC) {
            return Err(PixelError::Invalid("an adopted texture must be sampled and copied".into()));
        }
        let (w, h) = (texture.width(), texture.height());
        let bytes = bytes_for(w, h, storage);
        let mut s = self.state.borrow_mut();
        if s.stats.in_use + bytes > s.stats.budget {
            return Err(PixelError::Budget {
                needed: bytes,
                in_use: s.stats.in_use,
                budget: s.stats.budget,
            });
        }
        trim(&mut s, bytes);
        s.stats.resident += bytes;
        s.stats.in_use += bytes;
        s.stats.peak = s.stats.peak.max(s.stats.in_use);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(PoolTexture(Rc::new(Inner {
            texture: Some(texture),
            view,
            w,
            h,
            storage,
            pool: Rc::downgrade(&self.state),
        })))
    }
}

impl Drop for TexturePool {
    fn drop(&mut self) {
        self.clear_free();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> crate::device::GpuContext {
        crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter")
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn reuse_budget_and_eviction() {
        let gpu = gpu();
        // 64×64 f16 = 32 KiB; a budget of three of them.
        let unit = bytes_for(64, 64, Storage::F16);
        let pool = TexturePool::new(&gpu.device, unit * 3);
        let a = pool.acquire(64, 64, Storage::F16).unwrap();
        let b = a.clone();
        assert!(a.same(&b));
        assert_eq!(pool.stats().in_use, unit, "a clone shares the texture");
        let c = pool.acquire(64, 64, Storage::F16).unwrap();
        let d = pool.acquire(64, 64, Storage::F16).unwrap();
        assert!(matches!(pool.acquire(64, 64, Storage::F16), Err(PixelError::Budget { .. })));
        drop(c);
        assert_eq!(pool.stats().in_use, unit * 2);
        assert_eq!(pool.stats().resident, unit * 3);
        let e = pool.acquire(64, 64, Storage::F16).unwrap();
        assert_eq!(pool.stats().reused, 1, "the released texture comes back");
        drop((e, d));
        // A different size evicts free textures rather than exceeding the budget.
        let big = pool.acquire(64, 128, Storage::F16).unwrap();
        assert!(pool.stats().resident <= pool.stats().budget);
        assert_eq!(pool.stats().in_use, unit * 3);
        drop((a, b, big));
        assert_eq!(pool.stats().in_use, 0);
        assert_eq!(pool.stats().peak, unit * 3);
        assert!(matches!(pool.acquire(0, 4, Storage::F16), Err(PixelError::Invalid(_))));
        let max = pool.max_dimension();
        assert!(matches!(pool.acquire(max + 1, 1, Storage::F32), Err(PixelError::TooLarge { .. })));
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn budget_reduction_releases_free_textures_not_live_handles() {
        let gpu = gpu();
        let unit = bytes_for(64, 64, Storage::F16);
        let pool = TexturePool::new(&gpu.device, unit * 3);
        let live = pool.acquire(64, 64, Storage::F16).unwrap();
        let free = pool.acquire(64, 64, Storage::F16).unwrap();
        drop(free);
        pool.set_budget(unit);
        assert_eq!(pool.stats().resident, unit);
        assert_eq!(pool.stats().in_use, unit);
        assert!(matches!(pool.acquire(64, 64, Storage::F16), Err(PixelError::Budget { .. })));
        drop(live);
        assert!(pool.acquire(64, 64, Storage::F16).is_ok());
    }

    #[test]
    fn budget_env() {
        // Read-only: the env var isn't set in tests.
        if std::env::var_os("MODA_PIXEL_BUDGET_MB").is_none() {
            assert_eq!(TexturePool::budget_from_env(), DEFAULT_BUDGET_BYTES);
        }
    }
}
