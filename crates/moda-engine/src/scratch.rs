//! GPU buffers and textures a render makes and drops (uniforms, vertex streams, a layer capture's target and
//! filter passes), kept and handed out again once the GPU is done with them. Every engine's renders use it: a
//! retained engine keeps its outermost frame's buffers and targets itself, and its captures take theirs here.
//!
//! An export renders thousands of small captures a frame (a shadow per shard of a motion-blurred fracture, the
//! effects of a diorama's planes), and every one created its buffers and textures afresh: on a paravirtual GPU each
//! creation is a host round trip (shatter-1 on the macOS runner: 7.8 M buffers and 1.8 M textures, 154 s of 998 s).
//!
//! A resource is taken inside an outermost render (one generation) and goes back to the pool when that render's
//! work is done on the GPU (`Queue::on_submitted_work_done`, which a later poll runs). Keys are exact: the same
//! size, format and usage, so a reused resource is indistinguishable from a fresh one to everything that writes it
//! in full before reading it (every caller here: a buffer is written whole or bound at the length written, a
//! texture is a render target cleared or covered by its pass). Outside a render nothing is pooled. What rests
//! unused is kept up to the bytes a generation takes, most recently taken first ([`evict_resting`]). What must
//! outlive its generation (a capture result an engine keeps for later frames) is withdrawn ([`ScratchPool::keep`]).
//!
//! A bind group over pooled resources is kept under what it binds, its layout and those resources
//! ([`ScratchPool::bind`]), for as long as the pool holds them all: a frame that takes the same resources for the
//! same passes as an earlier one makes no bind group. A pooled texture has one default view
//! ([`ScratchPool::view`]), so a bind group names it the same way every time.
//!
//! A render says when it starts whether its command buffers are held ([`ScratchPool::enter`]). Held, the command
//! buffers it submits (its captures', their filters', its mask chunks', its own) go to the queue together, in the
//! order they were made, when the outermost render ends ([`submit`]): one submission a render instead of several
//! per capture (shatter-1: 287 k captures, 373 s of submits on the macOS runner). The GPU runs the same command
//! buffers in the same order. Not held, each goes to the queue as it is submitted, and the GPU works on a frame's
//! captures while the rest of the frame is recorded. A retained engine does not hold: on Vulkan a frame of a
//! hundred captures held to its end is about 1,200 native command buffers against the backend's 128 pending, so the
//! submit call waited on the GPU part by part, and the encoders held meanwhile took new native command storage
//! (#2071's L4 tests: 60 frames of 100 blurs went from 3.05 s to 4.72 s). What moves when held is the start of the writes the
//! queue makes on submit (`write_buffer`, `write_texture`): every one a render makes goes to a resource made or taken
//! from the pool for that one use, so none lands under a command buffer still waiting to read what was there. Anything
//! that waits for the GPU (a readback, a map, a timed agent-shader pass) submits what is held first ([`flush`]).
//!
//! The writes a held command buffer reads ([`write_buffer`]) are held with it. `Queue::write_buffer` makes a staging
//! buffer for every call, and the queue frees each once its submission is done: on Metal a kernel round trip apiece
//! (shatter-1 on the macOS runner: 13% of the render thread). Held writes are copied into one staging buffer, and a
//! command buffer at the front of the submission copies them into place, in the order they were made. They land
//! where the queue's own writes would, before every command buffer of that submission.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

thread_local! {
    /// Renders open on this thread that hold their command buffers, and the command buffers they have submitted so far.
    static HELD: RefCell<(u32, Vec<(wgpu::Queue, wgpu::CommandBuffer)>)> = const { RefCell::new((0, Vec::new())) };
    /// The writes held with them ([`write_buffer`]), in the order they were made, and their bytes end to end.
    static WRITES: RefCell<(Vec<HeldWrite>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// One held [`write_buffer`]: `len` bytes from `at` in [`WRITES`]' bytes, for `buffer` at `offset`.
struct HeldWrite {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffer: wgpu::Buffer,
    offset: u64,
    at: usize,
    len: usize,
}

/// `queue.write_buffer(buffer, offset, data)`: held with the command buffers while a render is open on this thread
/// (they are the ones that read it), at once otherwise.
pub(crate) fn write_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    if data.is_empty() || HELD.with(|h| h.borrow().0 == 0) {
        queue.write_buffer(buffer, offset, data);
        return;
    }
    WRITES.with(|w| {
        let mut w = w.borrow_mut();
        let at = w.1.len();
        w.1.extend_from_slice(data);
        w.0.push(HeldWrite { device: device.clone(), queue: queue.clone(), buffer: buffer.clone(), offset, at, len: data.len() });
    });
}

/// The held writes as command buffers, one per queue: each copies that queue's writes out of one staging buffer, in
/// order. Sizes and offsets are multiples of 4, as `write_buffer` requires.
fn held_writes() -> Vec<(wgpu::Queue, wgpu::CommandBuffer)> {
    let (writes, bytes) = WRITES.with(|w| std::mem::take(&mut *w.borrow_mut()));
    let mut out: Vec<(wgpu::Queue, wgpu::CommandBuffer)> = Vec::new();
    let mut queues: Vec<&wgpu::Queue> = Vec::new();
    for w in &writes {
        if !queues.contains(&&w.queue) {
            queues.push(&w.queue);
        }
    }
    for queue in queues {
        let mine: Vec<&HeldWrite> = writes.iter().filter(|w| w.queue == *queue).collect();
        let device = &mine[0].device;
        let size: usize = mine.iter().map(|w| w.len).sum();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("held writes"),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let mut data = Vec::with_capacity(size);
        for w in &mine {
            data.extend_from_slice(&bytes[w.at..w.at + w.len]);
        }
        staging.slice(..).get_mapped_range_mut().copy_from_slice(&data);
        staging.unmap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("held writes") });
        let mut at = 0u64;
        for w in &mine {
            encoder.copy_buffer_to_buffer(&staging, at, &w.buffer, w.offset, w.len as u64);
            at += w.len as u64;
        }
        out.push((queue.clone(), encoder.finish()));
    }
    out
}

/// Submit `buffers` to `queue`: held until the outermost holding render on this thread ends while one is open, at
/// once otherwise.
pub(crate) fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(queue: &wgpu::Queue, buffers: I) {
    let mut buffers: Vec<wgpu::CommandBuffer> = buffers.into_iter().collect();
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        if h.0 > 0 {
            h.1.extend(buffers.drain(..).map(|b| (queue.clone(), b)));
        }
    });
    if !buffers.is_empty() {
        queue.submit(buffers);
    }
}

/// Submit everything held on this thread, in order (before a wait for the GPU, and when the outermost render ends).
pub fn flush() {
    // A queue's held writes go first, where its own writes would land.
    let mut held = held_writes();
    held.extend(HELD.with(|h| std::mem::take(&mut h.borrow_mut().1)));
    let mut run: Vec<wgpu::CommandBuffer> = Vec::new();
    let mut to: Option<wgpu::Queue> = None;
    for (queue, buffer) in held {
        if to.as_ref().is_some_and(|q| *q != queue) {
            to.take().expect("some").submit(std::mem::take(&mut run));
        }
        to.get_or_insert(queue);
        run.push(buffer);
    }
    if let Some(q) = to {
        q.submit(run);
    }
}

/// A GPU resource an explicit `destroy` frees at once.
pub(crate) trait Destroy {
    fn destroy_now(&self);
}

impl Destroy for wgpu::Texture {
    fn destroy_now(&self) {
        self.destroy();
    }
}

impl Destroy for wgpu::Buffer {
    fn destroy_now(&self) {
        self.destroy();
    }
}

/// `resource.destroy()`, except while this thread holds command buffers ([`submit`]) or writes ([`write_buffer`]): one
/// of them may use it, so it is only dropped (freed once the GPU is done with it).
/// Destructors can run after either TLS key is destroyed, or while scratch is mutably borrowed.
/// Neither case proves that pending work is empty: leave release to wgpu's handle ownership.
pub(crate) fn destroy<R: Destroy>(resource: &R) {
    if HELD.try_with(|h| h.try_borrow().is_ok_and(|h| h.1.is_empty())).unwrap_or(false)
        && WRITES.try_with(|w| w.try_borrow().is_ok_and(|w| w.0.is_empty())).unwrap_or(false)
    {
        resource.destroy_now();
    }
}

/// One more `name` in the export profile (`MODA_RENDER_PROFILE`), for a pooled take: `reused`, or made because no
/// resource had its key (`new_key`) or every one with it was still in use on the GPU (`all_busy`).
fn count(name: &'static str) {
    if crate::prof::on() {
        crate::prof::add_work(name, 0, 1);
    }
}

/// How many generations an unused bind group is kept before it is dropped.
const KEEP_GENERATIONS: u64 = 64;

/// The bytes of a texture, near enough to weigh one against another (mips beyond the first are not counted).
fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    let texel = u64::from(texture.format().block_copy_size(None).unwrap_or(4));
    u64::from(texture.width())
        * u64::from(texture.height())
        * u64::from(texture.depth_or_array_layers())
        * u64::from(texture.sample_count())
        * texel
}

/// Drop the resting resources of `map` beyond `budget` bytes, least recently taken first. Resting: last taken
/// before generation `done`, the newest finished one. Whole generations are kept, newest first, while they fit.
fn evict_resting<K: std::hash::Hash + Eq, T>(
    map: &mut HashMap<K, VecDeque<Slot<T>>>,
    done: u64,
    budget: u64,
    bytes: impl Fn(&T) -> u64,
    mut dropped: impl FnMut(&T),
) {
    let mut by_gen: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for list in map.values() {
        for slot in list.iter().filter(|s| s.gen < done) {
            *by_gen.entry(slot.gen).or_default() += bytes(&slot.item);
        }
    }
    // Everything last taken at or before `cutoff` goes (generations count from 1).
    let (mut kept, mut cutoff) = (0u64, 0u64);
    for (&gen, &size) in by_gen.iter().rev() {
        if kept + size > budget {
            cutoff = gen;
            break;
        }
        kept += size;
    }
    if cutoff == 0 {
        return;
    }
    for list in map.values_mut() {
        list.retain(|slot| {
            let goes = slot.gen <= cutoff;
            if goes {
                dropped(&slot.item);
            }
            !goes
        });
    }
    map.retain(|_, list| !list.is_empty());
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct TexKey {
    w: u32,
    h: u32,
    layers: u32,
    mips: u32,
    samples: u32,
    dim: u8,
    format: wgpu::TextureFormat,
    usage: u32,
}

impl TexKey {
    /// The key `texture` is pooled under, if it is.
    fn of(texture: &wgpu::Texture) -> TexKey {
        TexKey {
            w: texture.width(),
            h: texture.height(),
            layers: texture.depth_or_array_layers(),
            mips: texture.mip_level_count(),
            samples: texture.sample_count(),
            dim: texture.dimension() as u8,
            format: texture.format(),
            usage: texture.usage().bits(),
        }
    }
}

/// One pooled resource: the generation that last took it, and whether it has been handed back since
/// ([`ScratchPool::release`]: free now, whatever the GPU has finished, because whatever uses it next is queued
/// behind what used it last).
struct Slot<T> {
    gen: u64,
    free: bool,
    item: T,
}

/// What the pool knows of a texture it holds beside its slot: its default view, made when first asked for
/// ([`ScratchPool::view`]), and the kept bind groups that name it (the binding map owns their keys: a key names
/// every resource of its group, so an owning reference here would keep a withdrawn texture alive through another).
#[derive(Default)]
struct HeldTexture {
    view: Option<wgpu::TextureView>,
    bindings: Vec<Weak<BindKey>>,
}

/// One resource of a bind group ([`ScratchPool::bind`]): a buffer bound whole, a texture view or a sampler.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum Bound {
    Buffer(wgpu::Buffer),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

/// What a kept bind group binds: its layout and resources, binding `i` the `i`th.
type BindKey = (wgpu::BindGroupLayout, Vec<Bound>);

/// A position in the open generation's takes ([`ScratchPool::mark`]).
#[derive(Clone, Copy)]
pub(crate) struct Scope(usize, usize);

/// A kept bind group and the generation that last used it.
struct Binding {
    gen: u64,
    group: wgpu::BindGroup,
}

#[derive(Default)]
pub struct ScratchPool {
    depth: Cell<u32>,
    gen: Cell<u64>,
    done: Arc<AtomicU64>,
    /// Per key, the pooled resources: those handed back ([`ScratchPool::release`]) at the front, then the rest in
    /// the order they were last taken. The front alone says whether any is free, and a take is the front moved to
    /// the back or a new one at the back.
    buffers: RefCell<HashMap<(u64, u32), VecDeque<Slot<wgpu::Buffer>>>>,
    textures: RefCell<HashMap<TexKey, VecDeque<Slot<wgpu::Texture>>>>,
    /// The buffers and the textures the open generation has taken and not handed back, in order: a position in
    /// each is a scope ([`ScratchPool::mark`], [`ScratchPool::release`]).
    scope: RefCell<(Vec<wgpu::Buffer>, Vec<wgpu::Texture>)>,
    /// The bytes of buffers and of textures the open generation has taken so far: what may rest in the pool once
    /// it ends ([`evict_resting`]).
    taken: Cell<[u64; 2]>,
    /// Every buffer and texture in the queues above, for a direct answer to "does the pool hold this?", each with
    /// the kept bind groups that name it (weakly: the binding map owns their keys).
    held_buffers: RefCell<HashMap<wgpu::Buffer, Vec<Weak<BindKey>>>>,
    held_textures: RefCell<HashMap<wgpu::Texture, HeldTexture>>,
    /// The buffer of each usage that stands for "no data" ([`ScratchPool::empty`]). Nothing writes or reads it, so
    /// every render shares it, in any generation; it counts as held, so its bind group is kept.
    empties: RefCell<HashMap<u32, wgpu::Buffer>>,
    /// Bind groups by what they bind. One is returned only while the pool holds every buffer and texture it names,
    /// and goes with them or once [`KEEP_GENERATIONS`] have not used it.
    bindings: RefCell<HashMap<Arc<BindKey>, Binding>>,
}

impl ScratchPool {
    /// A render starts. The outermost one opens a generation, after a non-blocking poll so the work that finished
    /// since hands its resources back. `hold`: its command buffers and the writes they read wait for the outermost
    /// holding render on this thread to end ([`submit`]); [`ScratchPool::leave`] takes the same value.
    pub(crate) fn enter(&self, device: &wgpu::Device, hold: bool) {
        if hold {
            HELD.with(|h| h.borrow_mut().0 += 1);
        }
        let d = self.depth.get();
        self.depth.set(d + 1);
        if d > 0 {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let gen = self.gen.get() + 1;
        self.gen.set(gen);
        // What rests in the pool (free, and not taken by the newest finished generation or since) is kept up to the
        // bytes the generation that just ended took, most recently taken first: sizes that come back after a while
        // are still here if they fit beside a frame's own working set, and a clip whose sizes never repeat holds
        // about one working set of them, not every size it has drawn.
        *self.scope.borrow_mut() = Default::default();
        let done = self.done.load(Ordering::Acquire);
        let [buffer_budget, texture_budget] = self.taken.replace([0, 0]);
        // An evicted resource's bind groups go with it: a kept group owns what it binds.
        {
            let mut held = self.held_buffers.borrow_mut();
            let mut bindings = self.bindings.borrow_mut();
            evict_resting(&mut self.buffers.borrow_mut(), done, buffer_budget, wgpu::Buffer::size, |buffer| {
                for key in held.remove(buffer).iter().flat_map(|keys| keys.iter().filter_map(Weak::upgrade)) {
                    bindings.remove(&key);
                }
            });
        }
        {
            let mut held = self.held_textures.borrow_mut();
            let mut bindings = self.bindings.borrow_mut();
            evict_resting(&mut self.textures.borrow_mut(), done, texture_budget, texture_bytes, |texture| {
                for key in held.remove(texture).iter().flat_map(|h| h.bindings.iter().filter_map(Weak::upgrade)) {
                    bindings.remove(&key);
                }
            });
        }
        if gen % KEEP_GENERATIONS == 0 {
            let mut bindings = self.bindings.borrow_mut();
            bindings.retain(|key, binding| binding.gen + KEEP_GENERATIONS >= gen && self.holds_all(&key.1));
            drop(bindings);
            for held in self.held_textures.borrow_mut().values_mut() {
                held.bindings.retain(|key| key.strong_count() > 0);
            }
            for keys in self.held_buffers.borrow_mut().values_mut() {
                keys.retain(|key| key.strong_count() > 0);
            }
        }
    }

    /// Let go of everything the pool holds (an engine dropping its retained GPU state): work already submitted
    /// keeps what it uses. Generations go on counting, so a finished one still reports itself correctly.
    pub(crate) fn clear(&self) {
        self.buffers.borrow_mut().clear();
        self.textures.borrow_mut().clear();
        self.held_buffers.borrow_mut().clear();
        *self.scope.borrow_mut() = Default::default();
        self.empties.borrow_mut().clear();
        self.held_textures.borrow_mut().clear();
        self.bindings.borrow_mut().clear();
    }

    /// Whether the pool holds every buffer and texture of `bound`.
    fn holds_all(&self, bound: &[Bound]) -> bool {
        bound.iter().all(|b| match b {
            Bound::Buffer(buffer) => self.held_buffers.borrow().contains_key(buffer),
            Bound::View(view) => self.held_textures.borrow().contains_key(view.texture()),
            Bound::Sampler(_) => true,
        })
    }

    /// A render ends (its work submitted). The outermost one closes its generation: its resources come back once the
    /// queue has done everything submitted so far.
    pub(crate) fn leave(&self, queue: &wgpu::Queue, hold: bool) {
        if hold {
            let open = HELD.with(|h| {
                let mut h = h.borrow_mut();
                h.0 = h.0.saturating_sub(1);
                h.0
            });
            if open == 0 {
                flush();
            }
        }
        let d = self.depth.get().saturating_sub(1);
        self.depth.set(d);
        if d > 0 {
            return;
        }
        let (gen, done) = (self.gen.get(), self.done.clone());
        queue.on_submitted_work_done(move || {
            done.fetch_max(gen, Ordering::AcqRel);
        });
    }

    fn active(&self) -> bool {
        self.depth.get() > 0
    }

    /// A buffer as `device.create_buffer(desc)` makes it (not mapped at creation).
    pub(crate) fn buffer(&self, device: &wgpu::Device, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::Buffer {
        let mappable = desc.usage.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE);
        if !self.active() || desc.mapped_at_creation || mappable {
            return device.create_buffer(desc);
        }
        let key = (desc.size, desc.usage.bits());
        let (gen, done) = (self.gen.get(), self.done.load(Ordering::Acquire));
        let [buffers, textures] = self.taken.get();
        self.taken.set([buffers + desc.size, textures]);
        let mut map = self.buffers.borrow_mut();
        let list = map.entry(key).or_default();
        if list.front().is_some_and(|s| s.free || s.gen <= done) {
            let item = list.pop_front().expect("a front").item;
            list.push_back(Slot { gen, free: false, item: item.clone() });
            self.scope.borrow_mut().0.push(item.clone());
            count("scratch.buffer.reused");
            return item;
        }
        count(if list.is_empty() { "scratch.buffer.new_key" } else { "scratch.buffer.all_busy" });
        let buf = device.create_buffer(desc);
        self.scope.borrow_mut().0.push(buf.clone());
        list.push_back(Slot { gen, free: false, item: buf.clone() });
        self.held_buffers.borrow_mut().insert(buf.clone(), Vec::new());
        buf
    }

    /// Where the open generation's takes stand: the start of a scope ([`ScratchPool::release`]).
    pub(crate) fn mark(&self) -> Scope {
        let scope = self.scope.borrow();
        Scope(scope.0.len(), scope.1.len())
    }

    /// Hand back the buffers and textures taken since `mark`, for the caller whose every command buffer that uses
    /// them has been submitted (a capture, once it is rendered and filtered), all but `result`, the texture
    /// something later still reads (it stays in the scope around this one). The next take of a key may be one of
    /// them, in this generation or any later: the queue runs its writes and command buffers in the order they were
    /// made, so a later write or pass cannot land under an earlier reader. That is how a retained engine's
    /// captures shared its labelled buffers and its attachments; otherwise a frame of a hundred captures holds a
    /// hundred captures' buffers and filter textures to its end, where it needs one capture's at a time. While
    /// command buffers are held on this thread nothing has been submitted, and nothing is handed back.
    pub(crate) fn release(&self, mark: Scope, result: Option<&wgpu::Texture>) {
        let (buffers, textures) = {
            let mut scope = self.scope.borrow_mut();
            let scope = &mut *scope;
            let from = (mark.0.min(scope.0.len()), mark.1.min(scope.1.len()));
            let buffers = scope.0.split_off(from.0);
            let mut textures = scope.1.split_off(from.1);
            if HELD.with(|h| h.borrow().0 > 0) {
                return;
            }
            if let Some(at) = result.and_then(|result| textures.iter().position(|t| t == result)) {
                scope.1.push(textures.swap_remove(at));
            }
            (buffers, textures)
        };
        // Each was taken in this scope, so it is near the back of its queue.
        let mut map = self.buffers.borrow_mut();
        for buffer in buffers {
            let Some(list) = map.get_mut(&(buffer.size(), buffer.usage().bits())) else { continue };
            let at = list.iter().rposition(|s| s.item == buffer);
            if let Some(slot) = at.and_then(|at| list.remove(at)) {
                list.push_front(Slot { free: true, ..slot });
            }
        }
        let mut map = self.textures.borrow_mut();
        for texture in textures {
            // One withdrawn since it was taken ([`ScratchPool::keep`]) is no longer the pool's to hand out.
            let Some(list) = map.get_mut(&TexKey::of(&texture)) else { continue };
            let at = list.iter().rposition(|s| s.item == texture);
            if let Some(slot) = at.and_then(|at| list.remove(at)) {
                list.push_front(Slot { free: true, ..slot });
            }
        }
    }

    /// The buffer a render binds where it has no data of `usage` to upload (four bytes, never written, never read:
    /// draws address only what was written). A frame of a hundred captures bound some two thousand of these, each
    /// its own buffer with its own bind group; they are one buffer.
    pub(crate) fn empty(&self, device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let desc = wgpu::BufferDescriptor { label: Some(label), size: 4, usage, mapped_at_creation: false };
        if !self.active() {
            return device.create_buffer(&desc);
        }
        self.empties
            .borrow_mut()
            .entry(usage.bits())
            .or_insert_with(|| {
                let buffer = device.create_buffer(&desc);
                self.held_buffers.borrow_mut().insert(buffer.clone(), Vec::new());
                buffer
            })
            .clone()
    }

    /// A texture as `device.create_texture(desc)` makes it. For a texture its users write whole before they read.
    pub(crate) fn texture(&self, device: &wgpu::Device, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::Texture {
        if !self.active() || !desc.view_formats.is_empty() {
            return device.create_texture(desc);
        }
        let key = TexKey {
            w: desc.size.width,
            h: desc.size.height,
            layers: desc.size.depth_or_array_layers,
            mips: desc.mip_level_count,
            samples: desc.sample_count,
            dim: desc.dimension as u8,
            format: desc.format,
            usage: desc.usage.bits(),
        };
        let (gen, done) = (self.gen.get(), self.done.load(Ordering::Acquire));
        let mut map = self.textures.borrow_mut();
        let list = map.entry(key).or_default();
        let tex = if list.front().is_some_and(|s| s.free || s.gen <= done) {
            count("scratch.texture.reused");
            list.pop_front().expect("a front").item
        } else {
            count(if list.is_empty() { "scratch.texture.new_key" } else { "scratch.texture.all_busy" });
            let tex = device.create_texture(desc);
            self.held_textures.borrow_mut().insert(tex.clone(), HeldTexture::default());
            tex
        };
        list.push_back(Slot { gen, free: false, item: tex.clone() });
        self.scope.borrow_mut().1.push(tex.clone());
        let [buffers, textures] = self.taken.get();
        self.taken.set([buffers, textures + texture_bytes(&tex)]);
        tex
    }

    /// `texture`'s default view (`create_view(&Default::default())`): the one view the pool keeps for a texture it
    /// holds, a new one for any other.
    pub(crate) fn view(&self, texture: &wgpu::Texture) -> wgpu::TextureView {
        match self.held_textures.borrow_mut().get_mut(texture) {
            Some(held) => held.view.get_or_insert_with(|| texture.create_view(&Default::default())).clone(),
            None => texture.create_view(&Default::default()),
        }
    }

    /// Withdraw `texture` from the pool: its holder keeps it past this generation (a capture result kept for later
    /// frames), so it is never handed out again. Nothing to do for a texture the pool does not hold.
    pub(crate) fn keep(&self, texture: &wgpu::Texture) {
        let Some(held) = self.held_textures.borrow_mut().remove(texture) else { return };
        let key = TexKey::of(texture);
        let mut map = self.textures.borrow_mut();
        if let Some(list) = map.get_mut(&key) {
            // A result is kept in the generation that took it, so it is at the back of its queue, behind only what
            // was taken under the same key since.
            if let Some(at) = list.iter().rposition(|s| s.item == *texture) {
                let _ = list.remove(at);
            }
            if list.is_empty() {
                map.remove(&key);
            }
        }
        // The bind groups that name it would keep it alive after its holder lets it go.
        let mut bindings = self.bindings.borrow_mut();
        for key in held.bindings.iter().filter_map(Weak::upgrade) {
            bindings.remove(&key);
        }
    }

    /// A bind group of `layout` with `bound[i]` at binding `i`. While a render is open and the pool holds every
    /// buffer and texture bound, the group is kept under the layout and those resources and returned for them again:
    /// a bind group is what it binds, whatever the buffers hold.
    pub(crate) fn bind(
        &self,
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        bound: &[Bound],
    ) -> wgpu::BindGroup {
        let make = || {
            let entries: Vec<wgpu::BindGroupEntry<'_>> = bound
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: match b {
                        Bound::Buffer(buffer) => buffer.as_entire_binding(),
                        Bound::View(view) => wgpu::BindingResource::TextureView(view),
                        Bound::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    },
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
        };
        if !self.active() || !self.holds_all(bound) {
            return make();
        }
        let gen = self.gen.get();
        let key: BindKey = (layout.clone(), bound.to_vec());
        let mut bindings = self.bindings.borrow_mut();
        if let Some(binding) = bindings.get_mut(&key) {
            binding.gen = gen;
            count("scratch.bind.reused");
            return binding.group.clone();
        }
        count("scratch.bind.new");
        let group = make();
        let key = Arc::new(key);
        let mut textures = self.held_textures.borrow_mut();
        let mut buffers = self.held_buffers.borrow_mut();
        for b in bound {
            match b {
                Bound::View(view) => {
                    if let Some(texture) = textures.get_mut(view.texture()) {
                        texture.bindings.push(Arc::downgrade(&key));
                    }
                }
                Bound::Buffer(buffer) => {
                    if let Some(keys) = buffers.get_mut(buffer) {
                        keys.push(Arc::downgrade(&key));
                    }
                }
                Bound::Sampler(_) => {}
            }
        }
        bindings.insert(key, Binding { gen, group: group.clone() });
        group
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    /// A kept texture is never handed out again; a pooled texture has one view; a bind group over pooled resources
    /// is the same group for the same resources, and goes when one of them leaves the pool.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn kept_textures_leave_the_pool_and_bind_groups_follow_their_resources() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let (device, queue) = (&gpu.device, &gpu.queue);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool test"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let desc = wgpu::TextureDescriptor {
            label: Some("pool test"),
            size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let pool = ScratchPool::default();
        pool.enter(device, false);
        let (a, b) = (pool.texture(device, &desc), pool.texture(device, &desc));
        assert!(a != b, "one generation handed a texture out twice");
        assert!(pool.view(&a) == pool.view(&a), "a pooled texture has one view");
        let group = |texture: &wgpu::Texture| pool.bind(device, "pool test", &layout, &[Bound::View(pool.view(texture))]);
        assert!(group(&a) == group(&a), "the same resources are the same bind group");
        assert!(group(&a) != group(&b));
        let kept = group(&a);
        // A group over both textures: withdrawing one must leave nothing owning it through the other.
        let pair_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool test pair"),
            entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }),
        });
        let pair = || pool.bind(device, "pool test pair", &pair_layout, &[Bound::View(pool.view(&a)), Bound::View(pool.view(&b))]);
        assert!(pair() == pair());
        let only_b = group(&b);
        pool.keep(&a);
        assert!(
            pool.bindings
                .borrow()
                .keys()
                .all(|key| !key.1.iter().any(|bound| matches!(bound, Bound::View(view) if *view.texture() == a))),
            "a kept bind group still names a withdrawn texture"
        );
        assert!(group(&b) == only_b, "withdrawing one texture dropped another's bind group");
        assert!(!pool.held_textures.borrow().contains_key(&a) && pool.held_textures.borrow().contains_key(&b));
        assert!(group(&a) != kept, "a bind group outlived its texture's place in the pool");
        pool.leave(queue, false);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        // The finished generation's texture comes back; the kept one never does.
        pool.enter(device, false);
        let c = pool.texture(device, &desc);
        assert!(c == b);
        assert!(pool.texture(device, &desc) != a);
        pool.leave(queue, false);
        // A buffer handed back is the next one taken, in the same generation; one not handed back is not.
        let uniform = wgpu::BufferDescriptor {
            label: Some("pool test"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        pool.enter(device, false);
        let scope = pool.mark();
        let first = pool.buffer(device, &uniform);
        assert!(pool.buffer(device, &uniform) != first, "one generation handed a buffer out twice");
        // Textures too, all but the one named as the scope's result.
        let (result, between) = (pool.texture(device, &desc), pool.texture(device, &desc));
        pool.release(scope, Some(&result));
        let again = pool.buffer(device, &uniform);
        assert!(again == first || pool.buffer(device, &uniform) == first, "a released buffer was not taken again");
        assert!(pool.texture(device, &desc) == between, "a released texture was not taken again");
        assert!(pool.texture(device, &desc) != result, "a scope's result was handed out while still read");
        pool.leave(queue, false);
        // An evicted buffer takes its bind group with it. A generation that takes nothing leaves no budget for
        // what rests, so once the buffers' generation is finished and another has passed, they go.
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool test uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        pool.enter(device, false);
        let bound = pool.buffer(device, &uniform);
        let _ = pool.bind(device, "pool test uniform", &uniform_layout, &[Bound::Buffer(bound.clone())]);
        assert!(pool.bindings.borrow().keys().any(|key| key.1.contains(&Bound::Buffer(bound.clone()))));
        pool.leave(queue, false);
        for _ in 0..3 {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            pool.enter(device, false);
            pool.leave(queue, false);
        }
        assert!(!pool.held_buffers.borrow().contains_key(&bound), "a resting buffer outlived an empty budget");
        assert!(
            !pool.bindings.borrow().keys().any(|key| key.1.contains(&Bound::Buffer(bound.clone()))),
            "a kept bind group still owns an evicted buffer"
        );
        // Cleared, the pool holds nothing and hands nothing it held out again.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        pool.clear();
        assert!(pool.held_textures.borrow().is_empty() && pool.bindings.borrow().is_empty());
        pool.enter(device, false);
        assert!(pool.texture(device, &desc) != b);
        pool.leave(queue, false);
    }
}

#[cfg(test)]
mod teardown_tests {
    use super::*;

    #[derive(Default)]
    struct Resource(Cell<usize>);
    impl Destroy for Resource {
        fn destroy_now(&self) { self.0.set(self.0.get() + 1); }
    }

    #[test]
    fn destroys_only_when_scratch_can_be_read() {
        let resource = Resource::default();
        destroy(&resource);
        assert_eq!(resource.0.get(), 1);
        HELD.with(|h| {
            let _borrow = h.borrow_mut();
            destroy(&resource);
        });
        WRITES.with(|w| {
            let _borrow = w.borrow_mut();
            destroy(&resource);
        });
        assert_eq!(resource.0.get(), 1);
    }

    struct AtExit {
        held_gone: bool,
        writes_gone: bool,
        checked: Arc<AtomicU64>,
    }
    impl Drop for AtExit {
        fn drop(&mut self) {
            assert_eq!(HELD.try_with(|_| ()).is_err(), self.held_gone);
            assert_eq!(WRITES.try_with(|_| ()).is_err(), self.writes_gone);
            let resource = Resource::default();
            destroy(&resource);
            assert_eq!(resource.0.get(), 0, "unavailable TLS cannot authorize explicit destruction");
            self.checked.fetch_add(1, Ordering::SeqCst);
        }
    }
    thread_local! {
        static AT_EXIT: RefCell<Option<AtExit>> = const { RefCell::new(None) };
    }

    #[test]
    fn destroy_after_either_or_both_scratch_keys_are_gone() {
        for (held_gone, writes_gone) in [(true, false), (false, true), (true, true)] {
            let checked = Arc::new(AtomicU64::new(0));
            let result = checked.clone();
            std::thread::spawn(move || {
                // TLS destructors run in reverse initialization order on our native targets.
                if !held_gone { HELD.with(|_| ()); }
                if !writes_gone { WRITES.with(|_| ()); }
                AT_EXIT.with(|slot| *slot.borrow_mut() = Some(AtExit { held_gone, writes_gone, checked }));
                HELD.with(|_| ());
                WRITES.with(|_| ());
            }).join().unwrap();
            assert_eq!(result.load(Ordering::SeqCst), 1);
        }
    }
}
