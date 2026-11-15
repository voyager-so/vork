//! Install-owned sink storage. A consumer lease is the credit for this entry.
use std::sync::{Arc, Condvar, Mutex};

#[derive(Default)]
pub(crate) struct Credit {
    busy: Mutex<bool>,
    free: Condvar,
}
impl Credit {
    pub(crate) fn wait(&self) {
        let mut busy = self.busy.lock().unwrap();
        while *busy {
            busy = self.free.wait(busy).unwrap();
        }
    }
    pub(crate) fn acquire(self: &Arc<Self>) -> Lease {
        self.wait();
        *self.busy.lock().unwrap() = true;
        Lease(self.clone())
    }
}
pub(crate) struct Lease(Arc<Credit>);
impl Drop for Lease {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap() = false;
        self.0.free.notify_one();
    }
}

/// All sink resources of a frame slot, allocated before any frames are submitted.
pub(crate) struct Entry {
    pub(crate) credit: Arc<Credit>,
    pub(crate) readback: wgpu::Buffer,
    pub(crate) encoder_surfaces: Vec<vork_encode::Surface>,
    pub(crate) frame: Option<usize>,
    pub(crate) output: Option<(wgpu::Texture, wgpu::BindGroup)>,
    copies: Vec<Copy>,
}
struct Copy {
    source: wgpu::Buffer,
    offset: u64,
    size: u64,
    sink: Sink,
}
#[derive(Clone, Copy)]
enum Sink {
    Color,
    Color16,
    DepthIds,
    Normal,
    Motion,
    FloatColor,
    FloatDepth,
    FloatNormal,
    FloatMotion,
    Aov,
}
impl Sink {
    fn enabled(self, outputs: crate::ReadbackOutputs) -> bool {
        match self {
            Self::Aov => outputs.aovs,
            Self::Color => outputs.color,
            Self::Color16 => outputs.color16,
            Self::DepthIds => outputs.depth_ids,
            Self::Normal => outputs.normals,
            Self::Motion => outputs.motion,
            Self::FloatColor => outputs.float.color,
            Self::FloatDepth => outputs.float.depth,
            Self::FloatNormal => outputs.float.normals,
            Self::FloatMotion => outputs.float.motion,
        }
    }
}
impl Entry {
    /// Staging covers the planes up to the highest one the install reads back.
    pub(crate) fn new(
        device: &wgpu::Device,
        pixels: &wgpu::Buffer,
        plane_bytes: u64,
        float: bool,
        outputs: crate::ReadbackOutputs,
        aovs: usize,
        // First display staging plane of a camera view; RGBA16 colour occupies two.
        display_base: Option<u32>,
    ) -> Self {
        use crate::constants::*;
        let planes = if float {
            FLOAT_OUTPUT_PLANES
        } else {
            PACKED_OUTPUT_PLANES
        };
        let copies = (0..planes as usize+aovs*4)
            .map(|plane| Copy {
                source: pixels.clone(),
                offset: plane as u64 * plane_bytes,
                size: plane_bytes,
                sink: match plane {
                    p if p >= planes as usize => Sink::Aov,
                    PACKED_COLOR_PLANE => Sink::Color,
                    PACKED_DEPTH_IDS_PLANE => Sink::DepthIds,
                    PACKED_NORMAL_PLANE => Sink::Normal,
                    PACKED_MOTION_PLANE => Sink::Motion,
                    p if (FLOAT_COLOR_PLANE..FLOAT_DEPTH_PLANE).contains(&p) => Sink::FloatColor,
                    FLOAT_DEPTH_PLANE => Sink::FloatDepth,
                    p if (FLOAT_NORMAL_PLANE..FLOAT_MOTION_PLANE).contains(&p) => Sink::FloatNormal,
                    _ => Sink::FloatMotion,
                },
            })
            .chain(display_base.map(|base| Copy {
                source: pixels.clone(),
                offset: u64::from(base) * plane_bytes,
                size: 2 * plane_bytes,
                sink: Sink::Color16,
            }))
            .filter(|copy| copy.sink.enabled(outputs))
            .collect::<Vec<_>>();
        let size = copies
            .iter()
            .map(|copy| copy.offset + copy.size)
            .max()
            .unwrap_or(4);
        Self {
            credit: Arc::default(),
            readback: crate::gpu_memory::buffer(&device, &wgpu::BufferDescriptor {
                label: Some("ring sink staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            encoder_surfaces: Vec::new(),
            frame: None,
            output: None,
            copies,
        }
    }
    /// An entry whose producer records its own copy into the staging: the 2D canvas, which has one plane.
    /// The credit, the encoder surfaces and the delivered frame are the ones a 3D slot's entry holds.
    pub(crate) fn staging(device: &wgpu::Device, bytes: u64) -> Self {
        Self {
            credit: Arc::default(),
            readback: crate::gpu_memory::buffer(device, &wgpu::BufferDescriptor {
                label: Some("ring sink staging"),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            encoder_surfaces: Vec::new(),
            frame: None,
            output: None,
            copies: Vec::new(),
        }
    }
    /// Preserve sink leases and staging while replacing the frame's producers.
    pub(crate) fn rebind_pixels(&mut self, pixels: &wgpu::Buffer) {
        for copy in &mut self.copies { copy.source = pixels.clone(); }
    }
    pub(crate) fn encode_video(&self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::Texture) {
        for surface in &self.encoder_surfaces {
            surface.record_copy(encoder, output);
        }
    }
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        outputs: crate::ReadbackOutputs,
    ) {
        for copy in &self.copies {
            if copy.sink.enabled(outputs) {
                encoder.copy_buffer_to_buffer(
                    &copy.source,
                    copy.offset,
                    &self.readback,
                    copy.offset,
                    copy.size,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn grown_slots_read_current_pixels_through_every_preserved_ring() -> Result<()> {
        let r = crate::test_renderer()?;
        let scene: Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,
            "composition":{"space":"3d","view":{"program":"std/agx","inputs":{}}},"render":{"dither":false,"resolution":[8,8],"settings":{"integrator":"raster","samples":1}},
            "buffers":{"p":{"type":"vec3","data":[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]]},"i":{"type":"u32","data":[0,1,2,0,2,3]}},
            "geometry":{"quad":{"attributes":{"POSITION":"p"},"indices":"i"}},
            "programs":{"color":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs i) { return unlit(ctx.time < .5 ? float3(1,0,0) : float3(0,1,0)); }"}},
            "materials":{"color":{"program":"color"}},
            "nodes":[{"type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2}}},{"type":"mesh","geometry":"quad","materials":["color"]}]
        }))?;
        let mut s = r.install(scene, Path::new("."))?;
        let mut slot = s.create_frame_slot(&r, ReadbackPrecision::Float)?;
        let external = r.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external redo output"), size: extent(8,8), mip_level_count:1,
            sample_count:1, dimension:wgpu::TextureDimension::D2, format:r.output_format(),
            usage:wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats:&[],
        });
        slot.set_output(&s,&r,external.clone())?;
        slot.write_color = true;
        fn read_output(r: &Renderer, texture: &wgpu::Texture) -> Result<Vec<u8>> {
            let buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
                label:Some("external redo pixels"), size:8*256,
                usage:wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation:false,
            });
            let mut encoder = r.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(wgpu::TexelCopyTextureInfo {
                texture,mip_level:0,origin:wgpu::Origin3d::ZERO,aspect:wgpu::TextureAspect::All,
            },wgpu::TexelCopyBufferInfo {buffer:&buffer,layout:wgpu::TexelCopyBufferLayout {
                offset:0,bytes_per_row:Some(256),rows_per_image:Some(8),
            }},extent(8,8));
            let submission = r.queue.submit([encoder.finish()]);
            let (tx,rx) = std::sync::mpsc::channel();
            buffer.map_async(wgpu::MapMode::Read,..,move |result| {tx.send(result).unwrap();});
            r.device.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;
            rx.recv()??;
            let mapped = buffer.get_mapped_range(..);
            let bytes = mapped.chunks_exact(256).flat_map(|row|row[..32].iter().copied()).collect();
            drop(mapped); buffer.unmap(); Ok(bytes)
        }
        for _ in 1..constants::OUTPUT_RING_DEPTH {
            slot.spare_ring.push_back(super::Entry::new(&r.device, &slot.view.post.pixels,
                8 * 8 * 4, slot.view.float_output, s.readback, 0, slot.view.post.display_base));
        }
        slot.submit(&s, &r, 0.)?;
        let red = slot.collect(&r)?;
        let red_texture = read_output(&r,&external)?;
        slot.submit(&s, &r, 1.)?;
        let green = slot.collect(&r)?;
        let green_texture = read_output(&r,&external)?;
        assert_ne!(red_texture,green_texture,"external texture must receive each colour");
        assert_ne!(red.linear_rgba, green.linear_rgba, "fixture must change colour");
        crate::test_display::assert_display_ne(&red.rgba, &green.rgba);
        s.redo_frame(&r, std::slice::from_mut(&mut slot), Reservations {
            visibility_pool:65536,transparent_fragments:65536,shading_evaluations:65536,..Default::default()
        }, 1)?;
        assert_eq!(slot.view.output,external,"growth detached the external output");
        for time in [1., 0.] {
            for _ in 0..constants::OUTPUT_RING_DEPTH {
                slot.advance_ring();
                slot.submit(&s, &r, time)?;
                let actual = slot.collect(&r)?;
                let expected = if time == 0. { &red } else { &green };
                crate::test_linear::assert_linear_eq(actual.linear_rgba.as_ref().unwrap(), expected.linear_rgba.as_ref().unwrap());
                // The float plane cannot detect a stale displayed CPU readback;
                // the external texture below is a separate output sink too.
                crate::test_display::assert_display_eq(&actual.rgba, &expected.rgba);
                assert_eq!(read_output(&r,&external)?,if time == 0. { &red_texture } else { &green_texture }.clone());
            }
        }
        Ok(())
    }
}
