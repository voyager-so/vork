//! Exposure report: shares of clipped and near-black pixels and the mean luma of the
//! final display image, reduced on the GPU after a frame's last colour sample.
use crate::*;
use vork_slang::registry::{EXPOSURE_LUMA_SCALE, EXPOSURE_WORDS};

const BYTES: u64 = (EXPOSURE_WORDS * 4) as u64;
/// Workgroup side (REG_COMPUTE_TILE) of display_stats, and width of display_stats_packed.
const TILE: u32 = 8;
const PACKED_GROUP: u64 = 64;
/// Workgroups per dispatch row of display_stats_packed, within every backend's dimension limit.
const PACKED_ROW: u32 = 32768;
const fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
/// display_stats.slang: the GPU colour output, then the counters.
pub(crate) const TEXTURE_LAYOUT: [wgpu::BindGroupLayoutEntry; 2] = [
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    },
    storage(1, false),
];
/// display_stats_packed.slang: the packed RGBA8 colour plane, the counters and the size.
pub(crate) const PACKED_LAYOUT: [wgpu::BindGroupLayoutEntry; 3] = [
    storage(0, true),
    storage(1, false),
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];
/// Where the frame's final display colour lives.
#[derive(Clone, Copy)]
pub(crate) enum Display<'a> {
    Texture(&'a wgpu::Texture),
    /// A buffer starting with one packed RGBA8 word per pixel.
    Packed(&'a wgpu::Buffer, u64),
}

/// One frame's display exposure: alpha-weighted shares of covered pixels in 0..1, and mean display luma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exposure {
    /// Mean alpha; zero coverage makes the exposure values undefined (JSON null).
    pub coverage: f64,
    /// Any display channel at or above 99% (ARRI false-colour white clipping): where a view ran out of display
    /// range, so brighter light is no longer told apart. None for a picture no view made: a canvas that shows
    /// its authored display colours as they are has nothing to clip, and a full channel there is the colour asked for.
    pub clipped: Option<f64>,
    /// Display Rec.709 luma below 2.5% (ARRI false-colour black clipping).
    pub near_black: f64,
    /// Mean display Rec.709 luma.
    pub mean: f64,
}

impl Exposure {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"coverage": self.coverage, "clipped": self.clipped, "near_black": self.near_black, "mean": self.mean})
    }
}

pub(crate) struct Stats {
    counters: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// display_stats_packed's size uniform, created at the first packed reduction.
    pixels: Option<(u64, wgpu::Buffer)>,
    /// The pending exposure reduced the display image.
    pub(crate) written: bool,
}
impl Stats {
    pub(crate) fn new(d: &wgpu::Device) -> Self {
        Self {
            counters: init(d, "exposure counters", &[0; BYTES as usize],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST),
            readback: crate::gpu_memory::buffer(d, &wgpu::BufferDescriptor {
                label: Some("exposure readback"),
                size: BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            pixels: None,
            written: false,
        }
    }
    /// Reduce the finished display image, then copy and reset the counters.
    pub(crate) fn encode(&mut self, r: &Renderer, encoder: &mut wgpu::CommandEncoder, display: Display<'_>, queries: Option<&wgpu::QuerySet>) {
        let [texture_pipeline, packed_pipeline] = pipelines(r);
        let (pipeline, view, groups) = match display {
            Display::Texture(texture) => (texture_pipeline, Some(texture.create_view(&Default::default())),
                [texture.width().div_ceil(TILE), texture.height().div_ceil(TILE)]),
            Display::Packed(_, pixels) => {
                let groups = pixels.div_ceil(PACKED_GROUP) as u32;
                (packed_pipeline, None, [groups.min(PACKED_ROW), groups.div_ceil(PACKED_ROW)])
            }
        };
        if let Display::Packed(_, pixels) = display
            && self.pixels.as_ref().is_none_or(|(count, _)| *count != pixels)
        {
            let words = [u32::try_from(pixels).unwrap_or(u32::MAX), PACKED_ROW, 0, 0];
            self.pixels = Some((pixels, init(&r.device, "exposure pixel count", bytemuck::cast_slice(&words), wgpu::BufferUsages::UNIFORM)));
        }
        let mut entries = vec![entry(1, self.counters.as_entire_binding())];
        match (&display, &view) {
            (Display::Texture(_), Some(view)) => entries.push(entry(0, wgpu::BindingResource::TextureView(view))),
            (Display::Packed(buffer, pixels), _) => {
                entries.push(entry(0, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer, offset: 0, size: wgpu::BufferSize::new(pixels * 4),
                })));
                entries.push(entry(2, self.pixels.as_ref().unwrap().1.as_entire_binding()));
            }
            _ => unreachable!(),
        }
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exposure report"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            // Canvas exports share Moda's opt-in query stream; 3D keeps its own query addresses.
            let canvas_queries = queries.is_none().then(moda_engine::gpu_profile::pass_timestamps).flatten();
            let timestamp_writes = queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(crate::timing::query("exposure")),
                end_of_pass_write_index: Some(crate::timing::query("exposure") + 1),
            }).or_else(|| moda_engine::gpu_profile::compute_writes(&canvas_queries));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("exposure report"), timestamp_writes });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        encoder.copy_buffer_to_buffer(&self.counters, 0, &self.readback, 0, BYTES);
        encoder.clear_buffer(&self.counters, 0, None);
        self.written = true;
    }
    /// The buffer to map at collection, when this exposure wrote one.
    pub(crate) fn pending(&self) -> Option<&wgpu::Buffer> {
        self.written.then_some(&self.readback)
    }
    /// Read the mapped counters of a `pixels`-pixel display image.
    pub(crate) fn read(&mut self, pixels: u64) -> Result<Option<Exposure>> {
        if !std::mem::take(&mut self.written) {
            return Ok(None);
        }
        let mapped = self.readback.get_mapped_range(..);
        let exposure = decode(bytemuck::cast_slice(&mapped), pixels);
        drop(mapped);
        self.readback.unmap();
        exposure.map(Some)
    }
}

/// Coverage may legitimately be zero, but every display pixel must be visited: a count
/// that disagrees is GPU work that did not complete, whatever the scene.
fn decode(words: &[u32], pixels: u64) -> Result<Exposure> {
    let wide = |i: usize| u64::from(words[2 * i]) | (u64::from(words[2 * i + 1]) << 32);
    let processed = wide(4);
    if processed != pixels {
        return Err(anyhow::anyhow!("incomplete GPU exposure readback: expected {pixels} pixels, got {processed}")
            .context(crate::failure::FailureKind::Device));
    }
    let coverage = wide(3) as f64;
    Ok(Exposure {
        coverage: coverage / (pixels as f64 * 255.),
        clipped: Some(wide(0) as f64 / coverage),
        near_black: wide(1) as f64 / coverage,
        mean: wide(2) as f64 / coverage / f64::from(EXPOSURE_LUMA_SCALE),
    })
}

/// Create the reduction pipelines before the first frame, with the frame slots' pipelines: preparation
/// owns pipeline creation. Left alone they are created while the frame's last sample is recorded.
pub(crate) fn prepare(r: &Renderer) {
    pipelines(r);
}

fn pipelines(r: &Renderer) -> &[wgpu::ComputePipeline; 2] {
    r.pipelines.display_stats.get_or_init(|| {
        let d = &r.device;
        let pipeline = |shader: &native::Compiled, entries: &[wgpu::BindGroupLayoutEntry]| {
            let module = native::shader_module(d, shader);
            let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("exposure report"), entries });
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("exposure report"),
                bind_group_layouts: &[Some(&group)],
                immediate_size: 0,
            });
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("exposure report"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(shader.entry),
                compilation_options: Default::default(),
                cache: r.cache.as_ref(),
            })
        };
        [
            pipeline(&native::DISPLAY_STATS_REDUCE, &TEXTURE_LAYOUT),
            pipeline(&native::DISPLAY_STATS_PACKED_REDUCE, &PACKED_LAYOUT),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_exposure_is_not_a_completed_transparent_image() {
        let mut words = [0u32; EXPOSURE_WORDS];
        assert!(decode(&words, 60).is_err(), "zero-filled exposure was accepted");
        words[8] = 60;
        let empty = decode(&words, 60).unwrap();
        assert_eq!(empty.coverage, 0.);
        assert!(empty.mean.is_nan());
        words[6] = 60 * 255; // opaque black is also a valid image
        words[2] = 60 * 255;
        let black = decode(&words, 60).unwrap();
        assert_eq!((black.coverage, black.near_black, black.mean), (1., 1., 0.));
        words[8] = 59;
        assert!(decode(&words, 60).is_err(), "partial reduction was accepted");
        words[8] = 61;
        assert!(decode(&words, 60).is_err(), "duplicate reduction was accepted");
    }

    /// A frame whose counters say fewer pixels were visited did not complete on the GPU:
    /// a device failure for the canvas and the 3D frame alike, even inside a scene boundary.
    #[test]
    fn an_incomplete_readback_is_a_device_failure() {
        let error = decode(&[0; 10], 60).err().expect("a zeroed readback was accepted");
        assert!(format!("{error:#}").contains("incomplete GPU exposure readback"), "{error:#}");
        assert_eq!(crate::failure::exit_code(&error), 4);
        assert_eq!(crate::failure::exit_code(&crate::failure::scene(error)), 4);
    }

    /// Both reductions preserve opaque statistics and weight mixed/empty coverage.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn exposure_counts_clipped_near_black_and_mean_luma() -> Result<()> {
        let r = crate::test_renderer()?;
        // 20x3 so partial workgroups and several groups are exercised.
        let (w, h) = (20u32, 3u32);
        let mut texels = vec![[128u8, 128, 128, 255]; (w * h) as usize];
        texels[0] = [255, 0, 0, 255]; // clipped red (low luma, not near-black)
        texels[1] = [255, 255, 255, 255]; // clipped white
        texels[2] = [0, 0, 0, 255]; // black
        texels[59] = [5, 5, 5, 255]; // 5/255 = 2.0% < 2.5%
        for case in ["opaque", "mixed", "empty"] {
            let mut texels = texels.clone();
            if case == "mixed" {
                texels[0][3] = 0; // hidden clipped RGB must not count
                texels[2][3] = 0; // transparent black must not count
                texels[1][3] = 128;
                texels[59][3] = 64;
            } else if case == "empty" {
                for t in &mut texels { t[3] = 0; }
            }
            let coverage: f64 = texels.iter().map(|t| f64::from(t[3])).sum();
            let format = r.output_format();
            if format == wgpu::TextureFormat::Bgra8Unorm {
                for t in &mut texels { t.swap(0, 2); }
            }
            let texture = r.device.create_texture(&wgpu::TextureDescriptor {
                label: None, size: extent(w, h), mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2, format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
            });
            r.queue.write_texture(texture.as_image_copy(), bytemuck::cast_slice(&texels),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: None }, extent(w, h));
            // The packed plane is RGBA8 whatever the output texture's channel order.
            let rgba: Vec<[u8; 4]> = texels.iter().map(|&t| if format == wgpu::TextureFormat::Bgra8Unorm { [t[2], t[1], t[0], t[3]] } else { t }).collect();
            let mut plane = bytemuck::cast_slice::<[u8; 4], u8>(&rgba).to_vec();
            plane.extend([7u8; 64]); // a following plane the reduction must not read
            let packed = init(&r.device, "packed plane", &plane, wgpu::BufferUsages::STORAGE);
            let mut stats = Stats::new(&r.device);
            for packed_source in [false, true, false] {
                let mut encoder = r.device.create_command_encoder(&Default::default());
                stats.encode(&r, &mut encoder, if packed_source { Display::Packed(&packed, u64::from(w * h)) } else { Display::Texture(&texture) }, None);
                let submission = crate::command_finish::Segments::submit(&r.queue, encoder);
                let (tx, rx) = std::sync::mpsc::channel();
                stats.pending().unwrap().map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
                r.wait(Some(submission))?;
                rx.recv()??;
                let e = stats.read(u64::from(w * h))?.unwrap();
                assert_eq!(e.coverage, coverage / (f64::from(w * h) * 255.));
                if coverage == 0. {
                    assert!(e.clipped.is_some_and(f64::is_nan) && e.near_black.is_nan() && e.mean.is_nan());
                    assert!(e.json()["mean"].is_null());
                    continue;
                }
                let clipped = if case == "opaque" { 510. } else { 128. };
                let black = if case == "opaque" { 510. } else { 64. };
                assert_eq!((e.clipped, e.near_black), (Some(clipped / coverage), black / coverage), "counters reset between exposures");
                let luma = |c: [u8; 4]| {
                    let c = if format == wgpu::TextureFormat::Bgra8Unorm { [c[2], c[1], c[0]] } else { [c[0], c[1], c[2]] };
                    let y = (0.2126 * f64::from(c[0]) + 0.7152 * f64::from(c[1]) + 0.0722 * f64::from(c[2])) / 255.;
                    (y * 65535.).round() / 65535.
                };
                let mean = texels.iter().map(|&t| luma(t) * f64::from(t[3])).sum::<f64>() / coverage;
                assert!((e.mean - mean).abs() < 1e-6, "{} != {mean}", e.mean);
                assert!(stats.pending().is_none() && stats.read(1)?.is_none());
            }
        }
        Ok(())
    }
}
