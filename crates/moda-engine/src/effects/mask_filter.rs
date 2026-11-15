use super::mask::CoveragePlacement;
use crate::seams::MaskPlan;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Params {
    origin: vec2<f32>, size: vec2<f32>,
    ancestorOrigin: vec2<f32>, ancestorSize: vec2<f32>,
    sigma: f32, vertical: f32, invert: f32, ancestor: f32,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var ancestor: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    let dimensions = vec2<f32>(textureDimensions(source));
    let uv = position.xy / dimensions;
    var coverage = textureSampleLevel(source, linearSampler, uv, 0.0).r;
    if params.sigma > 0.0 {
        let stride = max(1.0, 3.0 * params.sigma / 32.0);
        let reach = i32(min(32.0, ceil(3.0 * params.sigma)));
        let direction = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), params.vertical == 1.0);
        var total = 0.0;
        var weightSum = 0.0;
        for (var tap = -reach; tap <= reach; tap += 1) {
            let distance = f32(tap) * stride;
            let weight = exp(-0.5 * distance * distance / (params.sigma * params.sigma));
            let sampleUV = uv + direction * distance / dimensions;
            let sample = textureSampleLevel(source, linearSampler, sampleUV, 0.0).r;
            let inside = all(sampleUV >= vec2<f32>(0.0)) && all(sampleUV <= vec2<f32>(1.0));
            total += select(0.0, sample, inside) * weight;
            weightSum += weight;
        }
        coverage = total / weightSum;
    }
    if params.vertical == 1.0 {
        if params.invert == 1.0 { coverage = 1.0 - coverage; }
        if params.ancestor == 1.0 {
            let ancestorUV = (params.origin + uv * params.size - params.ancestorOrigin) / params.ancestorSize;
            let sampled = textureSampleLevel(ancestor, linearSampler, ancestorUV, 0.0).r;
            let inside = all(ancestorUV >= vec2<f32>(0.0)) && all(ancestorUV <= vec2<f32>(1.0));
            coverage *= select(0.0, sampled, inside);
        }
    }
    return coverage;
}
"#;

pub(super) fn apply(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    coverage: &wgpu::TextureView,
    placement: &CoveragePlacement,
    mask: &MaskPlan,
    ancestor: Option<&wgpu::TextureView>,
    ancestor_placement: Option<&CoveragePlacement>,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mask invert and feather"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
        label: Some("mask invert and feather"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let temporary = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mask feather intermediate"),
        size: wgpu::Extent3d {
            width: placement.tex[0],
            height: placement.tex[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let temporary_view = temporary.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    for vertical in [false, true] {
        let ancestor_bounds = ancestor_placement.unwrap_or(placement);
        let values = [
            placement.origin[0] as f32,
            placement.origin[1] as f32,
            placement.size[0] as f32,
            placement.size[1] as f32,
            ancestor_bounds.origin[0] as f32,
            ancestor_bounds.origin[1] as f32,
            ancestor_bounds.size[0] as f32,
            ancestor_bounds.size[1] as f32,
            (0.5 * mask.feather * placement.scale) as f32,
            vertical as u8 as f32,
            mask.invert as u8 as f32,
            ancestor.is_some() as u8 as f32,
        ];
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mask filter uniforms"),
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let source = if vertical { &temporary_view } else { coverage };
        let target = if vertical { coverage } else { &temporary_view };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask filter"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ancestor.unwrap_or(source)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        let profile_ts = crate::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mask filter"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
