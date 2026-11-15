//! cargo run --release -p vork-encode --example synthetic -- auto out.mp4 48
use anyhow::Result;
use vork_encode::{Backend, DeviceInfo, EncodeSettings, Encoder};
fn main() -> Result<()> {
    pollster::block_on(run())
}
async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let backend = match args.get(1).map(String::as_str) {
        Some("readback") => Backend::Readback,
        Some("hardware-readback") => Backend::HardwareReadback,
        _ => Backend::Auto,
    };
    let path = args.get(2).map(String::as_str).unwrap_or("synthetic.mp4");
    let n: u32 = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(48);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance.request_adapter(&Default::default()).await?;
    eprintln!("adapter={:?}", adapter.get_info());
    let (device, queue) = adapter.request_device(&Default::default()).await?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Frame {index:u32};
@group(0) @binding(0) var image:texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(1) var<uniform> frame:Frame;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let size=textureDimensions(image); if(id.x>=size.x || id.y>=size.y){return;}
 var rgb=vec3<u32>((id.x+frame.index*3u)%256u,id.y%256u,frame.index*5u%256u);
 // Large binary digits identify the frame even after lossy chroma subsampling.
 if(id.y<80u && id.x<640u) { let bit=(frame.index>>(id.x/80u))&1u;rgb=vec3<u32>(bit*255u); }
 textureStore(image,vec2<i32>(id.xy),vec4<f32>(vec3<f32>(rgb)/255.0,1.0));
}"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let start = std::time::Instant::now();
    let mut output = Encoder::open(
        DeviceInfo {
            device: device.clone(),
            queue: queue.clone(),
            backend,
        },
        1920,
        1080,
        vork_encode::Time::integer(24),
        EncodeSettings::default(),
        path,
        n as u64,
        3,
    )?;
    let pixel_readback = output.uses_pixel_readback();
    eprintln!("encoder_pixel_readback={pixel_readback}");
    let mut surfaces = (0..3)
        .map(|_| output.create_surface())
        .collect::<anyhow::Result<Vec<_>>>()?;
    for i in 0..n {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, &i.to_le_bytes());
        let view = texture.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut commands = device.create_command_encoder(&Default::default());
        {
            let mut pass = commands.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1920u32.div_ceil(16), 1080u32.div_ceil(16), 1);
        }
        let surface = &mut surfaces[i as usize % 3];
        if i >= 3 {
            output.release(i as u64 - 3, surface)?;
        }
        surface.record_copy(&mut commands, &texture);
        let submission = queue.submit([commands.finish()]);
        output.push(i as u64, texture, surface, &submission)?;
    }
    output.finish()?;
    drop(output); // Standalone benchmark includes native/GPU reclamation.
    println!(
        "backend={backend:?} pixel_readback={pixel_readback} frames={n} elapsed_ms={:.3} ms/frame={:.3}",
        start.elapsed().as_secs_f64() * 1000.,
        start.elapsed().as_secs_f64() * 1000. / n as f64
    );
    Ok(())
}
