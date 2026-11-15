use super::*;

/// A FIFO/vsync surface for one prepared session. Zero size suspends drawing.
/// Window resize scales the fixed render image with preserved aspect ratio;
/// changing internal render resolution requires preparing another session.
pub struct Presentation<'window> {
    surface: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    owner: u64,
    size: [u32; 2],
    submission: Option<wgpu::SubmissionIndex>,
    reconfigure: bool,
}

impl Session {
    /// Create a surface on the same instance/adapter as the retained renderer.
    /// Pass an owned window handle (such as Arc<Window>) to retain its lifetime.
    pub fn presentation<'w>(&self, target: impl Into<wgpu::SurfaceTarget<'w>>, size: [u32; 2]) -> Result<Presentation<'w>> {
        let r = &self.renderer;
        let surface = r.instance.create_surface(target).context("create window surface; run on a desktop with a supported display driver")?;
        let caps = surface.get_capabilities(&r.adapter);
        ensure!(!caps.formats.is_empty() && !caps.present_modes.is_empty(),
            "display driver could not query window surface capabilities; use a supported desktop display and graphics driver");
        let format = caps.formats.iter().copied().find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .context("display requires an RGBA8/BGRA8 unorm surface to preserve vork display-encoded colour; use another display adapter")?;
        ensure!(caps.present_modes.contains(&wgpu::PresentMode::Fifo), "display has no FIFO/vsync support; select another display adapter");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size[0].max(1), height: size[1].max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 1,
            alpha_mode: caps.alpha_modes.iter().copied().find(|m| *m == wgpu::CompositeAlphaMode::Opaque)
                .or_else(|| caps.alpha_modes.first().copied()).context("display has no alpha mode; select another display adapter")?,
            view_formats: vec![],
        };
        let scope = r.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let vertex = crate::native::shader_module(&r.device, &crate::native::MIP_VS);
        let fragment = crate::native::shader_module(&r.device, &crate::native::PRESENT_FS);
        let group = r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("session presentation"), entries: &crate::materials::MIP_LAYOUT,
        });
        let layout = r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("session presentation"), bind_group_layouts: &[Some(&group)], immediate_size: 0,
        });
        let pipeline = r.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("session presentation"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &vertex, entry_point: Some(crate::native::MIP_VS.entry), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &fragment, entry_point: Some(crate::native::PRESENT_FS.entry), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let texture = self.slot.view.output.create_view(&Default::default());
        let sampler = r.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
        });
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("session image"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&texture) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        if let Some(error) = pollster::block_on(scope.pop()) { anyhow::bail!("prepare presentation: {error}; use a supported desktop adapter"); }
        let mut presentation = Presentation { surface, config, pipeline, bind, owner: self.owner, size, submission: None, reconfigure: true };
        presentation.configure(r)?;
        Ok(presentation)
    }

    /// Acquire before rendering so minimized, occluded and timed-out surfaces
    /// consume no scene work. None means the batch was not published; retry it.
    pub fn present(&mut self, presentation: &mut Presentation<'_>, time: f32, updates: &[Update<'_>]) -> Result<Option<FrameTiming>> {
        ensure!(presentation.owner == self.owner, "presentation belongs to another session; create it from this session");
        ensure!(!self.failed, "session is invalid; recreate the renderer and prepare again");
        let r = &self.renderer;
        if let Some(submission) = presentation.submission.take() {
            r.wait(Some(submission)).context("presentation GPU failed; recreate the renderer and window surface")?;
        }
        if presentation.size.contains(&0) { return Ok(None); }
        presentation.configure(r)?;
        let surface = match presentation.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface) => surface,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface) => { presentation.reconfigure = true; surface },
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(None),
            wgpu::CurrentSurfaceTexture::Outdated => { presentation.reconfigure = true; return Ok(None); },
            wgpu::CurrentSurfaceTexture::Lost => anyhow::bail!("window surface lost; recreate presentation from the window; if the device is lost, recreate the renderer and prepare again"),
            wgpu::CurrentSurfaceTexture::Validation => anyhow::bail!("window surface validation failed; recreate presentation with a supported size and display adapter"),
        };
        let timing = self.update_frame(time, updates)?.timing;
        let r = &self.renderer;
        let target = surface.texture.create_view(&Default::default());
        let mut encoder = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present frame") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present frame"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target, depth_slice: None, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            let [x, y, w, h] = viewport(self.resolution(), presentation.size);
            pass.set_viewport(x, y, w, h, 0., 1.);
            pass.set_pipeline(&presentation.pipeline);
            pass.set_bind_group(0, &presentation.bind, &[]);
            pass.draw(0..3, 0..1);
        }
        presentation.submission = Some(r.queue.submit([encoder.finish()]));
        surface.present();
        r.errors.check().context("presentation GPU failed; recreate the renderer and prepare again")?;
        Ok(Some(timing))
    }
}

impl Presentation<'_> {
    pub fn resize(&mut self, size: [u32; 2]) { self.size = size; self.reconfigure = true; }
    fn configure(&mut self, r: &Renderer) -> Result<()> {
        if !self.reconfigure || self.size.contains(&0) { return Ok(()); }
        ensure!(self.size.iter().all(|v| *v <= r.device.limits().max_texture_dimension_2d),
            crate::gpu_memory::device_limit("window", format_args!("{}x{} pixels",self.size[0],self.size[1]), format_args!("{} pixels per axis",r.device.limits().max_texture_dimension_2d), "resize the window within the texture dimension limit"));
        self.config.width = self.size[0]; self.config.height = self.size[1];
        let scope = r.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.surface.configure(&r.device, &self.config);
        if let Some(error) = pollster::block_on(scope.pop()) { anyhow::bail!("configure window surface: {error}; recreate presentation on a supported display adapter"); }
        r.errors.check()?;
        self.reconfigure = false;
        Ok(())
    }
}

fn viewport(image: [u32; 2], window: [u32; 2]) -> [f32; 4] {
    let scale = (window[0] as f32 / image[0] as f32).min(window[1] as f32 / image[1] as f32);
    let w = image[0] as f32 * scale; let h = image[1] as f32 * scale;
    [(window[0] as f32 - w) * 0.5, (window[1] as f32 - h) * 0.5, w, h]
}

// The renderer output is already display encoded. A non-sRGB attachment avoids
// applying the transfer function twice. Texture coordinates preserve top-left Y.
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resize_letterboxes_without_distortion() {
        assert_eq!(viewport([1920,1080], [1280,720]), [0.,0.,1280.,720.]);
        assert_eq!(viewport([200,100], [100,100]), [0.,25.,100.,50.]);
        assert_eq!(viewport([100,200], [100,100]), [25.,0.,50.,100.]);
    }
}
