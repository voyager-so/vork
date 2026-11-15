//! device.rs — excerpt of moda-engine's `GpuContext`: the device and queue the text pipelines draw with (see
//! PROVENANCE.md).

#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// How the host creates driver pipelines on a device. vork's renderer installs its own creation here, so Moda's
/// pipelines go through the persistent pipeline cache (a Metal binary archive on macOS) and are timed and counted
/// with every other pipeline; without a host, creation is the device's own.
pub trait PipelineHost: Send + Sync {
    /// `descriptor.cache` is the host's to choose.
    fn render_pipeline(&self, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline;
    /// `descriptor.cache` is the host's to choose.
    fn compute_pipeline(&self, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline;
}

type Host = std::sync::Arc<dyn PipelineHost>;
/// One host per device, by device identity. An entry keeps its device alive until it is forgotten.
static PIPELINE_HOSTS: std::sync::Mutex<Vec<(wgpu::Device, Host)>> = std::sync::Mutex::new(Vec::new());

/// Create every later Moda pipeline on `device` through `host`.
pub fn set_pipeline_host(device: &wgpu::Device, host: Host) {
    let mut hosts = PIPELINE_HOSTS.lock().unwrap();
    hosts.retain(|(owner, _)| owner != device);
    hosts.push((device.clone(), host));
}

/// The host's device is going away: later creation on it is the device's own.
pub fn forget_pipeline_host(device: &wgpu::Device) {
    PIPELINE_HOSTS.lock().unwrap().retain(|(owner, _)| owner != device);
}

// The lock is released before the driver compiles.
fn pipeline_host(device: &wgpu::Device) -> Option<Host> {
    PIPELINE_HOSTS.lock().unwrap().iter().find(|(owner, _)| owner == device).map(|(_, host)| host.clone())
}

/// Every Moda render pipeline is created here.
pub fn create_render_pipeline(device: &wgpu::Device, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
    match pipeline_host(device) {
        Some(host) => host.render_pipeline(descriptor),
        None => device.create_render_pipeline(descriptor),
    }
}

/// Every Moda compute pipeline is created here.
pub fn create_compute_pipeline(device: &wgpu::Device, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline {
    match pipeline_host(device) {
        Some(host) => host.compute_pipeline(descriptor),
        None => device.create_compute_pipeline(descriptor),
    }
}
