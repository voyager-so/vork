//! Shared ray sampling and retained lighting bindings.
use crate::*;
#[derive(Clone)]
pub(crate) struct Persistent {
    _fibre: Option<std::sync::Arc<crate::fibre_transport::Storage>>,
    _probes: Option<std::sync::Arc<crate::probes::Storage>>,
    pub(crate) sequence: wgpu::Buffer,
    sequence_offsets: std::collections::BTreeMap<usize,u32>,
    pub(crate) changes: std::sync::Arc<std::sync::Mutex<values::LightingTracker>>,
}
impl Persistent {
    pub(crate) fn rebind_probes(&self, probes:std::sync::Arc<crate::probes::Storage>) -> Self {
        let mut replacement=self.clone();
        replacement._probes=Some(probes);
        replacement.changes.lock().unwrap().invalidate_probes();
        replacement
    }
    pub(crate) fn bytes(&self) -> u64 {self.sequence.size()}
    pub(crate) fn new(s:&InstalledScene,r:&Renderer)->Result<Self> {
        let mut counts = std::collections::BTreeSet::from([1usize]);
        for render in std::iter::once(s.render.clone()).chain(
            s.scene.images.values().filter_map(|i| i.resolved_render(&s.render))
        ) {

            counts.insert(render.settings.samples as usize); // One emitter ray per camera sample.

            let (steps, spatial) = crate::plan::sampling_counts(&render, true);
            let rounded = steps as usize * spatial as usize;
            counts.insert(rounded);

        }
        // Finite-exposure sampling rows remain immutable across frame slots.
        let mut words = Vec::new();
        let mut sequence_offsets = std::collections::BTreeMap::new();
        for count in counts {
            sequence_offsets.insert(count, words.len() as u32);
            words.extend(crate::sampling::sequence_parameters(count));
        }

        ensure!(words.len() as u64*4<=r.device.limits().max_storage_buffer_binding_size.min(r.device.limits().max_buffer_size),crate::gpu_memory::device_limit("shadow sampling rows", format_args!("{} bytes",words.len() as u64*4), format_args!("{} bytes",r.device.limits().max_storage_buffer_binding_size.min(r.device.limits().max_buffer_size)), "reduce shadow light count"));
        let sequence=init(&r.device,"shadow sampling rows",bytemuck::cast_slice(&words),wgpu::BufferUsages::STORAGE);
        let mut changes=values::LightingTracker::default();
        if s.environment.animated() {changes.changed(values::LightingKey::EnvironmentInputs,&s.environment.sample_key(0.,0.)?);}
        Ok(Self {_fibre:s.fibre.as_ref().map(|p|p.storage()),_probes:s.probes.as_ref().map(|p|p.storage()),sequence,sequence_offsets,changes:std::sync::Arc::new(std::sync::Mutex::new(changes))})
    }
    /// Camera-budget parameters live above the independent transport flag bits.
    pub(crate) fn ray_sequence_word(&self, exposure: usize) -> u32 {
        self.sequence_offsets[&exposure]<<vork_slang::registry::INTEGRATOR_RAY_SEQUENCE_SHIFT
    }
    /// Shared sampling and retained lighting bind alongside this slot's TLAS.
    pub(crate) fn bind(
        &self,
        s: &InstalledScene,
        d: &wgpu::Device,
        rays: Option<&crate::rt::FrameRays>,
    ) -> wgpu::BindGroup {
        let hit_buffers = rays.map(|r| r.hit_bindings()).unwrap_or_default();
        let mut entries=vec![entry(0,self.sequence.as_entire_binding())];
        let empty=s.transients.buffer(d,"empty camera resource record",512,wgpu::BufferUsages::STORAGE);
        if s.pipelines.ray_shadows {
            let rays=rays.expect("camera ray transport has frame acceleration structures");
            let installed=s.rays.as_ref().expect("camera ray transport is installed");
            entries.push(entry(1,rays.tlas.as_binding()));
            entries.push(entry(2,installed.alpha.as_entire_binding()));
            entries.push(entry(3,wgpu::BindingResource::BufferArray(&hit_buffers)));
        } else {entries.extend((1..4).map(|i|entry(i,empty.as_entire_binding())));}
        let texture=|label,format|s.transients.texture(d,&wgpu::TextureDescriptor {
            label:Some(label),size:wgpu::Extent3d {width:1,height:1,depth_or_array_layers:1},mip_level_count:1,sample_count:1,
            dimension:wgpu::TextureDimension::D2,format,usage:wgpu::TextureUsages::TEXTURE_BINDING,view_formats:&[]
        }).create_view(&wgpu::TextureViewDescriptor {dimension:Some(wgpu::TextureViewDimension::D2),..Default::default()});
        let placeholders=self._fibre.is_none().then(|| [
            texture("empty camera fibre depth",wgpu::TextureFormat::Depth32Float),
            texture("empty camera fibre counts",wgpu::TextureFormat::Rgba32Float),
            texture("empty camera fibre ids",wgpu::TextureFormat::Rg32Uint),
        ]);
        if let Some(fibre)=&self._fibre {entries.extend(fibre.read_bindings(4));} else {
            let [depth,counts,ids]=placeholders.as_ref().unwrap();
            entries.push(entry(4,empty.as_entire_binding()));
            for (binding,view) in [(5,depth),(6,depth),(7,counts),(8,ids)] {entries.push(entry(binding,wgpu::BindingResource::TextureView(view)));}
        }
        let empty_grid=self._probes.is_none().then(||s.transients.buffer(d,"empty probe directory",512,wgpu::BufferUsages::STORAGE));
        if let Some(probes)=&self._probes {entries.extend(probes.read_bindings(9));}
        else {
            entries.extend((9..11).map(|i|entry(i,empty.as_entire_binding())));
            // The writable request directory cannot alias read-only fallbacks
            // in the same dispatch, even when its zero header disables probes.
            entries.push(entry(11,empty_grid.as_ref().unwrap().as_entire_binding()));
        }

        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow read"),
            layout: &s.pipelines.shadow_layout,
            entries: &entries,
        })
    }
}


#[cfg(all(test, vork_private_tests))]
include!("shadows_tests_private.rs");

/// Immutable finite-exposure sampling rows.
pub(crate) fn sequence_entry() -> wgpu::BindGroupLayoutEntry {
    pipelines::buffer_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage {read_only:true})
}
