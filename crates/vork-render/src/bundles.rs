//! One persistent recorder for depth, forward, casters and transparent controls.
use crate::*;
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Depth,
    /// Decoupled visibility keys for one sample of the exposure (passes/decoupled.rs).
    Visibility(u32),
    /// Transparent draws' fragments for one sample of the exposure, after its keys.
    TransparentFragments(u32),
    Controls,
    Fibre(crate::fibre_transport::Stage, bool),
    /// Transparent draws' nearest-surface controls.
    TransparentControls,
}
#[derive(Clone, Copy)]
pub(crate) struct Recorder<'a> {
    pub transparent_instances: Option<&'a wgpu::Buffer>,
    pub camera_instances: Option<&'a wgpu::Buffer>,
    pub compute: Option<&'a wgpu::Buffer>,
    pub s: &'a InstalledScene,
    pub graph: &'a crate::graph::Graph,
    pub included: Option<&'a [bool]>,
    pub included_objects: Option<&'a [bool]>,
    pub deforms: &'a [Option<EvaluatedGeometry>],
    pub fibre_capture: Option<&'a wgpu::BindGroup>,
    /// The scene material table, or this frame's animated copy of it.
    pub shadow_bind: &'a wgpu::BindGroup,
    pub material_bind: &'a wgpu::BindGroup,
    /// The view's transparent fragment list (passes/decoupled.rs Fragments).
    pub visibility: Option<&'a crate::passes::decoupled::Sampling>,
    pub fragments: Option<&'a wgpu::BindGroup>,
    pub fragments_geometry: Option<&'a wgpu::BindGroup>,
    /// Its device, and its allocation errors: wgpu panics when a bundle binds a
    /// resource whose allocation failed, so recording reports that error instead.
    pub r: &'a Renderer,
}
impl Recorder<'_> {
    pub(crate) fn record(
        &self,
        label: &str,
        kind: Kind,
        global: &wgpu::BindGroup,
        indirect: Option<&wgpu::Buffer>,
        only: Option<usize>,
    ) -> Result<wgpu::RenderBundle> {
        self.r.errors.check()?;
        let s = self.s;
        let formats = pipelines::control_formats(s.pipelines.denoise);
        let fibre_near_formats=[Some(wgpu::TextureFormat::Rg32Uint)];
        let visibility_formats=[Some(crate::passes::decoupled::KEY_FORMAT)];
        let fibre_formats=[Some(wgpu::TextureFormat::Rgba16Float)];
        let mut b = self
            .r
            .device
            .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some(label),
                color_formats: match kind {
                    Kind::Depth | Kind::Fibre(crate::fibre_transport::Stage::Far,_) => &[],
                    Kind::Fibre(crate::fibre_transport::Stage::Near,_) => &fibre_near_formats,
                    Kind::Visibility(_) | Kind::TransparentFragments(_) => &visibility_formats,
                    Kind::Fibre(crate::fibre_transport::Stage::Accumulate,_) => &fibre_formats,
                    _ => &formats,
                },
                depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_read_only: matches!(kind, Kind::Controls | Kind::Fibre(crate::fibre_transport::Stage::Accumulate,_)) && !s.pipelines.field_controls,
                    stencil_read_only: true,
                }),
                sample_count: if matches!(kind, Kind::Fibre(_,_) | Kind::Visibility(_) | Kind::TransparentFragments(_)) {
                    1
                } else {
                    s.pipelines.raster_samples
                },
                multiview: None,
            });
        if !self.graph.draws(kind) || self.included.is_some_and(|draws|!draws.iter().any(|&draw|draw)) {return Ok(b.finish(&wgpu::RenderBundleDescriptor {label:Some(label)}));}
        let caster_filter = if let Kind::Fibre(_, moving) = kind {
            Some(moving)
        } else {
            None
        };
        b.set_bind_group(0, global, &[]);
        // Every material reads the one scene material table.
        b.set_bind_group(1, self.material_bind, &[]);
        if s.pipelines.program_trace && matches!(kind,Kind::Depth|Kind::Controls|Kind::TransparentControls) {
            b.set_bind_group(2,self.shadow_bind,&[]);
        }
        if matches!(kind, Kind::TransparentFragments(_) | Kind::Visibility(_)) || ((s.pipelines.field_controls || s.pipelines.software_raster) && matches!(kind,Kind::Controls|Kind::TransparentControls)) || (s.pipelines.software_raster && kind==Kind::Depth) {
            b.set_bind_group(2, self.fragments.expect("transparent fragment list"), &[]);
            b.set_bind_group(3, self.fragments_geometry.expect("transparent opacity geometry"), &[]);
            if s.pipelines.program_trace {b.set_bind_group(4,self.shadow_bind,&[]); }
        }
        if matches!(kind,Kind::Fibre(crate::fibre_transport::Stage::Accumulate,_)) {
            b.set_bind_group(2,self.fibre_capture.expect("fibre capture resources installed"),&[]);
        }
        let records: Vec<_> = if let Some(i) = only {
            vec![(i, i, s.draws[i].instances.clone())]
        } else if let Some(moving) = caster_filter.filter(|_| indirect.is_some()) {
            s.plan.casters.instances[usize::from(moving)]
                .iter()
                .enumerate()
                .filter_map(|(command, &(draw, object))| {
                    if s.compute.as_ref().is_some_and(|c| c.computed_draw(draw)) {
                        // Computed ranges are contiguous and share node links;
                        // keep one GPU count for the whole range in shadow bundles too.
                        (object == s.draws[draw].instances.start)
                            .then(|| (command, draw, s.draws[draw].instances.clone()))
                    } else {
                        Some((command, draw, object..object + 1))
                    }
                })
                .collect()
        } else {
            s.draws
                .iter()
                .enumerate()
                .map(|(i, draw)| (i, i, draw.instances.clone()))
                .collect()
        };
        for (command, i, instances) in records {
            // Packed transparency selects its draw slots after sorting; those slots
            // can lie outside the source subtree but contain only selected objects.
            if !(kind == Kind::TransparentControls && indirect.is_some())
                && self.included.is_some_and(|mask| !mask[i])
            {
                continue;
            }
            let draw = &s.draws[i];
            if !draw.rasterized() {continue;}
            if kind==Kind::Depth && s.pipelines.field_controls && draw.implicit.is_some() {continue;}
            let flags = &s.plan.draws[i];
            let moving = flags.moving;
            if matches!(kind, Kind::TransparentFragments(_)) && s.materials[draw.material].zero_opacity {
                continue;
            }
            if matches!(kind,Kind::Fibre(_, _)) && !flags.hair {continue;}
            if caster_filter.is_some() && !flags.casts_shadows {
                continue;
            }
            if caster_filter.is_some_and(|want| want != moving) {
                continue;
            }
            if caster_filter.is_none() && flags.transparent != matches!(kind, Kind::TransparentControls | Kind::TransparentFragments(_)) {
                continue;
            }
            b.set_pipeline(
                &s.geometry_pipelines[&(
                    draw.layout,
                    crate::pipelines::Role::of(kind, flags, &s.materials[draw.material]),
                )]
                    .current(true),
            );
            b.set_vertex_buffer(
                0,
                self.deforms[i].as_ref().map_or_else(
                    || draw.vertices.slice(draw.vertex_offset..),
                    |d| d.output.slice(),
                ),
            );
            b.set_vertex_buffer(
                1,
                self.deforms[i].as_ref().map_or_else(
                    || draw.vertices.slice(draw.vertex_offset..),
                    |d| d.previous_output.slice(),
                ),
            );
            b.set_vertex_buffer(
                2,
                if draw.layout.constant_attributes() {
                    s.vertex_defaults.slice(..)
                } else {
                    self.deforms[i].as_ref().and_then(|d|d.attributes.as_ref()).map_or_else(||draw.attributes.slice(draw.attribute_offset()..), |a|a.slice(..))
                },
            );
            let camera=matches!(kind,Kind::Depth|Kind::Visibility(_)|Kind::TransparentFragments(_)|Kind::Controls);
            b.set_vertex_buffer(3, if camera {
                self.camera_instances.unwrap_or(&s.vertex_defaults).slice(..)
            } else if kind == Kind::TransparentControls {
                self.transparent_instances.expect("transparent instance mapping").slice(..)
            } else {s.vertex_defaults.slice(..)});
            if let Kind::Visibility(sample) | Kind::TransparentFragments(sample) = kind {
                b.set_vertex_buffer(4, self.visibility.expect("view visibility sampling").pass_parameters(sample));
                b.set_vertex_buffer(5, s.vertex_defaults.slice(..));
            } else {
                b.set_vertex_buffer(4, s.vertex_defaults.slice(..));
            }

            b.set_index_buffer(
                draw.indices.slice(draw.index_offset..),
                wgpu::IndexFormat::Uint32,
            );
            if let Some(args) = indirect {
                b.draw_indexed_indirect(
                    args,
                    (command * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>()) as u64,
                );
            } else if let Some(args) = self.compute.filter(|_| {
                matches!(kind, Kind::Depth | Kind::Visibility(_) | Kind::TransparentFragments(_) | Kind::Controls)
                    && s.compute.as_ref().is_some_and(|c| c.computed_draw(i))
            }) {
                b.draw_indexed_indirect(
                    args,
                    (i * std::mem::size_of::<wgpu::util::DrawIndexedIndirectArgs>()) as u64,
                );
            } else {
                if let Some(mask) = self.included_objects {
                    // Retained instance groups can span subtree boundaries.
                    let mut start = None;
                    for object in instances.clone().chain(std::iter::once(instances.end)) {
                        if object < instances.end && mask[object as usize] {
                            start.get_or_insert(object);
                        } else if let Some(first) = start.take() {
                            b.draw_indexed(0..draw.count, 0, first..object);
                        }
                    }
                } else {
                    b.draw_indexed(0..draw.count, 0, instances);
                }
            }
        }
        Ok(b.finish(&wgpu::RenderBundleDescriptor { label: Some(label) }))
    }
}
