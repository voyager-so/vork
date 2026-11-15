//! Scene structure is decided at install. Frames supply values to this plan.
use crate::install::{Draw, MeshRoot};
use crate::*;

/// A shadow reservation must survive visibility tracks on any ancestor.
pub(crate) fn shadow_visibility_animated(nodes:&[EvalNode],node:usize,animation:&crate::animation::Plan)->bool {
    std::iter::successors(Some(node),|&id|nodes[id].parent).any(|id| {
        animation.targets().any(|target|matches!(target,vork_scene::TrackTarget::Node {
            node,property:vork_scene::NodeProperty::Visibility(vork_scene::Ray::Shadow|vork_scene::Ray::All),
        } if *node==nodes[id].node.name))
    })
}

/// Renderer preparation inputs; inferred from the scene, not an authored concept.
#[derive(Clone, Copy, Default)]
pub(crate) struct EffectPlan {
    pub(crate) outputs: Option<crate::compositor::Dependencies>,
    #[cfg(test)]
    pub(crate) force_standalone_controls: bool,
    pub(crate) path: bool,
    pub(crate) trace: bool,
    pub(crate) implicit: bool,
    pub(crate) subsurface: bool,
    /// A full-resolution reflection output exists, traced through scene rays.
    pub(crate) reflections: bool,
    pub(crate) indirect: bool,
    pub(crate) denoise: bool,
    pub(crate) compositor: bool,
    /// The hemisphere images come from hardware rays with hit shading (indirect: ray).
    pub(crate) ray_indirect: bool,
    /// A ray-traced irradiance volume replaces world diffuse light (probes.volume).
    pub(crate) probes: bool,
    pub(crate) fibre: bool,
    /// Seed render images before image-dependent scene transport.
    pub(crate) image_feedback: bool,
}
impl EffectPlan {
    /// Material facts survive integrator selection; only raster beauty lowers
    /// subsurface capability to screen-space diffusion resources and passes.
    pub(crate) fn raster_diffusion(self, material_subsurface: bool) -> bool {
        material_subsurface && !self.path
    }

    /// Without a screen-space neighbour consumer, controls can stay in registers
    /// through output resolve instead of crossing five intermediate textures.
    pub(crate) fn fuse_controls(self) -> bool {
        #[cfg(test)]
        if self.force_standalone_controls {return false;}
        !(self.implicit || self.indirect || self.reflections || self.denoise || self.compositor)
    }

    /// Shared admission for the standalone controls pass and its pipeline.
    /// Graph resource pruning may remove an admitted pass, but must never add
    /// one whose pipeline was excluded during installation.
    pub(crate) fn standalone_controls(self, outputs: Option<crate::compositor::Dependencies>) -> bool {
        !self.fuse_controls()
            && !(self.implicit && self.path)
            && outputs.is_none_or(|o| o.controls() || (!self.path && (o.beauty || o.aovs || o.catcher)))
    }

    pub(crate) fn for_scene(scene: &Scene) -> Self {
        let integrator = &scene.render.settings.integrator;
        let indirect = integrator.indirect().is_some();
        let ray_indirect = crate::rt::ray_indirect(scene);
        let features = Self {
            path: integrator.is_path(),
            implicit: crate::splats::present(scene) || scene.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit)),
            indirect,
            ray_indirect,
            reflections: matches!(
                integrator.reflections(),
                Some(vork_scene::Reflections::Auto | vork_scene::Reflections::Ray)
            ),
            denoise: integrator.denoiser().is_some() || scene.render.settings.reconstruct,
            compositor: crate::catcher::has_catchers(scene) || !scene.compositor().is_empty() || scene.images.values().any(|i| i.render.is_some() && !i.compositor().is_empty()),
            probes: integrator.probe_volume().is_some(),
            ..Default::default()
        };
        features
    }
}

pub(crate) struct DrawFlags {
    pub hair: bool,
    pub transparent: bool,
    pub moving: bool,
    pub casts_shadows: bool,
    pub alpha_mask: bool,
}
/// Fixed shadow command partitions, also used by opaque ray grouping.
#[derive(Default)]
pub(crate) struct CasterPlan {
    pub instances: [Vec<(usize, u32)>; 2],
}
impl CasterPlan {
    fn include(
        &mut self,
        id: usize,
        object: u32,
        local_bounds: (Vec3, Vec3),
        flags: &DrawFlags,
        value: &Object,
    ) {
        if !flags.moving {
            let bounds = crate::shadow_envelope::transformed(value.model, local_bounds);
            if !bounds.0.is_finite() || !bounds.1.is_finite() {
                return;
            }
        }
        if flags.casts_shadows
            && (flags.moving || value.metadata[1] & constants::RAY_SHADOW != 0)
        {
            self.instances[usize::from(flags.moving)].push((id, object));
        }
    }
}
pub(crate) struct ScenePlan {
    pub effects: EffectPlan,
    pub draws: Vec<DrawFlags>,
    pub casters: CasterPlan,
    pub transparent: bool,
    pub deform: bool,
    /// Ray structures exist: rebuild them after deformation, before any ray reads.
    pub rays: bool,
    pub cluster: bool,
    /// Recorded passes per frame: the exposure strata plus the nominal control
    /// pass when [`nominal_control_pass`] can record one. Per-sample upload and
    /// timestamp storage is sized from this.
    pub samples: u64,
    /// Maximum scene-time strata reserved at install; selected per exposure.
    pub exposure_samples: u32,
    /// Maximum spatial visibility passes per stratum; selected per exposure.
    pub visibility_samples: u32,
    pub aovs: bool,
    /// Raster colour is decoupled: each scene-time stratum collects spatial visibility and
    /// every visibility slot is shaded once per scene-time sample (passes/decoupled.rs).
    pub decoupled: bool,
    /// Decoupled transparent draws refract the opaque image (screen refraction):
    /// the shade splits into its opaque pass and the transparent composite.
    pub screen_refraction: bool,
}
/// Whether a frame recording `colour_samples` exposure strata can follow them with
/// one nominal (unjittered, nominal-time) control pass; the frame records it only
/// when a depth, id, normal or motion output needs it.
pub(crate) fn nominal_control_pass(colour_samples: u32, temporal: bool, motion_blur: bool) -> bool {
    colour_samples > 1 || temporal || motion_blur
}
/// Exposure strata and the visibility budget each stratum integrates.
pub(crate) fn sampling_counts(render: &vork_scene::Render, allow_shutter: bool) -> (u32, u32) {
    let per_stratum = render.settings.integrator.is_path() || (allow_shutter && render.motion_blur.as_ref().is_some_and(|m| m.shutter > 0.));
    (if per_stratum { render.settings.samples } else { 1 }, if per_stratum { 1 } else { render.settings.samples })
}

impl ScenePlan {
    pub(crate) fn new(
        effects: EffectPlan,
        render: &vork_scene::Render,
        draws: &[Draw],
        materials: &[Material],
        nodes: &[EvalNode],
        roots: &[MeshRoot],
        objects: &[Object],
        light_plan: &LightPlan,
        animation: &crate::animation::Plan,
        rays: bool,
        compute: Option<&crate::compute::Compute>,
    ) -> Self {
        let flags: Vec<_> = draws
            .iter()
            .map(|draw| {
                let node = &nodes[roots[draw.object as usize].node];
                DrawFlags {
                    hair: materials[draw.material].physical_hair,
                    transparent: materials[draw.material].transparent,
                    moving: draw.evaluated()
                        || node.dynamic
                        || materials[draw.material].shadow_animated,
                    // Fixed possible-caster reservation; active visibility is per-object data.
                    casts_shadows: draw.instances.clone().any(|object| {
                        (objects[object as usize].metadata[1] & constants::RAY_SHADOW) != 0
                            || (roots[object as usize].imported_pose.is_some() && animation.targets().any(|target| {
                                matches!(target, vork_scene::TrackTarget::Imported {asset,property:vork_scene::NodeProperty::Visibility(vork_scene::Ray::All),..}
                                    if *asset == nodes[roots[object as usize].node].node.name)
                            }))
                            || ((roots[object as usize].imported_visibility
                                & constants::RAY_SHADOW)
                                != 0
                                && shadow_visibility_animated(nodes,roots[object as usize].node,animation))
                    }),
                    alpha_mask: materials[draw.material].alpha_mask,
                }
            })
            .collect();
        let mut casters = CasterPlan::default();
        for (id, (draw, flags)) in draws.iter().zip(&flags).enumerate() {
            // Compute writes one indirect count for this contiguous range. The
            // per-frame CPU path evaluates only its parent; reserving one caster
            // per element would reintroduce linear work and duplicate GPU draws.
            let instances = if compute.is_some_and(|c| c.computed_draw(id)) {
                draw.instances.start..draw.instances.start + 1
            } else {
                draw.instances.clone()
            };
            for object in instances {
                casters.include(id, object, draw.bounds, flags, &objects[object as usize]);
            }
        }
        let deform = draws.iter().any(|d| d.evaluated());
        let transparent = flags.iter().any(|f| f.transparent);
        let (mut exposure_samples, mut visibility_samples) = sampling_counts(render, true);
        // These are install-time reservations, not the current exposure choice.
        // A shutter track can require either recorded visibility layout later.
        if animation.has_shutter_animation() {
            exposure_samples = render.settings.samples;
            if !effects.path { visibility_samples = render.settings.samples; }
        }
        Self {
            exposure_samples,
            visibility_samples,
            aovs: !render.aovs.is_empty(),
            samples: u64::from(exposure_samples)
                + u64::from(nominal_control_pass(exposure_samples, false, render.motion_blur.as_ref().is_some_and(|m| m.shutter > 0.) || animation.has_shutter_animation())),
            decoupled: !effects.path,
            screen_refraction: !effects.path && render.settings.integrator.raster().is_some_and(|r| match r.refraction {
                vork_scene::Refraction::Screen => draws.iter().any(|draw| materials[draw.material].transmissive),
                _ => false,
            }),
            effects,
            transparent,
            draws: flags,
            casters,
            deform,
            rays,
            cluster: light_plan.nodes.iter().any(|&i| !matches!(nodes[i].node.kind, NodeKind::Light { shape: vork_scene::LightShape::Sun, .. })) && !effects.path,
        }
    }
}

#[cfg(test)]
mod caster_plan_tests {
    use super::*;

    #[test]
    fn closed_shutter_uses_spatial_samples() -> Result<()> {
        let scene: Scene = vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"motion_blur":{"shutter":0},
                "settings":{"integrator":"raster","samples":64}},
            "nodes":[],
            "animation":[{"tracks":[{"path":["render","motion_blur","shutter"],
                "keys":[[0,0],[1,1]]}]}]}))?;
        assert_eq!(sampling_counts(&scene.render,true),(1,64));
        let mut render = scene.render.clone();
        render.motion_blur.as_mut().unwrap().shutter = 1.;
        assert_eq!(sampling_counts(&render,true),(64,1));
        assert_eq!(sampling_counts(&render,false),(1,64));
        render.settings.integrator = vork_scene::v2::Integrator::Path(Default::default());
        render.motion_blur.as_mut().unwrap().shutter = 0.;
        assert_eq!(sampling_counts(&render,true),(64,1));
        Ok(())
    }

    #[test]
    fn commands_exclude_hidden_casters_but_reserve_movers() {
        let mut plan = CasterPlan::default();
        let mut value = Object::zeroed();
        value.model = Mat4::from_translation(Vec3::X);
        value.metadata[1] = constants::RAY_CAMERA | constants::RAY_SHADOW;
        let mut flags = DrawFlags {
            hair: false,
            moving: false,
            transparent: false,
            casts_shadows: true,
            alpha_mask: false,
        };
        let bounds = (Vec3::ZERO, Vec3::ONE);
        plan.include(2, 8, bounds, &flags, &value);
        value.metadata[1] = constants::RAY_CAMERA;
        plan.include(3, 9, bounds, &flags, &value);
        flags.moving = true;
        value.metadata[1] = 0; // May become visible later: reserve a compact command.
        plan.include(4, 10, bounds, &flags, &value);
        assert_eq!(plan.instances, [vec![(2, 8)], vec![(4, 10)]]);
    }
}
