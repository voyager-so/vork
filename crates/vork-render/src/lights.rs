//! Light records, distributions and per-frame light evaluation.
pub(crate) mod distribution;
pub(crate) mod tree;
use super::*;
pub(crate) mod links;

pub(crate) fn needs_receiver_masks(lights: &[Light]) -> bool {
    let unrestricted = constants::RAY_DIFFUSE | constants::RAY_SPECULAR | constants::LIGHT_LINK_ALL;
    lights.iter().any(|light| {
        light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] & unrestricted != unrestricted
    })
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Light {
    pub shadow_offsets: [u32; constants::LIGHT_SHADOW_OFFSET_WORDS],
    pub position: [f32; 4],   // xyz, kind (sun=0, point=1, spot=2, area=3)
    pub direction: [f32; 4],  // xyz, outer cosine
    pub radiance: [f32; 4],   // rgb, inner cosine
    pub params: [f32; 4],     // reserved, emitter size, reserved, reserved
    pub emitter_u: [f32; 4],  // area edge + ellipse flag
    pub emitter_v: [f32; 4],  // area edge
    pub emission: [f32; 4], // source rotation quaternion xyz (positive w), profile table address
    pub projection: [f32; 4], // world width/tan(half angle), range, glTF flag, min/max offset
}
#[derive(Clone)]
pub(super) struct LightPlan {
    pub nodes: Vec<usize>,
    pub blackbody: Option<crate::light_temperature::Table>,
}
impl LightPlan {
    pub fn logical(nodes: &[EvalNode], limits: &wgpu::Limits, volume: bool) -> Result<Self> {
        let nodes: Vec<_> = nodes.iter().enumerate().filter_map(|(id, node)| {
            if !matches!(&node.node.kind, NodeKind::Light { shape, .. }
                if !matches!(shape, vork_scene::LightShape::Environment)) {
                return None;
            }
            let mut ancestor = Some(id);
            let mut dynamic = false;
            while let Some(i) = ancestor {
                dynamic |= nodes[i].dynamic;
                ancestor = nodes[i].parent;
            }
            let mask = buffers::node_visibility_mask(nodes, id);
            let consumers = constants::RAY_DIFFUSE | constants::RAY_SPECULAR
                | if volume { constants::RAY_VOLUME } else { 0 };
            (dynamic || mask & consumers != 0).then_some(id)
        }).collect();
        ensure!(nodes.len() <= limits.max_storage_buffer_binding_size as usize / std::mem::size_of::<Light>(),
            crate::gpu_memory::device_limit("light buffer", format_args!("{} bytes",nodes.len()*std::mem::size_of::<Light>()), format_args!("max_storage_buffer_binding_size={} bytes",limits.max_storage_buffer_binding_size), "reduce light count"));
        Ok(Self { nodes, blackbody: None })
    }
    pub fn evaluate(
        &self,
        nodes: &(impl std::ops::Index<usize, Output = EvalNode> + ?Sized),
        worlds: &[Mat4],
        links: &links::Links,
    ) -> Result<Vec<Light>> {

        self.nodes
            .iter()
            .map(|&id| {
                use vork_scene::LightShape;
                let NodeKind::Light {
                    shape,
                    color,
                    temperature,
                    energy: intensity,
                    angle,
                    radius,
                    spot_size,
                    spot_blend,
                    size: width,
                    size_y: height,
                    ..
                } = &nodes[id].node.kind
                else {
                    unreachable!()
                };
                let height = height.as_ref().unwrap_or(width);
                let intensity = intensity.unwrap_or(shape.default_energy());
                let mut color = color.unwrap_or(shape.default_color(false));
                if let Some(kelvin) = temperature {
                    let table = self.blackbody.as_deref().context("light temperature without an installed blackbody table")?;
                    let tint = crate::light_temperature::blackbody(table, *kelvin)?;
                    color = std::array::from_fn(|c| color[c] * tint[c]);
                }
                let (size, kind, inner, outer) = match shape {
                    LightShape::Sun => (angle * 0.5, 0., 0., 0.),
                    LightShape::Point => (*radius, 1., 0., std::f32::consts::FRAC_PI_4),
                    LightShape::Spot => (
                        *radius,
                        2.,
                        vork_scene::spot_inner_angle(*spot_size, *spot_blend),
                        spot_size * 0.5,
                    ),
                    LightShape::Rectangle | LightShape::Ellipse => (
                        0.5 * width.hypot(*height),
                        3.,
                        0.,
                        std::f32::consts::FRAC_PI_4,
                    ),
                    LightShape::Environment => unreachable!(),
                };
                let frame = values::node_view(id, nodes, worlds)?.inverse();
                let direction = frame.transform_vector3(Vec3::NEG_Z);
                let position = worlds[id].transform_point3(Vec3::ZERO);
                let mut emitter_u = [0.; 4];
                let mut emitter_v = [0.; 4];
                if kind == 3. {
                    let u =
                        frame.transform_vector3(Vec3::X) * worlds[id].x_axis.truncate().length();
                    let v =
                        frame.transform_vector3(Vec3::Y) * worlds[id].y_axis.truncate().length();
                    emitter_u = (u * *width)
                        .extend(if matches!(shape, LightShape::Ellipse) {
                            1.
                        } else {
                            0.
                        })
                        .to_array();
                    emitter_v = (v * *height).extend(0.).to_array();
                }
                let size = if kind == 3. {
                    0.5 * (Vec3::from_slice(&emitter_u[..3]).length_squared()
                        + Vec3::from_slice(&emitter_v[..3]).length_squared())
                    .sqrt()
                } else {
                    size
                };
                let mut light = Light {
                    shadow_offsets: [0; constants::LIGHT_SHADOW_OFFSET_WORDS],
                    emitter_u,
                    emitter_v,
                    emission: [0.;4],
                    position: position.extend(kind).to_array(),
                    direction: direction.extend(outer.cos()).to_array(),
                    radiance: [
                        color[0] * intensity,
                        color[1] * intensity,
                        color[2] * intensity,
                        inner.cos(),
                    ],
                    params: [0., size, 0., 0.],
                    projection: [
                        outer.tan(),
                        nodes[id].embedded.and_then(|e| e.3).unwrap_or(0.),
                        if nodes[id].embedded.is_some() { 1. } else { 0. },
                        0.,
                    ],
                };

                // A collapsed area has no emitting surface. Other analytic shapes
                // ignore node scale, including their authored radius/angle.
                if kind == 3. && Vec3::from_slice(&emitter_u[..3]).cross(Vec3::from_slice(&emitter_v[..3])).length_squared() == 0. {
                    light.radiance[..3].fill(0.);
                }
                light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] =
                    buffers::node_visibility_mask(nodes, id);
                let selection = links.illuminates(id);
                light.shadow_offsets[constants::LIGHT_LINK_WORD] = selection.mask;
                if selection.all {
                    light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] |=
                        constants::LIGHT_LINK_ALL;
                }
                if crate::rt::traces_shadows(&nodes[id].node.kind) {
                    let casters = links.shadowed_by(id);
                    light.shadow_offsets[constants::LIGHT_RAY_WORD] = constants::LIGHT_RAY_SHADOWED
                        | if casters.all {
                            constants::LIGHT_RAY_SHADOWED_BY_ALL
                        } else {
                            0
                        };
                    light.shadow_offsets[constants::LIGHT_RAY_LINK_WORD] = casters.mask;
                }
                Ok(light)
            })
            .collect::<Result<Vec<_>>>()
            .map(|mut lights| { publish_volume_slots(&mut lights); lights })
    }
}

/// Publish once with the current masks through the existing light upload path.
fn publish_volume_slots(lights: &mut [Light]) {
    let mut slot=0u32;
    for light in lights {
        let visible=light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD]&constants::RAY_VOLUME!=0;
        light.shadow_offsets[vork_slang::registry::LIGHT_VOLUME_SLOT_WORD]=if visible {slot} else {u32::MAX};
        slot+=u32::from(visible);
    }
}

#[test]
fn packed_volume_slots_follow_current_masks() {
    let mut lights=vec![Light::zeroed();5];
    for masks in [[true,false,true,false,true],[false,true,false,true,false]] {
        for (light,visible) in lights.iter_mut().zip(masks) {
            light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD]=if visible {constants::RAY_VOLUME} else {constants::RAY_CAMERA};
        }
        publish_volume_slots(&mut lights);
        let mut packed=0;
        for (light,visible) in lights.iter().zip(masks) {
            let slot=light.shadow_offsets[vork_slang::registry::LIGHT_VOLUME_SLOT_WORD];
            assert_eq!(slot,if visible {packed} else {u32::MAX});
            if visible {packed+=1;}
        }
        // Clustered/random access observes the same mapping as ordered traversal.
        for index in [4,1,3,0,2] {
            let expected=masks[..index].iter().filter(|&&v|v).count() as u32;
            if masks[index] {assert_eq!(lights[index].shadow_offsets[vork_slang::registry::LIGHT_VOLUME_SLOT_WORD],expected);}
        }
    }
}

pub(crate) fn visibility_from_mask(mask: u32) -> ::vork_scene::v2::Visibility {
    ::vork_scene::v2::Visibility {
        visible: mask != 0,
        camera: mask & constants::RAY_CAMERA != 0,
        shadow: mask & constants::RAY_SHADOW != 0,
        diffuse: mask & constants::RAY_DIFFUSE != 0,
        glossy: mask & constants::RAY_SPECULAR != 0,
        transmission: mask & constants::RAY_TRANSMISSION != 0,
    }
}

/// glTF photometric units: lux for suns, candela for local lights. The
/// renderer uses radiometric irradiance / isotropic power (Blender convention).
/// LUMENS_PER_WATT is the photometric conversion used for punctual-light import.
pub(super) fn import(
    doc: &gltf::Document,
    root: usize,
    asset: usize,
    world: Mat4,
    prefix: &str,
    overrides: &::vork_scene::v2::Table<::vork_scene::v2::AssetNodeOverrides>,
    out: &mut Vec<EvalNode>,
    camera_materials: &mut vork_scene::v2::Table<vork_scene::v2::Material>,
) -> Result<()> {
    let prefix = if prefix.is_empty() { format!("#{root}") } else { prefix.to_owned() };
    fn visit(
        node: gltf::Node<'_>,
        parent: Mat4,
        root: usize,
        asset: usize,
        world: Mat4,
        prefix: &str,
        animated: bool,
        overrides: &::vork_scene::v2::Table<::vork_scene::v2::AssetNodeOverrides>,
        path: &mut Vec<(usize, String)>,
        out: &mut Vec<EvalNode>,
        camera_materials: &mut vork_scene::v2::Table<vork_scene::v2::Material>,
    ) -> Result<()> {
        let local = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        path.push((node.index(), node.name().unwrap_or("").to_owned()));
        let name = format!("{prefix}/{}", node.name().map(str::to_owned).unwrap_or_else(|| format!("#{}", node.index())));
        if let Some(camera) = node.camera() {
            let lens = match camera.projection() {
                gltf::camera::Projection::Perspective(p) => {
                    let far = p.zfar();
                    if far.is_none() {crate::warnings::dropped("camera_far_clip", serde_json::json!({"camera":name,"clip_end":"infinite"}), format!("warning: camera {name}: ignored infinite far clip; using the camera default"));}
                    let mut inputs=serde_json::json!({"lens":24. / (2. * (p.yfov() * 0.5).tan()),"fit":"vertical","clip_start":p.znear()});
                    if let Some(far)=far {inputs["clip_end"]=far.into();}
                    inputs
                }
                gltf::camera::Projection::Orthographic(o) => {
                    let mut inputs=serde_json::json!({"projection":"ortho", "fit":"vertical", "ortho_scale":2. * o.ymag(),"clip_end":o.zfar()});
                    if o.znear()>0. {inputs["clip_start"]=o.znear().into();}
                    else {crate::warnings::dropped("camera_near_clip", serde_json::json!({"camera":name,"clip_start":0}), format!("warning: camera {name}: ignored zero near clip; using the camera default"));}
                    inputs
                }
            };
            let mut value = serde_json::json!({"type":"camera", "name":name});
            let material=format!("{name}/camera");
            camera_materials.insert(material.clone(),serde_json::from_value(serde_json::json!({"program":"std/camera","inputs":lens}))?);
            value["material"]=material.into();
            let mut imported: Node = serde_json::from_value(value)?;
            imported.visibility = visibility_from_mask(buffers::imported_visibility_mask(overrides, path));
            out.push(EvalNode { camera: None,
                node: imported, parent: Some(root), world: world * local, dynamic: animated,
                embedded: Some((asset, node.index(), root, None)),
            });
        }
        if let Some(light) = node.light() {
            use gltf::khr_lights_punctual::Kind;
            let (shape, intensity, cone) = match light.kind() {
                Kind::Directional => (
                    vork_scene::LightShape::Sun,
                    light.intensity() / constants::LUMENS_PER_WATT,
                    None,
                ),
                Kind::Point => (
                    vork_scene::LightShape::Point,
                    light.intensity() * (4. * std::f32::consts::PI / constants::LUMENS_PER_WATT),
                    None,
                ),
                Kind::Spot {
                    inner_cone_angle,
                    outer_cone_angle,
                } => (
                    vork_scene::LightShape::Spot,
                    light.intensity() * (4. * std::f32::consts::PI / constants::LUMENS_PER_WATT),
                    Some((outer_cone_angle * 2., (inner_cone_angle.cos() - outer_cone_angle.cos()) / (1. - outer_cone_angle.cos()))),
                ),
            };
            // An imported light is a v2 light node with glTF's values; it casts
            // shadows like an authored light (Blender's import default).
            let mut fields = serde_json::json!({
                "type": "light",
                "name": name,
                "shape": shape,
                "color": light.color(),
                "energy": intensity,
                // Punctual: glTF lights have no angular size.
                "angle": 0,
            });
            if let Some((size, blend)) = cone {
                fields["spot_size"] = size.into();
                fields["spot_blend"] = blend.into();
            }
            let mut imported: Node = serde_json::from_value(fields)?;
            imported.visibility =
                visibility_from_mask(buffers::imported_visibility_mask(overrides, path));
            out.push(EvalNode { camera: None,
                node: imported,
                parent: Some(root),
                world: world * local,
                dynamic: animated,
                embedded: Some((asset, node.index(), root, light.range())),
            });
        }
        for child in node.children() {
            visit(
                child, local, root, asset, world, prefix, animated, overrides, path, out, camera_materials,
            )?;
        }
        path.pop();
        Ok(())
    }
    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .context("glTF has no scene")?;
    for node in scene.nodes() {
        visit(
            node,
            Mat4::IDENTITY,
            root,
            asset,
            world,
            &prefix,
            doc.animations().next().is_some(),
            overrides,
            &mut Vec::new(),
            out, camera_materials,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Scaling a batch of individually faint lights must scale scene-linear
    // illumination even when every receiver lies outside the old cutoff.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn faint_lights_preserve_energy_under_display_gain() -> Result<()> {
        let r = crate::test_renderer()?;
        let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut input:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/analytic.json"))?;
        input["animation"]=serde_json::json!([]);
        input["render"]=serde_json::json!({"resolution":[8,8],"settings":{"samples":1,"integrator":"raster","probes":null,"denoiser":null}});
        input["materials"]=serde_json::json!({"mat":{"inputs":{"base_color":[1,1,1],"specular_weight":0}}});
        input["nodes"]=serde_json::json!([
            {"type":"camera","translation":[0,0,5],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1}}},
            {"type":"mesh","geometry":"shape_box","materials":["mat"]},
            {"type":"light","shape":"point","translation":[0,0,3],"energy":1,"shadowed_by":[]},
            {"type":"light","shape":"point","translation":[1,0,3],"energy":1,"shadowed_by":[]},
            {"type":"light","shape":"environment","energy":0}]);
        let mut frames=Vec::new();
        for power in [1.,0.01] {
            input["nodes"][2]["energy"]=serde_json::json!(power);
            input["nodes"][3]["energy"]=serde_json::json!(power);
            let scene=r.install(vork_scene::v2::from_test_value(input.clone())?,&root)?;
            let mut frame=scene.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            frame.submit_outputs(&scene,&r,0.,crate::ReadbackOutputs {float:crate::FloatOutputs {color:true,..Default::default()},..Default::default()})?;
            frames.push(frame.collect(&r)?.linear_rgba.unwrap());
        }
        assert!(frames[0].iter().any(|p|p[0]>0.001));
        for (bright,faint) in frames[0].iter().zip(&frames[1]) {
            for c in 0..3 {assert!((bright[c]-faint[c]*100.).abs()<bright[c]*0.002+1e-6,"lost faint illumination: {bright:?} vs {faint:?}");}
        }
        Ok(())
    }

    use super::*;
    #[test]
    fn receiver_masks_follow_evaluated_visibility_and_explicit_selection() {
        let mut light = Light::zeroed();
        let word = constants::LIGHT_CONTRIBUTION_WORD;
        let default = constants::RAY_VISIBILITY_ALL | constants::LIGHT_LINK_ALL;
        light.shadow_offsets[word] = default & !constants::RAY_SHADOW;
        assert!(!needs_receiver_masks(&[light]));
        light.shadow_offsets[word] = default & !constants::RAY_DIFFUSE;
        assert!(needs_receiver_masks(&[light]));
        light.shadow_offsets[word] = default & !constants::RAY_SPECULAR;
        assert!(needs_receiver_masks(&[light]));
        // An explicit selection of every named group still excludes ungrouped receivers.
        light.shadow_offsets[word] = default & !constants::LIGHT_LINK_ALL;
        light.shadow_offsets[constants::LIGHT_LINK_WORD] = u32::MAX;
        assert!(needs_receiver_masks(&[light]));
        light.shadow_offsets[word] = default;
        assert!(!needs_receiver_masks(&[light]));
        assert!(!needs_receiver_masks(&[]));
    }

    #[test]
    fn area_height_follows_size_unless_explicit() {
        for shape in ["rectangle", "ellipse"] {
            for authored_height in [None, Some(2.)] {
                let mut input = serde_json::json!({"type":"light","shape":shape,"size":4.});
                if let Some(height) = authored_height { input["size_y"] = height.into(); }
                let mut nodes = vec![EvalNode { camera: None,
                    node: vork_scene::v2::from_test_value(input).unwrap(), parent: None,
                    world: Mat4::IDENTITY, dynamic: false, embedded: None,
                }];
                for width in [4., 6.] {
                    let NodeKind::Light { size, .. } = &mut nodes[0].node.kind else { unreachable!() };
                    *size = width;
                    let links = links::Links::new(&nodes).unwrap();
                    let plan = LightPlan::logical(&nodes, &wgpu::Limits::default(), false).unwrap();
                    let lights = plan.evaluate(
                        &nodes, &[Mat4::IDENTITY], &links).unwrap();
                    assert_eq!(Vec3::from_slice(&lights[0].emitter_u).length(), width);
                    assert_eq!(Vec3::from_slice(&lights[0].emitter_v).length(), authored_height.unwrap_or(width));
                }
            }
        }
    }

    #[test]
    fn punctual_units_hierarchy_and_cones() {
        let doc = gltf::Gltf::from_slice(
            br#"{
            "asset":{"version":"2.0"}, "extensionsUsed":["KHR_lights_punctual"],
            "extensions":{"KHR_lights_punctual":{"lights":[
                {"type":"directional","intensity":683},
                {"type":"point","intensity":683,"range":7},
                {"type":"spot","intensity":683,"spot":{"innerConeAngle":0.2,"outerConeAngle":0.6}}
            ]}},
            "scene":0,"scenes":[{"nodes":[0]}],"nodes":[
                {"translation":[2,3,4],"children":[1,2,3]},
                {"extensions":{"KHR_lights_punctual":{"light":0}}},
                {"translation":[1,0,0],"extensions":{"KHR_lights_punctual":{"light":1}}},
                {"extensions":{"KHR_lights_punctual":{"light":2}}}
            ]}"#,
        )
        .unwrap()
        .document;
        let mut nodes = Vec::new();
        import(
            &doc,
            0,
            0,
            Mat4::from_translation(Vec3::X),
            "rig",
            &Default::default(),
            &mut nodes, &mut Default::default(),
        )
        .unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[1].node.name, "rig/#2");
        assert_eq!(
            nodes[1].world.transform_point3(Vec3::ZERO),
            Vec3::new(4., 3., 4.)
        );
        assert_eq!(nodes[1].embedded.unwrap().3, Some(7.));
        let overrides = vork_scene::v2::from_test_value(serde_json::json!({"#0":{"visible_diffuse":false},"#2":{"visible_glossy":false}})).unwrap();
        let mut masked = Vec::new();
        import(&doc, 0, 0, Mat4::IDENTITY, "rig", &overrides, &mut masked, &mut Default::default()).unwrap();
        assert!(!masked[1].node.visibility.diffuse && !masked[1].node.visibility.glossy);
        assert!(!masked[0].node.visibility.diffuse && masked[0].node.visibility.glossy);
        use vork_scene::LightShape;
        assert!(
            matches!(nodes[0].node.kind, NodeKind::Light { shape: LightShape::Sun, energy: Some(intensity), .. } if (intensity-1.).abs()<1e-5)
        );
        assert!(
            matches!(nodes[1].node.kind, NodeKind::Light { shape: LightShape::Point, energy: Some(intensity), .. } if (intensity-4.*std::f32::consts::PI).abs()<1e-5)
        );
        assert!(
            matches!(nodes[2].node.kind, NodeKind::Light { shape: LightShape::Spot, spot_size: size, spot_blend: blend, .. } if (size - 1.2).abs() < 1e-6 && (vork_scene::spot_inner_angle(size, blend) - 0.2).abs() < 1e-6)
        );
    }
}

/// Shared install-time packing for object ray visibility (light contributions are separate).
/// Per-ray refinements keep volume scattering; `visible` disables every ray.
pub(crate) fn object_visibility_mask(v: &::vork_scene::v2::Visibility) -> u32 {
    if !v.visible { return 0; }
    constants::RAY_VOLUME | [
        (v.camera, constants::RAY_CAMERA),
        (v.shadow, constants::RAY_SHADOW),
        (v.diffuse, constants::RAY_DIFFUSE),
        (v.glossy, constants::RAY_SPECULAR),
        (v.transmission, constants::RAY_TRANSMISSION),
    ]
    .into_iter()
    .fold(0, |bits, (visible, flag)| {
        bits | if visible { flag } else { 0 }
    })
}

/// Admit object transmission masks where transport consumes them: path transport
/// or raster ray-refraction objects. Light transmission masks remain reserved.
pub(crate) fn validate_visibility(scene: &::vork_scene::v2::Scene) -> Result<()> {
    use ::vork_scene::v2::{NodeKind, Visibility};
    if scene.delivery {
        fn ignored(nodes:&[::vork_scene::v2::Node], inherited:bool) {
            for node in nodes {
                let transmission=inherited && node.visibility.transmission;
                if matches!(node.kind,NodeKind::Light {..}) && (!transmission || !node.visibility.shadow) {
                    crate::warnings::dropped("light_visibility", serde_json::json!({"node":node,"inherited_transmission":inherited}), format!("warning: light {:?}: ignored visible_shadow/visible_transmission contribution masks; use shadowed_by or supported light visibility fields",node.name));
                }
                ignored(&node.children,transmission);
            }
        }
        ignored(&scene.nodes,true);
        for target in scene.track_targets()?.into_iter().flatten() {
            if let ::vork_scene::TrackTarget::Node {node,property: ::vork_scene::NodeProperty::Visibility(::vork_scene::Ray::Shadow|::vork_scene::Ray::Transmission)}=target {
                fn light(nodes:&[::vork_scene::v2::Node],name:&str)->bool {nodes.iter().any(|n|n.name==name && matches!(n.kind,NodeKind::Light {..}) || light(&n.children,name))}
                if light(&scene.nodes,&node) {crate::warnings::dropped("light_visibility_track", serde_json::json!({"node":node,"tracks":scene.animation}), format!("warning: light {node:?}: ignored shadow/transmission visibility track"));}
            }
        }
        return Ok(());
    }
    fn reserved(name: &str, value: &Visibility, transmission: bool) -> Result<()> {
        ensure!(
            value.transmission || transmission,
            "node {name}: visible_transmission is not yet supported by this renderer"
        );
        Ok(())
    }
    fn visit(nodes: &[::vork_scene::v2::Node], parent: &Visibility, transmission: bool) -> Result<()> {
        for node in nodes {
            let mut effective = node.visibility.clone();
            crate::buffers::inherit_visibility(&mut effective, parent);
            if matches!(node.kind, NodeKind::Light { .. }) {
                ensure!(
                    effective.transmission,
                    "light transmission visibility is not yet supported by this renderer"
                );
            } else {
                reserved(&node.name, &effective, transmission)?;
            }
            if matches!(node.kind, NodeKind::Light { .. }) {
                ensure!(
                    node.visibility.shadow,
                    "light {}: visible_shadow is not a light contribution field",
                    node.name
                );
            }
            if let NodeKind::Asset {
                nodes: overrides, ..
            } = &node.kind
            {
                for (name, node) in overrides {
                    if let Some(visibility) = &node.visibility {
                        reserved(name, visibility, transmission)?;
                    }
                }
            }
            visit(&node.children, &effective, transmission)?;
        }
        Ok(())
    }
    let path = matches!(scene.render.settings.integrator, vork_scene::Integrator::Path(_));
    let transmission = path || scene.render.settings.integrator.raster().is_some_and(|r|matches!(r.refraction, vork_scene::Refraction::Ray | vork_scene::Refraction::Auto));
    visit(&scene.nodes, &Visibility::default(), transmission)?;
    use ::vork_scene::{NodeProperty, Ray, TrackTarget};
    for target in scene.track_targets()?.into_iter().flatten() {
        let TrackTarget::Node {
            node: name,
            property: NodeProperty::Visibility(ray),
        } = target
        else {
            continue;
        };
        let path = format!("[\"nodes\",{name:?},\"visibility\",{ray:?}]").to_lowercase();
        ensure!(
            matches!(
                ray,
                Ray::All | Ray::Camera | Ray::Shadow | Ray::Diffuse | Ray::Glossy
            ) || (transmission && matches!(ray, Ray::Transmission)),
            "{path} is not yet supported by this renderer"
        );
        fn find<'a>(
            nodes: &'a [::vork_scene::v2::Node],
            name: &str,
        ) -> Option<&'a ::vork_scene::v2::Node> {
            nodes.iter().find_map(|node| {
                if node.name == name {
                    Some(node)
                } else {
                    find(&node.children, name)
                }
            })
        }
        if let Some(node) = find(&scene.nodes, &name) {
            ensure!(
                ray != Ray::Shadow || !matches!(node.kind, NodeKind::Light { .. }),
                "light visibility shadow is not a light contribution field: {path}"
            );
            ensure!(
                ray != Ray::Transmission || !matches!(node.kind, NodeKind::Light { .. }),
                "light transmission visibility is not yet supported by this renderer"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
mod visibility_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn visibility_payload_survives_half_float_storage() {
        // CPU mirror of visibility.wgsl: exhaustive packed RGB5/roughness payloads
        // check the actual f16 attachment precision, including -2048 at the edge.
        let constant = |name: &str| -> f32 {
            include_str!(concat!(env!("OUT_DIR"), "/constants.wgsl"))
                .lines()
                .find(|line| line.starts_with(&format!("const {name}:")))
                .unwrap()
                .split('=')
                .nth(1)
                .unwrap()
                .trim()
                .trim_end_matches(';')
                .parse()
                .unwrap()
        };
        let base = constant("REG_REFLECTION_PACK_BASE");
        let specular_offset = constant("REG_WORLD_SPECULAR_OFFSET");
        let offset = base * base;
        for payload in 0..offset as u32 {
            let encoded = -(payload as f32 + specular_offset);
            let stored = half::f16::from_f32(encoded).to_f32();
            assert!(stored < -1.);
            assert_eq!(-stored - specular_offset, payload as f32);
            for diffuse in [false, true] {
                for specular in [false, true] {
                    let value = payload as f32 + if diffuse { 0. } else { offset };
                    let encoded = if specular { value } else { -value - 1. };
                    let stored = half::f16::from_f32(encoded).to_f32();
                    assert_eq!(stored, encoded);
                    assert_eq!(stored >= 0., specular);
                    let decoded = if stored < 0. { -stored - 1. } else { stored };
                    assert_eq!(decoded < offset, diffuse);
                    assert_eq!(decoded % offset, payload as f32);
                }
            }
        }
    }

    #[test]
    fn reserved_ray_masks_fail_after_native_producer_activation() -> Result<()> {
        for field in ["transmission"] {
            let mut node = json!({"type":"node","name":"cube"});
            node[format!("visible_{field}")] = false.into();
            let scene = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[node],"render":{"settings":{"integrator":"raster","refraction":"screen"}}}))?;
            assert!(
                validate_visibility(&scene)
                    .unwrap_err()
                    .to_string()
                    .contains(field)
            );
        }
        Ok(())
    }

    #[test]
    fn path_accepts_object_transmission_visibility() -> Result<()> {
        let mut document = json!({"version":2,
            "render":{"settings":{"integrator":"path","denoiser":null}},
            "nodes":[{"type":"node","name":"rig","visible_transmission":false}],
            "animation":[{"tracks":[{"path":["nodes","rig","visible_transmission"],
                "keys":[[0,false],[1,true]],"interpolation":"STEP"}]}]});
        let scene = vork_scene::v2::from_test_value(document.clone())?;
        validate_visibility(&scene)?;
        document["nodes"][0] = json!({"type":"light","name":"rig","shape":"environment"});
        let scene = vork_scene::v2::from_test_value(document)?;
        assert!(validate_visibility(&scene).unwrap_err().to_string().contains("light transmission visibility"));
        Ok(())
    }

    #[test]
    fn environment_accepts_inherited_contribution_mask() -> Result<()> {
        let scene = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"node","visible_glossy":false,"children":[{"type":"light","shape":"environment"}]}]}),
        )?;
        validate_visibility(&scene)?;
        Ok(())
    }

    #[test]
    fn light_mask_inherits_and_fully_disabled_static_light_installs_no_record() -> Result<()> {
        let parent: Node =
            vork_scene::v2::from_test_value(json!({"type":"node","visible_glossy":false}))?;
        let lamp: Node =
            vork_scene::v2::from_test_value(json!({"type":"light","shape":"sun","shadowed_by":[]}))?;
        let mut nodes = vec![
            EvalNode { camera: None,
                node: parent,
                parent: None,
                world: Mat4::IDENTITY,
                dynamic: false,
                embedded: None,
            },
            EvalNode { camera: None,
                node: lamp,
                parent: Some(0),
                world: Mat4::IDENTITY,
                dynamic: false,
                embedded: None,
            },
        ];
        let plan = LightPlan::logical(&nodes, &wgpu::Limits::default(), false)?;
        let lights = plan.evaluate(
            &nodes,
            &[Mat4::IDENTITY; 2],
            &links::Links::new(&nodes).unwrap())?;
        assert_eq!(
            lights[0].shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD]
                & constants::RAY_VISIBILITY_ALL,
            constants::RAY_VISIBILITY_ALL & !constants::RAY_SPECULAR
        );
        nodes[0].node.visibility.diffuse = false;
        let plan = LightPlan::logical(&nodes, &wgpu::Limits::default(), false)?;
        assert!(plan.nodes.is_empty());
        Ok(())
    }

    #[test]
    fn light_camera_inheritance_and_own_shadow_semantics() -> Result<()> {
        let mut scene: ::vork_scene::v2::Scene = vork_scene::v2::from_test_value(json!({
            "version":2,"nodes":[{"name":"rig","type":"node",
            "visible_camera":false,"visible_shadow":false,
            "children":[{"name":"lamp","type":"light","shape":"point"}]}]}))?;
        validate_visibility(&scene)?;
        scene.nodes[0].children[0].visibility.shadow = false;
        assert!(
            validate_visibility(&scene)
                .unwrap_err()
                .to_string()
                .contains("not a light contribution field")
        );
        Ok(())
    }

    #[test]
    fn animated_shadow_visibility_reaches_the_native_uploader() -> Result<()> {
        let scene = vork_scene::v2::from_test_value(
            json!({"version":2,
            "nodes":[{"type":"node","name":"cube"}],
            "animation":[{"tracks":[{"path":["nodes","cube","visible_shadow"],"keys":[[0,true],[1,false]],"interpolation":"STEP"}]}]}),
        )?;
        validate_visibility(&scene)?;
        Ok(())
    }
}

#[cfg(test)]
mod imported_camera_tests {
    use super::*;
    #[test]
    fn gltf_cameras_keep_vertical_framing_clips_and_node_transforms() -> Result<()> {
        let document = gltf::Gltf::from_slice(include_bytes!("../tests/fixtures/imported-camera.gltf"))?.document;
        let mut nodes = Vec::new();
        let mut materials=Default::default();
        import(&document, 0, 0, Mat4::from_translation(Vec3::X), "rig", &Default::default(), &mut nodes, &mut materials)?;
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].node.name, "rig/shot");
        assert_eq!(nodes[1].node.name, "rig/#2");
        assert_eq!(nodes[0].world.transform_point3(Vec3::ZERO), Vec3::new(3., 3., 9.));
        let inputs=serde_json::to_value(&materials["rig/shot/camera"].inputs)?;
        assert_eq!(inputs["fit"],"vertical");
        assert!((inputs["lens"].as_f64().unwrap() as f32-24./(2.*0.5f32.tan())).abs()<1e-5);
        assert_eq!(inputs["clip_start"],0.2f32);assert_eq!(inputs["clip_end"],50.);
        let inputs=serde_json::to_value(&materials["rig/#2/camera"].inputs)?;
        assert_eq!(inputs["projection"],"ortho");assert_eq!(inputs["ortho_scale"],4.);
        Ok(())
    }
}
    #[test]
    fn delivery_light_masks_warn_for_static_and_tracked_requests() -> Result<()> {
        let mut scene = vork_scene::v2::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"},
            {"type":"light","name":"lamp","shape":"point","visible_shadow":false}],
            "animation":[{"tracks":[{"path":["nodes","lamp","visible_shadow"],"keys":[[0,true],[1,false]]}]}]}"#)?;
        scene.delivery = true;
        let collector = crate::warnings::Collector::default();
        let _scope = collector.enter();
        validate_visibility(&scene)?;
        let warnings = collector.snapshot();
        for code in ["light_visibility", "light_visibility_track"] {
            assert!(warnings.iter().any(|warning|warning.code == code), "{warnings:?}");
        }
        Ok(())
    }
