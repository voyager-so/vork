//! Install-time polyline tubes; the resulting mesh follows the ordinary retained path.
use super::*;

// Persist generated positions, normals, UVs and longitudinal strand tangents.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(non_snake_case)]
struct CachedVertex {
    POSITION: [f32; 3],
    NORMAL: [f32; 3],
    TEXCOORD_0: [f32; 2],
    fiber: [f32; 2],
    TANGENT: [f32; 4],
}

pub(crate) fn cached_tubes(
    positions: &[[f32; 3]],
    widths: &[f32],
    counts: &[u32],
    sides: usize,
    cache: Option<&crate::disk_cache::Store>,
) -> Result<crate::geometry::Geometry> {
    let mut hash = blake3::Hasher::new();
    hash.update(bytemuck::cast_slice(positions));
    hash.update(bytemuck::cast_slice(widths));
    hash.update(bytemuck::cast_slice(counts));
    hash.update(&(positions.len() as u64).to_le_bytes());
    hash.update(&(widths.len() as u64).to_le_bytes());
    hash.update(&(sides as u32).to_le_bytes());
    let path = cache.map(|c| c.path("curve-tubes", 8, hash.finalize()));
    if let Some(payload) = path.as_ref().and_then(|p| crate::disk_cache::read(p)) {
        if let Some(mesh) = decode(&payload) {
            return Ok(mesh);
        }
    }
    let mesh = tubes(positions, widths, counts, sides)?;
    if let Some(path) = path {
        let header = [mesh.vertices.len() as u64, mesh.indices.len() as u64];
        let vertices: Vec<_> = mesh
            .vertices
            .iter()
            .map(|v| CachedVertex {
                POSITION: v.POSITION,
                NORMAL: v.NORMAL,
                TEXCOORD_0: v.TEXCOORD_0,
                fiber: v.TEXCOORD_1,
                TANGENT: v.TANGENT,
            })
            .collect();
        let _ = crate::disk_cache::write_parts(
            &path,
            &[
                bytemuck::cast_slice(&header),
                bytemuck::cast_slice(&vertices),
                bytemuck::cast_slice(&mesh.indices),
            ],
        );
    }
    Ok(mesh)
}

fn decode(payload: &[u8]) -> Option<crate::geometry::Geometry> {
    let header = payload.get(..16)?;
    let vertices = usize::try_from(u64::from_ne_bytes(header[..8].try_into().ok()?)).ok()?;
    let indices = usize::try_from(u64::from_ne_bytes(header[8..].try_into().ok()?)).ok()?;
    let vertex_end =
        16usize.checked_add(vertices.checked_mul(std::mem::size_of::<CachedVertex>())?)?;
    let end = vertex_end.checked_add(indices.checked_mul(4)?)?;
    if end != payload.len() || indices % 3 != 0 {
        return None;
    }
    let vertices: Vec<Vertex> = payload[16..vertex_end]
        .chunks_exact(std::mem::size_of::<CachedVertex>())
        .map(|bytes| {
            let v: CachedVertex = bytemuck::pod_read_unaligned(bytes);
            Vertex {
                POSITION: v.POSITION,
                NORMAL: v.NORMAL,
                TEXCOORD_0: v.TEXCOORD_0,
                TEXCOORD_1: v.fiber,
                pad: [0.; 2],
                COLOR_0: [1.; 4],
                TANGENT: v.TANGENT,
            }
        })
        .collect();
    let indices: Vec<u32> = payload[vertex_end..]
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    if indices.iter().any(|&i| i as usize >= vertices.len()) {
        return None;
    }
    Some(crate::geometry::Geometry::new(vertices, indices, 3))
}

pub(crate) fn tubes(
    positions: &[[f32; 3]],
    widths: &[f32],
    counts: &[u32],
    sides: usize,
) -> Result<crate::geometry::Geometry> {
    ensure!(
        sides >= 3,
        "curve tubes require at least three radial segments"
    );
    let ring = sides
        .checked_add(1)
        .context("curve radial count overflow")?; // Duplicate the UV seam without adding a face.
    ensure!(!counts.is_empty(), "curves require at least one curve");
    let total = counts.iter().try_fold(0usize, |total, &count| {
        ensure!(count >= 2, "each curve requires at least two points");
        total
            .checked_add(count as usize)
            .context("curve point count overflow")
    })?;
    ensure!(
        total == positions.len() && total == widths.len(),
        "curve point counts and width count must match positions"
    );
    ensure!(
        positions.iter().flatten().all(|v| v.is_finite())
            && widths.iter().all(|r| r.is_finite() && *r >= 0.),
        "curve positions must be finite and widths finite and nonnegative"
    );
    let vertex_count = total
        .checked_mul(ring)
        .context("curve vertex count overflow")?;
    ensure!(
        vertex_count <= u32::MAX as usize,
        "curves exceed u32 vertex indices"
    );
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut indices = Vec::new();
    let mut offset = 0;
    for (strand, &count) in counts.iter().enumerate() {
        let points = &positions[offset..offset + count as usize];
        let fiber_triangles = (count as usize - 1) * sides * 2 + 2 * (sides - 2);
        ensure!(
            strand <= (1 << 24) && fiber_triangles <= (1 << 24),
            "curve shadow metadata exceeds exact f32 indices"
        );
        let mut previous_side = None;
        for (i, &point) in points.iter().enumerate() {
            let p = Vec3::from(point);
            let incoming = (i > 0).then(|| p - Vec3::from(points[i - 1]));
            let outgoing = (i + 1 < points.len()).then(|| Vec3::from(points[i + 1]) - p);
            ensure!(
                incoming
                    .iter()
                    .chain(outgoing.iter())
                    .all(|v| v.length_squared() > 0.),
                "consecutive curve points must be distinct"
            );
            let tangent = match (incoming, outgoing) {
                (Some(a), Some(b)) => (a.normalize() + b.normalize())
                    .try_normalize()
                    .unwrap_or(b.normalize()),
                (Some(a), None) => a.normalize(),
                (None, Some(b)) => b.normalize(),
                _ => unreachable!(),
            };
            // Project the preceding ring's axis for a rotation-minimizing frame.
            let side = previous_side
                .and_then(|s: Vec3| (s - tangent * s.dot(tangent)).try_normalize())
                .unwrap_or_else(|| tangent.any_orthonormal_vector());
            previous_side = Some(side);
            let binormal = tangent.cross(side);
            for j in 0..=sides {
                let angle = std::f32::consts::TAU * (j % sides) as f32 / sides as f32;
                let normal = side * angle.cos() + binormal * angle.sin();
                vertices.push(Vertex {
                    POSITION: (p + normal * (widths[offset + i] * 0.5)).to_array(),
                    NORMAL: normal.to_array(),
                    TEXCOORD_0: [j as f32 / sides as f32, i as f32 / (points.len() - 1) as f32],
                    TEXCOORD_1: [strand as f32, fiber_triangles as f32],
                    pad: [0.; 2],
                    COLOR_0: [1.; 4],
                    // Internal |w| >= 2 tags a longitudinal curve tangent.
                    // Normal maps reconstruct their UV frame.
                    TANGENT: [tangent.x, tangent.y, tangent.z, 2.],
                });
                if i > 0 && j < sides {
                    let a = ((offset + i - 1) * ring + j) as u32;
                    let b = a + 1;
                    let c = a + ring as u32;
                    let d = b + ring as u32;
                    indices.extend_from_slice(&[a, b, c, b, d, c]);
                }
            }
        }
        offset += count as usize;
    }
    Ok(crate::geometry::Geometry::new(vertices, indices, 3))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tubes_keep_curves_separate_and_wind_outward() -> Result<()> {
        let mesh = tubes(
            &[[0., 0., 0.], [0., 1., 0.], [3., 0., 0.], [3., 1., 0.]],
            &[0.1; 4],
            &[2, 2],
            8,
        )?;
        let sides = 8;
        let ring = sides + 1;
        assert_eq!(mesh.vertices.len(), 4 * ring);
        assert_eq!(mesh.indices.len(), 12 * sides);
        for tri in mesh.indices.chunks_exact(3) {
            assert!(
                tri.iter()
                    .all(|v| (*v as usize / (2 * ring)) == tri[0] as usize / (2 * ring))
            );
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| &mesh.vertices[i as usize]);
            let cross =
                (Vec3::from(b.POSITION) - Vec3::from(a.POSITION)).cross(Vec3::from(c.POSITION) - Vec3::from(a.POSITION));
            assert!(cross.dot(Vec3::from(a.NORMAL)) > 0.);
        }
        Ok(())
    }
    #[test]
    fn loader_uses_diameter_and_expands_point_attributes() -> Result<()> {
        let scene: Scene = vork_scene::v2::from_test_value(
            serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"buffers":{"p":{"type":"vec3","data":[[0,0,0],[0,1,0]]},"w":{"type":"f32","data":[0.4,0]},"c":{"type":"u32","data":[2]},"COLOR_0":{"type":"vec4","data":[[1,0,0,1],[0,1,0,1]]},"rough":{"type":"f32","data":[0.2,0.8]}},"geometry":{"hair":{"topology":"curves","curve_vertex_counts":"c","widths":"w","attributes":{"POSITION":"p","COLOR_0":"COLOR_0","rough":"rough"}}}}),
        )?;
        scene.validate()?;
        let data = crate::buffers::SceneBuffers::decode(&scene, Path::new("."))?;
        let mesh = crate::buffers::geometry_v2(
            &data,
            &scene.geometry["hair"],
            &["rough".into()],
            8,
            None,
        )?;
        let sides = 8 + 1;
        assert_eq!(mesh.vertices[0].POSITION, mesh.vertices[sides - 1].POSITION);
        assert_eq!(mesh.vertices[sides - 1].TEXCOORD_0[0], 1.);
        for v in &mesh.vertices[..sides] {
            assert!((Vec3::from(v.POSITION).length() - 0.2).abs() < 1e-6);
            assert_eq!(v.COLOR_0, [1., 0., 0., 1.]);
        }
        for v in &mesh.vertices[sides..] {
            assert_eq!(v.POSITION, [0., 1., 0.]);
        }
        assert_eq!(mesh.attributes["rough"][sides], [0.8, 0.8, 0.8, 1.]);
        let mut bad = scene.clone();
        bad.geometry.get_mut("hair").unwrap().widths = None;
        assert!(bad.validate().is_err());
        let mut bad = scene.clone();
        bad.geometry.get_mut("hair").unwrap().topology = vork_scene::v2::Topology::Triangles;
        assert!(bad.validate().is_err());
        Ok(())
    }

    #[test]
    fn cache_payload_reconstructs_mesh_and_rejects_truncation() -> Result<()> {
        let mesh = tubes(&[[0., 0., 0.], [0., 1., 0.]], &[0.2, 0.1], &[2], 8)?;
        let mut payload =
            bytemuck::cast_slice(&[mesh.vertices.len() as u64, mesh.indices.len() as u64]).to_vec();
        let packed: Vec<_> = mesh
            .vertices
            .iter()
            .map(|v| CachedVertex {
                POSITION: v.POSITION,
                NORMAL: v.NORMAL,
                TEXCOORD_0: v.TEXCOORD_0,
                fiber: v.TEXCOORD_1,
                TANGENT: v.TANGENT,
            })
            .collect();
        payload.extend_from_slice(bytemuck::cast_slice(&packed));
        payload.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
        let restored = decode(&payload).unwrap();
        assert_eq!(restored.indices, mesh.indices);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&restored.vertices),
            bytemuck::cast_slice::<_, u8>(&mesh.vertices)
        );
        assert!(decode(&payload[..payload.len() - 1]).is_none());
        Ok(())
    }

    #[test]
    fn rejects_invalid_curves() {
        assert!(tubes(&[[0.; 3]; 2], &[0.1; 2], &[2], 8).is_err());
        assert!(tubes(&[[0.; 3]], &[0.1], &[1], 8).is_err());
        assert!(tubes(&[[0.; 3]; 2], &[-1.; 2], &[2], 8).is_err());
        assert!(tubes(&[[0.; 3]; 2], &[0.1; 2], &[3], 8).is_err());
    }
}

// A circle's chord sagitta is r*(1-cos(pi/n)). Keep it below 1/8 pixel.
fn radial_sides(projected_radius: f32) -> usize {
    let angle = 2. * (0.125 / (2. * f64::from(projected_radius.max(0.125)))).sqrt().asin();
    ((std::f64::consts::PI / angle).ceil() as usize).max(8)
}

/// One retained mesh per geometry, sized for every camera, image resolution and
/// instance. Swept origin spheres bound animated distance independently of
/// orientation; animation envelopes include cubic overshoot and scale.
pub(crate) fn projected_sides(
    scene: &Scene,
    nodes: &[crate::install::EvalNode],
    buffers: &crate::buffers::SceneBuffers,
    animation: &crate::animation::Plan,
    views: &std::collections::BTreeMap<String,Vec<[u32;2]>>,
) -> Result<std::collections::HashMap<String, usize>> {
    use vork_scene::{NodeProperty as P, TrackTarget};
    let mut result = std::collections::HashMap::new();
    if !scene
        .geometry
        .values()
        .any(|g| matches!(g.topology, vork_scene::Topology::Curves | vork_scene::Topology::Points))
    {
        return Ok(result);
    }
    let moving: std::collections::HashSet<_> = animation
        .targets()
        .filter_map(|t| match t {
            TrackTarget::Node {
                node,
                property: P::Translation | P::Rotation | P::Scale,
            } => Some(node.as_str()),
            _ => None,
        })
        .collect();
    let changing_linear: std::collections::HashSet<_> = animation.targets().filter_map(|t| match t {
        TrackTarget::Node {node, property: P::Rotation | P::Scale} => Some(node.as_str()),
        _ => None,
    }).collect();
    // Unlike the broad swept-sphere flags, a fixed aim is a static pose.
    // Propagate animation through both evaluator dependency edges to a fixed point.
    let mut pose_dynamic: Vec<_> = nodes.iter().map(|n| moving.contains(n.node.name.as_str())).collect();
    loop {
        let mut changed = false;
        for (i, node) in nodes.iter().enumerate() {
            let target = match &node.node.target {
                Some(vork_scene::AimTarget::Node(name)) => nodes.iter().position(|n| n.node.name == *name),
                _ => None,
            };
            if !pose_dynamic[i] && (node.parent.is_some_and(|p| pose_dynamic[p])
                || target.is_some_and(|t| pose_dynamic[t])) {
                pose_dynamic[i] = true;
                changed = true;
            }
        }
        if !changed { break; }
    }
    // Planning precedes import/pose initialization. Resolve static authored
    // parent/aim dependencies through the evaluator, not the un-aimed TRS worlds.
    let mut planned_nodes:Vec<_>=nodes.iter().map(|n|crate::install::EvalNode {
        node:n.node.clone(),camera:n.camera.clone(),parent:n.parent,world:n.world,
        dynamic:n.dynamic,embedded:n.embedded,
    }).collect();
    let has_object_focus=nodes.iter().any(|n|views.contains_key(&n.node.name)
        && matches!(n.node.kind,NodeKind::Camera {focus_object:Some(_),..}));
    let targets=match crate::targets::Plan::compile(&mut planned_nodes) {
        Ok(plan)=>Some(plan),
        // Imported target names do not exist until geometry installation. Numeric
        // focus retains the existing orientation-independent swept-sphere path.
        Err(_) if !has_object_focus=>None,
        Err(error)=>return Err(error).context("curve/point camera bounds: cannot resolve parent/aim/focus dependencies; use numeric material.inputs.focus_distance or an authored focus/aim node"),
    };
    if let Some(targets)=targets {
        let mut worlds:Vec<_>=nodes.iter().map(|n|n.world).collect();
        for &i in &targets.order {
            if !pose_dynamic[i] {
                let world=nodes[i].parent.map_or(Mat4::IDENTITY,|p|worlds[p])*nodes[i].node.transform.matrix();
                worlds[i]=targets.aim(i,world,&worlds,&nodes[i].node,|_|false)?;
            }
        }
        for (i,node) in planned_nodes.iter_mut().enumerate() {
            if !pose_dynamic[i] {node.world=worlds[i];}
        }
        for &(camera,target) in &targets.focus {
            if !pose_dynamic[camera] && !pose_dynamic[target] {
                if let Some(state)=&mut planned_nodes[camera].camera {
                    state.focus=crate::targets::focus_distance(worlds[camera],worlds[target]);
                }
            }
        }
    }
    let nodes=&planned_nodes;
    let mut dynamic = vec![false; nodes.len()];
    let mut linear_dynamic = vec![false; nodes.len()];
    let mut max_scale = vec![1.; nodes.len()];
    let mut min_scale = vec![1.; nodes.len()];
    // Every node origin over the full animation. A rotation changes orientation,
    // not distance from its parent origin. Translation envelopes include layer
    // blending and spline overshoot; parent scale bounds every rotated offset.
    let mut origin_centres = vec![Vec3::ZERO; nodes.len()];
    let mut origin_radii = vec![0.; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        let scale = animation.node_envelope(&node.node.name, &P::Scale, &node.node.transform.scale);
        let hi = scale
            .iter()
            .map(|r| r[0].abs().max(r[1].abs()))
            .fold(0., f32::max);
        let lo = scale
            .iter()
            .map(|r| {
                if r[0] <= 0. && r[1] >= 0. {
                    0.
                } else {
                    r[0].abs().min(r[1].abs())
                }
            })
            .fold(f32::INFINITY, f32::min);
        max_scale[index] = hi * node.parent.map_or(1., |p| max_scale[p]);
        min_scale[index] = lo * node.parent.map_or(1., |p| min_scale[p]);
        let translation = animation.node_envelope(
            &node.node.name, &P::Translation, &node.node.transform.translation);
        let low = Vec3::new(translation[0][0], translation[1][0], translation[2][0]);
        let high = Vec3::new(translation[0][1], translation[1][1], translation[2][1]);
        let centre = (low + high) * 0.5;
        let extent = (high - low).length() * 0.5;
        linear_dynamic[index] = changing_linear.contains(node.node.name.as_str())
            || node.node.target.is_some() || node.parent.is_some_and(|p| linear_dynamic[p]);
        if let Some(parent) = node.parent {
            if linear_dynamic[parent] {
                origin_centres[index] = origin_centres[parent];
                origin_radii[index] = origin_radii[parent]
                    + max_scale[parent] * (centre.length() + extent);
            } else {
                origin_centres[index] = origin_centres[parent]
                    + nodes[parent].world.transform_vector3(centre);
                origin_radii[index] = origin_radii[parent] + max_scale[parent] * extent;
            }
        } else {
            origin_centres[index] = centre;
            origin_radii[index] = extent;
        }
        dynamic[index] =
            moving.contains(node.node.name.as_str())
                || node.node.target.is_some()
                || node.parent.is_some_and(|p| dynamic[p]);
    }
    for (mesh_index, mesh) in nodes.iter().enumerate() {
        let NodeKind::Mesh {
            geometry: name,
            instances,
            ..
        } = &mesh.node.kind
        else {
            continue;
        };
        let geometry = &scene.geometry[name];
        if !matches!(geometry.topology, vork_scene::Topology::Curves | vork_scene::Topology::Points) {
            continue;
        }
        let positions = buffers.data[&geometry.attributes["POSITION"]].floats::<3>()?;
        let width = buffers.data[geometry.widths.as_ref().context("curve/point widths")?]
            .floats::<1>()?
            .iter()
            .map(|v| v[0])
            .fold(0., f32::max);
        let computed = crate::compute::spatial_instances(scene, &mesh.node)
            || crate::computed_geometry::spatial(scene, geometry);
        let computed_width = geometry.widths.as_ref().is_some_and(|name|crate::compute::writes_buffer(scene,name))
            || (width>0. && crate::compute::writes_instance_attributes(scene,&mesh.node,&["SCALE"]));
        let instances = if let Some(instances) = instances {
            crate::buffers::instance_transforms(instances, buffers)?
        } else {
            vec![Mat4::IDENTITY]
        };
        let norm = |m: Mat4| {
            (m.x_axis.truncate().length_squared()
                + m.y_axis.truncate().length_squared()
                + m.z_axis.truncate().length_squared())
            .sqrt()
        };
        let mut radius_pixels = 0f32;
        for (camera_index, camera) in nodes.iter().enumerate() {
            if !matches!(camera.node.kind,NodeKind::Camera {..}) {continue;}
            let Some(resolutions)=views.get(&camera.node.name) else {continue;};
            for &resolution in resolutions {
                let [w,h]=resolution.map(|v|v as f32);
                let Some(c)=camera.camera.as_ref().map(|c|c.calibration(w/h)).transpose()
                    .with_context(|| format!("node {:?}, camera {:?}: cannot bound curve/point projection; check camera optics and focus depth, or use numeric material.inputs.focus_distance instead of object focus",
                        mesh.node.name, camera.node.name))?.flatten() else {
                    radius_pixels=radius_pixels.max(w.hypot(h));continue;
                };
                let optics_animated=matches!(&camera.node.kind,NodeKind::Camera {material:Some(name),..} if animation.material_inputs().iter().any(|(material,_)|material==name));
                // Unknown centres still have a finite projected bound when widths
                // are authored: the near plane below bounds their approach.
                // Unknown widths or instance scales need the screen-domain fallback.
                if computed_width || optics_animated {radius_pixels=radius_pixels.max(w.hypot(h));}
                let near=c[1][0];let pixels_per_unit=h*c[0][1]*0.5;let ortho=c[1][2]!=0.;
                // Focus uses world camera-axis metres, exactly as the evaluator,
                // rather than camera-space Z (which includes inherited scale).
                let focus = if let NodeKind::Camera {focus_object:Some(target),..} = &camera.node.kind {
                    let target_index = nodes.iter().position(|n| n.node.name == *target)
                        .with_context(|| format!("camera {:?}: unknown focus object {target:?}", camera.node.name))?;
                    if !pose_dynamic[camera_index] && !pose_dynamic[target_index] {
                        let depth = crate::targets::focus_distance(camera.world, nodes[target_index].world);
                        ensure!(c[2][0] == 0. || depth > crate::constants::MIN_FOCUS_DISTANCE,
                            "node {:?}, camera {:?}: focus object {:?} is on the camera plane; use numeric material.inputs.focus_distance or move the focus object away from the camera plane",
                            mesh.node.name, camera.node.name, target);
                        c[2][1]
                    } else {
                        let standard=match &camera.node.kind {
                            NodeKind::Camera {material:Some(name),..} => scene.materials[name].program == "std/camera",
                            _ => true,
                        };
                        ensure!(standard,
                            "node {:?}, camera {:?}: animated object focus {:?} has no certified custom-camera projection bound; use numeric material.inputs.focus_distance or std/camera",
                            mesh.node.name, camera.node.name, target);
                        let axis = camera.world.z_axis.truncate().normalize();
                        let lower = if !linear_dynamic[camera_index] || !pose_dynamic[camera_index] {
                            ((origin_centres[target_index] - origin_centres[camera_index]).dot(axis).abs()
                                - origin_radii[target_index] - origin_radii[camera_index]).max(0.)
                        } else if camera.node.transform.resolved_rotation() == [0., 0., 0., 1.]
                            && !animation.targets().any(|t| matches!(t,
                                TrackTarget::Node {node, property:P::Rotation} if node == &camera.node.name))
                            && camera.node.target.is_some()
                        {
                            // QR aiming retains parent stretch H. With no local
                            // rotation offset, |Hzz| / |H*z| >= sigma_min / sigma_max.
                            // Thus even animated affine parents retain this much
                            // axial separation. A distinct focus loses at most its
                            // distance to the aim origin.
                            let (aim_centre, aim_radius, focus_offset) = match camera.node.target.as_ref().unwrap() {
                                vork_scene::AimTarget::Node(name) => {
                                    let i=nodes.iter().position(|n| n.node.name == *name)
                                        .context("unknown camera aim target")?;
                                    let offset=if i==target_index {0.} else {
                                        origin_centres[i].distance(origin_centres[target_index])
                                            + origin_radii[i] + origin_radii[target_index]
                                    };
                                    (origin_centres[i],origin_radii[i],offset)
                                }
                                vork_scene::AimTarget::Point(point) => {
                                    let centre=Vec3::from_array(*point);
                                    (centre,0.,centre.distance(origin_centres[target_index])+origin_radii[target_index])
                                }
                            };
                            let alignment=camera.parent.map_or(1.,|p|
                                min_scale[p] / max_scale[p].max(f32::MIN_POSITIVE));
                            ((aim_centre.distance(origin_centres[camera_index])
                                - aim_radius - origin_radii[camera_index]).max(0.) * alignment
                                - focus_offset).max(0.)
                        } else { 0. };
                        ensure!(c[2][0] == 0. || lower > crate::constants::MIN_FOCUS_DISTANCE,
                            "node {:?}, camera {:?}: animated focus object {:?} has no positive camera-axis depth bound; use numeric material.inputs.focus_distance, constrain the camera/target motion, or disable DOF",
                            mesh.node.name, camera.node.name, target);
                        if c[2][0] > 0. {
                            // A loose but positive envelope must not resurrect
                            // the near-zero-focus allocation blowup. Certify the
                            // lower endpoint against the camera's own optics.
                            let mut bound=camera.camera.as_ref().unwrap().clone();
                            bound.focus=lower;
                            bound.calibration(w/h).with_context(|| format!(
                                "node {:?}, camera {:?}: cannot certify animated focus object {:?} within valid camera optics; use numeric material.inputs.focus_distance or constrain the camera/target motion",
                                mesh.node.name,camera.node.name,target))?;
                        }
                        lower.max(crate::constants::MIN_FOCUS_DISTANCE)
                    }
                } else {c[2][1]};
                let blur = if c[2][0]>0. {c[2][0]*(1./near+1./focus)} else {0.};
                let perspective_bound=if ortho {1.} else {
                    (1.+((1.+c[0][2].abs())/c[0][0]+blur).powi(2)
                        +((1.+c[0][3].abs())/c[0][1]+blur).powi(2)).sqrt()
                };
                for instance in &instances {
                    let camera_view = if crate::buffers::node_collapsed(nodes, camera_index) {
                        let worlds: Vec<_> = nodes.iter().map(|node| node.world).collect();
                        crate::values::node_view(camera_index, nodes, &worlds)?
                    } else {camera.world.inverse()};
                    let view_mesh = camera_view * mesh.world * *instance;
                    let animated = dynamic[mesh_index] || dynamic[camera_index] || computed;
                    let scale = if animated {
                        max_scale[mesh_index] * norm(*instance)
                            / min_scale[camera_index].max(f32::MIN_POSITIVE)
                    } else {
                        norm(view_mesh)
                    };
                    let radius = width * 0.5 * scale;
                    // Compute placements have no authored motion envelope.
                    let depth = if computed {near} else if animated {
                        // For any visible point, radial distance / the view-cone
                        // secant bounds camera-space depth. Swept spheres retain
                        // real approach motion without putting every rotation at
                        // the near plane. Only intersecting bounds fall back there.
                        let (centre, reach) = if linear_dynamic[mesh_index] {
                            let reach = positions.iter().map(|p|
                                instance.transform_point3(Vec3::from(*p)).length())
                                .fold(0., f32::max) * max_scale[mesh_index];
                            (origin_centres[mesh_index], reach)
                        } else {
                            // A static linear transform (including translation-only
                            // animation) can centre its bounds on the geometry,
                            // even when vertices are far from the object's origin.
                            let mut low=Vec3::splat(f32::INFINITY);
                            let mut high=Vec3::splat(f32::NEG_INFINITY);
                            for p in &positions {
                                let q=mesh.world.transform_vector3(instance.transform_point3(Vec3::from(*p)));
                                low=low.min(q);high=high.max(q);
                            }
                            (origin_centres[mesh_index]+(low+high)*0.5,(high-low).length()*0.5)
                        };
                        let world_radius = width * 0.5 * max_scale[mesh_index] * norm(*instance);
                        let separation = centre.distance(origin_centres[camera_index])
                            - origin_radii[mesh_index] - origin_radii[camera_index]
                            - reach - world_radius;
                        (separation / (max_scale[camera_index] * perspective_bound)).max(near)
                    } else {
                        positions
                            .iter()
                            .map(|p| -view_mesh.transform_point3(Vec3::from(*p)).z - radius)
                            .fold(f32::INFINITY, f32::min)
                            .max(near)
                    };
                    radius_pixels = radius_pixels.max(
                        radius * pixels_per_unit * perspective_bound
                            / if ortho { 1. } else { depth },
                    );
                }
            }
        }
        ensure!(
            radius_pixels.is_finite(),
            "curve projected width is unbounded; camera/scale animation crosses a singularity"
        );
        // Both angular directions contribute sphere chord error. Refining
        // each against twice the radius bounds their combined radial loss.
        if matches!(geometry.topology, vork_scene::Topology::Points) { radius_pixels *= 2.; }
        result
            .entry(name.clone())
            .and_modify(|s: &mut usize| *s = (*s).max(radial_sides(radius_pixels)))
            .or_insert_with(|| radial_sides(radius_pixels));
    }
    Ok(result)
}

#[cfg(test)]
mod projected_tests {
    use super::*;
    #[test]
    fn chord_error_is_subpixel_across_projected_widths() {
        for radius in [0., 0.25, 1., 10., 100., 1000., 10000.] {
            let sides = radial_sides(radius);
            let error = f64::from(radius) * (1. - (std::f64::consts::PI / sides as f64).cos());
            assert!(
                error <= 0.12500001,
                "radius {radius}, sides {sides}, error {error}"
            );
        }
        assert!(radial_sides(100.) > radial_sides(1.));
    }
    #[test]
    fn point_sphere_faces_obey_the_projected_chord_error() -> Result<()> {
        for radius in [0.1, 1., 32., 512., 8192.] {
            let sides = radial_sides(radius * 2.);
            let sphere = crate::points::spheres(&[[0.; 3]], &[2.], sides)?;
            for face in sphere.indices.chunks_exact(3) {
                let [a,b,c] = [face[0],face[1],face[2]].map(|i|
                    glam::DVec3::from_array(sphere.vertices[i as usize].POSITION.map(f64::from)));
                let normal = (b-a).cross(c-a).normalize();
                let error = f64::from(radius) * (1. - a.dot(normal).abs());
                assert!(error <= 0.126, "radius {radius}, sides {sides}, error {error}");
            }
        }
        // No fixed angular ceiling, including widths beyond f64 acos precision.
        assert!(radial_sides(1e20) > radial_sides(1e10));
        assert!(crate::points::triangles_per_point(usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn object_focus_depth_bounds_cover_curves_points_and_capture_views() -> Result<()> {
        let program = std::sync::Arc::new(vork_slang::Compiler::discover().compile_camera(
            &crate::programs::abi(), vork_slang::module_source("std/camera").unwrap(),
            Path::new("std/camera.slang"), "objectfocusbounds")?);
        let base = serde_json::json!({
            "version":2,"render":{"resolution":[640,480]},
            "nodes":[{"type":"camera","name":"cam","translation":[0,0,3]},
                {"type":"node","name":"focus","translation":[1,0,0]},
                {"type":"mesh","name":"groom","geometry":"g"}],
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[0,1,0]]},
                "w":{"type":"f32","data":[0.02,0.02]},"n":{"type":"u32","data":[2]}},
            "geometry":{"g":{"topology":"curves","attributes":{"POSITION":"p"},
                "widths":"w","curve_vertex_counts":"n"}}
        });
        let detail = |mut value:serde_json::Value, focus:f32, resolution:[u32;2]| -> Result<usize> {
            value["nodes"].as_array_mut().unwrap().push(serde_json::json!({
                "type":"camera","name":"main","translation":[0,0,1000]}));
            value["render"]["camera"]=serde_json::json!("main");
            value["images"]=serde_json::json!({"feed":{"render":{"camera":"cam","resolution":resolution}}});
            let scene:Scene = vork_scene::v2::from_test_value(value)?;
            let words = program.inputs.pack(&serde_json::from_value(serde_json::json!({
                "use_dof":true,"aperture_fstop":3.5,"focus_distance":focus}))?,
                |name,_| anyhow::bail!("unexpected image {name}"))?;
            fn flatten(defs:&[vork_scene::Node], parent:Option<usize>, nodes:&mut Vec<crate::install::EvalNode>) {
                for node in defs {
                    let i=nodes.len();
                    nodes.push(crate::install::EvalNode {
                        node:node.clone(),parent,world:parent.map_or(Mat4::IDENTITY,|p|nodes[p].world)*node.transform.matrix(),
                        dynamic:false,embedded:None,camera:None,
                    });
                    flatten(&node.children,Some(i),nodes);
                }
            }
            let mut nodes=Vec::new();
            flatten(&scene.nodes,None,&mut nodes);
            for node in &mut nodes {
                if matches!(node.node.kind,NodeKind::Camera {..}) {
                    node.camera=Some(crate::camera_program::State {
                        program:program.clone(),words:words.clone(),offset:0,focus:-1.,time:0.,evaluated:Default::default(),
                    });
                }
            }
            let buffers=crate::buffers::SceneBuffers::decode(&scene,Path::new("."))?;
            let plan=crate::animation::Plan::compile(&scene,Path::new("."))?;
            let views=crate::camera_program::views(&scene,&nodes,&plan,0.);
            assert_eq!(views["cam"],vec![resolution]);
            Ok(projected_sides(&scene,&nodes,&buffers,&plan,&views)?["g"])
        };
        for points in [false,true] {
            let mut numeric=base.clone();
            if points {
                numeric["geometry"]["g"]["topology"]=serde_json::json!("points");
                numeric["geometry"]["g"].as_object_mut().unwrap().remove("curve_vertex_counts");
            }
            let mut object=numeric.clone();
            object["nodes"][0]["focus_object"]=serde_json::json!("focus");
            for resolution in [[640,480],[2048,1024]] {
                assert_eq!(detail(object.clone(),10.,resolution)?,detail(numeric.clone(),3.,resolution)?);
            }
            let mut aimed_numeric=numeric.clone();
            let mut aimed_object=object.clone();
            aimed_numeric["nodes"][0]["target"]=serde_json::json!([0,1,0]);
            aimed_object["nodes"][0]["target"]=serde_json::json!([0,1,0]);
            assert_eq!(detail(aimed_object,10.,[640,480])?,
                detail(aimed_numeric,9./10f32.sqrt(),[640,480])?,"static aiming must be evaluated before focus calibration");
            // Focus origins include parent motion and inherited camera scale.
            let mut hierarchy=object.clone();
            let focus_node=hierarchy["nodes"][1].take();
            hierarchy["nodes"][1]=serde_json::json!({"type":"node","name":"rig","children":[focus_node]});
            hierarchy["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","rig","translation"],"keys":[[0,[0,0,0]],[1,[0,0,2]]]
            }]}]);
            assert_eq!(detail(hierarchy.clone(),10.,[640,480])?,detail(numeric.clone(),1.,[640,480])?);
            let mut scaled_numeric=numeric.clone();
            let mut scaled_object=object.clone();
            scaled_numeric["nodes"][0]["scale"]=serde_json::json!([2,2,2]);
            scaled_object["nodes"][0]["scale"]=serde_json::json!([2,2,2]);
            assert_eq!(detail(scaled_object,10.,[640,480])?,detail(scaled_numeric,3.,[640,480])?);
            let mut approach=object.clone();
            approach["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","cam","translation"],"keys":[[0,[0,0,3]],[1,[0,0,0.1]]]
            }]}]);
            assert!(detail(approach.clone(),10.,[640,480]).is_ok());
            approach["animation"][0]["tracks"][0]["keys"][1][1]=serde_json::json!([0,0,0.01]);
            assert!(detail(approach.clone(),10.,[640,480]).is_err(),"a positive near-zero bound must be rejected before expansion too");
            approach["animation"][0]["tracks"][0]["keys"][1][1]=serde_json::json!([0,0,0]);
            assert!(detail(approach,10.,[640,480]).is_err());
            // Translation-only focus motion has a finite axis-depth envelope.
            object["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","focus","translation"],"keys":[[0,[1,0,0]],[1,[1,0,2.9]]]
            }]}]);
            let near=detail(object.clone(),10.,[640,480])?;
            assert!(near>=detail(numeric.clone(),3.,[640,480])?);
            assert_eq!(near,detail(numeric.clone(),0.1,[640,480])?);
            object["animation"][0]["tracks"][0]["keys"][1][1]=serde_json::json!([1,0,3]);
            let error=detail(object.clone(),10.,[640,480]).unwrap_err().to_string();
            for text in ["groom","cam","focus","focus_distance"] { assert!(error.contains(text),"{error}"); }
            let mut cubic=object.clone();
            cubic["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","focus","translation"],"interpolation":"CUBICSPLINE",
                "keys":[[0,[1,0,0],[0,0,0],[0,0,16]], [1,[1,0,0],[0,0,-16],[0,0,0]]]
            }]}]);
            assert!(detail(cubic,10.,[640,480]).is_err(),"cubic overshoot crosses the camera plane between safe keys");
            // An animated aim may rotate while retaining a useful separation bound.
            object["nodes"][0]["target"]=serde_json::json!("focus");
            object["animation"][0]["tracks"][0]["keys"][1][1]=serde_json::json!([1,0,0.5]);
            assert!(detail(object.clone(),10.,[640,480]).is_ok());
            let mut aimed_hierarchy=object.clone();
            let cam=aimed_hierarchy["nodes"][0].take();
            aimed_hierarchy["nodes"][0]=serde_json::json!({
                "type":"node","name":"camera-rig","scale":[2,1,1],"children":[cam]});
            aimed_hierarchy["nodes"][1]["translation"]=serde_json::json!([0,0,-10]);
            aimed_hierarchy["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","camera-rig","rotation"],
                "keys":[[0,{"euler":[0,0,0]}],[1,{"euler":[0,0.2,0]}]]
            }]}]);
            assert!(detail(aimed_hierarchy.clone(),10.,[640,480]).is_ok(),"aimed camera retains a depth bound through affine parents");
            aimed_hierarchy["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","camera-rig","scale"],"keys":[[0,[2,1,1]],[1,[0,1,1]]]
            }]}]);
            assert!(detail(aimed_hierarchy,10.,[640,480]).is_err(),"singular parent stretch has no positive alignment bound");
            // Independent rotation cannot promise any positive focus-axis depth.
            object["nodes"][0].as_object_mut().unwrap().remove("target");
            object["animation"]=serde_json::json!([{"tracks":[{
                "path":["nodes","cam","rotation"],
                "keys":[[0,{"euler":[0,0,0]}],[1,{"euler":[0,1.57,0]}]]
            }]}]);
            assert!(detail(object,10.,[640,480]).is_err());
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn computed_cloud_detail_tracks_only_spatial_writers() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut scene:Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[640,480]},
            "buffers":{"p":{"type":"vec3","data":[[0,0,0]]},"w":{"type":"f32","data":[0.001]},"color":{"type":"vec3","data":[[1,1,1]]},"live":{"type":"u32","data":[1]}},
            "geometry":{"cloud":{"topology":"points","attributes":{"POSITION":"p","COLOR_0":"color"},"widths":"w"}},
            "nodes":[{"type":"camera","translation":[0,0,5]},{"type":"mesh","geometry":"cloud"}]}))?;
        for name in ["p","w","color"] {
            let buffer=scene.buffers.get_mut(name).unwrap();
            buffer.data=Some(vec![buffer.data.as_ref().unwrap()[0].clone();96]);
        }
        let detail=|scene:&Scene|->Result<usize> {
            let mut nodes:Vec<_>=scene.nodes.iter().map(|node|crate::install::EvalNode {node:node.clone(),camera:None,parent:None,world:Mat4::from_translation(Vec3::from(node.transform.translation)),dynamic:false,embedded:None}).collect();
            let buffers=crate::buffers::SceneBuffers::decode(scene,Path::new("."))?;
            let plan=crate::animation::Plan::compile(scene,Path::new("."))?;
            let views=crate::camera_program::views(scene,&nodes,&plan,0.);
            crate::camera_program::install(&r,scene,&mut nodes,Path::new("."),&views)?;
            let sides=projected_sides(scene,&nodes,&buffers,&plan,&views)?;
            crate::buffers::admit_points(scene,&buffers,&sides)?;
            Ok(sides["cloud"])
        };
        let authored=detail(&scene)?;
        for binding in ["color","live"] {
            scene.programs=vork_scene::v2::from_test_value(serde_json::json!({"write":{"schedule":"frame","dispatch":1,"bindings":{"value":binding},"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {}"}}))?;
            scene.geometry.get_mut("cloud").unwrap().count=Some("live".into());
            assert_eq!(detail(&scene)?,authored,"{binding} must retain authored bounds");
        }
        scene.programs=vork_scene::v2::from_test_value(serde_json::json!({"write":{"schedule":"frame","dispatch":1,"bindings":{"value":"p"},"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {}"}}))?;
        let moving=detail(&scene)?;
        assert!(moving>=authored && moving<radial_sides(2.*640f32.hypot(480.)),"moving fixed-width points use their near-plane radius");
        scene.buffers.get_mut("w").unwrap().data=Some(vec![serde_json::json!(0.);96]);
        assert_eq!(detail(&scene)?,8,"zero-width motion has no screen-sized sphere");
        scene.buffers.get_mut("w").unwrap().data=Some(vec![serde_json::json!(0.001);96]);
        scene.buffers.insert("scale".into(),vork_scene::v2::from_test_value(serde_json::json!({"type":"vec3","data":vec![[1.,1.,1.];96]}))?);
        scene.nodes[1]=vork_scene::v2::from_test_value(serde_json::json!({"type":"mesh","geometry":"cloud","instances":{"attributes":{"SCALE":"scale"},"count":"live"}}))?;
        scene.programs=vork_scene::v2::from_test_value(serde_json::json!({"write":{"schedule":"frame","dispatch":1,"bindings":{"value":"live"},"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {}"}}))?;
        assert_eq!(detail(&scene)?,authored,"instance membership retains static transforms");
        scene.programs=vork_scene::v2::from_test_value(serde_json::json!({"write":{"schedule":"frame","dispatch":96,"bindings":{"value":"scale"},"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {}"}}))?;
        assert!(detail(&scene)?>=radial_sides(2.*640f32.hypot(480.)),"computed instance scale invalidates world width");
        scene.buffers.get_mut("w").unwrap().data=Some(vec![serde_json::json!(0.);96]);
        assert_eq!(detail(&scene)?,8,"finite scale cannot open a zero-width point");
        Ok(())
    }

    #[test]
    fn detail_accounts_for_resolution_instances_and_animated_approach() -> Result<()> {
        let mut scene: Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,
            "render":{"resolution":[128,128]},
            "nodes":[{"type":"camera","name":"camera","translation":[0,0,5]},{"type":"mesh","name":"hair","geometry":"hair"}],
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[0,1,0]]},"w":{"type":"f32","data":[0.2,0.2]},"c":{"type":"u32","data":[2]},"instances":{"type":"vec3","data":[[4,4,4]]}},
            "geometry":{"hair":{"topology":"curves","curve_vertex_counts":"c","widths":"w","attributes":{"POSITION":"p"}}}
        }))?;
        let abi=crate::programs::abi();
        let program=std::sync::Arc::new(vork_slang::Compiler::discover().compile_camera(
            &abi, vork_slang::module_source("std/camera").unwrap(),
            Path::new("std/camera.slang"), "curvesizingregression")?);
        let words=program.inputs.pack(&Default::default(),|name,_|anyhow::bail!("unexpected image input {name}"))?;
        let detail = |scene: &Scene| -> Result<usize> {
            let nodes: Vec<_> = scene
                .nodes
                .iter()
                .map(|n| crate::install::EvalNode { camera: matches!(n.kind,NodeKind::Camera {..}).then(||crate::camera_program::State {
                    program:program.clone(), words:words.clone(), offset:0, focus:-1., time:0., evaluated:Default::default(),
                }),
                    node: n.clone(),
                    parent: None,
                    world: n.transform.matrix(),
                    dynamic: false,
                    embedded: None,
                })
                .collect();
            let buffers = crate::buffers::SceneBuffers::decode(scene, Path::new("."))?;
            let plan = crate::animation::Plan::compile(scene, Path::new("."))?;
            let views=crate::camera_program::views(scene,&nodes,&plan,0.);
            Ok(projected_sides(scene, &nodes, &buffers, &plan, &views)?["hair"])
        };
        let small = detail(&scene)?;
        scene.nodes.push(vork_scene::v2::from_test_value(serde_json::json!({"type":"node","name":"spin"}))?);
        for interpolation in ["STEP", "LINEAR", "CUBICSPLINE"] {
            scene.animation = vork_scene::v2::from_test_value(serde_json::json!([{"tracks":[{
                "path":["nodes","spin","rotation"],
                "keys":[[0,{"euler":[0,0,0]}],[1,{"euler":[0,1,0]}]],
                "interpolation":interpolation
            }]}]))?;
            assert_eq!(detail(&scene)?, small);
        }
        scene.animation.clear();
        scene.render.resolution = [2048, 2048];
        let large = detail(&scene)?;
        assert!(large > small);
        scene.nodes[1] = vork_scene::v2::from_test_value(
            serde_json::json!({"type":"mesh","name":"hair","geometry":"hair","instances":{"attributes":{"SCALE":"instances"}}}),
        )?;
        let scaled=detail(&scene)?;
        assert!(scaled > large);
        scene.animation = vork_scene::v2::from_test_value(
            serde_json::json!([{"tracks":[{"path":["nodes","camera","translation"],"keys":[[0,[0,0,5]],[1,[0,0,0.5]]]}]}]),
        )?;
        let approaching=detail(&scene)?;
        assert!(approaching > large && approaching > scaled);

        // A distant fine groom must not inherit near-plane detail from a fixed
        // aim or a rotation about its own origin. Real camera approach still
        // requires detail; this is a distance bound, not a tessellation limit.
        scene.animation.clear();
        scene.nodes[1] = vork_scene::v2::from_test_value(serde_json::json!({
            "type":"mesh","name":"hair","geometry":"hair"}))?;
        scene.buffers.get_mut("w").unwrap().data = Some(vec![serde_json::json!(0.001); 2]);
        let fine = detail(&scene)?;
        assert_eq!(fine, 8);
        scene.nodes[0].target = Some(vork_scene::AimTarget::Point([0.,0.,0.]));
        assert_eq!(detail(&scene)?, fine);
        for interpolation in ["STEP", "LINEAR", "CUBICSPLINE"] {
            scene.animation = vork_scene::v2::from_test_value(serde_json::json!([{"tracks":[{
                "path":["nodes","hair","rotation"],
                "keys":[[0,{"euler":[0,-0.15,0]}],[1,{"euler":[0,0.15,0]}]],
                "interpolation":interpolation
            }]}]))?;
            assert_eq!(detail(&scene)?, fine);
        }
        scene.animation = vork_scene::v2::from_test_value(serde_json::json!([{"tracks":[{
            "path":["nodes","camera","translation"],
            "keys":[[0,[0,0,5]],[1,[0,0,0.1]]]
        }]}]))?;
        assert!(detail(&scene)? > fine);
        scene.animation.clear();
        scene.buffers.get_mut("p").unwrap().data = Some(vec![
            serde_json::json!([100.,0.,0.]),serde_json::json!([100.,1.,0.])]);
        scene.nodes[0].transform.translation=[100.,0.,5.];
        scene.nodes[0].target=Some(vork_scene::AimTarget::Point([100.,0.,0.]));
        assert_eq!(detail(&scene)?,fine,"fixed aim must not depend on mesh origin placement");
        Ok(())
    }
}
