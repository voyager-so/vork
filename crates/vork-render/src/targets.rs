//! Install-time dependency order for node aiming and object focus.
use super::{EvalNode, Mat4, NodeKind, Result, Vec3, ensure};

#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub order: Vec<usize>,
    pub aims: Vec<Option<Aim>>,
    pub focus: Vec<(usize, usize)>,
    parents: Vec<Option<usize>>,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Aim { Node(usize), Point(Vec3) }
impl Aim {
    pub(crate) fn node(self) -> Option<usize> { match self { Self::Node(index)=>Some(index),Self::Point(_)=>None } }
}
impl Plan {
    pub fn compile(nodes: &mut [EvalNode]) -> Result<Self> {
        let names: std::collections::HashMap<_, _> = nodes.iter().enumerate()
            .filter(|(_, n)| !n.node.name.is_empty()).map(|(i, n)| (n.node.name.as_str(), i)).collect();
        let resolve = |name: &str, node: &EvalNode, field: &str| -> Result<usize> {
            names.get(name).copied().ok_or_else(|| anyhow::anyhow!(
                "node {:?}: {field}: {}", node.node.name,
                vork_scene::v2::diagnostics::unknown("node", name, names.keys().copied())))
        };
        let mut aims = vec![None; nodes.len()];
        let mut focus = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            if let NodeKind::Camera { focus_object, .. } = &node.node.kind {
                if let Some(name) = focus_object.as_ref() { focus.push((i, resolve(name, node, "focus_object")?)); }
            }
            if let Some(target) = &node.node.target { aims[i] = Some(match target {
                vork_scene::AimTarget::Node(name)=>Aim::Node(resolve(name,node,"target")?),
                vork_scene::AimTarget::Point(point)=>Aim::Point(Vec3::from_array(*point)),
            }); }
        }
        let mut order = Vec::new();
        if aims.iter().any(Option::is_some) {
            fn visit(i: usize, nodes: &[EvalNode], aims: &[Option<Aim>], state: &mut [u8], order: &mut Vec<usize>) -> Result<()> {
                ensure!(state[i] != 1, "node {:?}: parent/target dependency cycle", nodes[i].node.name);
                if state[i] == 2 { return Ok(()); }
                state[i] = 1;
                for dependency in [nodes[i].parent, aims[i].and_then(Aim::node)].into_iter().flatten() {
                    visit(dependency, nodes, aims, state, order)?;
                }
                state[i] = 2;
                order.push(i);
                Ok(())
            }
            let mut state = vec![0; nodes.len()];
            for i in 0..nodes.len() { visit(i, nodes, &aims, &mut state, &mut order)?; }
        }
        drop(names);
        // Aim changes propagate through both the target and transform-parent edges.
        for &i in &order {
            nodes[i].dynamic |= aims[i].is_some() || nodes[i].parent.is_some_and(|p| nodes[p].dynamic)
                || aims[i].and_then(Aim::node).is_some_and(|target| nodes[target].dynamic);
        }
        for &(camera, _) in &focus { nodes[camera].dynamic = true; }
        Ok(Self { order, aims, focus, parents: nodes.iter().map(|node| node.parent).collect() })
    }
    /// `absent` says which nodes do not exist at this sample: a node never aims at one (rule 1).
    pub fn aim(&self, i: usize, world: Mat4, worlds: &[Mat4], node: &vork_scene::Node, absent: impl Fn(usize) -> bool) -> Result<Mat4> {
        let Some(target) = self.aims[i] else { return Ok(world); };
        let eye = world.w_axis.truncate();
        let target = match target {
            Aim::Node(index) => {
                let name = match &node.target { Some(vork_scene::AimTarget::Node(name)) => name.as_str(), _ => "" };
                ensure!(!absent(index), "node {:?}: target {name:?} is absent (active); aim at a node that exists", node.name);
                worlds[index].w_axis.truncate()
            }
            Aim::Point(point) => point,
        };
        let direction = target - eye;
        ensure!(direction.length_squared() > super::constants::ORIENTATION_EPSILON,
            "node {:?}: target coincides with the node origin", node.name);
        let direction = super::values::view_axis(eye, direction.normalize());
        let up = if direction.dot(Vec3::Y).abs() > super::constants::ORIENTATION_UP_PARALLEL { Vec3::Z } else { Vec3::Y };
        // Remove the parent's orientation, retaining its affine stretch/shear.
        // Factor before the local rotation so a rotation track cannot change stretch.
        let parent = glam::Mat3::from_mat4(self.parents[i].map_or(Mat4::IDENTITY, |p| worlds[p]));
        let x = parent.x_axis.try_normalize()
            .or_else(|| parent.y_axis.cross(parent.z_axis).try_normalize()).unwrap_or(Vec3::X);
        let y = (parent.y_axis - x * x.dot(parent.y_axis)).try_normalize()
            .or_else(|| parent.z_axis.cross(x).try_normalize())
            .unwrap_or_else(|| {
                let axis = if x.dot(Vec3::Y).abs() < 0.9 { Vec3::Y } else { Vec3::Z };
                (axis - x * x.dot(axis)).normalize()
            });
        let frame = glam::Mat3::from_cols(x, y, x.cross(y));
        let stretch = frame.transpose() * parent;
        Ok(Mat4::look_to_rh(eye, direction, up).inverse()
            * Mat4::from_quat(glam::Quat::from_array(node.transform.resolved_rotation()).normalize())
            * Mat4::from_mat3(stretch)
            * Mat4::from_scale(Vec3::from_array(node.transform.scale)))
    }
}

pub(crate) fn focus_distance(camera: Mat4, target: Mat4) -> f32 {
    let axis = camera.z_axis.truncate().normalize();
    (target.w_axis.truncate() - camera.w_axis.truncate()).dot(axis).abs().max(super::constants::MIN_FOCUS_DISTANCE)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn nodes(value: serde_json::Value) -> Vec<EvalNode> {
        vork_scene::v2::from_test_value::<Vec<vork_scene::Node>>(value).unwrap().into_iter()
            .map(|node| EvalNode { camera: None, world: node.transform.matrix(), node, parent: None, dynamic: false, embedded: None }).collect()
    }
    #[test]
    fn aim_preserves_parent_stretch_without_local_rotation_contamination() {
        let mut nodes = nodes(serde_json::json!([
            {"type":"node","name":"parent"},
            {"type":"node","name":"child","target":[0,0,-1],"scale":[1,2,3]}
        ]));
        nodes[1].parent = Some(0);
        let plan = Plan::compile(&mut nodes).unwrap();
        // QR residual has nonuniform scale, shear and reflection.
        let stretch = Mat4::from_cols(glam::vec4(2.,0.,0.,0.),glam::vec4(0.6,1.,0.,0.),
            glam::vec4(0.2,0.3,-1.,0.),glam::Vec4::W);
        let parent = Mat4::from_rotation_y(0.7) * stretch;
        for angle in [0.,std::f32::consts::FRAC_PI_4,std::f32::consts::FRAC_PI_2] {
            nodes[1].node.transform.rotation = Some(glam::Quat::from_rotation_z(angle).to_array());
            let world = parent * nodes[1].node.transform.matrix();
            let actual = plan.aim(1,world,&[parent,world],&nodes[1].node,|_|false).unwrap();
            let expected = Mat4::from_rotation_z(angle) * stretch * Mat4::from_scale(Vec3::new(1.,2.,3.));
            assert!(actual.abs_diff_eq(expected,1e-5),"angle {angle}: {actual:?}");
        }
    }
    #[test]
    fn every_node_kind_composes_aim_with_local_rotation() {
        for kind in ["node","mesh","asset","camera","light"] {
            let mut value=serde_json::json!({"type":kind,"name":"aimed","target":[1,0,-1],"rotation":{"euler":[0,0,0.4]},"scale":[2,3,4]});
            match kind {
                "mesh" => {value["geometry"]="g".into();},
                "asset" => {value["file"]="model.gltf".into();},
                "light" => {value["shape"]="environment".into();},
                _ => {}
            }
            let mut nodes=nodes(serde_json::json!([value]));
            let plan=Plan::compile(&mut nodes).unwrap();
            let expected=Mat4::look_to_rh(Vec3::ZERO,Vec3::new(1.,0.,-1.).normalize(),Vec3::Y).inverse()
                * Mat4::from_rotation_z(0.4) * Mat4::from_scale(Vec3::new(2.,3.,4.));
            let actual=plan.aim(0,nodes[0].world,&[nodes[0].world],&nodes[0].node,|_|false).unwrap();
            assert!(actual.abs_diff_eq(expected,1e-6),"{kind}: {actual:?}");
        }
    }
    #[test]
    fn aiming_preserves_direction_at_large_world_translations() {
        let mut nodes=nodes(serde_json::json!([{"type":"camera","name":"camera","target":[100000096,100000064,99999872],"translation":[100000000,100000000,100000000]}]));
        let plan=Plan::compile(&mut nodes).unwrap();let world=nodes[0].world;
        let aimed=plan.aim(0,world,&[world],&nodes[0].node,|_|false).unwrap();
        assert!(aimed.transform_vector3(Vec3::NEG_Z).normalize().abs_diff_eq(Vec3::new(96.,64.,-128.).normalize(),1e-6));
    }

    #[test]
    fn point_targets_remain_in_world_space_when_the_parent_moves() {
        let mut nodes=nodes(serde_json::json!([
            {"type":"camera","name":"camera","target":[0,0,0],"translation":[1,0,3],"scale":[2,3,4]},
            {"type":"node","name":"rig"}]));
        nodes[0].parent=Some(1);
        nodes[1].dynamic=true;
        let plan=Plan::compile(&mut nodes).unwrap();
        for x in [5.,10.] {
            let parent=Mat4::from_translation(Vec3::new(x,0.,0.));
            let world=parent*nodes[0].world;
            let aimed=plan.aim(0,world,&[world,parent],&nodes[0].node,|_|false).unwrap();
            assert!(aimed.transform_vector3(Vec3::NEG_Z).normalize().abs_diff_eq(-world.w_axis.truncate().normalize(),1e-6));
            assert!(aimed.to_scale_rotation_translation().0.abs_diff_eq(Vec3::new(2.,3.,4.),1e-6));
        }
    }
    #[test]
    fn targets_order_dependencies_propagate_motion_and_keep_scale() {
        let mut nodes = nodes(serde_json::json!([
            {"type":"camera","name":"camera","target":"subject","scale":[2,3,4]},
            {"type":"node","name":"subject","translation":[1,0,-1]}
        ]));
        nodes[1].dynamic = true;
        let plan = Plan::compile(&mut nodes).unwrap();
        assert_eq!(plan.order, [1, 0]);
        assert!(nodes[0].dynamic);
        let worlds = nodes.iter().map(|n| n.world).collect::<Vec<_>>();
        let aimed = plan.aim(0, worlds[0], &worlds, &nodes[0].node, |_| false).unwrap();
        let direction = aimed.transform_vector3(Vec3::NEG_Z).normalize();
        assert!(direction.abs_diff_eq(Vec3::new(1.,0.,-1.).normalize(), 1e-6));
        assert!(aimed.to_scale_rotation_translation().0.abs_diff_eq(Vec3::new(2.,3.,4.), 1e-6));
    }
    #[test]
    fn missing_and_cyclic_targets_are_errors() {
        for value in [
            serde_json::json!([{"type":"camera","name":"cam","target":"missing"}]),
            serde_json::json!([{"type":"camera","name":"a","target":"b"},{"type":"light","name":"b","shape":"sun","target":"a"}]),
        ] { assert!(Plan::compile(&mut nodes(value)).is_err()); }
        let mut nodes = nodes(serde_json::json!([{"type":"camera","name":"cam","target":"child"},{"type":"node","name":"child"}]));
        nodes[1].parent = Some(0);
        assert!(Plan::compile(&mut nodes).unwrap_err().to_string().contains("cycle"));
    }
    #[test]
    fn focus_uses_absolute_view_axis_distance_instead_of_radial_distance() {
        let camera = Mat4::from_translation(Vec3::new(1.,2.,3.));
        assert_eq!(focus_distance(camera, Mat4::from_translation(Vec3::new(8.,9.,-2.))), 5.);
        assert_eq!(focus_distance(camera, Mat4::from_translation(Vec3::new(8.,9.,8.))), 5.);
    }
}
