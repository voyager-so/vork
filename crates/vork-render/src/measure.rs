//! Questions answered from an installed scene without rendering: a node's
//! axis-aligned box in its parent's space. Internal: tests reach it; no scene
//! field, command or SDK call does.
use crate::InstalledScene;
use glam::{Mat4, Vec3};

/// An axis-aligned box in metres in the frame of the node's parent
/// (`parent` None: the scene root). The parent's own name may be empty.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub parent: Option<usize>,
    pub parent_name: Option<String>,
}

/// Bounds of the authored node `name` and everything under it, in its
/// parent's space: the union of every installed draw's bounds under every
/// instance placement the install evaluated (rest pose for skinned, morphed
/// and computed geometry; visibility ignored). None when nothing under the
/// node draws, or when the parent's frame has no inverse (a zero scale).
pub fn node_bounds(scene: &InstalledScene, name: &str) -> Option<NodeBounds> {
    let index = scene.nodes.iter().position(|n| n.node.name == name)?;
    node_bounds_at(scene, index)
}

/// `node_bounds` by the node's preorder index in the authored tree.
pub fn node_bounds_at(scene: &InstalledScene, index: usize) -> Option<NodeBounds> {
    let node = scene.nodes.get(index)?;
    let to_parent = node.parent.map_or(Mat4::IDENTITY, |p| scene.nodes[p].world).inverse();
    if !to_parent.is_finite() {
        return None;
    }
    let under = |mut n: usize| loop {
        if n == index {
            return true;
        }
        match scene.nodes[n].parent {
            Some(p) => n = p,
            None => return false,
        }
    };
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for draw in &scene.draws {
        if !(draw.bounds.0.is_finite() && draw.bounds.1.is_finite()) {
            continue;
        }
        for object in draw.instances.clone() {
            if !under(scene.roots[object as usize].node) {
                continue;
            }
            let placed = crate::shadow_envelope::transformed(
                to_parent * scene.objects[object as usize].model,
                draw.bounds,
            );
            bounds = Some(bounds.map_or(placed, |(lo, hi)| (lo.min(placed.0), hi.max(placed.1))));
        }
    }
    let (lo, hi) = bounds?;
    Some(NodeBounds {
        min: lo.to_array(),
        max: hi.to_array(),
        parent: node.parent,
        parent_name: node.parent.map(|p| scene.nodes[p].node.name.clone()),
    })
}
