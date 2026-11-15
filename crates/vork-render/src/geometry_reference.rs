//! Read-only geometry handles. Connectivity and row identities are immutable;
//! the compute cursor refreshes poses at each canonical input sample.
use crate::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn names(scene: &vork_scene::v2::Scene) -> BTreeSet<String> {
    scene
        .materials
        .values()
        .filter(|material| {
            scene
                .programs
                .get(&material.program)
                .is_some_and(|program| program.is_compute())
        })
        .flat_map(|material| material.inputs.values())
        .filter_map(|input| match input {
            vork_scene::v2::Input::Geometry(binding) => Some(binding.geometry.clone()),
            _ => None,
        })
        .collect()
}
pub(crate) fn handle(scene: &vork_scene::v2::Scene, name: &str) -> Result<u32> {
    names(scene)
        .iter()
        .position(|candidate| candidate == name)
        .map(|i| i as u32)
        .with_context(|| format!("geometry input {name:?}: bind a named mesh object"))
}
pub(crate) fn validate_dependencies(
    scene: &vork_scene::v2::Scene,
    nodes: &[EvalNode],
) -> Result<()> {
    let mut graph = BTreeMap::<String, BTreeSet<String>>::new();
    for node in nodes {
        let vork_scene::v2::NodeKind::Mesh { geometry, .. } = &node.node.kind else {
            continue;
        };
        let shape = &scene.geometry[geometry];
        let (programs, _) = compute::geometry_solver_dependencies(
            scene,
            shape.attributes.values().cloned().collect(),
        );
        let targets = scene
            .materials
            .values()
            .filter(|material| programs.contains(&material.program))
            .flat_map(|material| material.inputs.values())
            .filter_map(|input| match input {
                vork_scene::v2::Input::Geometry(binding) => Some(binding.geometry.clone()),
                _ => None,
            })
            .collect();
        graph.insert(node.node.name.clone(), targets);
    }
    fn visit(
        name: &str,
        graph: &BTreeMap<String, BTreeSet<String>>,
        stack: &mut Vec<String>,
        done: &mut BTreeSet<String>,
    ) -> Result<()> {
        if let Some(at) = stack.iter().position(|item| item == name) {
            let mut cycle = stack[at..].to_vec();
            cycle.push(name.into());
            bail!(
                "geometry dependency cycle: {}; remove a geometry reference from this cycle or use an independent collider",
                cycle.join(" -> ")
            );
        }
        if done.contains(name) {
            return Ok(());
        }
        stack.push(name.into());
        for target in graph.get(name).into_iter().flatten() {
            visit(target, graph, stack, done)?;
        }
        stack.pop();
        done.insert(name.into());
        Ok(())
    }
    let mut done = BTreeSet::new();
    for name in graph.keys() {
        visit(name, &graph, &mut Vec::new(), &mut done)?;
    }
    Ok(())
}
/// Only referenced programs acquire a table declaration or storage binding.
pub(crate) fn header(binding: u32, count: usize) -> String {
    let stride = DeformedVertex::WORDS;
    let normal = DeformedVertex::NORMAL_WORD;
    format!(
        r#"
[[vk::binding({binding},0)]] StructuredBuffer<uint> vork_reference_words;
uint _vork_geometry_word(Geometry g,uint offset) {{if(vork_slot(g)>={count}u) return 0u; return vork_reference_words[vork_slot(g)*24u+offset];}}
uint geometry_vertex_count(Geometry g) {{return _vork_geometry_word(g,0u);}}
uint geometry_object_id(Geometry g) {{return _vork_geometry_word(g,20u);}}
uint geometry_source_vertex_load(Geometry g,uint i) {{return i<geometry_vertex_count(g) ? i : 0xffffffffu;}}
uint geometry_triangle_count(Geometry g) {{return _vork_geometry_word(g,1u);}}
float3 geometry_POSITION_load(Geometry g,uint i) {{
    if(i>=geometry_vertex_count(g)) return float3(0.);
    let at=_vork_geometry_word(g,2u)+i*{stride}u;
    return asfloat(uint3(vork_reference_words[at],vork_reference_words[at+1u],vork_reference_words[at+2u]));
}}
float3 geometry_NORMAL_load(Geometry g,uint i) {{
    if(i>=geometry_vertex_count(g)) return float3(0.);
    let at=_vork_geometry_word(g,2u)+i*{stride}u+{normal}u;
    return asfloat(uint3(vork_reference_words[at],vork_reference_words[at+1u],vork_reference_words[at+2u]));
}}
uint3 geometry_triangle_load(Geometry g,uint i) {{
    if(i>=geometry_triangle_count(g)) return uint3(0u);
    let at=_vork_geometry_word(g,3u)+i*3u;
    return uint3(vork_reference_words[at],vork_reference_words[at+1u],vork_reference_words[at+2u]);
}}
float3 geometry_to_world(Geometry g,float3 p) {{
    if(vork_slot(g)>={count}u) return float3(0.);
    let at=vork_slot(g)*24u+4u;
    return p.x*asfloat(uint3(vork_reference_words[at],vork_reference_words[at+1u],vork_reference_words[at+2u]))
        +p.y*asfloat(uint3(vork_reference_words[at+4u],vork_reference_words[at+5u],vork_reference_words[at+6u]))
        +p.z*asfloat(uint3(vork_reference_words[at+8u],vork_reference_words[at+9u],vork_reference_words[at+10u]))
        +asfloat(uint3(vork_reference_words[at+12u],vork_reference_words[at+13u],vork_reference_words[at+14u]));
}}
"#
    )
}

pub(crate) struct Reference {
    pub object: u32,
    pub root: install::MeshRoot,
    pub row: u64,
    pub vertex_word: u32,
    pub count: u32,
}
pub(crate) struct Table {
    pub buffer: wgpu::Buffer,
    pub references: Vec<Reference>,
}
impl Table {
    pub(crate) fn new(
        r: &Renderer,
        scene: &vork_scene::v2::Scene,
        decoded: &buffers::SceneBuffers,
        nodes: &[EvalNode],
        roots: &[install::MeshRoot],
        draws: &[install::Draw],
        active: &BTreeSet<String>,
    ) -> Result<Option<Self>> {
        validate_dependencies(scene, nodes)?;
        let names = names(scene);
        if active.is_empty() {
            return Ok(None);
        }
        let mut words = vec![0; names.len() * 24];
        let mut references = Vec::new();
        for (index, name) in names.iter().enumerate().filter(|(_, name)| active.contains(*name)) {
            let node = nodes
                .iter()
                .position(|node| node.node.name == *name)
                .with_context(|| {
                    format!("geometry reference {name:?}: object does not exist; name a mesh node")
                })?;
            let object = draws
                .iter()
                .find(|draw| roots[draw.object as usize].node == node)
                .map(|draw| draw.object)
                .with_context(|| format!(
                    "geometry reference {name:?}: requires one mesh object; put each collider instance on its own named node"
                ))?;
            let root = &roots[object as usize];
            // Material ranges have separate render object slots, but reference
            // the same complete source mesh. Distinguish actual instances even
            // when their placements happen to be identical.
            ensure!(
                draws.iter()
                    .filter(|draw| roots[draw.object as usize].node == node)
                    .flat_map(|draw| draw.instances.clone())
                    .map(|object| &roots[object as usize])
                    .filter(|other| other.node == node)
                    .all(|other| root.same_placement(other)
                        && root.instance_index == other.instance_index),
                "geometry reference {name:?}: requires one mesh object; put each collider instance on its own named node"
            );
            let vork_scene::v2::NodeKind::Mesh { geometry, .. } = &nodes[node].node.kind else {
                bail!("geometry reference {name:?}: use a native mesh node with named geometry");
            };
            let node_draws: Vec<_> = draws.iter()
                .filter(|draw| roots[draw.object as usize].node == node).collect();
            ensure!(
                node_draws.iter().all(|draw| draw.displacement.is_none())
                    || node_draws.iter().map(|draw| draw.material).collect::<BTreeSet<_>>().len() == 1,
                "canonical displacement on node {name:?} requires one material; split the node into separately named geometry objects"
            );
            ensure!(!compute::places_instances(scene,&nodes[node].node),
                "geometry reference {name:?}: GPU instance placement/counts require canonical placement sampling; use node animation tracks on a separately named native mesh collider");
            let shape = &scene.geometry[geometry];
            ensure!(
                matches!(shape.topology, vork_scene::v2::Topology::Triangles)
                    && shape.subdivision.is_none()
                    && shape.face_vertex_counts.is_none(),
                "geometry reference {name:?}: use fixed triangles or a triangle collision proxy"
            );
            ensure!(
                !computed_geometry::dynamic(scene, shape),
                "geometry reference {name:?}: computed collider dependencies need an ordered state snapshot; use an animated mesh collision proxy"
            );
            ensure!(shape.attributes.contains_key("NORMAL"),
                "geometry reference {name:?}: source-row NORMAL is required; author NORMAL (split triangle vertices for face normals) before binding this collider");
            let positions = decoded.data[&shape.attributes["POSITION"]].floats::<3>()?;
            let normals = decoded.data[&shape.attributes["NORMAL"]].floats::<3>()?;
            let vertex_word = u32::try_from(words.len())?;
            let count = u32::try_from(positions.len())?;
            for (position, normal) in positions.into_iter().zip(normals) {
                let vertex = DeformedVertex { POSITION: position, NORMAL: normal, TANGENT: [1., 0., 0., 1.] };
                words.extend_from_slice(bytemuck::cast_slice(std::slice::from_ref(&vertex)));
            }
            let triangle_word = u32::try_from(words.len())?;
            let indices = shape.indices.as_ref().map_or_else(
                || (0..count).collect(),
                |name| decoded.data[name].words.clone(),
            );
            words[index * 24..index * 24 + 4].copy_from_slice(&[
                count,
                indices.len() as u32 / 3,
                vertex_word,
                triangle_word,
            ]);
            words[index * 24 + 20] = object;
            words.extend(indices);
            references.push(Reference {
                object,
                root: root.clone(),
                row: index as u64 * 96,
                vertex_word,
                count,
            });
        }
        let bytes = words.len() as u64 * 4;
        let limit = u64::from(r.device.limits().max_storage_buffer_binding_size)
            .min(r.device.limits().max_buffer_size);
        ensure!(
            bytes <= limit,
            crate::gpu_memory::device_limit("geometry reference table", format_args!("{bytes} bytes"), format_args!("{limit} bytes"), "use smaller collision proxies")
        );
        let buffer = init(
            &r.device,
            "canonical geometry references",
            bytemuck::cast_slice(&words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        Ok(Some(Self { buffer, references }))
    }
}
