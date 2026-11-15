use super::evaluation::{CpuScene, Pose};
use anyhow::{Context, Result};
use serde_json::{Value, json};

fn lookup(mut value: Value, path: &str) -> Result<Value> {
    if path.is_empty() {
        return Ok(value);
    }
    for key in path.split('.') {
        value = value
            .get(key)
            .or_else(|| key.parse::<usize>().ok().and_then(|i| value.get(i)))
            .cloned()
            .with_context(|| format!("ask: field {path:?} does not exist on this value"))?;
    }
    Ok(value)
}

fn named<'a>(path: &'a str, names: impl Iterator<Item = &'a str>) -> Option<(&'a str, &'a str)> {
    names
        .filter(|name| !name.is_empty())
        .filter_map(|name| {
            path.strip_prefix(name)
                .and_then(|tail| tail.strip_prefix('.').map(|tail| (name, tail)))
        })
        .max_by_key(|(name, _)| name.len())
}

fn node(pose: &Pose, index: usize) -> Result<Value> {
    let source = &pose.nodes[index].node;
    let mut value = serde_json::to_value(source)?;
    // These fields deliberately omit defaults when writing a scene; an answer
    // still exposes the decoded Rust value, including an identity rotation.
    value["rotation"] = json!(source.transform.resolved_rotation());
    if let Some(composition) = &source.composition {
        value["composition"]["compositor"] = compositor(&composition.compositor)?;
    }
    value["children"] = Value::Array(
        pose.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == Some(index))
            .map(|(i, _)| node(pose, i))
            .collect::<Result<_>>()?,
    );
    Ok(value)
}

fn material(
    scene: &CpuScene,
    source: &vork_scene::v2::Material,
    tail: Option<&str>,
) -> Result<Value> {
    let mut value = serde_json::to_value(source)?;
    if source.program == "std/openpbr" {
        for parameter in vork_scene::v2::OPENPBR {
            if !source.inputs.contains_key(parameter.name) {
                value["inputs"][parameter.name] =
                    if parameter.kind == vork_scene::v2::OpenPbrKind::Bool {
                        json!(parameter.default[0] != 0.)
                    } else if parameter.default.len() == 1 {
                        json!(parameter.default[0])
                    } else {
                        json!(parameter.default)
                    };
            }
        }
    }
    if source.program != "std/openpbr" {
        let wanted = tail
            .and_then(|t| t.strip_prefix("inputs."))
            .map(|t| t.split('.').next().unwrap());
        let all = tail.is_none() || tail == Some("inputs");
        if all || wanted.is_some_and(|name| !source.inputs.contains_key(name)) {
            for (name, default) in scene.input_defaults(&source.program)? {
                if source.inputs.contains_key(&name) || (!all && wanted != Some(name.as_str())) {
                    continue;
                }
                value["inputs"][&name]=default.with_context(||format!(
                    "not built yet: this answer needs program {:?} input {name:?}; author its input value",source.program))?;
            }
        }
    }
    Ok(value)
}

fn compositor(entries: &[vork_scene::v2::CompositeEntry]) -> Result<Value> {
    let mut value = serde_json::to_value(entries)?;
    for (value, entry) in value.as_array_mut().unwrap().iter_mut().zip(entries) {
        if let vork_scene::v2::CompositeEntry::Node(step) = entry {
            value["name"] = json!(step.name);
            value["resolution"] = serde_json::to_value(&step.resolution)?;
        }
    }
    Ok(value)
}

pub(super) fn read(
    scene: &CpuScene,
    pose: &Pose,
    request: &vork_time::Request,
    path: &str,
) -> Result<Value> {
    if let Some(path) = path.strip_prefix("nodes.") {
        let (name, tail) = named(path, scene.nodes.iter().map(|n| n.node.name.as_str()))
            .context("ask: field needs a named node and a property")?;
        return lookup(node(pose, scene.node(name)?)?, tail);
    }
    if let Some(path) = path.strip_prefix("materials.") {
        let (name, tail) = named(path, pose.materials.keys().map(String::as_str))
            .context("ask: field needs a named material and a property")?;
        return lookup(material(scene, &pose.materials[name], Some(tail))?, tail);
    }
    let mut value = serde_json::to_value(&scene.scene)?;
    value["$schema"] = json!(scene.scene.schema);
    value["simulation"] = serde_json::to_value(&scene.scene.simulation)?;
    value["propagation"] = serde_json::to_value(&scene.scene.propagation)?;
    value["render"] = serde_json::to_value(&pose.render)?;
    // A composition's entries read back with their nullable fields: the scene's, each rendered image's and
    // (in `node`) each node's.
    if let Some(composition) = &scene.scene.composition {
        value["composition"]["compositor"] = compositor(&composition.compositor)?;
    }
    for (name, image) in &scene.scene.images {
        if let Some(composition) = &image.composition {
            value["images"][name]["composition"]["compositor"] = compositor(&composition.compositor)?;
        }
    }
    if path == "nodes" {
        return Ok(Value::Array(
            pose.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.parent.is_none())
                .map(|(i, _)| node(pose, i))
                .collect::<Result<_>>()?,
        ));
    }
    if path == "materials" {
        return Ok(Value::Object(
            pose.materials
                .iter()
                .map(|(name, m)| Ok((name.clone(), material(scene, m, None)?)))
                .collect::<Result<_>>()?,
        ));
    }
    for (name, geometry) in &scene.scene.geometry {
        value["geometry"][name]["crease_edge"] = json!(geometry.crease_edge);
        value["geometry"][name]["status"] = serde_json::to_value(&geometry.status)?;
    }
    for (name, program) in &scene.scene.programs {
        value["programs"][name]["source_origin"] = json!(program.source_origin);
    }
    if path == "images" || path.starts_with("images.") {
        for name in scene.scene.images.keys() {
            value["images"][name]["time"] =
                json!(scene.image_time(name, pose, request)?.numeric()?);
        }
    }
    for table in ["images", "buffers", "geometry", "programs"] {
        if let Some(path) = path.strip_prefix(&format!("{table}.")) {
            let (name, tail) = named(
                path,
                value[table].as_object().unwrap().keys().map(String::as_str),
            )
            .with_context(|| format!("ask: field needs a named {table} entry and a property"))?;
            return lookup(value[table][name].clone(), tail);
        }
    }
    lookup(value, path)
}
