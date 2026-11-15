//! Generated authoring schema shared by the CLI and SDK.
/// Shared fields on each definition; source-exclusive alternatives name only their source fields.
/// The published form.
pub(super) fn authored_schema() -> schemars::Schema {
    // Everything derived below iterates object keys; see `sorted_keys`.
    let mut schema: schemars::Schema = serde_json::from_value(sorted_keys(full_schema().to_value())).unwrap();
    let values = schema.clone().to_value();
    let reference = reference(&values);
    schema.as_object_mut().unwrap().insert(
        "x-vork-differences".into(),
        serde_json::json!(reference.lines().collect::<Vec<_>>()),
    );
    schema
}
/// Every authored type: what the decode diagnostics walk.
pub(super) fn full_schema() -> schemars::Schema {
    let mut schema = schemars::generate::SchemaSettings::draft2020_12()
        .for_deserialize()
        .into_generator()
        .into_root_schema_for::<super::Scene>();
    schema.as_object_mut().unwrap().insert("$id".into(), serde_json::json!(crate::SCHEMA_URL));
    schema.get_mut("properties").unwrap()["version"]["const"] = serde_json::json!(2);
    schema.as_object_mut().unwrap().insert(
        "x-standards".into(),
        serde_json::json!({
            "materials":"OpenPBR Surface 1.1.1",
            "hair":"Blender 5.2.1 Principled Hair; Chiang scattering in raster and path",
            "cameras":"Blender 5.2.1 physical camera, except the names fit, shift, projection",
            "camera JSON":"OpenCV",
            "colour":"AgX as Blender 5.2.1",
            "assets":"glTF 2.0",
            "animation":"glTF 2.0 interpolation",
        "lights":"Blender 5.2.1 lights, except radius (Blender: shadow_soft_size)",
        "geometry":"glTF 2.0 topology and attribute semantics",
        "textures":"glTF 2.0 samplers; KHR_texture_transform lowered on import"
        }),
    );
    // Render settings are not scene content; profiles hold them. Publish their
    // contract beside the scene's for help, SDK types and profile authors.
    let mut settings = schemars::generate::SchemaSettings::draft2020_12().for_deserialize()
        .into_generator().into_root_schema_for::<super::ProfileFile>().to_value();
    let settings_defs = settings.as_object_mut().unwrap().remove("$defs").unwrap_or_default();
    for (name, definition) in settings_defs.as_object().unwrap() {
        schema.get_mut("$defs").unwrap()[name.as_str()] = definition.clone();
    }
    settings.as_object_mut().unwrap().remove("$schema");
    schema.get_mut("$defs").unwrap()["Profile"] = settings;
    let mut destination = schemars::generate::SchemaSettings::draft2020_12().for_deserialize()
        .into_generator().into_root_schema_for::<super::OutputDestination>().to_value();
    let definitions = destination.as_object_mut().unwrap().remove("$defs").unwrap();
    for (name, definition) in definitions.as_object().unwrap() {
        schema.get_mut("$defs").unwrap()[name.as_str()] = definition.clone();
    }
    destination.as_object_mut().unwrap().remove("$schema");
    schema.get_mut("$defs").unwrap()["OutputDestination"] = destination;
    // Resolved settings serialize for in-memory round trips; they are not scene content.
    schema.get_mut("$defs").unwrap()["Render"]["default"].as_object_mut().map(|d| d.remove("settings"));
    schema.get_mut("properties").unwrap()["render"]["default"].as_object_mut().map(|d| d.remove("settings"));
    // Every settings field is optional; omitted ones take the integrator's defaults.
    let settings = &mut schema.get_mut("$defs").unwrap()["Settings"];
    settings.as_object_mut().unwrap().remove("required");
    settings["additionalProperties"] = serde_json::json!(false);
    let defs = schema.get_mut("$defs").unwrap();
    // Only the plain group branch may omit the discriminator.
    for branch in defs["Node"]["oneOf"].as_array_mut().unwrap() {
        if branch["properties"]["type"]["const"] == "node" {
            branch.as_object_mut().unwrap().remove("required");
            branch["properties"]["type"]["default"] = serde_json::json!("node");
        }
    }
    defs["Node"]["unevaluatedProperties"] = serde_json::json!(false);
    for definition in ["Node", "AssetNodeOverrides"] {
        defs[definition]["properties"]["pass_index"]["minimum"] = serde_json::json!(1);
        defs[definition]["properties"]["pass_index"]["maximum"] = serde_json::json!(65535);
    }
    for (name, a, b, title_a, title_b) in [
        ("Buffer", "data", "file", "InlineBuffer", "FileBuffer"),
        ("Program", "file", "source", "FileProgram", "SourceProgram"),
        ("Layer", "tracks", "clip", "TrackLayer", "ClipLayer"),
        ("NodeLayer", "tracks", "clip", "NodeTrackLayer", "NodeClipLayer"),
    ] {
        // A definition absent from the document schema (a type hidden with `schemars(skip)`) has no variants to split.
        if defs.get(name).is_none() { continue; }
        // Optional fields serialize as null; exactly one must contain a value.
        fn nonnull(mut property: serde_json::Value) -> serde_json::Value {
            if let Some(types) = property
                .get_mut("type")
                .and_then(serde_json::Value::as_array_mut)
            {
                types.retain(|t| t != "null");
            }
            if let Some(branches) = property
                .get_mut("anyOf")
                .and_then(serde_json::Value::as_array_mut)
            {
                branches.retain(|b| b["type"] != "null");
            }
            property
        }
        let pa = nonnull(defs[name]["properties"][a].clone());
        let pb = nonnull(defs[name]["properties"][b].clone());
        defs[name]["oneOf"] = serde_json::json!([
            {"title":title_a,"required":[a],"properties":{a:pa,b:{"type":"null"}}},
            {"title":title_b,"required":[b],"properties":{b:pb,a:{"type":"null"}}}
        ]);
    }
    // Dense storage is the third buffer source; optional inline cells seed it.
    for branch in defs["Buffer"]["oneOf"].as_array_mut().unwrap() {
        branch["properties"]["dims"]=serde_json::json!({"type":"null"});
    }
    let mut dims=defs["Buffer"]["properties"]["dims"].clone();
    dims["type"]=serde_json::json!("array");
    dims["items"]["minimum"]=serde_json::json!(1);
    defs["Buffer"]["oneOf"].as_array_mut().unwrap().push(serde_json::json!({
        "title":"DenseBuffer","required":["dims"],"properties":{
            "dims":dims,"file":{"type":"null"},"grid":{"type":"null"},
            "offset":{"type":"integer","minimum":0,"maximum":0},"count":{"type":"null"}
        }
    }));
    let mut dense_file = defs["Buffer"]["oneOf"][1].clone();
    dense_file["title"] = serde_json::json!("DenseFileBuffer");
    dense_file["required"] = serde_json::json!(["file", "dims"]);
    dense_file["properties"]["dims"] = defs["Buffer"]["oneOf"][2]["properties"]["dims"].clone();
    dense_file["properties"]["grid"] = serde_json::json!({"type":"null"});
    defs["Buffer"]["oneOf"].as_array_mut().unwrap().push(dense_file);
    defs["Program"]["oneOf"][0]["properties"]["source_origin"]=serde_json::json!({"type":"null"});
    schema
}

/// Complete named alternatives for SDK generators.
pub fn schema() -> schemars::Schema {
    let mut schema = authored_schema();
    let defs = schema.get_mut("$defs").unwrap();
    // Give every object alternative its full shape. SDK generators then consume
    // the same titled variants without separate shared-field merging rules.
    for definition in defs.as_object_mut().unwrap().values_mut() {
        if definition.get("properties").is_none() || definition.get("oneOf").is_none() { continue; }
        let common = definition.as_object_mut().unwrap();
        let properties = common.remove("properties").unwrap();
        let required = common.remove("required").unwrap_or(serde_json::json!([]));
        let additional = common.remove("additionalProperties");
        common.remove("type");
        for branch in common["oneOf"].as_array_mut().unwrap() {
            let mut fields = properties.as_object().unwrap().clone();
            fields.extend(branch["properties"].as_object().unwrap().clone());
            branch["properties"] = fields.into();
            let mut fields = required.as_array().unwrap().clone();
            for field in branch["required"].as_array().into_iter().flatten() {
                if !fields.contains(field) { fields.push(field.clone()); }
            }
            branch["required"] = fields.into();
            branch["type"] = serde_json::json!("object");
            if let Some(additional) = &additional { branch["additionalProperties"] = additional.clone(); }
        }
    }
    // The public editor schema includes authoring shorthand, before std expansion.
    let mut geometry = defs["Geometry"].take();
    geometry["title"] = serde_json::json!("BufferGeometry");
    let mut builder = schemars::schema_for!(super::StdGeometry).to_value();
    builder.as_object_mut().unwrap().remove("$schema");
    builder["not"] = serde_json::json!({"required": ["name"]});
    defs["Geometry"] = serde_json::json!({"anyOf": [geometry, builder]});
    // Merged branch properties above follow map order; publish sorted keys.
    serde_json::from_value(sorted_keys(schema.to_value())).unwrap()
}

fn help_default(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) if map.contains_key("program") => serde_json::json!({"program":map["program"]}).to_string(),
        serde_json::Value::Object(_) => "{}".to_owned(),
        serde_json::Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(help_default)
                .collect::<Vec<_>>()
                .join(",")
        ),
        // Scene floating-point fields are f32; preserve their shortest round-trip spelling.
        serde_json::Value::Number(number) if number.is_f64() => {
            (number.as_f64().unwrap() as f32).to_string()
        }
        serde_json::Value::String(value) if !value.is_empty() => value.clone(),
        _ => serde_json::to_string(value).unwrap(),
    }
}

/// Follow actual schema properties through tagged branches, rather than treating
/// variant-qualified annotation paths as JSON object paths.
fn tagged_help(
    schema: &serde_json::Value,
) -> (
    std::collections::BTreeMap<String, (String, String)>,
    Vec<String>,
) {
    fn walk(
        schema: &serde_json::Value,
        value: &serde_json::Value,
        path: &str,
        condition: &str,
        annotation_root: &str,
        stack: &mut Vec<String>,
        fields: &mut std::collections::BTreeMap<String, (String, String)>,
    ) {
        if let Some(annotation) = value["x-help-path"].as_str() {
            if value["x-help"].is_string()
                && (annotation == annotation_root
                    || annotation.starts_with(&format!("{annotation_root}.")))
            {
                fields
                    .entry(annotation.into())
                    .and_modify(|(wire, conditions)| {
                        assert_eq!(wire, path, "one annotation must describe one wire path");
                        if !conditions.split(" or ").any(|c| c == condition) {
                            conditions.push_str(&format!(" or {condition}"));
                        }
                    })
                    .or_insert_with(|| (path.into(), condition.into()));
            }
        }
        if let Some(reference) = value["$ref"].as_str() {
            if !stack.iter().any(|r| r == reference) {
                stack.push(reference.into());
                walk(
                    schema,
                    schema.pointer(reference.trim_start_matches('#')).unwrap(),
                    path,
                    condition,
                    annotation_root,
                    stack,
                    fields,
                );
                stack.pop();
            }
        }
        if let Some(properties) = value["properties"].as_object() {
            for (name, property) in properties {
                walk(
                    schema,
                    property,
                    &format!("{path}.{name}"),
                    condition,
                    annotation_root,
                    stack,
                    fields,
                );
            }
        }
        for keyword in ["allOf", "anyOf", "oneOf"] {
            if let Some(branches) = value[keyword].as_array() {
                for branch in branches {
                    walk(
                        schema,
                        branch,
                        path,
                        condition,
                        annotation_root,
                        stack,
                        fields,
                    );
                }
            }
        }
        if let Some(items) = value.get("additionalProperties").filter(|v| v.is_object()) {
            walk(
                schema,
                items,
                &format!("{path}.<name>"),
                condition,
                annotation_root,
                stack,
                fields,
            );
        }
    }
    let mut fields = std::collections::BTreeMap::new();
    let mut lines = Vec::new();
    for definition in schema["$defs"].as_object().unwrap().values() {
        let tagged = ["oneOf", "anyOf"].iter().any(|keyword| {
            definition[*keyword].as_array().is_some_and(|branches| {
                branches.iter().any(|branch| {
                    branch["properties"].as_object().is_some_and(|properties| {
                        properties.values().any(|p| p.get("const").is_some())
                    })
                })
            })
        });
        if !tagged {
            continue;
        }
        let path = definition["x-help-path"]
            .as_str()
            .expect("tagged types need a help object path");
        for keyword in ["oneOf", "anyOf"] {
            let Some(branches) = definition[keyword].as_array() else {
                continue;
            };
            for branch in branches {
                let Some(properties) = branch["properties"].as_object() else {
                    continue;
                };
                let Some((tag, value)) = properties
                    .iter()
                    .find_map(|(name, property)| property.get("const").map(|value| (name, value)))
                else {
                    continue;
                };
                let condition = format!("{path}.{tag}={value}");
                let names = properties
                    .keys()
                    .filter(|name| *name != tag)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!("{path} [{tag}={value}]: {{\"{tag}\":{value}, …}}; flat sibling fields: {}; no variant wrapper", if names.is_empty() { "none" } else { &names }));
                for (name, property) in properties {
                    if let Some(annotation) = property["x-help-path"].as_str() {
                        walk(
                            schema,
                            property,
                            &format!("{path}.{name}"),
                            &condition,
                            annotation,
                            &mut Vec::new(),
                            &mut fields,
                        );
                    }
                }
            }
        }
    }
    (fields, lines)
}

fn present_help(
    reference: &str,
    fields: &std::collections::BTreeMap<String, (String, String)>,
) -> String {
    reference
        .lines()
        .map(|line| {
            if let Some((path, help)) = line.split_once(": ") {
                if let Some((annotation, (wire, condition))) = fields
                    .iter()
                    .filter(|(annotation, _)| {
                        path == annotation.as_str()
                            || path
                                .strip_prefix(annotation.as_str())
                                .is_some_and(|suffix| {
                                    suffix.starts_with('.') || suffix.starts_with(' ')
                                })
                    })
                    .max_by_key(|(annotation, _)| annotation.len())
                {
                    let suffix = &path[annotation.len()..];
                    // Compact variant tag: `node.energy (light): ...`; omitted when every variant applies.
                    let tags: Vec<&str> = condition.split(" or ")
                        .map(|c| c.rsplit_once('=').map_or(c, |(_, v)| v.trim_matches('"'))).collect();
                    if tags == ["raster", "path"] || tags == ["path", "raster"] {
                        return format!("{wire}{suffix}: {help}");
                    }
                    return format!("{wire}{suffix} ({}): {help}", tags.join("|"));
                }
            }
            line.to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Terse field facts come from attributes on the Rust input fields.
pub fn reference(schema: &serde_json::Value) -> String {
    fn visit(value: &serde_json::Value, lines: &mut std::collections::BTreeMap<String, String>) {
        if let (Some(path), Some(help)) = (value["x-help-path"].as_str(), value["x-help"].as_str())
        {
            let base = format!("{path}: {help}");
            let mut line = base.clone();
            if let Some(default) = value.get("default") {
                line.push_str(&format!("; default {}", help_default(default)));
            }
            // Integrator variants can share a field with different defaults; the help states them.
            if lines.get(path).is_some_and(|seen| *seen != line) {
                line = base;
            }
            lines.insert(path.to_owned(), line);
        }
        if let Some(extra) = value["x-help-extra"].as_str() {
            lines.insert(
                format!("~{}", extra.split(':').next().unwrap()),
                extra.to_owned(),
            );
        }
        match value {
            serde_json::Value::Object(map) => {
                for child in map.values() {
                    visit(child, lines);
                }
            }
            serde_json::Value::Array(array) => {
                for child in array {
                    visit(child, lines);
                }
            }
            _ => {}
        }
    }
    let mut lines = std::collections::BTreeMap::new();
    visit(schema, &mut lines);
    for p in super::OpenPbrParam::supported() {
        let (ty, binding) = match p.kind {
            super::OpenPbrKind::Float => ("number", "constant|texture|attribute; animatable"),
            super::OpenPbrKind::Color3 => ("color3", "constant|texture|attribute; animatable"),
            super::OpenPbrKind::Bool => ("bool", "constant only; not animatable"),
            super::OpenPbrKind::Vector3 => ("vector3", "constant or texture; animatable"),
        };
        let range = match p.kind {
            super::OpenPbrKind::Bool => "{false,true}".to_owned(),
            super::OpenPbrKind::Vector3 => "N/A".to_owned(),
            _ => format!("{}{},{}{}{}", if p.open_min { '(' } else { '[' },
                p.min, p.max, if p.open_max { ')' } else { ']' },
                if p.kind == super::OpenPbrKind::Color3 { " per channel" } else { "" }),
        };
        let path = format!("material.inputs.{}", p.name);
        lines.insert(path.clone(), format!("{path}: {ty}; {}; range {range}; units {}; {binding}; default {}",
            p.meaning, match p.unit { "" => "dimensionless", "length" => "metres", unit => unit }, p.default_text));
        let program_path = format!("std/openpbr.{}", p.name);
        lines.insert(program_path.clone(), lines[&path].replacen(&path, &program_path, 1));
    }
    lines.insert("material.inputs (std/openpbr)".into(), format!(
        "material.inputs (std/openpbr): OpenPBR Surface; shaded inputs: {}",
        super::OpenPbrParam::supported().map(|p| p.name).collect::<Vec<_>>().join(" ")));
    let (fields, variants) = tagged_help(schema);
    let reference = lines
        .into_values()
        .chain(
            include_str!("reference-notes.txt")
                .lines()
                .map(str::to_owned),
        )
        .chain(variants)
        .collect::<Vec<_>>()
        .join("\n");
    present_help(&reference, &fields)
}

/// A nonempty Rust field annotation marks advanced help and explains when to tune it.
/// It is documentation metadata only: validation and the complete reference are unchanged.
fn advanced_fields(schema: &serde_json::Value) -> std::collections::BTreeMap<String, String> {
    fn visit(value: &serde_json::Value, fields: &mut std::collections::BTreeMap<String, (String, String, std::collections::BTreeSet<String>)>) {
        if let Some(reason) = value["x-help-advanced"].as_str() {
            assert!(
                !reason.trim().is_empty(),
                "advanced fields need tuning guidance"
            );
            let path = value["x-help-path"]
                .as_str()
                .expect("advanced field help path");
            let help = value["x-help"].as_str().expect("advanced field help");
            let default = value
                .get("default")
                .map(help_default)
                .unwrap_or_else(|| "omitted".into());
            let entry = fields.entry(path.into()).or_insert_with(||
                (help.into(), reason.into(), Default::default()));
            assert_eq!(entry.0, help, "conflicting advanced help for {path}");
            assert_eq!(entry.1, reason, "conflicting advanced guidance for {path}");
            entry.2.insert(default);
        }
        match value {
            serde_json::Value::Object(map) => {
                for child in map.values() {
                    visit(child, fields);
                }
            }
            serde_json::Value::Array(array) => {
                for child in array {
                    visit(child, fields);
                }
            }
            _ => {}
        }
    }
    let mut fields = std::collections::BTreeMap::new();
    // Only visit the input definitions, never generated reference metadata.
    visit(&schema["$defs"], &mut fields);
    visit(&schema["properties"], &mut fields);
    let (tagged, _) = tagged_help(schema);
    fields
        .into_iter()
        .map(|(path, (help, reason, defaults))| {
            let default=defaults.into_iter().collect::<Vec<_>>().join(" | ");
            let line=format!("{path}: {help}; default {default}; when to change: {reason}");
            let presented = present_help(&line, &tagged);
            (
                tagged
                    .get(&path)
                    .map(|(wire, _)| wire.clone())
                    .unwrap_or(path),
                presented,
            )
        })
        .collect()
}

/// Exact records whose inputs/options are already enumerated on the API page.
/// Keep their full contracts available through field lookup, outside topic listings.
pub fn lookup_only_field(line: &str) -> bool {
    let Some((path, _)) = line.split_once(": ") else { return false; };
    path.strip_prefix("material.inputs.")
        .is_some_and(|name| super::OpenPbrParam::supported().any(|p| p.name == name))
}

/// Ordinary authoring fields live on the API page; field lookup retains the full store.
pub fn api_field(line: &str) -> bool {
    let Some((path, _)) = line.split_once(": ") else { return false; };
    // Keep the starting page small. These supported fields remain in topic --all
    // and exact field lookup; this only controls documentation placement.
    if ["scene.propagation", "scene.simulation", "node.is_holdout", "node.is_shadow_catcher",
        "node.light_link_tags", "node.pass_index", "node.visible_diffuse", "node.visible_glossy",
        "node.visible_shadow", "node.visible_transmission", "node.instances", "node.illuminates",
        "node.shadowed_by", "node.image", "material.step", "material.aov", "render.aovs", "render.root",
        "render.motion_blur.position", "geometry", "buffer", "animation.layer", "animation.track",
        "material.attribute", "node.angle", "node.radius", "node.size_y", "node.spot_blend",
        "node.spot_size", "node.temperature", "image.time", "image.wrap_s", "image.wrap_t", "image.composition",
        "node.animation.layer"].iter().any(|prefix| path == *prefix ||
            path.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('.') || rest.starts_with(" ("))) {
        return false;
    }
    if path.starts_with("scene.") { return true; }
    if path.starts_with("node [") { return true; }
    if path.starts_with("node.") {
        // CSS/SVG properties keep their full contracts in field lookup and --all;
        // the root page names them once in the Text and Path blocks.
        if (path.ends_with(" (path)") || path.ends_with(" (text)"))
            && !matches!(path, "node.content (text)" | "node.font (text)" | "node.width (text)" | "node.fill (text)" | "node.d (path)" | "node.fill (path)") {
            return false;
        }
        // instances.count is a compute-written buffer, not an instance number; it lives in the nodes topic.
        return !["skin", "morph", "nodes", "instances.count"].iter().any(|part| path.contains(part))
            && (!path.ends_with(" (asset)") || path == "node.file (asset)");
    }
    if path.starts_with("material.") {
        if matches!(path, "material.attribute" | "material.texture") || path.starts_with("material.inputs.") { return false; }
        return !path.starts_with("material.grid") && !path.starts_with("material.pass") && !path.starts_with("material.channel")
            && (!path.contains(" (") || path == "material.inputs (std/openpbr)");
    }
    if path.starts_with("geometry.") {
        return !path.contains("morph") && !path.contains("crease_edge") && !path.contains("material_ranges.")
            && !path.contains("face_vertex_counts") && !path.starts_with("geometry.subdivision.");
    }
    if path.starts_with("buffer.") { return path != "buffer.grid"; }
    if path.starts_with("image.") { return !path.starts_with("image.render") && !path.ends_with("_filter"); }
    if path.starts_with("animation.layer.") { return path != "animation.layer.clip"; }
    if path.starts_with("animation.track.") { return !path.contains(' '); }
    if path.starts_with("render.") {
        return true;
    }
    matches!(path, "coordinates" | "names" | "paths" | "structure" | "normal maps" | "colour")
}

fn api_page_fields(store: &str) -> String {
    let groups = [
        ("Scene", "scene.", ""), ("Node", "node.", "common"),
        ("Mesh (type: mesh)", "node.", "mesh"),
        ("Asset (type: asset)", "node.", "asset"),
        ("Camera (type: camera)", "node.", "camera"),
        ("Light (type: light)", "node.", "light"),
        ("Text (type: text, canvas)", "node.", "text"),
        ("Path (type: path, canvas)", "node.", "path"),
        ("Material", "material.", ""),
        ("Texture binding", "material.texture.", ""),
        ("Attribute binding", "material.attribute.", ""),
        ("AOV binding", "material.aov.", ""), ("Geometry", "geometry.", ""),
        ("Buffer", "buffer.", ""), ("Image", "image.", ""),
        ("Layer", "animation.layer.", ""), ("Track", "animation.track.", ""),
        ("Render", "render.", ""),
    ];
    let mut result = String::new();
    for (title, prefix, variant) in groups {
        let group_start = result.len();
        result.push_str(&format!("\n{title} {{\n"));
        let fields_start = result.len();
        for line in store.lines().filter(|line| api_field(line)) {
            let (path, fact) = line.split_once(": ").unwrap();
            if path == "material.inputs (std/openpbr)" { continue; }
            let Some(name) = path.strip_prefix(prefix) else { continue; };
            if prefix == "material." && ["texture.", "attribute.", "aov."].iter().any(|p| name.starts_with(p)) { continue; }
            let (name, tag) = name.split_once(" (").map_or((name, ""), |(n,t)| (n,t.trim_end_matches(')')));
            if prefix == "node." && !(tag == variant || variant == "common" && tag.is_empty()) { continue; }
            let (fact, _) = fact.split_once("; when to change:").unwrap_or((fact, ""));
            let (fact, default) = fact.rsplit_once("; default ").map_or((fact, None), |(f,d)| (f, Some(d)));
            let (ty, note) = fact.split_once("; ").map_or((fact, ""), |(t,n)| (t,n));
            result.push_str(&format!("  {name}: {ty}"));
            if let Some(default) = default { result.push_str(&format!(" = {default}")); }
            // Standard names already specify these meanings and ranges.
            let note = match path {
                "material.double_sided" | "node.dof.aperture_ratio (camera)" | "node.spot_blend (light)"
                    | "geometry.subdivision.render_levels" | "material.program" => "",
                "geometry.subdivision" => "one material range",
                _ => note,
            };
            if !note.is_empty() { result.push_str(&format!("; {note}")); }
            result.push('\n');
        }
        if result.len() == fields_start { result.truncate(group_start); continue; }
        result.push_str("}\n");
        if title == "Material" {
            let line = store.lines().find(|line| line.starts_with("material.inputs (std/openpbr): ")).expect("OpenPBR inputs");
            result.push_str(&line.replace("material.inputs (std/openpbr): OpenPBR Surface; shaded inputs:", "std/openpbr inputs:"));
            result.push('\n');
        }
    }
    result
}

/// Compact authoring syntax, independent of the lowered Rust storage type.
/// Named objects stay named so resource definitions do not recursively expand at
/// each use. Structural containers and unions retain their actual JSON shape.
fn help_shape(schema: &serde_json::Value, value: &serde_json::Value) -> String {
    fn shape(schema: &serde_json::Value, value: &serde_json::Value, stack: &mut Vec<String>, depth: usize) -> String {
        if let Some(reference) = value["$ref"].as_str() {
            let name = reference.rsplit('/').next().unwrap_or("object");
            let Some(target) = schema.pointer(reference.trim_start_matches('#')) else { return name.into(); };
            if target.get("properties").is_some() { return format!("{name} object"); }
            if stack.iter().any(|seen| seen == reference) || depth >= 6 { return name.into(); }
            stack.push(reference.into());
            let result = shape(schema, target, stack, depth + 1);
            stack.pop();
            return result;
        }
        if depth >= 6 { return "…".into(); }
        if let Some(value) = value.get("const") { return value.to_string(); }
        if let Some(values) = value["enum"].as_array() {
            return values.iter().map(|value| value.to_string()).collect::<Vec<_>>().join("|");
        }
        for keyword in ["oneOf", "anyOf"] {
            if let Some(branches) = value[keyword].as_array() {
                let mut alternatives = Vec::new();
                for branch in branches {
                    let next = shape(schema, branch, stack, depth + 1);
                    if !alternatives.contains(&next) { alternatives.push(next); }
                }
                return alternatives.join("|");
            }
        }
        if let Some(types) = value["type"].as_array() {
            return types.iter().map(|kind| {
                let mut branch = value.clone();
                branch["type"] = kind.clone();
                shape(schema, &branch, stack, depth + 1)
            }).collect::<Vec<_>>().join("|");
        }
        match value["type"].as_str() {
            Some("object") | None if value.get("properties").is_some() => {
                // Tagged objects show their discriminator, never a fictitious wrapper.
                let tags = value["properties"].as_object().into_iter().flatten()
                    .filter_map(|(name, property)| property.get("const").map(|v| format!("\"{name}\":{v}")))
                    .collect::<Vec<_>>();
                if tags.is_empty() { "object".into() }
                else { format!("{{{}, …}}", tags.join(", ")) }
            }
            Some("object") => {
                match value.get("additionalProperties") {
                    Some(items) if items.is_object() => format!("{{name: {}}}", shape(schema, items, stack, depth + 1)),
                    _ => "object".into(),
                }
            }
            Some("array") => {
                if let Some(items) = value["prefixItems"].as_array() {
                    format!("[{}]", items.iter().map(|item| shape(schema, item, stack, depth + 1)).collect::<Vec<_>>().join(", "))
                } else {
                    let item = shape(schema, &value["items"], stack, depth + 1);
                    if let Some(count) = value["minItems"].as_u64().filter(|count| *count <= 8 && value["maxItems"].as_u64() == Some(*count)) {
                        format!("[{}]", vec![item; count as usize].join(", "))
                    } else if item.contains('|') {
                        format!("({item})[]")
                    } else {
                        format!("{item}[]")
                    }
                }
            }
            Some(kind) => kind.into(),
            None => "JSON value".into(),
        }
    }
    shape(schema, value, &mut Vec::new(), 0)
}

/// Use exactly the same author-schema annotations and wire-path presentation as
/// field facts. Shapes augment the advanced/exact store, not the default root.
fn help_shapes(schema: &serde_json::Value) -> std::collections::BTreeMap<String, String> {
    fn visit(schema: &serde_json::Value, value: &serde_json::Value, shapes: &mut std::collections::BTreeMap<String, String>) {
        if let Some(path) = value["x-help-path"].as_str().filter(|_| value["x-help"].is_string()) {
            shapes.insert(path.into(), help_shape(schema, value));
        }
        match value {
            serde_json::Value::Object(map) => for child in map.values() { visit(schema, child, shapes); },
            serde_json::Value::Array(array) => for child in array { visit(schema, child, shapes); },
            _ => {},
        }
    }
    let mut shapes = std::collections::BTreeMap::new();
    visit(schema, &schema["$defs"], &mut shapes);
    visit(schema, &schema["properties"], &mut shapes);
    let (fields, _) = tagged_help(schema);
    shapes.into_iter().map(|(path, shape)| {
        let line = present_help(&format!("{path}: {shape}"), &fields);
        let (path, shape) = line.split_once(": ").unwrap();
        (path.into(), shape.into())
    }).collect()
}

/// Checked-in documentation produced without requiring the TypeScript toolchain.
/// The schema with every object's keys in sorted order. serde_json iterates maps
/// in insertion order whenever any crate in the build enables its
/// `preserve_order` feature (the Moda crates do), and sorted otherwise; generated
/// documents must not depend on which packages a build happens to include.
fn sorted_keys(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(entries.into_iter().map(|(key, value)| (key, sorted_keys(value))).collect())
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(items.into_iter().map(sorted_keys).collect()),
        other => other,
    }
}

pub fn documents() -> Vec<(&'static str, String)> {
    let schema = sorted_keys(authored_schema().to_value());
    let reference = schema["x-vork-differences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|line| line.as_str().unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    let advanced = advanced_fields(&schema);
    let shapes = help_shapes(&schema);
    // One field store for `vork help TOPIC --all` and `vork help field PATH`:
    // advanced lines carry their tuning guidance in place of the plain line.
    let store = reference
        .lines()
        .map(|line| {
            advanced
                .iter()
                .find(|(path, _)| line.strip_prefix(path.as_str()).is_some_and(|rest| rest.starts_with(':') || rest.starts_with(" (")))
                .map_or(line, |(_, advanced)| advanced.as_str())
        })
        .collect::<Vec<_>>()
        .join("\n");
    let root_store = store.clone();
    let store = store.lines().map(|line| {
        let path = line.split_once(": ").map(|(path, _)| path).unwrap_or("");
        match shapes.get(path) {
            Some(shape) => format!("{line}; JSON shape: {shape}"),
            None => line.into(),
        }
    }).collect::<Vec<_>>().join("\n");
    let mut rows = Vec::new();
    fn fields(area: &str, object: &serde_json::Value, rows: &mut Vec<String>) {
        if let Some(properties) = object["properties"].as_object() {
            for (name, property) in properties {
                let required = object["required"]
                    .as_array()
                    .is_some_and(|fields| fields.iter().any(|f| f == name));
                let default = property
                    .get("const")
                    .or_else(|| property.get("default"))
                    .map(help_default)
                    .unwrap_or_else(|| if required { "required" } else { "omitted" }.into());
                let source = property["x-source"]
                    .as_str()
                    .unwrap_or("vork (structural contract)");
                rows.push(format!(
                    "| `{area}.{name}` | {} | {source} |",
                    default.replace('|', "\\|")
                ));
            }
        }
    }
    fields("scene", &schema, &mut rows);
    for (name, definition) in schema["$defs"].as_object().unwrap() {
        fields(name, definition, &mut rows);
        for keyword in ["oneOf", "anyOf"] {
            for branch in definition[keyword].as_array().into_iter().flatten() {
                for (tag, property) in branch["properties"].as_object().into_iter().flatten() {
                    if let Some(value) = property.get("const") {
                        let path = definition["x-help-path"].as_str().unwrap();
                        fields(&format!("{path} [{tag}={value}]"), branch, &mut rows);
                    }
                }
            }
        }
    }
    let index = include_str!("../../../../docs/help/index.md");
    let start = "<!-- generated-standards:start -->";
    let end = "<!-- generated-standards:end -->";
    let (before, rest) = index.split_once(start).expect("standards start marker");
    let (_, after) = rest.split_once(end).expect("standards end marker");
    let standards = schema["x-standards"]
        .as_object()
        .unwrap()
        .iter()
        // Hair and the camera JSON sidecar are documented in their own topics.
        .filter(|(area, _)| !matches!(area.as_str(), "hair" | "camera JSON"))
        .map(|(area, standard)| format!("{area}: follows {}", standard.as_str().unwrap()))
        .collect::<Vec<_>>()
        .join("\n");
    let index = format!("{before}{start}\n{standards}\n{end}{after}");
    let start = "<!-- generated-api:start -->";
    let end = "<!-- generated-api:end -->";
    let (before, rest) = index.split_once(start).expect("API start marker");
    let (_, after) = rest.split_once(end).expect("API end marker");
    let index = format!("{before}{start}\n{}{end}{after}", api_page_fields(&root_store));

    fn call_keys(text: &str) -> String {
        let (before, rest) = text.split_once("<!-- generated-call-keys:start -->").expect("call keys start");
        let (_, after) = rest.split_once("<!-- generated-call-keys:end -->").expect("call keys end");
        format!("{before}<!-- generated-call-keys:start -->\n--set KEY=VALUE accepts {}.\n<!-- generated-call-keys:end -->{after}", super::CallOverrides::KEYS.join(", "))
    }
    vec![
        ("docs/help/index.md", call_keys(&index)),
        ("docs/help/render.md", call_keys(include_str!("../../../../docs/help/render.md"))),
        ("packages/ts/src/lib/call-keys.ts", format!("// Generated from CallOverrides::KEYS; run scripts/generate-docs.\nexport const callKeys = {} as const;\n", serde_json::to_string(super::CallOverrides::KEYS).unwrap())),
        ("packages/python/vork/_call_keys.py", format!("# Generated from CallOverrides::KEYS; run scripts/generate-docs.\nCALL_KEYS = {}\n", serde_json::to_string(super::CallOverrides::KEYS).unwrap())),
        (
            "packages/ts/scene.schema.json",
            format!("{}\n", serde_json::to_string_pretty(&self::schema().to_value()).unwrap()),
        ),
        (
            "crates/vork-scene/src/v2/reference.txt",
            format!("{store}\n"),
        ),
        (
            "docs/DEFAULTS.md",
            format!(
                "# Defaults and standards\n\nGenerated from Rust scene types; run `scripts/generate-docs`. See [defaults policy](design/defaults.md) for precedence and rationale.\nStd compositor socket defaults: [COMPOSITOR.md](COMPOSITOR.md).\n\nRender calls default to `std/final`: scene-ray reflections, diffuse indirect light and physical shadows, with a smaller camera budget and bundled OIDN denoising by default. Retained probes are removed. Raster reconstruction preserves transmitting, strongly reflecting and mixed-surface pixels; scene effects requiring path transport retain the call's denoise intent at the same camera budget. `denoise=false` turns OIDN off and uses the profile reconstruction. Without a fast GPU denoiser, draft and final announce it and use 16 and 32 samples undenoised; explicit `denoise=true` uses the CPU denoiser. This affects glass, hair and interior lighting. `std/draft` uses OIDN at four samples; `std/final` uses eight; `--set samples=N` overrides the profile budget; `std/path` uses 64 samples independently of denoising; custom level-2 profiles without an explicit sample count use the same budget. Converged references must request `samples=4096`. `std/previs` is the explicit blocking-look exception: four samples, direct lights and ray shadows, environment reflections, no GI or probes, and no denoising by default; it does not estimate final transport. These quality-level choices are separate from the bare input defaults below.\n\nThe disk cache defaults to a 16 GiB ceiling, further limited to half of cache bytes plus available filesystem space (vork storage policy). `VORK_CACHE_BYTES` overrides the ceiling in bytes; `0` disables persistence. Invalid values use the default. `VORK_CACHE_DIR` overrides the platform cache directory. All artifact kinds share least-recently-used eviction at startup and before publication. Entries larger than the budget are not persisted.\n\nRaster quality levels use triangle visibility with physical emitter sampling shared\nwith path transport.\n\n`std/hair` evaluates Chiang scattering at every quality level. Raster no longer substitutes the EEVEE Color-only diffuse response, which discarded Melanin and Direct Absorption inputs. This changes raster hair appearance in `scenes/hair/`, curve hair and custom scenes using the bundled material; explicit custom diffuse Hair closures retain their meaning.\n\n<!-- generated-defaults:start -->\n| Field | Input default | Origin |\n|---|---|---|\n{}\n<!-- generated-defaults:end -->\n",
                rows.join("\n")
            ),
        ),
    ].into_iter()
        // The development-only defaults report is not part of the public help surface.
        .filter(|(path, _)| *path != "docs/DEFAULTS.md" || cfg!(vork_private_tests))
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn starter_schema_matches_release() {
        let starter: serde_json::Value = serde_json::from_str(crate::STARTER_SCENE).unwrap();
        assert_eq!(starter["$schema"], crate::SCHEMA_URL);
    }

    #[test]
    fn author_help_shapes_preserve_containers_and_bound_recursion() {
        use serde_json::json;
        let schema = json!({"$defs": {
            "Buffer": {"type":"object", "properties":{"data":{"type":"array","items":{"type":"number"}}}},
            "Resource": {"anyOf":[{"type":"string"},{"$ref":"#/$defs/Buffer"}]},
            "Dictionary": {"type":"object","additionalProperties":{"$ref":"#/$defs/Resource"}},
            "Cycle": {"$ref":"#/$defs/Cycle"}
        }});
        for (value, expected) in [
            (json!({"type":"object","additionalProperties":{"$ref":"#/$defs/Resource"}}), "{name: string|Buffer object}"),
            (json!({"$ref":"#/$defs/Resource"}), "string|Buffer object"),
            (json!({"$ref":"#/$defs/Dictionary"}), "{name: string|Buffer object}"),
            (json!({"anyOf":[{"$ref":"#/$defs/Resource"},{"type":"null"}]}), "string|Buffer object|null"),
            (json!({"type":"array","prefixItems":[{"type":"number"},{"type":"string"}],"items":false}), "[number, string]"),
            (json!({"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":3}), "[integer, integer, integer]"),
            (json!({"oneOf":[{"type":"object","properties":{"type":{"const":"mesh"},"geometry":{"$ref":"#/$defs/Resource"}}},{"type":"object","properties":{"type":{"const":"light"}}}]}), "{\"type\":\"mesh\", …}|{\"type\":\"light\", …}"),
            (json!({"$ref":"#/$defs/Cycle"}), "Cycle"),
        ] {
            assert_eq!(super::help_shape(&schema, &value), expected);
        }
        let mut deep = json!({"type":"number"});
        for _ in 0..20 { deep = json!({"type":"array","items":deep}); }
        assert!(super::help_shape(&schema, &deep).contains('…'));
    }

    #[test]
    fn generated_help_uses_authorable_resource_edges_without_growing_root() {
        let schema = super::authored_schema().to_value();
        let shapes = super::help_shapes(&schema);
        assert_eq!(shapes["program.bindings"], "{name: string|Buffer object}");
        assert_eq!(shapes["node.instances.attributes (mesh)"], "{name: string|Buffer object}");
        let documents = super::documents();
        let reference = &documents.iter().find(|(path, _)| *path == "crates/vork-scene/src/v2/reference.txt").unwrap().1;
        assert!(reference.lines().any(|line| line.starts_with("program.bindings: ") && line.ends_with("; JSON shape: {name: string|Buffer object}")));
        let index = &documents.iter().find(|(path, _)| *path == "docs/help/index.md").unwrap().1;
        assert!(!index.contains("JSON shape:"));
    }

    #[test]
    fn tagged_help_uses_wire_fields_and_discriminators_through_references() {
        let schema = super::authored_schema().to_value();
        let reference = super::reference(&schema);
        for line in [
            "node.geometry (mesh): geometry-name|inline geometry",
            "node.skin.joints (mesh): node-name[]",
            "node.focus_object (camera): ",
            "node.visible_camera: bool; default true",
        ] {
            assert!(
                reference.lines().any(|entry| entry.starts_with(line)),
                "missing {line}"
            );
        }
        assert!(!reference.contains("node.mesh."));
        assert!(!reference.contains("camera.lens:"));
        assert!(!reference.contains("node.asset."));
        assert!(!reference.lines().any(|line| line.starts_with("light.")));
        assert!(reference.contains("node.nodes.<name>.visible_camera (asset): bool; default true"));
        // Variant conditions print as a compact tag, and vanish when every integrator applies.
        assert!(!reference.contains("; when node.") && !reference.contains("; when settings."));
        assert!(!reference.lines().any(|line| line.starts_with("settings.") && line.contains(" (raster): ")));
        assert!(!reference.lines().any(|line| line.starts_with("settings.") && line.contains(" (path): ")));
        for (name, definition) in schema["$defs"].as_object().unwrap() {
            for keyword in ["oneOf", "anyOf"] {
                for branch in definition[keyword].as_array().into_iter().flatten() {
                    for (tag, property) in branch["properties"].as_object().into_iter().flatten() {
                        if let Some(value) = property.get("const") {
                            let path = definition["x-help-path"].as_str().unwrap();
                            assert!(
                                reference.contains(&format!("{path} [{tag}={value}]:")),
                                "missing {name} variant"
                            );
                        }
                    }
                }
            }
        }
        // Presentation metadata never broadens the accepted scene shape.
        assert!(
            crate::v2::from_test_value::<super::super::Node>(
                serde_json::json!({"type":"mesh", "mesh":{"geometry":"g"}})
            )
            .is_err()
        );
    }

    #[test]
    fn settings_schema_contains_only_public_intent() {
        let schema = super::schema().to_value();
        let settings = &schema["$defs"]["Settings"];
        assert_eq!(settings["additionalProperties"], false);
        assert!(settings.get("oneOf").is_none());
        assert_eq!(settings["properties"]["seed"]["default"], 0);
        assert_eq!(settings["properties"]["denoise"]["default"], false);
    }

    #[test]
    fn published_schemas_list_keys_in_sorted_order() {
        // Order must not depend on serde_json's preserve_order (enabled by any
        // crate in a build), or generated documents differ between builds.
        fn check(value: &serde_json::Value, path: &str) {
            match value {
                serde_json::Value::Object(map) => {
                    let keys: Vec<_> = map.keys().collect();
                    assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]), "{path}: {keys:?}");
                    for (key, value) in map { check(value, &format!("{path}/{key}")); }
                }
                serde_json::Value::Array(items) => for (index, item) in items.iter().enumerate() { check(item, &format!("{path}/{index}")); },
                _ => {}
            }
        }
        check(&super::schema().to_value(), "schema");
        check(&super::authored_schema().to_value(), "authored");
    }

    #[test]
    fn generated_documents_are_current() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (path, expected) in super::documents() {
            assert_eq!(
                std::fs::read_to_string(root.join(path)).unwrap(),
                expected,
                "{path} is stale; run scripts/generate-docs"
            );
        }
    }
    #[test]
    fn field_store_holds_every_reference_line_once_with_tuning_guidance() {
        let schema = super::authored_schema().to_value();
        let advanced = super::advanced_fields(&schema);
        let documents = super::documents();
        let store = &documents
            .iter()
            .find(|(path, _)| *path == "crates/vork-scene/src/v2/reference.txt")
            .unwrap()
            .1;
        let full = super::reference(&schema);
        assert_eq!(store.lines().count(), full.lines().count());
        for (path, line) in &advanced {
            assert!(store.lines().any(|entry| entry.split_once("; JSON shape:").map_or(entry, |(facts, _)| facts) == line), "missing advanced field: {path}: {line:?} vs {:?}",
                store.lines().find(|entry| entry.contains(line.split(';').nth(1).unwrap_or(line))));
            assert!(line.contains("; default "), "missing default: {path}");
            assert!(line.contains("; when to change: "), "missing guidance: {path}");
        }
        for removed in ["settings.shadow_resolution:", "settings.denoiser:"] {
            assert!(!store.contains(removed), "internal technique leaked: {removed}");
        }
        for path in [
            "node.translation",
            "node.rotation",
            "node.focus_object",
            "node.energy",
            "render.resolution",
            "settings.seed",
            "settings.samples",
            "settings.denoise",
        ] {
            assert!(
                store.lines().any(|line| line.strip_prefix(path).is_some_and(|rest| rest.starts_with(':') || rest.starts_with(" ("))),
                "missing field: {path}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "advanced fields need tuning guidance")]
    fn advanced_annotation_requires_guidance() {
        super::advanced_fields(&serde_json::json!({"properties": {"test": {
            "x-help-path": "test", "x-help": "number", "x-help-advanced": " "
        }}}));
    }

    #[test]
    fn image_has_its_own_render_schema() {
        let value = super::schema().to_value();
        assert!(value["$defs"].get("ImageRender").is_some());
        assert_eq!(value["properties"]["render"]["$ref"], "#/$defs/Render");
        assert_eq!(
            value["$defs"]["Image"]["properties"]["render"]["anyOf"][0]["$ref"],
            "#/$defs/ImageRender"
        );
        assert!(value["$defs"]["Render"]["properties"].get("root").is_some());
    }
}

#[cfg(test)]
mod variant_advanced_defaults {
    #[test]
    fn shared_help_retains_distinct_variant_defaults() {
        let mut schema=serde_json::json!({"$defs":{}});
        for (name,default) in [("local",serde_json::Value::Null),("spatial",serde_json::json!("random_walk"))] {
            schema["$defs"][name]=serde_json::json!({"x-help-path":"test.method","x-help":"method",
                "x-help-advanced":"choose transport", "default":default});
        }
        let help=super::advanced_fields(&schema);
        assert!(help["test.method"].contains("default null | random_walk"));
    }
}
