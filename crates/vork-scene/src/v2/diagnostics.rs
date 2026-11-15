//! Error-only diagnostics: successful scene decoding does not generate or walk a schema.
use anyhow::{Result, bail, ensure};
use serde_json::Value;
use std::collections::BTreeMap;

/// Edits between names, counting a swapped adjacent pair ("grb"/"rgb") as one.
pub fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<_>, Vec<_>) = (a.chars().collect(), b.chars().collect());
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 0..=a.len() { d[i][0] = i; }
    for j in 0..=b.len() { d[0][j] = j; }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]));
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

/// One step of an authored path: `parent.key` for an identifier key, `parent["key"]` (escaped) for any other,
/// the form inline resource paths write table keys in. A name with a dot or a line break stays one step on one line.
pub fn member(parent: &str, key: &str) -> String {
    let identifier = key.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !identifier { format!("{parent}[{key:?}]") } else if parent.is_empty() { key.to_owned() } else { format!("{parent}.{key}") }
}

/// Nearest spelling by Unicode edit distance; ties resolve lexically for stable errors.
/// Dotted paths compare segment by segment, so an omitted intermediate object costs
/// one (`render.shutter` finds `render.motion_blur.shutter`).
pub fn unknown<'a>(kind: &str, name: &str, valid: impl IntoIterator<Item = &'a str>) -> String {
    if kind == "field" && matches!(name, "preservesPitch" | "preserves_pitch") {
        return format!("unknown field {name:?}; use time.pitch: keep; see `vork help upgrade`");
    }
    if kind == "program" && let Some(message) = crate::libraries::removed_module(name) {
        return message;
    }
    // blender.nodes MathOperation: an author's own enum that still declares the old member never reaches here.
    let valid: Vec<&str> = valid.into_iter().collect();
    for (old, new) in [("FRACTION", "FRACT"), ("fraction", "fract")] {
        if matches!(kind, "member" | "token") && name == old && valid.contains(&new) {
            return format!("unknown {kind} {name:?}; it was renamed {new:?}; see `vork help upgrade`");
        }
    }
    if kind == "program" && matches!(name,
        "std/fog" | "blender/principled_volume" | "blender.volume" |
        "post/vignette" | "post/lens_distortion" | "post/glare" | "std/highlight") {
        return format!("unknown program {name:?}; this program was removed; see `vork help upgrade`");
    }
    // Error-only Blender vocabulary; these names are never accepted as aliases.
    const BLENDER_HINTS: &[(&str, &str, &str)] = &[
        ("field", "world", "environment light (nodes[].shape: environment)"),
        ("field", "location", "translation"),
        ("field", "hide_render", "visible (inverse of hide_render)"),
        ("field", "use_shadow", "shadowed_by: [] to disable shadows"),
        ("field", "film_transparent", "render.background: \"transparent\""),
        ("field", "resolution_percentage", "render.resolution (final pixel dimensions)"),
        ("field", "look", "composition.view with the std/agx or blender/filmic display program"),
        ("value", "Filmic", "composition.view with the blender/filmic display program"),
        ("value", "ORTHO", "ortho (lowercase)"),
        ("field", "extension", "wrap_s/wrap_t on the image"),
        ("program", "blender/principled", "std/openpbr"),
    ];
    if let Some((_, _, hint)) = BLENDER_HINTS.iter().find(|(k, n, _)| *k == kind && *n == name) {
        return format!("unknown {kind} {name:?}; use {hint}{}",
            if kind == "program" { "; see `vork help upgrade`" } else { "" });
    }
    fn path_distance(a: &str, b: &str) -> usize {
        if !a.contains('.') && !b.contains('.') {
            return distance(a, b);
        }
        let b: Vec<_> = b.split('.').collect();
        let mut row: Vec<_> = (0..=b.len()).collect();
        for a in a.split('.') {
            let mut diagonal = row[0];
            row[0] += a.chars().count().max(1);
            for (j, b) in b.iter().enumerate() {
                let above = row[j + 1];
                row[j + 1] = (above + a.chars().count().max(1))
                    .min(row[j] + 1)
                    .min(diagonal + distance(a, b));
                diagonal = above;
            }
        }
        row[b.len()]
    }
    let distance = path_distance;
    let mut valid: Vec<_> = valid.into_iter().collect();
    valid.sort_unstable();
    valid.dedup();
    valid.sort_by_cached_key(|candidate| (distance(name, candidate), *candidate));
    let total = valid.len();
    let shown = valid
        .iter()
        .take(super::constants::SUGGESTION_DISPLAY_LIMIT)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    let label = if total > super::constants::SUGGESTION_DISPLAY_LIMIT {
        format!(
            "Nearest valid names ({} of {total})",
            super::constants::SUGGESTION_DISPLAY_LIMIT
        )
    } else {
        "Valid names".to_owned()
    };
    // Error-only guidance for the Blender RNA spelling; never accepted by serde
    // or track resolution. Edit distance cannot connect it to emitter radius.
    if name == "shadow_soft_size" && valid.contains(&"radius") {
        return format!("unknown {kind} {name:?}; did you mean \"radius\"? {label}: {shown}");
    }
    match valid.first() {
        Some(nearest)
            if distance(name, nearest)
                <= super::constants::SUGGESTION_MIN_DISTANCE
                    .max(name.chars().count() / super::constants::SUGGESTION_LENGTH_DIVISOR) =>
        {
            format!("unknown {kind} {name:?}; did you mean {nearest:?}? {label}: {shown}")
        }
        Some(_) => format!("unknown {kind} {name:?}; {label}: {shown}"),
        None => format!("unknown {kind} {name:?}; no valid names are declared"),
    }
}

/// JSON text from users without a leading UTF-8 byte-order mark.
pub fn without_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)
}

/// A user's JSON document as a value: without a byte-order mark, and with the reader's nesting limit said
/// in scene terms. Every loader that reads authored scene text before expanding it starts here.
pub fn read_document(bytes: &[u8]) -> anyhow::Result<Value> {
    serde_json::from_slice(without_bom(bytes)).map_err(|error| nesting_error(&error, "scene").unwrap_or_else(|| error.into()))
}

/// The JSON reader admits [`super::MAX_JSON_DEPTH`] nested arrays and objects. A node's children nest two
/// each, so a hierarchy about 62 nodes deep reaches it: say that, not only the reader's words. `None` for
/// any other error.
pub fn nesting_error(error: &serde_json::Error, what: &str) -> Option<anyhow::Error> {
    error.to_string().contains("recursion limit exceeded").then(|| anyhow::anyhow!(
        "the {what} nests arrays and objects more than {} deep (about 62 nested nodes); flatten the hierarchy ({error})",
        super::MAX_JSON_DEPTH))
}

pub(super) fn decode_error(bytes: &[u8], original: serde_json::Error) -> anyhow::Error {
    if let Ok(mut value) = serde_json::from_slice::<Value>(bytes) {
        value.sort_all_objects();
        if value.get("animation").and_then(Value::as_array)
            .is_some_and(|layers| layers.iter().any(|layer| layer.get("blend").and_then(Value::as_str) == Some("add"))) {
            return anyhow::anyhow!("animation.layer.blend: use \"combine\" instead of \"add\"; see `vork help upgrade`");
        }
        // The walker expects shared fields on the parent (see inspect).
        let schema = super::schema::full_schema().to_value();
        if let Err(error) = inspect(&value, &schema, &schema, "scene") {
            return error;
        }
    }
    original.into()
}

pub(super) fn profile_error(bytes: &[u8], original: serde_json::Error) -> anyhow::Error {
    if let Ok(mut value) = serde_json::from_slice::<Value>(bytes) {
        value.sort_all_objects();
        let schema = super::schema::authored_schema().to_value();
        if let Err(error) = inspect(&value, &schema["$defs"]["Profile"], &schema, "profile") {
            return error;
        }
    }
    original.into()
}

fn resolve<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|r| r.strip_prefix('#'))
        .and_then(|p| root.pointer(p))
        .unwrap_or(schema)
}
fn accepts(value: &Value, schema: &Value, root: &Value) -> bool {
    let schema = resolve(schema, root);
    let matches = |kind: &str| match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    };
    match schema.get("type") {
        Some(Value::String(kind)) => matches(kind),
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).any(matches),
        _ => true,
    }
}
fn inspect(value: &Value, schema: &Value, root: &Value, path: &str) -> Result<()> {
    let schema = resolve(schema, root);
    if schema.get("x-clock").and_then(Value::as_bool)==Some(true) {
        // Output rates cross the i64 ABI; authored clocks retain arbitrary precision.
        if schema.get("x-output-rate").and_then(Value::as_bool)==Some(true) {
            serde_json::from_value::<super::Time>(value.clone()).map_err(|error|anyhow::anyhow!("{path}: {error}"))?;
        } else {
            serde_json::from_value::<vork_time::Clock>(value.clone()).map_err(|error|anyhow::anyhow!("{path}: {error}"))?;
        }
    }
    if let Some(variants) = schema.get("enum").and_then(Value::as_array) {
        if !variants.contains(value) {
            let name = value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string());
            bail!("{path}: {}", unknown("value", &name, variants.iter().filter_map(Value::as_str)));
        }
    }
    ensure!(accepts(value, schema, root), "{path}: expected {}, got {}",
        schema.get("type").unwrap_or(&Value::Null), value);
    if let Some(expected) = schema.get("minItems").and_then(Value::as_u64) {
        ensure!(value.as_array().is_none_or(|a| a.len() >= expected as usize),
            "{path}: expected at least {expected} items");
    }
    if let Some(expected) = schema.get("maxItems").and_then(Value::as_u64) {
        ensure!(value.as_array().is_none_or(|a| a.len() <= expected as usize),
            "{path}: expected at most {expected} items");
    }
    // Node is a flattened tagged union: combine common properties with its selected kind.
    let mut properties = BTreeMap::new();
    if let Some(fields) = schema.get("properties").and_then(Value::as_object) {
        properties.extend(fields.iter().map(|(k, v)| (k.as_str(), v)));
    }
    for union in ["oneOf", "anyOf"] {
        if let Some(branches) = schema.get(union).and_then(Value::as_array) {
            let tag = if branches.iter().any(|b| b.pointer("/properties/std/const").is_some()) { "std" }
                else if branches.iter().any(|b| b.pointer("/properties/integrator/const").is_some()) { "integrator" } else { "type" };
            let tag_pointer = format!("/properties/{tag}/const");
            let tagged: Vec<_> = branches
                .iter()
                .filter_map(|b| b.pointer(&tag_pointer).and_then(Value::as_str))
                .collect();
            if !tagged.is_empty() {
                // Only a branch with an optional discriminator supplies a default.
                let kind = value.get(tag).and_then(Value::as_str)
                    .or_else(|| value.get(tag).is_none().then(|| branches.iter().find(|b|
                        !b.get("required").and_then(Value::as_array).is_some_and(|r| r.iter().any(|v| v == tag)))
                        .and_then(|b| b.pointer(&tag_pointer).and_then(Value::as_str))).flatten());
                if kind.is_none() {
                    bail!(
                        "{path}: missing or invalid {tag:?}; choose {}",
                        tagged.join(", ")
                    );
                }
                if let Some(kind) = kind {
                    let Some(branch) = branches.iter().find(|b| {
                        b.pointer(&tag_pointer).and_then(Value::as_str) == Some(kind)
                    }) else {
                        bail!("{path}.{tag}: {}", unknown(if tag == "type" { "node kind" } else { "value" }, kind, tagged));
                    };
                    if let Some(fields) = branch.get("properties").and_then(Value::as_object) {
                        properties.extend(fields.iter().map(|(k, v)| (k.as_str(), v)));
                    }
                }
            } else if properties.is_empty() {
                // Source-exclusive oneOf constraints repeat properties; the parent handles them.
                let mut candidates: Vec<_> = branches.iter().filter(|b| accepts(value, b, root)).collect();
                if let Some(object) = value.as_object() {
                    let fields: std::collections::BTreeSet<_> = candidates.iter()
                        .filter_map(|branch| resolve(branch, root).get("properties").and_then(Value::as_object))
                        .flat_map(|fields| fields.keys().map(String::as_str)).collect();
                    let open = candidates.iter().any(|branch| resolve(branch, root)
                        .get("additionalProperties").is_some_and(|v| v != &Value::Bool(false)));
                    // Nested alternatives validate their own fields recursively. Their
                    // properties are not visible in this direct-field union.
                    let nested = candidates.iter().any(|branch| {
                        let branch = resolve(branch, root);
                        branch.get("oneOf").is_some() || branch.get("anyOf").is_some()
                    });
                    if !open && !nested && !fields.is_empty() {
                        for key in object.keys().filter(|key| !fields.contains(key.as_str())) {
                            bail!("{path}: {}", unknown("field", key, fields.iter().copied()));
                        }
                    }
                }
                // Untagged object alternatives may share a JSON type. Prefer the
                // branch with the author's keys, rather than naming a sibling's fields.
                candidates.sort_by_key(|branch| {
                    let branch = resolve(branch, root);
                    let matched = value.as_object().zip(branch.get("properties").and_then(Value::as_object))
                        .map_or(0, |(object, fields)| object.keys().filter(|key| fields.contains_key(*key)).count());
                    std::cmp::Reverse(matched)
                });
                let mut first_error = None;
                for branch in candidates {
                    match inspect(value, branch, root, path) {
                        Ok(()) => return Ok(()),
                        Err(error) if first_error.is_none() => first_error = Some(error),
                        Err(_) => {}
                    }
                }
                if let Some(error) = first_error { return Err(error); }
                bail!("{path}: value does not match any permitted type");
            }
        }
    }
    // Sibling fields and items are independent: report every one that fails.
    let mut errors = Vec::new();
    if let Some(object) = value.as_object() {
        let additional = schema.get("additionalProperties").filter(|v| v.is_object());
        for (name, value) in object {
            if let Some(field) = properties.get(name.as_str()) {
                errors.extend(inspect(value, field, root, &format!("{path}.{name}")).err());
            } else if let Some(additional) = additional {
                errors.extend(inspect(value, additional, root, &format!("{path}.{name}")).err());
            } else if !properties.is_empty()
                || schema.get("additionalProperties") == Some(&Value::Bool(false))
            {
                if let Some(branches) = schema.get("oneOf").and_then(Value::as_array) {
                    let owners: Vec<_> = branches.iter().filter(|b| b["properties"].get(name).is_some())
                        .filter_map(|b| b.pointer("/properties/type/const").and_then(Value::as_str)).collect();
                    if !owners.is_empty() {
                        errors.push(anyhow::anyhow!("{path}: unknown field {name:?}; {name} belongs to type {}", owners.join(" or ")));
                        continue;
                    }
                }
                errors.push(anyhow::anyhow!(
                    "{path}: {}",
                    unknown("field", name, properties.keys().copied())
                ));
            }
        }
    }
    if let (Some(array), Some(items)) = (value.as_array(), schema.get("prefixItems").and_then(Value::as_array)) {
        for (i, (value, item)) in array.iter().zip(items).enumerate() {
            errors.extend(inspect(value, item, root, &format!("{path}[{i}]")).err());
        }
    }
    if let (Some(array), Some(items)) = (value.as_array(), schema.get("items")) {
        for (i, value) in array.iter().enumerate() {
            errors.extend(inspect(value, items, root, &format!("{path}[{i}]")).err());
        }
    }
    joined(errors)
}
/// One error listing every problem, one per line, in document order.
pub fn joined(errors: Vec<anyhow::Error>) -> Result<()> {
    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.into_iter().next().unwrap()),
        _ => {
            let messages = errors.iter().map(|e| format!("{e:#}")).collect::<Vec<_>>().join("\n");
            // Keep the primary typed cause (and its CLI category), as a
            // single-error result does; the context lists every sibling.
            Err(errors.into_iter().next().unwrap().context(messages))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joined_diagnostics_preserve_primary_typed_cause() {
        let error = joined(vec![
            anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound)).context("first file"),
            anyhow::anyhow!("second file"),
        ]).unwrap_err();
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
        let text = format!("{error:#}");
        assert!(text.contains("first file") && text.contains("second file"), "{text}");
    }

    #[test]
    fn decimal_timestep_does_not_mask_invalid_field() {
        let bytes = br#"{"version":2,"nodes":[{"type":"light","shape":"rectangle","size":[0.16,0.22]}],"simulation":{"dt":0.0020833333333333333}}"#;
        let error = super::super::Scene::from_json(bytes).unwrap_err().to_string();
        assert!(error.contains("scene.nodes[0].size: expected"), "{error}");
        assert!(!error.contains("output rate"), "{error}");
    }

    #[test]
    fn real_output_rate_overflow_keeps_its_diagnostic() {
        let bytes = br#"{"version":2,"nodes":[],"render":{"fps":0.0020833333333333333}}"#;
        let error = super::super::Scene::from_json(bytes).unwrap_err().to_string();
        assert!(error.contains("output rate denominator exceeds i64 ABI"), "{error}");
    }

    #[test]
    fn obsolete_light_radius_field_is_rejected() {
        let bytes = br#"{"version":2,"nodes":[{"type":"light","shape":"point","shadow_soft_size":0.2}]}"#;
        let error = super::super::Scene::from_json(bytes).unwrap_err().to_string();
        assert!(error.contains("unknown field") && error.contains("did you mean \"radius\""), "{error}");
    }
    #[test]
    fn short_typos_suggest_and_far_names_list_choices() {
        assert!(unknown("value", "paht", ["path", "raster"]).contains("did you mean \"path\""));
        // A swap is one edit, so short tokens find their transposition, not a shorter neighbour.
        assert!(unknown("channel", "grb", ["r", "g", "b", "a", "rgb"]).contains("did you mean \"rgb\""));
        assert!(unknown("buffer type", "ve3c", ["vec2", "vec3", "vec4"]).contains("did you mean \"vec3\""));
        assert_eq!(unknown("field", "zzzzzzzzzzzz", ["samples", "integrator"]),
            "unknown field \"zzzzzzzzzzzz\"; Valid names: integrator, samples");
    }
    #[test]
    fn dotted_paths_rank_by_segments_and_find_nested_fields() {
        let valid = ["render.camera", "render.root", "render.motion_blur.shutter", "render.fps", "image.render"];
        let message = unknown("field", "render.shutter", valid);
        assert!(message.contains("did you mean \"render.motion_blur.shutter\"?"), "{message}");
        assert!(unknown("field", "rendr.fps", valid).contains("did you mean \"render.fps\"?"));
    }
    #[test]
    fn removed_light_shadow_reports_valid_names() {
        let bytes = br#"{"version":2,"nodes":[{"type":"light","shape":"sun","shadow":{}}]}"#;
        let original = serde_json::from_slice::<super::super::Scene>(bytes).unwrap_err();
        let error = decode_error(bytes, original).to_string();
        assert!(error.contains("unknown field \"shadow\""), "{error}");
        assert!(error.contains("Nearest valid names"), "{error}");
    }

    #[test]
    fn failed_decode_suggests_fields_and_values_through_tagged_and_untagged_types() {
        let value = serde_json::json!({"version":2,"world":{},"nodes":[{"type":"camera"}]});
        let bytes = serde_json::to_vec(&value).unwrap();
        let original = serde_json::from_slice::<super::super::Scene>(&bytes).unwrap_err();
        let error = decode_error(&bytes, original).to_string();
        assert!(error.contains("environment light (nodes[].shape: environment)"), "{error}");
        for (scene, expected) in [
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","focus_objec":"subject","material":{"program":"std/camera","inputs":{"projection":"persp"}}}]}),
                "focus_object",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camra"}]}),
                "camera",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","visible_shdow":false,"material":{"program":"std/camera","inputs":{"projection":"persp"}}}]}),
                "visible_shadow",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"materials":{"m":{"inputs":{"base_color":{"image":"i","chnnel":"r"}}}}}),
                "channel",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"buffers":{"p":{"type":"vec3","data":[],"offest":0}}}),
                "offset",
            ),
        ] {
            let bytes = serde_json::to_vec(&scene).unwrap();
            let original = serde_json::from_slice::<super::super::Scene>(&bytes).unwrap_err();
            let error = decode_error(&bytes, original).to_string();
            assert!(
                error.contains(&format!("did you mean {expected:?}")),
                "{error}"
            );
            assert!(error.contains("scene"), "{error}");
        }
    }
    #[test]
    fn missing_targets_and_naming_rules_are_specific() {
        for (scene, expected) in [
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam/one","material":{"program":"std/camera","inputs":{"projection":"persp"}}}]}),
                "'/' is not allowed",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"render":{"camera":"camra"}}),
                "did you mean \"camera\"",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"animation":[{"tracks":[{"path":["nodes","camra","translation"],"keys":[[0,[0,0,0]]]}]}]}),
                "segment 1: unknown node \"camra\"; did you mean \"camera\"",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"materials":{"m":{"inputs":{"base_colr":[1,1,1]}}}}),
                "did you mean \"base_color\"",
            ),
        ] {
            let scene: super::super::Scene = crate::v2::from_test_value(scene).unwrap();
            let error = format!("{:#}", scene.validate().unwrap_err());
            assert!(error.contains(expected), "{error}");
        }
    }
    #[test]
    fn suggestions_are_unicode_safe_and_stable() {
        assert!(unknown("name", "éa", ["éb", "éc"]).contains("did you mean \"éb\""));
        assert!(unknown("name", "missing", []).contains("no valid names"));
    }
}

#[cfg(test)]
mod review_regressions {
    use super::super::Scene;
    use super::*;
    #[test]
    fn group_fields_name_the_owning_type_and_invalid_tags_still_error() {
        for (field, owner) in [("geometry", "mesh"), ("focus_object", "camera"), ("energy", "light"), ("file", "asset")] {
            let value = serde_json::json!({"version":2,"nodes":[{field:"unexpected"}]});
            let error = Scene::from_json(&serde_json::to_vec(&value).unwrap()).unwrap_err().to_string();
            assert!(error.contains(&format!("{field} belongs to type {owner}")), "{error}");
        }
        let error = Scene::from_json(br#"{"version":2,"nodes":[{"type":"meshh"}]}"#).unwrap_err().to_string();
        assert!(error.contains("meshh") && error.contains("mesh"), "{error}");
    }
    #[test]
    fn std_geometry_is_rejected_by_kernel_diagnostics() {
        let bytes = br#"{"version":2,"nodes":[{"type":"camera"}],"geometry":{"log":{"std":"rounded_box","size":[0.21,1.2,0.21],"radius":0.105,"segments":24}}}"#;
        let error = Scene::from_json(bytes).unwrap_err().to_string();
        assert!(error.contains("std") && error.contains("unknown field"), "{error}");
    }
    #[test]
    fn omitted_type_preserves_unknown_field_errors() {
        let bytes = br#"{"version":2,"nodes":[{"name":"cam","focal_length":50}]}"#;
        let original = serde_json::from_slice::<Scene>(bytes).unwrap_err();
        let error = decode_error(bytes, original).to_string();
        assert!(error.contains("unknown field"), "{error}");
    }
    #[test]
    fn resolve_morphs_from_geometry_and_distinguish_structure_and_null_values() {
        let base = serde_json::json!({"version":2,"buffers":{"p":{"type":"vec3","data":[[0,0,0]]}},"geometry":{"g":{"attributes":{"POSITION":"p"},"morphs":{"smile":{"POSITION":"p"}}}},"nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"mesh","name":"face","geometry":"g"}],"animation":[{"tracks":[{"path":["nodes","face","morph_weights","smile"],"keys":[[0,1]]}]}]});
        crate::v2::from_test_value::<Scene>(base.clone())
            .unwrap()
            .validate()
            .unwrap();
        for (path, expected) in [
            (
                &["nodes", "face", "geometry"][..],
                "segment 2: unknown keyable node property",
            ),
            (&["nodes", "face", "light_link_tags", "0"][..], "segment 2"),
            (
                &["render", "samples"][..],
                "segment 1: unknown keyable render property",
            ),
            (&["render", "resolution", "0"][..], "segment 1"),
            (&["nodes", "cam", "shutter", "angle"][..], "segment 2"),
            (
                &["nodes", "face", "morph_weights", "smil"][..],
                "did you mean \"smile\"",
            ),
        ] {
            let mut scene = base.clone();
            scene["animation"][0]["tracks"][0]["path"] = serde_json::json!(path);
            let error = format!(
                "{:#}",
                crate::v2::from_test_value::<Scene>(scene)
                    .unwrap()
                    .validate()
                    .unwrap_err()
            );
            assert!(error.contains(expected), "{error}");
        }
    }
    #[test]
    fn reject_versions_layer_names_and_ambiguous_clip_selectors() {
        for (value, expected) in [
            (
                serde_json::json!({"version":3,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}]}),
                "only scene version 2",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"animation":[{"name":"walk/fast","tracks":[]}]}),
                "'/' is not allowed",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"animation":[{"clip":{"target":"cam"}}]}),
                "must name an asset node",
            ),
            (
                serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"animation":[{"clip":{"target":"cam","name":"Walk","index":0}}]}),
                "at most one of name or index",
            ),
        ] {
            let error = format!(
                "{:#}",
                crate::v2::from_test_value::<Scene>(value)
                    .unwrap()
                    .validate()
                    .unwrap_err()
            );
            assert!(error.contains(expected), "{error}");
        }
        assert!(!unknown("field", "zzzz", ["geometry"]).contains("did you mean"));
    }
}

#[cfg(test)]
mod final_review_regressions {
    use super::super::Scene;
    use super::*;
    #[test]
    fn candidate_lists_are_bounded() {
        let names = (0..100).map(|i| format!("node_{i}")).collect::<Vec<_>>();
        let error = unknown("node", "node_1", names.iter().map(String::as_str));
        assert!(error.contains("5 of 100"), "{error}");
        assert!(error.len() < 180, "{error}");
    }
    #[test]
    fn clip_sources_layer_order_cubic_tangents_and_ids_have_specific_errors() {
        let base = serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"asset","name":"fox","file":"fox.glb"}]});
        for (animation, expected) in [
            (
                serde_json::json!([{"clip":{"target":"cam"}}]),
                "must name an asset node",
            ),
            (
                serde_json::json!([{"clip":{"target":"fox","name":"Walk","index":0}}]),
                "at most one of name or index",
            ),
            (
                serde_json::json!([{"tracks":[{"path":["nodes","cam","dof","aperture_fstop"],"keys":[[0,2]]}]}]),
                "unknown keyable node property",
            ),
            (
                serde_json::json!([{"name":"walk","clip":{"target":"fox"}},{"tracks":[{"path":["animation","walk","weight"],"keys":[[0,1]]}]}]),
                "from a lower layer",
            ),
        ] {
            let mut scene = base.clone();
            scene["animation"] = animation;
            let error = format!(
                "{:#}",
                crate::v2::from_test_value::<Scene>(scene)
                    .unwrap()
                    .validate()
                    .unwrap_err()
            );
            assert!(error.contains(expected), "{error}");
        }
        let mut scene = base;
        scene["nodes"][0]["pass_index"] = 0.into();
        assert!(
            crate::v2::from_test_value::<Scene>(scene)
                .unwrap()
                .validate()
                .unwrap_err()
                .to_string()
                .contains("reserved for background")
        );
    }
}

#[cfg(test)]
mod field_path_regressions {
    use super::*;
    fn diagnose(scene: &serde_json::Value) -> String {
        let bytes = serde_json::to_vec(scene).unwrap();
        let original = serde_json::from_slice::<super::super::Scene>(&bytes).unwrap_err();
        decode_error(&bytes, original).to_string()
    }
    #[test]
    fn malformed_values_name_their_actual_field() {
        for (value, path) in [
            (serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"render":{"background":[1,"bad",0]}}), "scene.render.background[1]"),
            (serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"render":{"fps":"fast"}}), "scene.render.fps"),
            (serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"materials":{"m":{"inputs":{"base_color":{"attribute":"COLOR_0","channel":"invalid"}}}}}), "scene.materials.m.inputs.base_color.channel"),
        ] {
            let error = diagnose(&value);
            assert!(error.contains(path), "{error}");
        }
    }
    #[test]
    fn source_alternatives_report_only_the_authored_fields_problem() {
        let file_buffer = |key: &str, value: serde_json::Value| {
            let mut buffer = serde_json::json!({"type":"u32","file":"geometry.bin","offset":0,"count":6});
            buffer[key] = value;
            serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"buffers":{"b":buffer}})
        };
        for (scene, expected) in [
            (file_buffer("type", "zz_bogus".into()), "scene.buffers.b.type: unknown value \"zz_bogus\""),
            (file_buffer("offset", 1.5.into()), "scene.buffers.b.offset: expected \"integer\", got 1.5"),
            (file_buffer("count", 1.5.into()), "scene.buffers.b.count: expected [\"integer\",\"null\"], got 1.5"),
            (serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"buffers":{"b":{"type":"zz_bogus","data":[1]}}}),
                "scene.buffers.b.type: unknown value \"zz_bogus\""),
            (serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"animation":[{"clip":{"target":"a"},"weight":"heavy"}]}),
                "scene.animation[0].weight: expected"),
        ] {
            let error = diagnose(&scene);
            assert!(error.starts_with(expected) && !error.contains('\n'), "{error}");
        }
    }
}

#[cfg(test)]
mod all_errors {
    #[test]
    fn decode_and_openpbr_errors_list_every_problem() {
        let bytes = serde_json::to_vec(&serde_json::json!({"version":2,
            "nodes":[{"type":"camera","focus_objec":"subject","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"light","shape":"sun","powr":3}],
            "render":{"fps":"fast"}})).unwrap();
        let original = serde_json::from_slice::<super::super::Scene>(&bytes).unwrap_err();
        let error = super::decode_error(&bytes, original).to_string();
        for expected in ["scene.nodes[0]", "\"focus_object\"", "scene.nodes[1]", "powr", "scene.render.fps"] {
            assert!(error.contains(expected), "{expected} missing from {error}");
        }
        let scene: super::super::Scene = crate::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "materials":{"a":{"inputs":{"metallic":1,"roughness":0.2}},"b":{"inputs":{"transmission":1}}}})).unwrap();
        let error = format!("{:#}", scene.validate().unwrap_err());
        for expected in ["base_metalness", "specular_roughness", "material b", "transmission_weight"] {
            assert!(error.contains(expected), "{expected} missing from {error}");
        }
    }
}

#[cfg(test)]
mod blender_hints {
    use super::*;

    #[test]
    fn blender_names_explain_the_vork_contract() {
        for (kind, name, valid, expected) in [
            ("field", "location", "translation", "translation"),
            ("field", "hide_render", "visible", "inverse of hide_render"),
            ("field", "use_shadow", "shadowed_by", "shadowed_by: []"),
            ("field", "film_transparent", "background", "render.background: \"transparent\""),
            ("field", "resolution_percentage", "resolution", "render.resolution"),
            ("field", "look", "view", "std/agx or blender/filmic"),
            ("value", "Filmic", "AgX", "blender/filmic"),
            ("value", "ORTHO", "ortho", "ortho (lowercase)"),
            ("field", "extension", "wrap_s", "wrap_s/wrap_t"),
        ] {
            let error = unknown(kind, name, [valid]);
            assert!(error.contains(expected), "{error}");
            assert!(error.starts_with(&format!("unknown {kind}")), "{error}");
        }
        assert!(!unknown("node", "location", ["camera"]).contains("translation"));
    }

    #[test]
    fn decoder_routes_blender_fields_and_values_to_hints() {
        for (scene, expected) in [
            (serde_json::json!({"nodes":[{"type":"camera","location":[0,0,0]}]}), "use translation"),
            (serde_json::json!({"nodes":[{"type":"camera","hide_render":true}]}), "use visible"),
            (serde_json::json!({"nodes":[{"type":"light","shape":"sun","use_shadow":false}]}), "shadowed_by: []"),
            (serde_json::json!({"render":{"film_transparent":true}}), "render.background"),
            (serde_json::json!({"render":{"resolution_percentage":50}}), "render.resolution"),
            (serde_json::json!({"render":{"look":"Filmic"}}), "std/agx or blender/filmic"),
            (serde_json::json!({"images":{"i":{"file":"i.png","extension":"REPEAT"}}}), "wrap_s/wrap_t"),
        ] {
            let mut scene = scene;
            scene["version"] = 2.into();
            if scene.get("nodes").is_none() { scene["nodes"] = serde_json::json!([]); }
            let bytes = serde_json::to_vec(&scene).unwrap();
            let original = serde_json::from_slice::<super::super::Scene>(&bytes).unwrap_err();
            let error = decode_error(&bytes, original).to_string();
            assert!(error.contains(expected), "{error}");
        }
    }
}

#[cfg(test)]
mod upgrade_tests {
    #[test]
    fn removed_programs_point_to_upgrade_without_accepting_aliases() {
        for name in ["std/fog", "blender/principled_volume", "blender.volume", "post/vignette", "post/lens_distortion", "post/glare", "std/highlight"] {
            assert!(super::unknown("program", name, ["std/openpbr"]).contains("vork help upgrade"));
        }
        assert!(!super::unknown("program", "std/typo", ["std/openpbr"]).contains("was removed"));
    }
}

/// Resolve normalized resource identifiers in preparation and validation failures.
/// Origins are runtime metadata and do not participate in resource/cache identity.
pub fn authored_error(origins: &BTreeMap<String, Vec<String>>, error: anyhow::Error) -> anyhow::Error {
    let original = format!("{error:#}");
    let mut text = original.clone();
    let mut locations = Vec::new();
    fn mentions(text: &str, label: &str) -> bool {
        text.match_indices(label).any(|(i, _)| text[i + label.len()..].chars().next()
            .is_none_or(|c| !c.is_alphanumeric() && !matches!(c, '_' | '-' | '/' | '.')))
    }
    for (key, paths) in origins {
        let (table, name) = key.split_once('.').unwrap();
        let kind = table.trim_end_matches('s');
        let unique_generated = name.starts_with('~') && origins.keys()
            .filter(|key| key.split_once('.').is_some_and(|(_, n)| n == name)).count() == 1;
        let relevant = unique_generated && mentions(&text, name)
            || mentions(&text, key)
            || [format!("{kind} {name}"), format!("{kind} {name:?}")].iter().any(|label| mentions(&text, label));
        if relevant {
            let users = paths.join(", ");
            if name.starts_with('~') {
                // Compiler virtual filenames include the scene directory. Replace
                // that entire filename before expanding its resource key.
                if table == "programs" {
                    for path in paths {
                        if let Some((file, _)) = path.split_once(": scene.") {
                            let parent = std::path::Path::new(file).parent().unwrap_or(std::path::Path::new(""));
                            let virtual_file = parent.join(key).to_string_lossy().into_owned();
                            text = text.replace(&virtual_file, &users).replace(&virtual_file.replace('\\', "/"), &users);
                        }
                    }
                }
                text = text.replace(key, &users)
                    .replace(&format!("{kind} {name:?}"), &format!("{kind} {users}"))
                    .replace(&format!("{kind} {name}"), &format!("{kind} {users}"));
                if unique_generated { text = text.replace(name, &users); }
            } else { locations.push(format!("{table}: {users}")); }
        }
    }
    if text == original {
        if locations.is_empty() { error } else { error.context(locations.join("; ")) }
    } else {
        let text = if locations.is_empty() { text } else { format!("{}: {text}", locations.join("; ")) };
        // Filesystem failure classification must survive identifier rewriting.
        match error.chain().find_map(|cause| cause.downcast_ref::<std::io::Error>()) {
            Some(io) => std::io::Error::new(io.kind(), text).into(),
            None => anyhow::anyhow!(text),
        }
    }
}

#[cfg(test)]
mod provenance_io_tests {
    #[test]
    fn compiler_virtual_paths_do_not_duplicate_the_scene_directory() {
        let origins = [("programs.~abc".into(), vec!["/shots/caption.json: scene.materials[\"ink\"].program (line 4)".into()])].into();
        let error = super::authored_error(&origins, anyhow::anyhow!("/shots/programs.~abc:3:1: missing semicolon"));
        assert_eq!(error.to_string(), "/shots/caption.json: scene.materials[\"ink\"].program (line 4):3:1: missing semicolon");
    }

    #[test]
    fn program_name_prefixes_do_not_add_unrelated_origins() {
        let origins = [("programs.ink".into(), vec!["unrelated".into()]),
            ("programs.ink-blue".into(), vec!["authored program".into()])].into();
        let error = super::authored_error(&origins, anyhow::anyhow!("programs.ink-blue:3: bad syntax"));
        assert!(error.to_string().contains("authored program"));
        assert!(!error.to_string().contains("unrelated"));
    }

    #[test]
    fn equal_hashes_in_different_resource_tables_keep_their_owners() {
        let origins = [("buffers.~abc".into(), vec!["buffer slot".into()]),
            ("materials.~abc".into(), vec!["authored material slot".into()])].into();
        let error = super::authored_error(&origins, anyhow::anyhow!("material ~abc: invalid input"));
        assert_eq!(error.to_string(), "material authored material slot: invalid input");
    }

    #[test]
    fn resource_rewriting_keeps_filesystem_failure_category() {
        let origins = [("programs.~abc".into(), vec!["shot.json: scene.materials[\"ink\"].program (line 4)".into()])].into();
        let error = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound))
            .context("program ~abc: read missing.slang");
        let error = super::authored_error(&origins, error);
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
        assert!(!error.to_string().contains("~abc"));
        assert!(error.to_string().contains("shot.json"));
    }
}
