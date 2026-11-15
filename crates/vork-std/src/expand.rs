//! Author-side standard geometry expansion. Output is an ordinary buffer scene.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

pub fn expand_scene(document: &mut Value, base: &Path) -> Result<()> {
    // Nothing here reads the order a map holds its keys in: a builder's identity and its options are written
    // as sorted text, and the geometry table is visited by name. The document is sorted once, where it is
    // admitted.
    fn expand(value: &mut Value, base: &Path, built: &mut BTreeMap<String, Value>, apply: bool) -> Result<()> {
        let Some(shape) = value.get("std") else {
            return Ok(());
        };
        let shape = shape
            .as_str()
            .context("std must name a geometry builder")?
            .to_owned();
        let key = serde_json::to_string(value)?;
        if let Some(result) = built.get(&key) {
            if apply { *value = result.clone(); }
            return Ok(());
        }
        let mut options = value.as_object().context("std geometry object")?.clone();
        options.remove("std");
        ensure!(
            !options.contains_key("name"),
            "std geometry uses its table key or inline position, not name"
        );
        options.insert("name".into(), "shape".into());
        if shape == "text" {
            if let Some(font) = options.get("font").and_then(Value::as_str) {
                options.insert(
                    "font".into(),
                    base.join(font).to_string_lossy().into_owned().into(),
                );
            }
        }
        let fragment: Value = serde_json::from_str(&super::evaluate(
            &shape,
            &serde_json::to_string(&Value::Object(options))?,
        )?)?;
        let mut geometry = fragment
            .pointer("/geometry/shape")
            .context("std builder must return geometry")?
            .clone();
        for attribute in geometry["attributes"]
            .as_object_mut()
            .context("std geometry attributes")?
            .values_mut()
        {
            let name = attribute.as_str().context("std attribute buffer name")?;
            *attribute = fragment["buffers"][name].clone();
        }
        if let Some(indices) = geometry.get_mut("indices") {
            let name = indices.as_str().context("std index buffer name")?;
            *indices = fragment["buffers"][name].clone();
        }
        if apply { *value = geometry.clone(); }
        built.insert(key, geometry);
        Ok(())
    }
    fn nodes(value: &mut Value, base: &Path, built: &mut BTreeMap<String, Value>, errors: &mut Vec<anyhow::Error>, apply: bool) {
        for node in value.as_array_mut().into_iter().flatten() {
            if let Some(Err(error)) = node.get_mut("geometry").map(|geometry| expand(geometry, base, built, apply)) {
                // The node is named only for a builder that failed.
                let name = node.get("name").and_then(Value::as_str).unwrap_or("unnamed");
                errors.push(error.context(format!("node {name:?}: geometry")));
            }
            if let Some(children) = node.get_mut("children") {
                nodes(children, base, built, errors, apply);
            }
        }
    }
    let mut built = BTreeMap::new();
    let mut errors = Vec::new();
    if let Some(table) = document.get_mut("geometry").and_then(Value::as_object_mut) {
        let mut names: Vec<String> = table.keys().cloned().collect();
        names.sort_unstable();
        for name in &names {
            if let Err(error) = expand(&mut table[name.as_str()], base, &mut built, false).with_context(|| format!("geometry {name}")) {
                errors.push(error);
            }
        }
    }
    if let Some(value) = document.get_mut("nodes") {
        nodes(value, base, &mut built, &mut errors, false);
    }
    if !errors.is_empty() {
        let messages = errors.iter().map(|error| format!("{error:#}").replace(['\r', '\n'], " ")).collect::<Vec<_>>().join("; ");
        // One line for every builder, keeping a filesystem failure's I/O kind:
        // an unreadable font is an I/O failure (exit 5), not an invalid scene.
        let io = errors.iter().find_map(|error| error.chain().find_map(|cause| cause.downcast_ref::<std::io::Error>()).map(std::io::Error::kind));
        if let Some(kind) = io { return Err(std::io::Error::new(kind, messages).into()); }
        anyhow::bail!("{messages}");
    }
    // Publish only after every builder succeeds. The existing deduplication map
    // owns the prepared geometry, so this walk does not evaluate builders again.
    if let Some(table) = document.get_mut("geometry").and_then(Value::as_object_mut) {
        for geometry in table.values_mut() { expand(geometry, base, &mut built, true)?; }
    }
    if let Some(value) = document.get_mut("nodes") {
        nodes(value, base, &mut built, &mut errors, true);
    }
    debug_assert!(errors.is_empty());
    Ok(())
}

pub fn load_scene(path: &Path) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    let mut document =
        serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes))?;
    expand_scene(&mut document, path.parent().unwrap_or(Path::new(".")))?;
    Ok(serde_json::to_vec(&document)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expansion_reports_independent_builder_failures_together() {
        let mut scene = json!({"version":2,"geometry":{
            "first":{"std":"not_a_builder"},"second":{"std":"also_not_a_builder"},"valid":{"std":"box"}},
            "nodes":[{"geometry":{"std":"third_invalid_builder"}},
                {"children":[{"geometry":{"std":"fourth_invalid_builder"}}]}]});
        let original = scene.clone();
        let error = format!("{:#}", expand_scene(&mut scene, Path::new(".")).unwrap_err());
        assert_eq!(scene, original, "a failed expansion must not publish successful siblings");
        for name in ["not_a_builder", "also_not_a_builder", "third_invalid_builder", "fourth_invalid_builder"] {
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn unreadable_font_stays_one_line_and_an_io_failure() {
        let directory = std::env::temp_dir().join(format!("vork-font-directory-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for font in [directory.to_string_lossy().into_owned(), directory.join("missing.ttf").to_string_lossy().into_owned()] {
            let mut scene = json!({"nodes":[{"name":"title","geometry":{"std":"text","body":"A","font":font}}]});
            let error = expand_scene(&mut scene, Path::new(".")).unwrap_err();
            let text = format!("{error:#}");
            assert_eq!(text.lines().count(), 1, "{text}");
            assert!(text.contains("title") && text.contains(&font), "{text}");
            assert!(error.chain().any(|cause| cause.is::<std::io::Error>()), "filesystem failures keep their I/O kind: {text}");
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn text_refusal_is_one_actionable_line_without_font_advice() {
        let mut scene = json!({"nodes":[{"name":"title","geometry":{"std":"text","body":"",
            "font":"tests/fonts/LiberationSans-Regular.ttf"}}]});
        let error = format!("{:#}", expand_scene(&mut scene, Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap_err());
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(error.contains("title"), "{error}");
        assert_eq!(error.matches("text has no outlined glyphs").count(), 1, "{error}");
        assert!(!error.contains("restore its font"), "{error}");
    }

    #[test]
    fn expansion_owns_identity_and_preserves_unvalidated_content() -> Result<()> {
        let definition = json!({"std":"rounded_box","radius":0});
        let mut document = json!({"version":"invalid", "programs":{"broken":{"unknown":true}},
            "geometry":{"custom":definition},
            "nodes":[{"children":[{"geometry":definition}]}]});
        let expanded: Value =
            serde_json::from_str(&super::super::evaluate("expand", &document.to_string())?)?;
        let table = &expanded["geometry"]["custom"];
        assert_eq!(table, &expanded["nodes"][0]["children"][0]["geometry"]);
        assert!(table.get("std").is_none());
        assert!(table["attributes"]["POSITION"]["data"].is_array());
        assert!(table["indices"]["data"].is_array());
        assert_eq!(expanded["version"], document["version"]);
        assert_eq!(expanded["programs"], document["programs"]);
        for pointer in ["/geometry/custom", "/nodes/0/children/0/geometry"] {
            document.pointer_mut(pointer).unwrap()["name"] = "forbidden".into();
            let error = format!(
                "{:#}",
                expand_scene(&mut document, Path::new(".")).unwrap_err()
            );
            assert!(error.contains("table key or inline position"), "{error}");
            document
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove("name");
        }
        Ok(())
    }
}
