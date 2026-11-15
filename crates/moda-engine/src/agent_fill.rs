//! Agent shaders as fills (docs/design/effects-code-first.md §4.3, the fill placement).
//!
//! A shader paint whose `shaderId` is a code key (`b3:…`) names a WGSL code object in the document's code table
//! instead of a built-in fill. It draws through the production procedural-fill pipelines (instanced shapes, and the
//! vector-text pipeline for text), so it gets their MSAA, clip masks, strokes and opacity for free: moda-wgsl wraps
//! the agent's `shade(p)` in those modules' `shadeMain` ([`moda_wgsl::module`], [`moda_wgsl::fill_text_module`]).
//!
//! The modules are made at plan time and kept here, content-addressed (`agent:<hash of the WGSL>`, and its
//! `.vector-text-fill` variant), so the engine's module and pipeline caches treat them like the extracted ones. Their
//! pipeline descriptors are a production procedural fill's ([`MANIFEST_TEMPLATE`]): the hosts are byte-identical
//! copies of that module around its shader body (pinned by moda-wgsl's tests).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, Result};
use moda_scene::Scene;
use moda_scene::value::Value;

/// The production fill whose pipeline descriptors and bind layouts agent fills use.
pub const MANIFEST_TEMPLATE: &str = "plasma-field";
const TEXT_SUFFIX: &str = ".vector-text-fill";
const PREFIX: &str = "agent:";

fn modules() -> &'static Mutex<HashMap<String, Arc<str>>> {
    static M: OnceLock<Mutex<HashMap<String, Arc<str>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// A fill's `shaderId` names an agent shader (a code key) rather than a built-in fill.
pub fn is_code_ref(shader_id: &str) -> bool {
    shader_id.starts_with("b3:")
}

/// An agent fill's module id (`agent:…`, or its text variant).
pub fn is_agent_module(module_id: &str) -> bool {
    module_id.starts_with(PREFIX)
}

/// The WGSL of an agent module made by [`pack`].
pub fn module_source(module_id: &str) -> Option<Arc<str>> {
    modules().lock().ok()?.get(module_id).cloned()
}

/// The manifest module whose pipeline descriptors an agent module uses.
pub fn manifest_id(module_id: &str) -> String {
    if module_id.ends_with(TEXT_SUFFIX) {
        format!("{MANIFEST_TEMPLATE}{TEXT_SUFFIX}")
    } else {
        MANIFEST_TEMPLATE.to_string()
    }
}

fn fx(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    s.hash(&mut h);
    h.finish()
}

/// The size of the frame `node` sits in (its outermost transform parent), or the node's own size.
pub(crate) fn frame_size(scene: &Scene, node: &moda_scene::types::RawNode) -> [f32; 2] {
    let size = |n: &moda_scene::types::RawNode| {
        [n.f64("width").unwrap_or(0.0) as f32, n.f64("height").unwrap_or(0.0) as f32]
    };
    scene
        .transform_parent_chain(node.id())
        .last()
        .and_then(|id| scene.effective_node(id))
        .map(size)
        .unwrap_or_else(|| size(node))
}

/// A colour value as moda-wgsl takes it: `[r, g, b, a]` linear 0–1 straight, from a list or any CSS colour string.
pub(crate) fn css_color(v: &serde_json::Value) -> Option<[f32; 4]> {
    if let Some(a) = v.as_array() {
        let n: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
        return match n.len() {
            3 => Some([n[0], n[1], n[2], 1.0]),
            4 => Some([n[0], n[1], n[2], n[3]]),
            _ => None,
        };
    }
    let c = moda_text::color::css_color_to_normalized_rgba(v.as_str()?)?;
    let lin = |e: f64| {
        let e = e.clamp(0.0, 1.0);
        (if e <= 0.04045 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) }) as f32
    };
    Some([lin(c[0]), lin(c[1]), lin(c[2]), c[3] as f32])
}

/// A code object's source and schema from the document's code table, checked by moda-wgsl for `placement`
/// (cached by content: plans are rebuilt every frame of an animation). Errors read
/// `shader `b3:…` at line L, column C: … (code)`.
pub fn checked_code(
    scene: &Scene,
    code_key: &str,
    placement: moda_wgsl::Placement,
) -> Result<(String, moda_wgsl::Schema)> {
    let def = scene
        .doc()
        .code
        .get(code_key)
        .ok_or_else(|| anyhow!("no code object `{code_key}` in the document"))?;
    if def.get("lang").and_then(Value::as_str) != Some("wgsl") || def.get("kind").and_then(Value::as_str) != Some("shader") {
        bail!("code `{code_key}` isn't a WGSL shader");
    }
    let source = def.get("source").and_then(Value::as_str).unwrap_or_default();
    let schema_json = def.get("schema").map(Value::to_json).unwrap_or_default();
    static CHECKED: OnceLock<Mutex<HashMap<(u64, &'static str), Result<(), String>>>> = OnceLock::new();
    let key = (fx(&format!("{source}\u{0}{schema_json}")), placement.name());
    let cached = CHECKED.get_or_init(Default::default).lock().ok().and_then(|m| m.get(&key).cloned());
    let verdict = cached.unwrap_or_else(|| {
        let report = moda_wgsl::check(source, &schema_json, placement);
        let v = match report.errors.first() {
            None => Ok(()),
            Some(d) => {
                let at = match (d.line, d.column) {
                    (Some(l), Some(c)) => format!(" at line {l}, column {c}"),
                    _ => String::new(),
                };
                Err(format!("shader `{code_key}`{at}: {} ({})", d.message, d.code))
            }
        };
        if let Some(mut m) = CHECKED.get_or_init(Default::default).lock().ok() {
            m.insert(key, v.clone());
        }
        v
    });
    verdict.map_err(|e| anyhow!(e))?;
    let schema = moda_wgsl::parse_schema(&schema_json).map_err(|d| anyhow!("shader `{code_key}`: {}", d.message))?;
    Ok((source.to_string(), schema))
}

/// Pack an agent fill: check the code object (moda-wgsl's fill checks), make and register its shape and text
/// modules, and lay its parameters and host values into the instance (`shaderParams[0].x` is the clock).
pub fn pack(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    code_key: &str,
    params: &Value,
    shader_time_s: f64,
) -> Result<moda_shaders::runtime::PackedFill> {
    let (source, schema) =
        checked_code(scene, code_key, moda_wgsl::Placement::Fill).map_err(|e| anyhow!("shader fill: {e}"))?;
    let source = source.as_str();
    let shape = moda_wgsl::module(source, &schema, moda_wgsl::Placement::Fill);
    let text = moda_wgsl::fill_text_module(source, &schema);
    let id = format!("{PREFIX}{:016x}", fx(&shape.wgsl));
    {
        let mut m = modules().lock().map_err(|_| anyhow!("agent fill registry poisoned"))?;
        m.entry(id.clone()).or_insert_with(|| shape.wgsl.into());
        m.entry(format!("{id}{TEXT_SUFFIX}")).or_insert_with(|| text.wgsl.into());
    }
    let seed = params.get("seed").and_then(Value::as_f64).unwrap_or(0.0).max(0.0) as u32;
    let host = moda_wgsl::HostSlots {
        frame_size: frame_size(scene, node),
        time: shader_time_s as f32,
        px_scale: 1.0,
        seed: moda_wgsl::mix_seed(seed, code_key),
        progress: 0.0,
        bounds: [0.0; 4],
        to_out: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };
    let slots = moda_wgsl::encode_params(&schema, &|n| params.get(n).map(Value::to_json), &css_color);
    let (sp, cd) = moda_wgsl::encode_fill(&host, &slots);
    let mut packed = moda_shaders::runtime::PackedFill {
        module_id: id,
        shader_params: [0.0; moda_shaders::runtime::SHADER_PARAM_FLOAT_CAPACITY],
        color_data: [0.0; moda_shaders::runtime::COLOR_DATA_FLOAT_CAPACITY],
    };
    for (i, v) in sp.iter().flatten().enumerate() {
        packed.shader_params[i] = *v;
    }
    for (i, v) in cd.iter().flatten().enumerate() {
        packed.color_data[i] = *v;
    }
    Ok(packed)
}

/// The instance slots the clock drives: `shaderParams[0].x`.
pub fn clock_offsets() -> Option<Vec<u16>> {
    Some(vec![0])
}
