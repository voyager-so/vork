use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Retained source span after install-time constant folding; never authored.
    #[serde(skip)]
    pub source_span: Option<(f32, f32)>,
    #[serde(skip)]
    pub source_clock_span: Vec<vork_time::Clock>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.name", "x-help" = "string|null"))]
    pub name: Option<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.tracks", "x-help" = "track[]|null; XOR clip"))]
    pub tracks: Option<Vec<Track>>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.clip", "x-help" = "clip|null; XOR tracks"))]
    pub clip: Option<Clip>,
    #[serde(default = "constants::layer_unit")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.weight", "x-help" = "number"))]
    pub weight: f32,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.blend", "x-help" = "replace|combine"))]
    pub blend: Blend,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.offset", "x-help" = "number; source time = offset + time*speed"))]
    pub offset: f32,
    #[serde(default = "constants::layer_unit")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.speed", "x-help" = "number"))]
    pub speed: f32,
    #[serde(default, rename = "loop")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.layer.loop", "x-help" = "bool"))]
    pub looping: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    #[default]
    Replace,
    #[serde(rename = "combine")]
    Add,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Clip {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.clip.target", "x-help" = "asset-node-name"))]
    pub target: String,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.clip.file", "x-help" = "file-path|null; null = the target asset's clips; else a glTF file retargeted by node name"))]
    pub file: Option<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.clip.name", "x-help" = "string|null; XOR index"))]
    pub name: Option<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.clip.index", "x-help" = "u32|null; XOR name"))]
    pub index: Option<u32>,
}
/// A keyed property.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "AuthoredTrack")]
#[schemars(with = "AuthoredTrack")]
pub struct Track {
    pub path: Vec<String>,
    pub keys: Keys,
    pub interpolation: Interpolation,
    #[serde(skip)]
    pub times: Times,
    #[serde(skip)]
    pub values: Values,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(title = "Track")]
struct AuthoredTrack {
    /// Segments naming the keyed property, e.g. ["nodes","Bone.001","rotation"].
    #[serde(deserialize_with = "path_segments")]
    #[schemars(with = "Vec<PathSegment>")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "animation.track.path", "x-help" = "(string|index)[]; [\"nodes\",node,property...] | [\"nodes\",asset,\"nodes\",imported-name|#index,property...] | [\"materials\",material,\"inputs\",input] | [\"animation\",layer,\"weight\"] | [\"render\",property...] | [\"nodes\",text-node,\"glyphs\"|\"words\"|\"lines\",index,translation|rotation|scale|opacity|fill|stroke|transform-origin]", "x-help-extra" = "animation.track.path properties: node translation|rotation|scale|visible|visible_camera|visible_shadow|visible_diffuse|visible_glossy|visible_transmission|morph_weights,<morph>, camera/light numeric fields (camera optics are material inputs); canvas: filter, clip-path, mask, all path/text style fields; imported nodes: translation|rotation|scale|morph_weights|visible\nanimation.track.path render properties: camera|background (RGB; CSS colour on a canvas)|motion_blur,shutter\nanimation.track.path text pieces: fill and stroke [r,g,b,a] 0..1 or CSS colour/gradient or image paint; transform-origin CSS, default the ink box centre"))]
    path: Vec<String>,
    #[schemars(extend("x-source" = "vork; glTF 2.0 animation samplers", "x-help-path" = "animation.track.keys", "x-help" = "[[time, value], ...] | {times: buffer, values: buffer}; time increasing; CUBICSPLINE [time, value, in_tangent, out_tangent] (tangents per second) or [time, value] (zero tangents); rotation values as in node.rotation"))]
    keys: Keys,
    #[serde(default)]
    #[schemars(extend("x-source" = "glTF 2.0 LINEAR", "x-help-path" = "animation.track.interpolation", "x-help" = "STEP|LINEAR|CUBICSPLINE; omitted: LINEAR for numeric and CSS/SVG fields; STEP for bools and resource tokens"))]
    interpolation: Option<Interpolation>,
}
/// A track path segment: a name, or a zero-based index (`["nodes", "title", "glyphs", 3, "opacity"]`).
#[derive(JsonSchema)]
#[serde(untagged)]
#[allow(dead_code)]
enum PathSegment {
    Name(String),
    Index(u32),
}
/// Path segments as strings; an index segment is kept as its decimal digits.
fn path_segments<'de, D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Vec<String>, D::Error> {
    Vec::<serde_json::Value>::deserialize(deserializer)?.into_iter().map(|segment| match segment {
        serde_json::Value::String(name) => Ok(name),
        serde_json::Value::Number(index) if index.is_u64() => Ok(index.to_string()),
        other => Err(serde::de::Error::custom(format!("a track path segment is a name or a zero-based index, not {other}"))),
    }).collect()
}
/// Keyframes inline as `[time, value]` pairs, or two buffers for bulk data.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Keys {
    Inline(Vec<Vec<serde_json::Value>>),
    Buffers {
        #[schemars(with = "NameOr<Buffer>")]
        times: String,
        #[schemars(with = "NameOr<Buffer>")]
        values: String,
    },
}
impl TryFrom<AuthoredTrack> for Track {
    type Error = String;
    fn try_from(t: AuthoredTrack) -> std::result::Result<Self, String> {
        let path = display_path(&t.path);
        let (times, values, interpolation) = match &t.keys {
            Keys::Buffers { times, values } => (Times::Buffer(times.clone()), Values::Buffer(values.clone()),
                t.interpolation.unwrap_or_default()),
            Keys::Inline(keys) => {
                // Names and switches hold until the next key: STEP is the only meaning.
                let clock_value=matches!(t.path.as_slice(),[root,_,field] if root=="images" && field=="time")
                    || matches!(t.path.as_slice(),[.., time, speed] if time=="time" && speed=="speed");
                let css = css_path(&t.path);
                let discrete = !clock_value && !css && keys.first().and_then(|k| k.get(1)).is_some_and(|v| v.is_string() || v.is_boolean());
                let interpolation = t.interpolation.unwrap_or(if discrete { Interpolation::Step } else { Interpolation::Linear });
                let cubic = matches!(interpolation, Interpolation::Cubic);
                let mut times = Vec::with_capacity(keys.len());
                let mut values = Vec::with_capacity(keys.len() * if cubic { 3 } else { 1 });
                for (i, key) in keys.iter().enumerate() {
                    let time = key.first().ok_or_else(||format!("track {path}: key {i}: expected [time, value]"))?;
                    let time=if let Some(number)=time.as_f64() {number as f32} else {
                        serde_json::from_value::<vork_time::Clock>(time.clone())
                            .map_err(|e|format!("track {path}: key {i}: expected [time, value]: {e}"))?.as_f32()
                    };
                    match (cubic, key.len()) {
                        (_, 2) if !cubic => values.push(key[1].clone()),
                        (true, 2) => {
                            let zero = zero_like(&key[1]);
                            values.extend([zero.clone(), key[1].clone(), zero]);
                        }
                        (true, 4) => values.extend([key[2].clone(), key[1].clone(), key[3].clone()]),
                        _ => return Err(format!("track {path}: key {i}: expected [time, value]{}",
                            if cubic { " or [time, value, in_tangent, out_tangent]" } else { "" })),
                    }
                    times.push(time);
                }
                (Times::Inline(times), Values::Inline(values), interpolation)
            }
        };
        Ok(Self { path: t.path, keys: t.keys, interpolation, times, values })
    }
}
/// CSS inference follows target grammar, never program-owned input names.
fn css_path(path: &[String]) -> bool {
    if matches!(path, [root, field] if root == "render" && field == "background") { return true; }
    let mut path = match path {
        [root, _, rest @ ..] if root == "nodes" => rest,
        _ => path, // Node-owned tracks start at a property or a child.
    };
    while let [root, _, rest @ ..] = path {
        if root != "children" { break; }
        path = rest;
    }
    match path {
        [field] => StyleProperty::parse(field).is_some() || matches!(field.as_str(), "fill" | "stroke"),
        [unit, index, field] => TextUnit::parse(unit).is_some() && index.parse::<usize>().is_ok()
            && matches!(field.as_str(), "fill" | "stroke" | "transform-origin"),
        _ => false,
    }
}

/// A zero tangent shaped like a key's value.
fn zero_like(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => items.iter().map(zero_like).collect(),
        serde_json::Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), zero_like(v))).collect(),
        _ => 0.into(),
    }
}
#[derive(Clone, Debug)]
pub enum Times {
    Inline(Vec<f32>),
    Buffer(String),
}
impl Default for Times { fn default() -> Self { Self::Inline(Vec::new()) } }
#[derive(Clone, Debug)]
pub enum Values {
    Inline(Vec<serde_json::Value>),
    Buffer(String),
}
impl Default for Values { fn default() -> Self { Self::Inline(Vec::new()) } }
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Interpolation {
    Step,
    #[default]
    Linear,
    #[serde(rename = "CUBICSPLINE")]
    Cubic,
}

#[cfg(test)]
mod key_tests {
    use super::*;
    #[test]
    fn css_defaults_follow_absolute_and_node_owned_targets() {
        for prefix in [vec!["nodes", "title"], vec![], vec!["children", "title"], vec!["children", "group", "children", "title"]] {
            for suffix in [vec!["fill"], vec!["filter"], vec!["direction"], vec!["glyphs", "1", "fill"], vec!["words", "0", "stroke"], vec!["lines", "0", "transform-origin"]] {
                let path: Vec<_> = prefix.iter().chain(&suffix).copied().collect();
                let t = track(serde_json::json!({"path":path,"keys":[[0,"a"],[1,"b"]]})).unwrap();
                assert!(matches!(t.interpolation, Interpolation::Linear), "{path:?}");
                let t = track(serde_json::json!({"path":path,"interpolation":"STEP","keys":[[0,"a"],[1,"b"]]})).unwrap();
                assert!(matches!(t.interpolation, Interpolation::Step));
            }
        }
        for field in ["fill", "direction", "width", "filter", "transform-origin"] {
            let t = track(serde_json::json!({"path":["materials","m","inputs",field],"keys":[[0,0],[1,1]]})).unwrap();
            assert!(matches!(t.interpolation, Interpolation::Linear));
        }
    }

    #[test]
    fn combine_is_the_only_serialized_spelling_for_combining_layers() {
        let blend: Blend = serde_json::from_str("\"combine\"").unwrap();
        assert_eq!(blend, Blend::Add);
        assert_eq!(serde_json::to_string(&blend).unwrap(), "\"combine\"");
        assert_eq!(serde_json::from_str::<Blend>("\"replace\"").unwrap(), Blend::Replace);
        assert!(serde_json::from_str::<Blend>("\"add\"").is_err());
        for count in [1, 3] {
            let bytes = serde_json::to_vec(&serde_json::json!({
                "version": 2, "animation": vec![serde_json::json!({"blend":"add","tracks":[]}); count],
            })).unwrap();
            let error = Scene::parse_json_with_source(&bytes, b"", None).unwrap_err().to_string();
            assert_eq!(error.lines().count(), 1, "{error}");
            assert!(error.ends_with("use \"combine\" instead of \"add\"; see `vork help upgrade`"), "{error}");
        }
    }
    #[test]
    fn buffer_keys_reject_unknown_fields() {
        let valid = serde_json::json!({"path":["nodes","a","translation"],"keys":{"times":"t","values":"v"}});
        assert!(track(valid.clone()).is_ok());
        for field in ["interpolation", "value", "extra"] {
            let mut invalid = valid.clone();
            invalid["keys"][field] = "STEP".into();
            assert!(track(invalid).is_err(), "accepted keys.{field}");
        }
    }
    fn track(value: serde_json::Value) -> std::result::Result<Track, String> {
        serde_json::from_value(value).map_err(|e| e.to_string())
    }
    #[test]
    fn keys_are_time_value_pairs_and_cubic_tangents_are_optional() {
        let t = track(serde_json::json!({"path":["nodes","a","translation"],"keys":[[0,[0,0,0]],[1,[1,2,3]]]})).unwrap();
        let (Times::Inline(times), Values::Inline(values)) = (&t.times, &t.values) else { panic!() };
        assert_eq!((times.as_slice(), values.len()), (&[0., 1.][..], 2));
        let cubic = track(serde_json::json!({"path":["nodes","a","translation"],"interpolation":"CUBICSPLINE",
            "keys":[[0,[0,0,0]],[1,[1,2,3],[4,5,6],[7,8,9]]]})).unwrap();
        let Values::Inline(packed) = &cubic.values else { panic!() };
        assert_eq!(packed, &[serde_json::json!([0,0,0]), serde_json::json!([0,0,0]), serde_json::json!([0,0,0]),
            serde_json::json!([4,5,6]), serde_json::json!([1,2,3]), serde_json::json!([7,8,9])], "glTF in, value, out; omitted tangents are zero");
        let euler = track(serde_json::json!({"path":["nodes","a","rotation"],"interpolation":"CUBICSPLINE","keys":[[0,{"euler":[0,0,1]}]]})).unwrap();
        let Values::Inline(packed) = &euler.values else { panic!() };
        assert_eq!(packed[0], serde_json::json!({"euler":[0,0,0]}), "tangents take the value's form");
        for bad in [serde_json::json!([[0]]), serde_json::json!([[0,1,2]]), serde_json::json!([["a",1]])] {
            assert!(track(serde_json::json!({"path":["nodes","a","translation"],"keys":bad})).unwrap_err().contains("[time, value]"));
        }
        let names = track(serde_json::json!({"path":["render","camera"],"keys":[[0,"a"],[1,"b"]]})).unwrap();
        assert!(matches!(names.interpolation, Interpolation::Step), "names and switches default to STEP");
        assert!(track(serde_json::json!({"path":["nodes","a","translation"],"times":[0],"values":[[0,0,0]]})).is_err());
    }
}

#[cfg(test)]
mod output_rate_tests {
    use super::*;
    #[test]
    fn output_rate_decoding_preserves_numeric_inputs() -> anyhow::Result<()> {
        let scene: Scene = serde_json::from_str(r#"{"version":2,"composition":{"space":"3d","compositor":[{"material":"m","resolution":0.5}]},"render":{"fps":29.970000000000001,"background":[0.1,0.2,0.3]},"materials":{"m":{"inputs":{"base_weight":0.1,"base_color":[0.125,0.25,0.5]}}},"nodes":[{"type":"camera","translation":[0.001,0.2,0.3],"scale":[0.001,0.001,0.001],"rotation":{"euler":[0.01,0.02,0.03]},"target":[0.1,0.2,0.3]},{"type":"light","shape":"sun","energy":1e-3,"angle":1e-5}]}"#)?;
        assert_eq!(scene.render.fps, "29.970000000000001".parse()?);
        assert_ne!(scene.render.fps, "29.97".parse()?);
        assert_eq!(scene.nodes[0].transform.translation, [0.001,0.2,0.3]);
        assert_eq!(scene.nodes[0].transform.scale, [0.001;3]);
        assert!(scene.nodes[0].transform.resolved_rotation().iter().all(|v|v.is_finite()));
        assert_eq!(scene.materials["m"].inputs["base_weight"], Input::Number(0.1));
        assert_eq!(scene.materials["m"].inputs["base_color"], Input::Vector(vec![0.125,0.25,0.5]));
        Ok(())
    }

}
