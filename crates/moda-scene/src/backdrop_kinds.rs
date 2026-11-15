//! The backdrop filter kinds (live-nodes stage 7b): grain, vignette, glaze, displace and grade, the kinds the
//! one backdrop effect gained beside background blur, glass, pixelate and dither; and the adjustment kinds
//! (docs/design/adjustment-layers.md: levels, curves, brightness-contrast, hue-saturation, color-balance,
//! vibrance, invert, desaturate, gradient-map, channel-mixer), Photoshop's adjustment layers as backdrops.
//!
//! Each kind's parameters are defined ONCE, in `backdrop_kinds.json` (read here and by the editor's
//! `core/scene/backdropKinds.ts`): names, types, ranges, defaults, the markup order and the working colour
//! space. This module only reads that schema: a stored value is taken when it is a finite number (or a known
//! enum value), clamped to the schema's range, and the schema's default stands in otherwise, so a hand-edited
//! document never draws out of range. MOVES TO the shared `moda-code` crate with the JSON (live-nodes §18 point 9).
//!
//! A kind is inert when its `amount` is 0 (grain, vignette, glaze, displace), or, for a kind without an amount
//! (grade), when every number is at its default: an inert entry plans no split, as a zero-radius blur doesn't.

use crate::value::{Map, Value};
use std::sync::OnceLock;

/// The schema, embedded.
pub const SCHEMA_JSON: &str = include_str!("backdrop_kinds.json");

/// Where a kind computes (motion-model §12.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingSpace {
    /// The stored 8-bit gamma sRGB values.
    Gamma,
    /// Linear light: decoded at the kernel's input, encoded at its output.
    Linear,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    Number {
        min: f64,
        max: f64,
        fraction: bool,
    },
    Integer {
        min: f64,
        max: f64,
    },
    Enum(Vec<String>),
    Color,
    /// A CSS colour, or a stored `gradient` object.
    Paint,
    Boolean,
    /// A fixed-length list of numbers, each clamped to [min, max] (colour balance's shifts, a mixer row).
    Numbers {
        len: usize,
        min: f64,
        max: f64,
    },
    /// Curve points `[[x, y], …]`, each coordinate clamped to [min, max].
    Points {
        min: f64,
        max: f64,
    },
    /// Gradient stops `[{ at: 0..1, color }, …]`.
    Stops,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: ParamType,
    pub default: Value,
    pub unit: Option<String>,
    pub about: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kind {
    pub name: String,
    pub about: String,
    pub working_space: WorkingSpace,
    /// The markup order of the bare numbers after the kind name.
    pub positional: Vec<String>,
    pub params: Vec<Param>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub version: u64,
    /// The glaze blend modes, in the schema's order.
    pub blend_modes: Vec<String>,
    /// In the schema's order.
    pub kinds: Vec<Kind>,
}

fn parse_schema() -> Schema {
    let v: Value = crate::value::from_str(SCHEMA_JSON).expect("backdrop_kinds.json parses");
    let strings = |v: &Value| -> Vec<String> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let blend_modes = strings(&v["blendModes"]);
    let kinds = v["kinds"]
        .as_object()
        .expect("backdrop_kinds.json has kinds")
        .iter()
        .map(|(name, k)| Kind {
            name: name.to_string(),
            about: k["about"].as_str().unwrap_or("").to_string(),
            working_space: match k["workingSpace"].as_str() {
                Some("linear") => WorkingSpace::Linear,
                _ => WorkingSpace::Gamma,
            },
            positional: strings(&k["positional"]),
            params: k["params"]
                .as_array()
                .expect("a kind has params")
                .iter()
                .map(|p| {
                    let num = |key: &str| p[key].as_f64().unwrap_or(f64::NAN);
                    let ty = match p["type"].as_str() {
                        Some("number") => ParamType::Number {
                            min: num("min"),
                            max: num("max"),
                            fraction: p["fraction"].as_bool().unwrap_or(false),
                        },
                        Some("integer") => ParamType::Integer {
                            min: num("min"),
                            max: num("max"),
                        },
                        Some("enum") => ParamType::Enum(match &p["values"] {
                            Value::String(s) if s == "blendModes" => blend_modes.clone(),
                            other => strings(other),
                        }),
                        Some("color") => ParamType::Color,
                        Some("paint") => ParamType::Paint,
                        Some("boolean") => ParamType::Boolean,
                        Some("numbers") => ParamType::Numbers {
                            len: p["length"].as_u64().unwrap_or(0) as usize,
                            min: num("min"),
                            max: num("max"),
                        },
                        Some("points") => ParamType::Points { min: num("min"), max: num("max") },
                        Some("stops") => ParamType::Stops,
                        other => panic!("backdrop_kinds.json: unknown param type {other:?}"),
                    };
                    Param {
                        name: p["name"].as_str().expect("a param has a name").to_string(),
                        ty,
                        default: p["default"].clone(),
                        unit: p["unit"].as_str().map(str::to_string),
                        about: p["about"].as_str().unwrap_or("").to_string(),
                    }
                })
                .collect(),
        })
        .collect();
    Schema {
        version: v["version"].as_u64().unwrap_or(1),
        blend_modes,
        kinds,
    }
}

/// The parsed schema.
pub fn schema() -> &'static Schema {
    static SCHEMA: OnceLock<Schema> = OnceLock::new();
    SCHEMA.get_or_init(parse_schema)
}

/// The kind named `name` (an effect `type`), if it is a backdrop filter kind.
pub fn kind(name: &str) -> Option<&'static Kind> {
    schema().kinds.iter().find(|k| k.name == name)
}

/// The backdrop filter kinds' effect types, in the schema's order.
pub const FILTER_KINDS: &[&str] = &["grain", "vignette", "glaze", "displace", "grade"];

/// The adjustment kinds (adjustment-layers.md), in the schema's order after the filter kinds. Their parameters are
/// `paint.adjust`'s, so a destructive adjustment and a live one read the same.
pub const ADJUSTMENT_KINDS: &[&str] = &[
    "levels",
    "curves",
    "brightness-contrast",
    "hue-saturation",
    "color-balance",
    "vibrance",
    "invert",
    "desaturate",
    "gradient-map",
    "channel-mixer",
];

/// Whether `name` is an adjustment kind.
pub fn is_adjustment(name: &str) -> bool {
    ADJUSTMENT_KINDS.contains(&name)
}

impl Kind {
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }

    /// A number or integer param of `effect`: its finite stored value clamped to the range (an integer rounded),
    /// or the default. Panics on a name the schema doesn't define as a number (a programming error).
    pub fn number(&self, effect: &Map, name: &str) -> f64 {
        let p = self
            .param(name)
            .unwrap_or_else(|| panic!("backdrop kind {} has no param {name}", self.name));
        let default = p.default.as_f64().unwrap_or(0.0);
        let stored = effect
            .get(name)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite());
        match p.ty {
            ParamType::Number { min, max, .. } => stored.unwrap_or(default).clamp(min, max),
            ParamType::Integer { min, max } => stored.unwrap_or(default).round().clamp(min, max),
            _ => panic!("backdrop kind {} param {name} is not a number", self.name),
        }
    }

    /// An enum, colour or paint param of `effect` as text: the stored string when it is valid for the param (an
    /// enum value; any non-empty string for a colour), or the default.
    pub fn text<'a>(&'a self, effect: &'a Map, name: &str) -> &'a str {
        let p = self
            .param(name)
            .unwrap_or_else(|| panic!("backdrop kind {} has no param {name}", self.name));
        let default = p.default.as_str().unwrap_or("");
        let stored = effect
            .get(name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match (&p.ty, stored) {
            (ParamType::Enum(values), Some(s)) if values.iter().any(|v| v == s) => s,
            (ParamType::Color | ParamType::Paint, Some(s)) => s,
            _ => default,
        }
    }

    /// A boolean param: the stored boolean, or the default.
    pub fn boolean(&self, effect: &Map, name: &str) -> bool {
        let p = self.param(name).unwrap_or_else(|| panic!("backdrop kind {} has no param {name}", self.name));
        effect.get(name).and_then(Value::as_bool).unwrap_or_else(|| p.default.as_bool().unwrap_or(false))
    }

    /// A `numbers` param: the stored list when it has the right length and every entry is a finite number (each
    /// clamped to the range), or the default.
    pub fn numbers(&self, effect: &Map, name: &str) -> Vec<f64> {
        let p = self.param(name).unwrap_or_else(|| panic!("backdrop kind {} has no param {name}", self.name));
        let ParamType::Numbers { len, min, max } = p.ty else {
            panic!("backdrop kind {} param {name} is not a list of numbers", self.name)
        };
        let read = |v: &Value| -> Option<Vec<f64>> {
            let a = v.as_array()?;
            if a.len() != len {
                return None;
            }
            a.iter().map(|x| x.as_f64().filter(|f| f.is_finite()).map(|f| f.clamp(min, max))).collect()
        };
        effect.get(name).and_then(read).or_else(|| read(&p.default)).unwrap_or_else(|| vec![0.0; len])
    }

    /// A `points` param: the stored `[[x, y], …]` (at least one point, every coordinate finite, clamped), or the
    /// default.
    pub fn points(&self, effect: &Map, name: &str) -> Vec<(f64, f64)> {
        let p = self.param(name).unwrap_or_else(|| panic!("backdrop kind {} has no param {name}", self.name));
        let ParamType::Points { min, max } = p.ty else {
            panic!("backdrop kind {} param {name} is not points", self.name)
        };
        let read = |v: &Value| -> Option<Vec<(f64, f64)>> {
            let a = v.as_array().filter(|a| !a.is_empty())?;
            a.iter()
                .map(|pt| {
                    let xy = pt.as_array().filter(|xy| xy.len() == 2)?;
                    let (x, y) = (xy[0].as_f64()?, xy[1].as_f64()?);
                    (x.is_finite() && y.is_finite()).then(|| (x.clamp(min, max), y.clamp(min, max)))
                })
                .collect()
        };
        effect.get(name).and_then(read).or_else(|| read(&p.default)).unwrap_or_default()
    }

    /// A `stops` param: the stored `[{ at, color }, …]` (at least one stop with a finite `at` and a colour string;
    /// `at` clamped to 0..1), or the default.
    pub fn stops(&self, effect: &Map, name: &str) -> Vec<(f64, String)> {
        let p = self.param(name).unwrap_or_else(|| panic!("backdrop kind {} has no param {name}", self.name));
        let read = |v: &Value| -> Option<Vec<(f64, String)>> {
            let a = v.as_array().filter(|a| !a.is_empty())?;
            a.iter()
                .map(|st| {
                    let at = st.get("at")?.as_f64().filter(|f| f.is_finite())?;
                    let color = st.get("color")?.as_str()?.trim();
                    (!color.is_empty()).then(|| (at.clamp(0.0, 1.0), color.to_string()))
                })
                .collect()
        };
        effect.get(name).and_then(read).or_else(|| read(&p.default)).unwrap_or_default()
    }

    /// Whether `effect` changes any pixel (see the module doc). A kind with an `amount` is active above 0; one
    /// without params (invert) always; any other when some param differs from its default.
    pub fn is_active(&self, effect: &Map) -> bool {
        if self.param("amount").is_some() {
            return self.number(effect, "amount") > 0.0;
        }
        // No params, or only choices (desaturate's method): the kind always changes pixels.
        if self.params.iter().all(|p| matches!(p.ty, ParamType::Enum(_) | ParamType::Color | ParamType::Paint)) {
            return true;
        }
        self.params.iter().any(|p| match &p.ty {
            ParamType::Number { .. } | ParamType::Integer { .. } => {
                (self.number(effect, &p.name) - p.default.as_f64().unwrap_or(0.0)).abs() > 1e-9
            }
            ParamType::Boolean => self.boolean(effect, &p.name) != p.default.as_bool().unwrap_or(false),
            ParamType::Numbers { .. } => {
                let d: Vec<f64> = p.default.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
                self.numbers(effect, &p.name).iter().zip(&d).any(|(a, b)| (a - b).abs() > 1e-9)
            }
            ParamType::Points { .. } => {
                // The identity curve (every point on the diagonal, ends at 0 and 255) changes nothing.
                let pts = self.points(effect, &p.name);
                !(pts.iter().all(|(x, y)| (x - y).abs() < 1e-9) && pts.iter().any(|p| p.0 <= 0.0) && pts.iter().any(|p| p.0 >= 255.0))
            }
            // A gradient map always recolours; the channel / method / colour choices only matter with another param.
            ParamType::Stops => true,
            ParamType::Enum(_) | ParamType::Color | ParamType::Paint => false,
        })
    }
}

/// The glaze blend mode's index in the schema's `blendModes` (0 = normal), or 0 for an unknown one.
pub fn blend_mode_index(mode: &str) -> usize {
    schema()
        .blend_modes
        .iter()
        .position(|m| m == mode)
        .unwrap_or(0)
}

/// Whether a raw effect entry is a backdrop filter kind that changes pixels.
pub fn filter_is_active(effect: &Map) -> bool {
    effect
        .get("type")
        .and_then(Value::as_str)
        .and_then(kind)
        .is_some_and(|k| k.is_active(effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn obj(v: Value) -> Map {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn the_schema_lists_every_filter_kind_in_order() {
        let names: Vec<&str> = schema().kinds.iter().map(|k| k.name.as_str()).collect();
        let all: Vec<&str> = FILTER_KINDS.iter().chain(ADJUSTMENT_KINDS).copied().collect();
        assert_eq!(names, all);
        assert_eq!(schema().blend_modes[0], "normal");
        assert_eq!(schema().blend_modes.len(), 16);
        for k in &schema().kinds {
            for name in &k.positional {
                assert!(matches!(
                    k.param(name).map(|p| &p.ty),
                    Some(ParamType::Number { .. } | ParamType::Integer { .. })
                ));
            }
            for p in &k.params {
                match &p.ty {
                    ParamType::Number { min, max, .. } | ParamType::Integer { min, max } => {
                        let d = p.default.as_f64().unwrap();
                        assert!(
                            min <= &d && &d <= max,
                            "{}.{} default in range",
                            k.name,
                            p.name
                        );
                    }
                    ParamType::Enum(values) => assert!(values
                        .iter()
                        .any(|v| Some(v.as_str()) == p.default.as_str())),
                    ParamType::Color | ParamType::Paint => assert!(p.default.is_string()),
                    ParamType::Boolean => assert!(p.default.is_boolean()),
                    ParamType::Numbers { len, .. } => assert_eq!(p.default.as_array().map(Vec::len), Some(*len)),
                    ParamType::Points { .. } | ParamType::Stops => assert!(p.default.is_array()),
                }
            }
        }
        assert_eq!(kind("grade").unwrap().working_space, WorkingSpace::Linear);
        assert_eq!(kind("glaze").unwrap().working_space, WorkingSpace::Gamma);
    }

    #[test]
    fn values_clamp_and_fall_back_to_defaults() {
        let grain = kind("grain").unwrap();
        let e = obj(
            json!({ "type": "grain", "amount": 7, "size": "big", "seed": 2.6, "mode": "sparkle" }),
        );
        assert_eq!(grain.number(&e, "amount"), 1.0);
        assert_eq!(grain.number(&e, "size"), 1.0);
        assert_eq!(grain.number(&e, "seed"), 3.0);
        assert_eq!(grain.text(&e, "mode"), "mono");
        let glaze = kind("glaze").unwrap();
        let g = obj(json!({ "type": "glaze", "mode": "screen" }));
        assert_eq!(glaze.text(&g, "mode"), "screen");
        assert_eq!(glaze.text(&g, "color"), "#f4c27a");
        assert_eq!(blend_mode_index("screen"), 2);
        assert_eq!(blend_mode_index("luminosity"), 15);
    }

    #[test]
    fn activity() {
        assert!(filter_is_active(&obj(json!({ "type": "grain" }))));
        assert!(!filter_is_active(&obj(
            json!({ "type": "vignette", "amount": 0 })
        )));
        assert!(!filter_is_active(&obj(json!({ "type": "grade" }))));
        assert!(!filter_is_active(&obj(
            json!({ "type": "grade", "gamma": 1 })
        )));
        assert!(filter_is_active(&obj(
            json!({ "type": "grade", "exposure": 0.5 })
        )));
        assert!(!filter_is_active(&obj(
            json!({ "type": "blur", "radius": 4 })
        )));
    }

    #[test]
    fn adjustment_kinds_read_their_lists_and_know_when_they_change_nothing() {
        for k in ADJUSTMENT_KINDS {
            assert!(is_adjustment(k) && kind(k).is_some(), "{k}");
            assert_eq!(kind(k).unwrap().working_space, WorkingSpace::Gamma);
        }
        let curves = kind("curves").unwrap();
        let c = obj(json!({ "type": "curves", "points": [[0, 0], [128, 400], [255, 255]] }));
        assert_eq!(curves.points(&c, "points"), vec![(0.0, 0.0), (128.0, 255.0), (255.0, 255.0)]);
        assert_eq!(curves.points(&obj(json!({ "points": "bad" })), "points"), vec![(0.0, 0.0), (255.0, 255.0)]);
        assert!(!filter_is_active(&obj(json!({ "type": "curves" }))), "the identity curve");
        assert!(filter_is_active(&c));
        let cb = kind("color-balance").unwrap();
        assert_eq!(cb.numbers(&obj(json!({ "shadows": [10, -500, 3] })), "shadows"), vec![10.0, -100.0, 3.0]);
        assert_eq!(cb.numbers(&obj(json!({ "shadows": [1, 2] })), "shadows"), vec![0.0, 0.0, 0.0], "wrong length: default");
        assert!(cb.boolean(&obj(json!({})), "preserveLuminosity"));
        assert!(!filter_is_active(&obj(json!({ "type": "color-balance" }))));
        assert!(filter_is_active(&obj(json!({ "type": "color-balance", "midtones": [0, 20, 0] }))));
        assert!(filter_is_active(&obj(json!({ "type": "invert" }))), "invert has no params and always changes pixels");
        assert!(filter_is_active(&obj(json!({ "type": "gradient-map" }))));
        assert!(!filter_is_active(&obj(json!({ "type": "levels" }))));
        assert!(filter_is_active(&obj(json!({ "type": "levels", "inBlack": 20 }))));
        assert!(filter_is_active(&obj(json!({ "type": "hue-saturation", "colorize": true }))));
        let gm = kind("gradient-map").unwrap();
        assert_eq!(gm.stops(&obj(json!({ "stops": [{ "at": 2, "color": "#f00" }] })), "stops"), vec![(1.0, "#f00".to_string())]);
    }
}
