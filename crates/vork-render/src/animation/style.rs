//! CSS/SVG values are decoded once, then sampled on the ordinary layer clocks.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::sync::Arc;
use vork_scene::v2::{self, StyleInterpolation as I};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Paint {
    None,
    Color([f32; 4]),
    Gradient(Arc<str>),
    Image(Value),
    Mix(Arc<Paint>, Arc<Paint>, f32),
}
impl Paint {
    /// The form this paint keeps while its values change, when it has one.
    pub(crate) fn form(&self) -> Option<crate::canvas_draw::PaintForm> {
        use crate::canvas_draw::{gradient_paint, PaintForm};
        match self {
            Self::Color(_) => Some(PaintForm::Solid),
            Self::Gradient(text) => PaintForm::of_gradient(&gradient_paint(text, [0., 0., 100., 100.]).ok()?),
            _ => None,
        }
    }
    pub fn parse(v: &Value) -> Result<Self> {
        if let Some(s) = v.as_str() {
            if s.contains("gradient(") {
                crate::canvas_draw::gradient_paint(s, [0., 0., 100., 100.])?;
                return Ok(Self::Gradient(s.into()));
            }
            return Ok(v2::text::paint(s)?.map_or(Self::None, Self::Color));
        }
        let _: v2::ImagePaint = serde_json::from_value(v.clone())?;
        Ok(Self::Image(v.clone()))
    }
    pub fn json(&self) -> Value {
        match self {
            Self::None => json!("none"),
            Self::Color(c) => json!(color_css(*c)),
            Self::Gradient(s) => json!(s.as_ref()),
            Self::Image(v) => v.clone(),
            Self::Mix(a, b, t) => {
                if *t < 0.5 {
                    a.json()
                } else {
                    b.json()
                }
            }
        }
    }
    pub fn mix(a: &Self, b: &Self, t: f32) -> Self {
        if t <= 0. || a == b {
            return a.clone();
        }
        if t >= 1. {
            return b.clone();
        }
        match (a, b) {
            (Self::Color(a), Self::Color(b)) => Self::Color(color_mix(*a, *b, t)),
            (Self::Gradient(a), Self::Gradient(b))
                if crate::canvas_draw::gradient_discrete(a, b) =>
            {
                if t < 0.5 {
                    Self::Gradient(a.clone())
                } else {
                    Self::Gradient(b.clone())
                }
            }
            (Self::Gradient(_) | Self::Mix(..), Self::Gradient(_) | Self::Mix(..)) => {
                Self::Mix(Arc::new(a.clone()), Arc::new(b.clone()), t)
            }
            _ => {
                if t < 0.5 {
                    a.clone()
                } else {
                    b.clone()
                }
            }
        }
    }
}
pub(crate) fn color_css(c: [f32; 4]) -> String {
    format!(
        "rgb({} {} {} / {})",
        c[0] * 255.,
        c[1] * 255.,
        c[2] * 255.,
        c[3]
    )
}
/// CSS premultiplied-alpha interpolation in the canvas's encoded sRGB space.
pub(crate) fn color_mix(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let alpha = a[3] + (b[3] - a[3]) * t;
    let mut out = [0., 0., 0., alpha];
    if alpha != 0. {
        for i in 0..3 {
            out[i] = (a[i] * a[3] * (1. - t) + b[i] * b[3] * t) / alpha;
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum StyleValue {
    Json(Value),
    Origin([(f32, f32); 2]),
    Path(Vec<(char, Vec<f64>)>),
    Filter(crate::canvas_draw::Filter),
    Paint(Paint),
    Variations(Vec<(String, f64)>),
}
impl StyleValue {
    pub fn parse(kind: I, value: &Value) -> Result<Self> {
        Ok(match kind {
            I::Origin => Self::Origin(v2::transform_origin(
                value
                    .as_str()
                    .context("transform-origin keys are CSS strings")?,
            )?),
            I::FontWeight => Self::Json(match value.as_str() {
                Some("normal") => json!(400.),
                Some("bold") => json!(700.),
                _ => value.clone(),
            }),
            I::Paint => Self::Paint(Paint::parse(value)?),
            I::Filter => Self::Filter(crate::canvas_draw::parse_filter(
                value.as_str().context("filter keys are CSS strings")?,
            )?),
            I::Path => {
                let s = value.as_str().context("d keys are SVG path strings")?;
                let mut commands = Vec::new();
                if s != "none" && !s.trim().is_empty() {
                    for command in svgtypes::PathParser::from(s) {
                        use svgtypes::PathSegment::*;
                        let (abs, cmd, args) = match command? {
                            MoveTo { abs, x, y } => (abs, 'M', vec![x, y]),
                            LineTo { abs, x, y } => (abs, 'L', vec![x, y]),
                            HorizontalLineTo { abs, x } => (abs, 'H', vec![x]),
                            VerticalLineTo { abs, y } => (abs, 'V', vec![y]),
                            CurveTo {
                                abs,
                                x1,
                                y1,
                                x2,
                                y2,
                                x,
                                y,
                            } => (abs, 'C', vec![x1, y1, x2, y2, x, y]),
                            SmoothCurveTo { abs, x2, y2, x, y } => (abs, 'S', vec![x2, y2, x, y]),
                            Quadratic { abs, x1, y1, x, y } => (abs, 'Q', vec![x1, y1, x, y]),
                            SmoothQuadratic { abs, x, y } => (abs, 'T', vec![x, y]),
                            EllipticalArc {
                                abs,
                                rx,
                                ry,
                                x_axis_rotation,
                                large_arc,
                                sweep,
                                x,
                                y,
                            } => (
                                abs,
                                'A',
                                vec![
                                    rx,
                                    ry,
                                    x_axis_rotation,
                                    u8::from(large_arc) as f64,
                                    u8::from(sweep) as f64,
                                    x,
                                    y,
                                ],
                            ),
                            ClosePath { abs } => (abs, 'Z', vec![]),
                        };
                        ensure!(
                            args.iter().all(|v| v.is_finite()),
                            "d keys need finite coordinates"
                        );
                        commands.push((if abs { cmd } else { cmd.to_ascii_lowercase() }, args));
                    }
                }
                Self::Path(commands)
            }
            I::Variations if value.as_str() != Some("normal") => {
                let s = value
                    .as_str()
                    .context("font-variation-settings is a CSS string")?;
                let mut axes = std::collections::BTreeMap::new();
                for axis in s.split(',') {
                    let mut parts = axis.split_whitespace();
                    let tag = parts.next().context("missing axis tag")?;
                    let bytes = tag.as_bytes();
                    ensure!(
                        bytes.len() == 6
                            && matches!(bytes[0], b'\'' | b'"')
                            && bytes[5] == bytes[0]
                            && bytes[1..5].iter().all(|b| (0x20..=0x7e).contains(b)),
                        "font variation axes use quoted four-character ASCII tags"
                    );
                    let number = parts.next().context("missing axis value")?.parse::<f64>()?;
                    ensure!(
                        number.is_finite() && parts.next().is_none(),
                        "invalid font variation axis"
                    );
                    axes.insert(tag[1..5].to_string(), number);
                }
                let axes = axes.into_iter().collect();
                Self::Variations(axes)
            }
            _ => Self::Json(value.clone()),
        })
    }
    pub fn path_commands(&self) -> Result<(Vec<vork_std::moda::PathCmd>, Vec<bool>)> {
        let Self::Path(commands) = self else {
            unreachable!("path interpolation value")
        };
        vork_std::moda::path_segments(commands.iter().map(|(command, v)| {
            use svgtypes::PathSegment::*;
            let abs = command.is_ascii_uppercase();
            Ok(match command.to_ascii_uppercase() {
                'M' => MoveTo {
                    abs,
                    x: v[0],
                    y: v[1],
                },
                'L' => LineTo {
                    abs,
                    x: v[0],
                    y: v[1],
                },
                'H' => HorizontalLineTo { abs, x: v[0] },
                'V' => VerticalLineTo { abs, y: v[0] },
                'C' => CurveTo {
                    abs,
                    x1: v[0],
                    y1: v[1],
                    x2: v[2],
                    y2: v[3],
                    x: v[4],
                    y: v[5],
                },
                'S' => SmoothCurveTo {
                    abs,
                    x2: v[0],
                    y2: v[1],
                    x: v[2],
                    y: v[3],
                },
                'Q' => Quadratic {
                    abs,
                    x1: v[0],
                    y1: v[1],
                    x: v[2],
                    y: v[3],
                },
                'T' => SmoothQuadratic {
                    abs,
                    x: v[0],
                    y: v[1],
                },
                'A' => EllipticalArc {
                    abs,
                    rx: v[0],
                    ry: v[1],
                    x_axis_rotation: v[2],
                    large_arc: v[3] != 0.,
                    sweep: v[4] != 0.,
                    x: v[5],
                    y: v[6],
                },
                'Z' => ClosePath { abs },
                _ => unreachable!("validated SVG command"),
            })
        }))
    }
    pub fn json(&self) -> Value {
        match self {
            Self::Origin(_) => unreachable!("origins are consumed as length-percentage pairs"),
            Self::Json(v) => v.clone(),
            Self::Paint(p) => p.json(),
            Self::Path(cmds) => json!(
                cmds.iter()
                    .map(|(cmd, args)| {
                        let args = args
                            .iter()
                            .enumerate()
                            .map(|(i, n)| {
                                if cmd.eq_ignore_ascii_case(&'A') && (i == 3 || i == 4) {
                                    if *n != 0. { "1".into() } else { "0".into() }
                                } else {
                                    n.to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        format!("{cmd}{args}")
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Self::Filter(f) => {
                let mut s = Vec::new();
                if let Some((xy, r, c)) = f.drop_shadow {
                    s.push(format!(
                        "drop-shadow({}px {}px {}px {})",
                        xy[0],
                        xy[1],
                        r,
                        color_css(c)
                    ));
                }
                if let Some(r) = f.blur {
                    s.push(format!("blur({r}px)"));
                }
                json!(if s.is_empty() {
                    "none".into()
                } else {
                    s.join(" ")
                })
            }
            Self::Variations(axes) => json!(
                axes.iter()
                    .map(|(name, v)| format!("'{name}' {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    pub fn discrete_with(&self, other: &Self, kind: I) -> bool {
        if self == other {
            return false;
        }
        match (self, other) {
            (Self::Origin(_), Self::Origin(_)) => false,
            (Self::Paint(Paint::Color(_)), Self::Paint(Paint::Color(_))) => false,
            (Self::Paint(Paint::Gradient(a)), Self::Paint(Paint::Gradient(b))) => {
                crate::canvas_draw::gradient_discrete(a, b)
            }
            (Self::Path(a), Self::Path(b)) => {
                a.len() != b.len() || !a.iter().zip(b).all(|(a, b)| a.0 == b.0)
            }
            (Self::Variations(a), Self::Variations(b)) => {
                a.len() != b.len() || !a.iter().zip(b).all(|(a, b)| a.0 == b.0)
            }
            (Self::Filter(a), Self::Filter(b)) => {
                let kinds = |f: &crate::canvas_draw::Filter| {
                    [
                        f.drop_shadow.is_some().then_some(0),
                        f.blur.is_some().then_some(1),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                };
                !kinds(a).iter().zip(kinds(b)).all(|(a, b)| *a == b)
            }
            (Self::Json(a), Self::Json(b)) if matches!(kind, I::Number | I::FontWeight) => {
                !(a.is_number() && b.is_number())
            }
            (Self::Json(Value::Array(a)), Self::Json(Value::Array(b))) if kind == I::Repeatable => {
                a.is_empty() || b.is_empty()
            }
            _ => true,
        }
    }
    pub fn mix(&self, other: &Self, t: f32, kind: I) -> Self {
        if t <= 0. || self == other {
            return self.clone();
        }
        if t >= 1. {
            return other.clone();
        }
        let discrete = || if t < 0.5 { self.clone() } else { other.clone() };
        let mix = |a: f64, b: f64| a + (b - a) * f64::from(t);
        match (self, other) {
            (Self::Origin(a), Self::Origin(b)) => Self::Origin(std::array::from_fn(|i| {
                (
                    a[i].0 + (b[i].0 - a[i].0) * t,
                    a[i].1 + (b[i].1 - a[i].1) * t,
                )
            })),
            (Self::Paint(a), Self::Paint(b)) => Self::Paint(Paint::mix(a, b, t)),
            (Self::Path(a), Self::Path(b))
                if a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.0 == b.0) =>
            {
                Self::Path(
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| {
                            (
                                a.0,
                                a.1.iter().zip(&b.1).map(|(&a, &b)| mix(a, b)).collect(),
                            )
                        })
                        .collect(),
                )
            }
            (Self::Filter(a), Self::Filter(b)) => {
                let kinds = |f: &crate::canvas_draw::Filter| {
                    [
                        f.drop_shadow.is_some().then_some(0),
                        f.blur.is_some().then_some(1),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                };
                if !kinds(a).iter().zip(kinds(b)).all(|(a, b)| *a == b) {
                    return discrete();
                }
                let mut out = crate::canvas_draw::Filter::default();
                if a.blur.is_some() || b.blur.is_some() {
                    out.blur = Some(mix(a.blur.unwrap_or(0.), b.blur.unwrap_or(0.)));
                }
                if a.drop_shadow.is_some() || b.drop_shadow.is_some() {
                    let (a, b) = (
                        a.drop_shadow.unwrap_or(([0.; 2], 0., [0.; 4])),
                        b.drop_shadow.unwrap_or(([0.; 2], 0., [0.; 4])),
                    );
                    out.drop_shadow = Some((
                        [mix(a.0[0], b.0[0]), mix(a.0[1], b.0[1])],
                        mix(a.1, b.1),
                        color_mix(a.2, b.2, t),
                    ));
                }
                Self::Filter(out)
            }
            (Self::Variations(a), Self::Variations(b))
                if a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.0 == b.0) =>
            {
                Self::Variations(
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| (a.0.clone(), mix(a.1, b.1)))
                        .collect(),
                )
            }
            (Self::Json(a), Self::Json(b)) if matches!(kind, I::Number | I::FontWeight) => {
                let number = |v: &Value| v.as_f64();
                if let (Some(a), Some(b)) = (number(a), number(b)) {
                    Self::Json(json!(mix(a, b)))
                } else {
                    discrete()
                }
            }
            (Self::Json(Value::Array(a)), Self::Json(Value::Array(b)))
                if kind == I::Repeatable && !a.is_empty() && !b.is_empty() =>
            {
                fn gcd(a: usize, b: usize) -> usize {
                    if b == 0 { a } else { gcd(b, a % b) }
                }
                let n = a.len() / gcd(a.len(), b.len()) * b.len();
                Self::Json(
                    (0..n)
                        .map(|i| {
                            json!(mix(
                                a[i % a.len()].as_f64().unwrap(),
                                b[i % b.len()].as_f64().unwrap()
                            ))
                        })
                        .collect(),
                )
            }
            _ => discrete(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct StyleTrack {
    pub times: Vec<f32>,
    pub values: Vec<Arc<StyleValue>>,
    pub kind: I,
    pub step: bool,
    pub clocks: Option<Vec<vork_time::Clock>>,
}
impl StyleTrack {
    pub fn sample(
        &self,
        time: f32,
        upper: Option<usize>,
        exact: Option<&vork_time::exact::Germ>,
    ) -> Arc<StyleValue> {
        let upper = upper.unwrap_or_else(|| self.times.partition_point(|t| *t <= time));
        let lo = upper.saturating_sub(1);
        if self.step || upper == 0 || upper >= self.times.len() {
            return self.values[lo].clone();
        }
        if self.values[lo].discrete_with(&self.values[upper], self.kind) {
            if let (Some(clocks), Some(exact)) = (&self.clocks, exact) {
                let midpoint = (clocks[lo].exact() + clocks[upper].exact())
                    / vork_time::exact::Rational::from_integer(2.into());
                return self.values[if exact.compare_point(&midpoint).is_lt() {
                    lo
                } else {
                    upper
                }]
                .clone();
            }
        }
        let t = (time - self.times[lo]) / (self.times[upper] - self.times[lo]);
        Arc::new(self.values[lo].mix(&self.values[upper], t, self.kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(kind: I, a: Value, b: Value, t: f32) -> Value {
        StyleValue::parse(kind, &a)
            .unwrap()
            .mix(&StyleValue::parse(kind, &b).unwrap(), t, kind)
            .json()
    }
    #[test]
    fn paths_keep_commands_and_interpolate_parameters() {
        let path = sample(
            I::Path,
            json!("M0 0H10V20A4 5 0 0 0 30 40z"),
            json!("M10 20H30V40A8 9 20 1 1 50 60z"),
            0.5,
        );
        assert_eq!(path, json!("M5 10 H20 V30 A6 7 10 1 1 40 50 z"));
        let a = json!("M0 0L10 10");
        let b = json!("M0 0Q5 10 10 10");
        assert_eq!(
            sample(I::Path, a.clone(), b.clone(), 0.49),
            json!("M0 0 L10 10")
        );
        assert_eq!(sample(I::Path, a, b, 0.5), json!("M0 0 Q5 10 10 10"));
        assert!(StyleValue::parse(I::Path, &json!("M0 nonsense")).is_err());
    }
    #[test]
    fn sampled_paths_use_the_same_svg_normalization() {
        for path in [
            "M1 2h3v4c1 2 3 4 5 6s7 8 9 10q1 2 3 4t5 6a4 5 30 1 0 8 9z l2 3",
            "M1 2L3 4H5V6C1 2 3 4 5 6S7 8 9 10Q1 2 3 4T5 6A4 5 30 0 1 8 9Z",
        ] {
            let value = StyleValue::parse(I::Path, &json!(path)).unwrap();
            assert_eq!(
                value.path_commands().unwrap(),
                vork_std::moda::path_commands(path).unwrap()
            );
        }
    }
    #[test]
    fn filters_pad_with_neutral_functions_and_mismatch_is_discrete() {
        assert_eq!(
            sample(I::Filter, json!("none"), json!("blur(8px)"), 0.25),
            json!("blur(2px)")
        );
        let a = StyleValue::parse(I::Filter, &json!("drop-shadow(2px 4px 6px red)")).unwrap();
        let b = StyleValue::parse(
            I::Filter,
            &json!("drop-shadow(6px 8px 10px blue) blur(4px)"),
        )
        .unwrap();
        let StyleValue::Filter(f) = a.mix(&b, 0.5, I::Filter) else {
            panic!()
        };
        assert_eq!(f.blur, Some(2.));
        assert_eq!(f.drop_shadow, Some(([4., 6.], 8., [0.5, 0., 0.5, 1.])));
        let a = json!("blur(4px)");
        let b = json!("drop-shadow(0 0 2px red)");
        assert_eq!(sample(I::Filter, a.clone(), b.clone(), 0.49), a);
        assert_eq!(
            sample(I::Filter, a, b, 0.5),
            json!("drop-shadow(0px 0px 2px rgb(255 0 0 / 1))")
        );
    }
    #[test]
    fn alpha_is_premultiplied_and_none_is_discrete() {
        assert_eq!(
            color_mix([1., 0., 0., 0.], [0., 0., 1., 1.], 0.5),
            [0., 0., 1., 0.5]
        );
        assert_eq!(
            sample(I::Paint, json!("none"), json!("red"), 0.49),
            json!("none")
        );
        assert_eq!(
            sample(I::Paint, json!("none"), json!("red"), 0.5),
            json!("rgb(255 0 0 / 1)")
        );
    }
    #[test]
    fn dash_lists_repeat_and_variation_axes_match_by_tag() {
        assert_eq!(
            sample(I::Repeatable, json!([2, 4]), json!([4, 8, 12]), 0.5),
            json!([3., 6., 7., 4., 5., 8.])
        );
        assert_eq!(
            sample(
                I::Variations,
                json!("'wght' 100, 'wdth' 50"),
                json!("'wdth' 100, 'wght' 900"),
                0.5
            ),
            json!("'wdth' 75, 'wght' 500")
        );
        assert_eq!(
            sample(I::Number, json!("normal"), json!(2), 0.25),
            json!("normal")
        );
        assert_eq!(
            sample(I::FontWeight, json!("normal"), json!("bold"), 0.5),
            json!(550.)
        );
    }
    #[test]
    fn gradient_keywords_turn_shortest_and_mixed_stops_are_discrete() {
        let bounds = [0., 0., 100., 100.];
        let actual = crate::canvas_draw::interpolate_gradient(
            "linear-gradient(to left, red, blue)",
            "linear-gradient(to top, red, blue)",
            0.5,
            bounds,
        )
        .unwrap();
        let expected =
            crate::canvas_draw::gradient_paint("linear-gradient(315deg, red, blue)", bounds)
                .unwrap();
        assert_eq!(actual, expected);
        let a = json!("linear-gradient(red 10px, blue 100%)");
        let b = json!("linear-gradient(white 20px, black 90%)");
        assert_eq!(sample(I::Paint, a.clone(), b.clone(), 0.49), a);
        assert_eq!(sample(I::Paint, a, b.clone(), 0.5), b);
    }
    #[test]
    fn discrete_midpoints_keep_the_approached_side() {
        let one = vork_time::exact::Rational::from_integer(1.into());
        let track = StyleTrack {
            times: vec![0., 2.],
            kind: I::Discrete,
            step: false,
            values: vec![
                Arc::new(StyleValue::Json(json!("left"))),
                Arc::new(StyleValue::Json(json!("right"))),
            ],
            clocks: Some(vec!["0".parse().unwrap(), "2".parse().unwrap()]),
        };
        assert_eq!(
            track
                .sample(
                    1.,
                    Some(1),
                    Some(&vork_time::exact::Germ::left(one.clone()))
                )
                .json(),
            json!("left")
        );
        assert_eq!(
            track
                .sample(1., Some(1), Some(&vork_time::exact::Germ::point(one)))
                .json(),
            json!("right")
        );
    }
    #[test]
    fn step_holds_and_linear_discrete_switches_mid_segment() {
        let track = StyleTrack {
            times: vec![0., 2.],
            values: vec![
                Arc::new(StyleValue::Json(json!("left"))),
                Arc::new(StyleValue::Json(json!("right"))),
            ],
            kind: I::Discrete,
            step: false,
            clocks: None,
        };
        assert_eq!(track.sample(0.9, None, None).json(), json!("left"));
        assert_eq!(track.sample(1., None, None).json(), json!("right"));
        let track = StyleTrack {
            step: true,
            ..track
        };
        assert_eq!(track.sample(1.9, None, None).json(), json!("left"));
        assert_eq!(track.sample(2., None, None).json(), json!("right"));
    }
}
