//! Interpolating paint stacks between two blend objects.
//!
//! Layers pair bottom to top by index (hidden layers are left out first). A layer only one side has fades in or out:
//! it keeps its paint and its opacity scales by `t` (or `1 - t`). A paired layer interpolates:
//!
//! - **solid ↔ solid**: the colours, per channel with alpha, in sRGB (Illustrator's RGB blend).
//! - **gradient ↔ gradient** of one kind: the placement (ends, centre, radii, rotation, angle) and the stops. Stops
//!   are re-sampled at the union of both gradients' stop offsets, so neither loses a stop.
//! - **gradient ↔ solid**: the solid acts as a gradient of the other's kind and placement with every stop its colour.
//! - **gradient ↔ gradient of another kind**: the step takes the nearer object's kind and placement; the stops still
//!   interpolate.
//! - **image, video or shader** on either side: not interpolable, so the step takes the nearer object's paint
//!   (Illustrator's blends of patterns do the same).
//!
//! Layer opacity interpolates; blend mode and every non-numeric stroke setting (alignment, caps, joins, arrowheads,
//! brushes) come from the nearer object. Stroke width, miter limit, dash offset and, when both dash arrays have the
//! same length, the dashes interpolate. A colour that is not a string (an unresolved variable) can't be
//! interpolated and also comes from the nearer object.

use super::color::{parse_rgba_string, rgba_to_css, Rgba255};
use crate::paint::{
    self, ConicGradientPaint, Fill, GradientStop, LinearGradientPaint, Paint, PaintPoint,
    RadialGradientPaint, SolidPaint, Stroke,
};
use crate::value::{json, Value};

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn nearer<T: Clone>(a: &T, b: &T, t: f64) -> T {
    if t < 0.5 {
        a.clone()
    } else {
        b.clone()
    }
}

fn lerp_point(a: PaintPoint, b: PaintPoint, t: f64) -> PaintPoint {
    PaintPoint {
        x: lerp(a.x, b.x, t),
        y: lerp(a.y, b.y, t),
    }
}

/// A colour value as RGBA (0-255 channels, 0-1 alpha), when it is a parsable string.
pub(crate) fn color_of(v: &Value) -> Option<Rgba255> {
    parse_rgba_string(Some(v.as_str()?))
}

fn lerp_rgba(a: Rgba255, b: Rgba255, t: f64) -> Rgba255 {
    Rgba255 {
        r: lerp(a.r, b.r, t),
        g: lerp(a.g, b.g, t),
        b: lerp(a.b, b.b, t),
        a: lerp(a.a, b.a, t),
    }
}

fn lerp_color(a: &Value, b: &Value, t: f64) -> Value {
    match (color_of(a), color_of(b)) {
        (Some(x), Some(y)) => json!(rgba_to_css(lerp_rgba(x, y, t))),
        _ => nearer(a, b, t),
    }
}

/// Stops as (offset, colour), sorted, offsets clamped to 0..1; unparsable colours are opaque black.
fn stops_rgba(stops: &[GradientStop]) -> Vec<(f64, Rgba255)> {
    let black = Rgba255 {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    let mut out: Vec<(f64, Rgba255)> = stops
        .iter()
        .map(|s| {
            let o = if s.offset.is_finite() {
                s.offset.clamp(0.0, 1.0)
            } else {
                0.0
            };
            (o, color_of(&s.color).unwrap_or(black))
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

fn sample_stops(stops: &[(f64, Rgba255)], t: f64) -> Rgba255 {
    let Some(first) = stops.first() else {
        return Rgba255 {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
    };
    if t <= first.0 {
        return first.1;
    }
    for w in stops.windows(2) {
        let ((ap, ac), (bp, bc)) = (w[0], w[1]);
        if t <= bp {
            let span = bp - ap;
            return if span > 0.0 {
                lerp_rgba(ac, bc, (t - ap) / span)
            } else {
                bc
            };
        }
    }
    stops[stops.len() - 1].1
}

/// Stops re-sampled at the union of both sets' offsets, then interpolated.
fn lerp_stops(a: &[GradientStop], b: &[GradientStop], t: f64) -> Vec<GradientStop> {
    let (sa, sb) = (stops_rgba(a), stops_rgba(b));
    let mut offsets: Vec<f64> = sa.iter().chain(&sb).map(|s| s.0).collect();
    offsets.sort_by(f64::total_cmp);
    offsets.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
    offsets
        .into_iter()
        .map(|o| GradientStop {
            offset: o,
            color: json!(rgba_to_css(lerp_rgba(
                sample_stops(&sa, o),
                sample_stops(&sb, o),
                t
            ))),
        })
        .collect()
}

fn flat_stops(color: &Value) -> Vec<GradientStop> {
    vec![
        GradientStop {
            offset: 0.0,
            color: color.clone(),
        },
        GradientStop {
            offset: 1.0,
            color: color.clone(),
        },
    ]
}

/// A gradient paint with its stops replaced.
fn with_stops(p: &Paint, stops: Vec<GradientStop>) -> Paint {
    match p {
        Paint::Linear(g) => Paint::Linear(LinearGradientPaint { stops, ..g.clone() }),
        Paint::Radial(g) => Paint::Radial(RadialGradientPaint { stops, ..g.clone() }),
        Paint::Diamond(g) => Paint::Diamond(RadialGradientPaint { stops, ..g.clone() }),
        Paint::Conic(g) => Paint::Conic(ConicGradientPaint { stops, ..g.clone() }),
        other => other.clone(),
    }
}

fn is_gradient(p: &Paint) -> bool {
    paint::paint_stops(p).is_some()
}

/// Interpolate one paint (see the module docs).
pub fn lerp_paint(a: &Paint, b: &Paint, t: f64) -> Paint {
    match (a, b) {
        (Paint::Solid(x), Paint::Solid(y)) => Paint::Solid(SolidPaint {
            color: lerp_color(&x.color, &y.color, t),
        }),
        (Paint::Solid(x), g) if is_gradient(g) => {
            lerp_paint(&with_stops(g, flat_stops(&x.color)), g, t)
        }
        (g, Paint::Solid(y)) if is_gradient(g) => {
            lerp_paint(g, &with_stops(g, flat_stops(&y.color)), t)
        }
        (Paint::Linear(x), Paint::Linear(y)) => Paint::Linear(LinearGradientPaint {
            start: lerp_point(x.start, y.start, t),
            end: lerp_point(x.end, y.end, t),
            stops: lerp_stops(&x.stops, &y.stops, t),
            options: if t < 0.5 { x.options } else { y.options },
        }),
        (Paint::Radial(x), Paint::Radial(y)) => Paint::Radial(lerp_radial(x, y, t)),
        (Paint::Diamond(x), Paint::Diamond(y)) => Paint::Diamond(lerp_radial(x, y, t)),
        (Paint::Conic(x), Paint::Conic(y)) => Paint::Conic(ConicGradientPaint {
            center: lerp_point(x.center, y.center, t),
            angle: lerp(x.angle, y.angle, t),
            stops: lerp_stops(&x.stops, &y.stops, t),
            options: if t < 0.5 { x.options } else { y.options },
        }),
        (x, y) if is_gradient(x) && is_gradient(y) => {
            let stops = lerp_stops(
                paint::paint_stops(x).unwrap_or(&[]),
                paint::paint_stops(y).unwrap_or(&[]),
                t,
            );
            with_stops(&nearer(x, y, t), stops)
        }
        _ => nearer(a, b, t),
    }
}

fn lerp_radial(x: &RadialGradientPaint, y: &RadialGradientPaint, t: f64) -> RadialGradientPaint {
    RadialGradientPaint {
        center: lerp_point(x.center, y.center, t),
        radius_x: lerp(x.radius_x, y.radius_x, t),
        radius_y: lerp(x.radius_y, y.radius_y, t),
        rotation: lerp(x.rotation, y.rotation, t),
        stops: lerp_stops(&x.stops, &y.stops, t),
        options: if t < 0.5 { x.options } else { y.options },
    }
}

fn layer_opacity(o: Option<f64>) -> f64 {
    o.filter(|v| v.is_finite()).unwrap_or(1.0)
}

/// Opacity `None` when it is 1 (the stored default).
fn opacity_field(o: f64) -> Option<f64> {
    if (o - 1.0).abs() < 1e-12 {
        None
    } else {
        Some(o.clamp(0.0, 1.0))
    }
}

/// Pair two visible layer lists by index; `f` gets both (or one) and `t`.
fn pair_layers<L: Clone, O>(
    a: &[L],
    b: &[L],
    mut f: impl FnMut(Option<&L>, Option<&L>) -> O,
) -> Vec<O> {
    (0..a.len().max(b.len()))
        .map(|i| f(a.get(i), b.get(i)))
        .collect()
}

/// Interpolate two fill stacks (bottom to top).
pub fn lerp_fills(a: &[Fill], b: &[Fill], t: f64) -> Vec<Fill> {
    let shown = |v: &[Fill]| -> Vec<Fill> {
        v.iter()
            .filter(|f| paint::shown(f.visible, f.opacity))
            .cloned()
            .collect()
    };
    let (a, b) = (shown(a), shown(b));
    pair_layers(&a, &b, |x, y| match (x, y) {
        (Some(x), Some(y)) => Fill {
            id: String::new(),
            visible: None,
            opacity: opacity_field(lerp(layer_opacity(x.opacity), layer_opacity(y.opacity), t)),
            blend_mode: nearer(&x.blend_mode, &y.blend_mode, t),
            overprint: nearer(&x.overprint, &y.overprint, t),
            paint: lerp_paint(&x.paint, &y.paint, t),
            // Entry effects switch at the midpoint, as other unmatched settings do.
            effects: if t < 0.5 { x.effects.clone() } else { y.effects.clone() },
        },
        (Some(x), None) => Fill {
            opacity: opacity_field(layer_opacity(x.opacity) * (1.0 - t)),
            ..x.clone()
        },
        (None, Some(y)) => Fill {
            opacity: opacity_field(layer_opacity(y.opacity) * t),
            ..y.clone()
        },
        (None, None) => unreachable!(),
    })
    .into_iter()
    .enumerate()
    .map(|(i, f)| Fill {
        id: format!("fill-{i}"),
        ..f
    })
    .collect()
}

fn lerp_opt(a: Option<f64>, b: Option<f64>, t: f64) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(lerp(x, y, t)),
        _ => nearer(&a, &b, t),
    }
}

/// Interpolate two stroke stacks (bottom to top).
pub fn lerp_strokes(a: &[Stroke], b: &[Stroke], t: f64) -> Vec<Stroke> {
    let shown = |v: &[Stroke]| -> Vec<Stroke> {
        v.iter()
            .filter(|s| paint::shown(s.visible, s.opacity) && s.width > 0.0)
            .cloned()
            .collect()
    };
    let (a, b) = (shown(a), shown(b));
    pair_layers(&a, &b, |x, y| match (x, y) {
        (Some(x), Some(y)) => {
            let base = nearer(x, y, t);
            let dash = match (&x.dash, &y.dash) {
                (Some(dx), Some(dy)) if dx.len() == dy.len() => {
                    Some(dx.iter().zip(dy).map(|(p, q)| lerp(*p, *q, t)).collect())
                }
                _ => base.dash.clone(),
            };
            Stroke {
                id: String::new(),
                visible: None,
                opacity: opacity_field(lerp(layer_opacity(x.opacity), layer_opacity(y.opacity), t)),
                paint: lerp_paint(&x.paint, &y.paint, t),
                width: lerp(x.width, y.width, t),
                miter_limit: lerp_opt(x.miter_limit, y.miter_limit, t),
                dash_offset: lerp_opt(x.dash_offset, y.dash_offset, t),
                arrow_scale: lerp_opt(x.arrow_scale, y.arrow_scale, t),
                dash,
                ..base
            }
        }
        (Some(x), None) => Stroke {
            opacity: opacity_field(layer_opacity(x.opacity) * (1.0 - t)),
            ..x.clone()
        },
        (None, Some(y)) => Stroke {
            opacity: opacity_field(layer_opacity(y.opacity) * t),
            ..y.clone()
        },
        (None, None) => unreachable!(),
    })
    .into_iter()
    .enumerate()
    .map(|(i, s)| Stroke {
        id: format!("stroke-{i}"),
        ..s
    })
    .collect()
}

/// Every colour a stack paints (solid colours and gradient stops) — for the smooth-colour step count.
pub fn stack_colors(fills: &[Fill], strokes: &[Stroke]) -> Vec<Rgba255> {
    let mut out = Vec::new();
    let mut add = |p: &Paint| match p {
        Paint::Solid(s) => out.extend(color_of(&s.color)),
        other => {
            if let Some(stops) = paint::paint_stops(other) {
                out.extend(stops.iter().filter_map(|s| color_of(&s.color)));
            }
        }
    };
    for f in fills.iter().filter(|f| paint::shown(f.visible, f.opacity)) {
        add(&f.paint);
    }
    for s in strokes
        .iter()
        .filter(|s| paint::shown(s.visible, s.opacity))
    {
        add(&s.paint);
    }
    out
}
