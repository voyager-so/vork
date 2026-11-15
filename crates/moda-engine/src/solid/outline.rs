//! A layer's filled outline as flat contours, in its own coordinates, for extrusion (motion-model.md §12.6 phase 2).
//!
//! Shapes (rectangles with their corner radii, ellipses with their arcs, polygons, stars and paths, vector networks
//! included) come from the renderer's own path builders; text from the vector text engine's glyph outlines, a fill
//! per colour run; a group or frame is its children's outlines through their transforms, each in its own colour.
//!
//! A visible stroke is a part of its own, in the stroke's colour, after (on top of) the fill: the stroke's outline
//! (`tess::outline::stroke_contours`, with its width, caps, joins and width profile), clipped to the shape for an
//! inside / outside stroke. A line is its body's outline plus its arrowheads. Dashes, brushes and text strokes are
//! not extruded (a dashed stroke extrudes solid; a text's stroke is left out).

use moda_scene::types::{Affine, RawNode};
use moda_scene::Scene;

use crate::tess::moda_engine_path_types::PathCmd;

/// Filled contours of one colour (straight sRGB, 0–1).
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub contours: Vec<Vec<[f64; 2]>>,
    pub color: [f32; 4],
}

/// Curves are cut into lines no more than this far from the curve (layer px).
pub const TOLERANCE: f64 = 0.25;

fn flatten_cubic(out: &mut Vec<[f64; 2]>, p0: [f64; 2], c1: [f64; 2], c2: [f64; 2], p3: [f64; 2], tol: f64) {
    // Wang's bound on the second difference: segments so each stays within `tol` of the curve.
    let dd = |a: f64, b: f64, c: f64| (a - 2.0 * b + c).abs();
    let m = (dd(p0[0], c1[0], c2[0]).max(dd(c1[0], c2[0], p3[0])))
        .hypot(dd(p0[1], c1[1], c2[1]).max(dd(c1[1], c2[1], p3[1])));
    let n = ((0.75 * m / tol).sqrt().ceil() as usize).clamp(1, 256);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        let u = 1.0 - t;
        let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
        out.push([
            a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0],
            a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1],
        ]);
    }
}

/// Path commands as closed polylines (each subpath closed; repeated points dropped; fewer than 3 points dropped).
/// Cleanup removes exact repeats only: `tol` bounds curve error in the caller's units,
/// and consumers apply their own output-precision policy to the flattened contours.
pub fn flatten(cmds: &[PathCmd], tol: f64) -> Vec<Vec<[f64; 2]>> {
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    let mut at = [0.0, 0.0];
    let finish = |cur: &mut Vec<[f64; 2]>, out: &mut Vec<Vec<[f64; 2]>>| {
        let mut c: Vec<[f64; 2]> = Vec::with_capacity(cur.len());
        for p in cur.drain(..) {
            if p.iter().all(|v| v.is_finite()) && c.last() != Some(&p) {
                c.push(p);
            }
        }
        while c.len() > 1 && c.first() == c.last() {
            c.pop();
        }
        if c.len() >= 3 {
            out.push(c);
        }
    };
    for cmd in cmds {
        match *cmd {
            PathCmd::Move { x, y } => {
                finish(&mut cur, &mut out);
                at = [x, y];
                cur.push(at);
            }
            PathCmd::Line { x, y } => {
                if cur.is_empty() {
                    cur.push(at);
                }
                at = [x, y];
                cur.push(at);
            }
            PathCmd::Curve { x, y, cp1, cp2 } => {
                if cur.is_empty() {
                    cur.push(at);
                }
                flatten_cubic(&mut cur, at, [cp1.0, cp1.1], [cp2.0, cp2.1], [x, y], tol);
                at = [x, y];
            }
            PathCmd::Quadratic { x, y, cp } => {
                if cur.is_empty() {
                    cur.push(at);
                }
                let c1 = [at[0] + 2.0 / 3.0 * (cp.0 - at[0]), at[1] + 2.0 / 3.0 * (cp.1 - at[1])];
                let c2 = [x + 2.0 / 3.0 * (cp.0 - x), y + 2.0 / 3.0 * (cp.1 - y)];
                flatten_cubic(&mut cur, at, c1, c2, [x, y], tol);
                at = [x, y];
            }
            PathCmd::Arc { x, y, rx, ry, x_axis_rotation, large_arc, sweep } => {
                if cur.is_empty() {
                    cur.push(at);
                }
                let arc = moda_scene::path_geom::arc::ArcParams { rx, ry, x_axis_rotation, large_arc, sweep, x, y };
                let beziers = moda_scene::path_geom::arc::arc_to_beziers(at[0], at[1], &arc);
                if beziers.is_empty() {
                    cur.push([x, y]);
                }
                let mut from = at;
                for b in beziers {
                    flatten_cubic(&mut cur, from, b.cp1, b.cp2, b.end, tol);
                    from = b.end;
                }
                at = [x, y];
            }
        }
    }
    finish(&mut cur, &mut out);
    out
}

fn blend_to_cmds(points: &[moda_scene::blend::BlendPathPoint]) -> Vec<PathCmd> {
    points.iter().map(crate::tess::path_cmd_from_blend_point).collect()
}

/// `#rgb[a]`, `#rrggbb[aa]`, `rgb()` / `rgba()` as straight sRGB 0–1.
pub fn parse_color(s: &str) -> Option<[f32; 4]> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix('#') {
        let d = |i: usize, w: usize| u8::from_str_radix(h.get(i..i + w)?, 16).ok().map(|v| {
            if w == 1 { f32::from(v) * 17.0 / 255.0 } else { f32::from(v) / 255.0 }
        });
        return match h.len() {
            3 => Some([d(0, 1)?, d(1, 1)?, d(2, 1)?, 1.0]),
            4 => Some([d(0, 1)?, d(1, 1)?, d(2, 1)?, d(3, 1)?]),
            6 => Some([d(0, 2)?, d(2, 2)?, d(4, 2)?, 1.0]),
            8 => Some([d(0, 2)?, d(2, 2)?, d(4, 2)?, d(6, 2)?]),
            _ => None,
        };
    }
    let inner = s.strip_prefix("rgba(").or_else(|| s.strip_prefix("rgb("))?.strip_suffix(')')?;
    let v: Vec<f32> = inner.split(',').map(|p| p.trim().parse().ok()).collect::<Option<_>>()?;
    match v.as_slice() {
        [r, g, b] => Some([r / 255.0, g / 255.0, b / 255.0, 1.0]),
        [r, g, b, a] => Some([r / 255.0, g / 255.0, b / 255.0, *a]),
        _ => None,
    }
}

/// A node's paint colour: its first visible solid fill, else a gradient's first stop, else a mid grey.
pub fn node_color(node: &RawNode) -> [f32; 4] {
    let fills = node.json_field("fills").and_then(|f| f.as_array()).cloned().unwrap_or_default();
    for f in fills.iter().filter(|f| f.get("visible").and_then(|v| v.as_bool()) != Some(false)) {
        let Some(p) = f.get("paint") else { continue };
        let c = match p.get("type").and_then(|t| t.as_str()) {
            Some("solid") => p.get("color").and_then(|c| c.as_str()).and_then(parse_color),
            _ => p
                .get("stops")
                .and_then(|s| s.as_array())
                .and_then(|s| s.first())
                .and_then(|s| s.get("color"))
                .and_then(|c| c.as_str())
                .and_then(parse_color),
        };
        if let Some(mut c) = c {
            if let Some(o) = f.get("opacity").and_then(|o| o.as_f64()) {
                c[3] *= o as f32;
            }
            return c;
        }
    }
    [0.6, 0.6, 0.6, 1.0]
}

/// A node's top visible stroke colour (its solid colour, else a gradient's first stop, else a mid grey) times the
/// stroke's opacity, and its width; `None` without a visible stroke of positive width.
pub fn stroke_color(node: &RawNode) -> Option<([f32; 4], f64)> {
    use moda_scene::paint::Paint;
    let stroke = moda_scene::paint::top_stroke(node)?;
    if !(stroke.width.is_finite() && stroke.width > 0.0) {
        return None;
    }
    let color = match &stroke.paint {
        Paint::Solid(p) => p.color.as_str().and_then(parse_color),
        p => moda_scene::paint::paint_stops(p)
            .and_then(|s| s.first())
            .and_then(|s| s.color.as_str())
            .and_then(parse_color),
    };
    let mut color = color.unwrap_or([0.6, 0.6, 0.6, 1.0]);
    if let Some(o) = stroke.opacity.filter(|o| o.is_finite()) {
        color[3] *= o.clamp(0.0, 1.0) as f32;
    }
    Some((color, stroke.width))
}

/// `contours` as clean, non-overlapping contours (nonzero union): what the extruder's walls and bevel expect.
fn simplify(contours: Vec<Vec<[f64; 2]>>) -> Vec<Vec<[f64; 2]>> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::float::simplify::SimplifyShape;
    if contours.is_empty() {
        return contours;
    }
    contours.simplify_shape(FillRule::NonZero).into_iter().flatten().filter(|c| c.len() >= 3).collect()
}

/// The node's stroke along `cmds` (its outline, node-local) as a part in the stroke colour (see the module docs).
/// A closed shape's inside / outside stroke is a centred stroke twice as wide clipped to the inside / outside.
fn stroke_shape(node: &RawNode, cmds: &[PathCmd], closed: bool) -> Option<Shape> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::simplify::SimplifyShape;
    use i_overlay::float::single::SingleFloatOverlay;
    let (color, width) = stroke_color(node)?;
    let style = moda_scene::paint::node_stroke_style(node);
    let align = if closed { style.align.as_deref() } else { None };
    let aligned = matches!(align, Some("inside" | "outside"));
    let band = crate::tess::outline::stroke_contours(
        node,
        cmds,
        closed,
        if aligned { 2.0 * width } else { width },
        None,
        None,
        true,
        TOLERANCE,
    );
    let band = simplify(band);
    let contours = if aligned && !band.is_empty() {
        let even_odd = node.str("fillRule") == Some("evenodd");
        let region: Vec<Vec<[f64; 2]>> = flatten(cmds, TOLERANCE)
            .simplify_shape(if even_odd { FillRule::EvenOdd } else { FillRule::NonZero })
            .into_iter()
            .flatten()
            .collect();
        let rule = if align == Some("outside") { OverlayRule::Difference } else { OverlayRule::Intersect };
        band.overlay(&region, rule, FillRule::NonZero).into_iter().flatten().filter(|c| c.len() >= 3).collect()
    } else {
        band
    };
    (!contours.is_empty()).then_some(Shape { contours, color })
}

/// A line's stroke as one part: its body's outline and its arrowheads (filled or open), as the line draws them.
fn line_shape(node: &RawNode) -> Option<Shape> {
    use crate::tess::line_to_path as ltp;
    let (color, _) = stroke_color(node)?;
    let mut spec = ltp::LineSpec::from_json(|k| node.json_field(k));
    spec.dash = None;
    if crate::tess::outline::needs_outline(node, |_| None::<&moda_scene::value::Value>) {
        crate::tess::outline::outlined_line_spec(node, &mut spec);
    }
    let parts = if spec.has_arrows() {
        let mut v = vec![ltp::convert_line_body(&spec)];
        v.extend(ltp::convert_arrow_heads(&spec));
        v
    } else {
        vec![ltp::convert_line_to_path(&spec)]
    };
    let mut contours = Vec::new();
    for part in &parts {
        match part.kind {
            ltp::LinePartKind::ArrowFill => contours.extend(flatten(&part.cmds, TOLERANCE)),
            kind if part.stroke_width > 0.0 => contours.extend(crate::tess::outline::stroke_contours(
                node,
                &part.cmds,
                part.closed,
                part.stroke_width,
                part.line_cap.as_deref(),
                part.line_join.as_deref(),
                kind == ltp::LinePartKind::Body,
                TOLERANCE,
            )),
            _ => {}
        }
    }
    let contours = simplify(contours);
    (!contours.is_empty()).then_some(Shape { contours, color })
}

fn transform(c: &[[f64; 2]], a: &Affine) -> Vec<[f64; 2]> {
    c.iter().map(|p| [a.a * p[0] + a.c * p[1] + a.e, a.b * p[0] + a.d * p[1] + a.f]).collect()
}

fn invert(a: &Affine) -> Option<Affine> {
    let det = a.a * a.d - a.b * a.c;
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    Some(Affine {
        a: a.d / det,
        b: -a.b / det,
        c: -a.c / det,
        d: a.a / det,
        e: (a.c * a.f - a.d * a.e) / det,
        f: (a.b * a.e - a.a * a.f) / det,
    })
}

/// The node's filled outlines in its own coordinates (see the module docs). Hidden children are left out.
pub fn node_shapes(scene: &Scene, id: &str) -> Vec<Shape> {
    let mut out = Vec::new();
    collect(scene, id, None, &mut out, 0);
    out
}

fn collect(scene: &Scene, id: &str, to_root: Option<&Affine>, out: &mut Vec<Shape>, depth: usize) {
    let Some(node) = scene.effective_node(id) else { return };
    if depth > 0 && node.bool("hidden") == Some(true) {
        return;
    }
    let place = |c: Vec<Vec<[f64; 2]>>| -> Vec<Vec<[f64; 2]>> {
        match to_root {
            Some(a) => c.iter().map(|c| transform(c, a)).collect(),
            None => c,
        }
    };
    match node.node_type() {
        "group" | "container" | "section" | "boolean" if depth < 32 => {
            // A frame's own fill is a shape too, behind its children.
            if node.node_type() != "group" && node.node_type() != "boolean" {
                let (w, h) = (node.f64("width").unwrap_or(0.0), node.f64("height").unwrap_or(0.0));
                let has_fill = node.json_field("fills").and_then(|f| f.as_array()).is_some_and(|f| !f.is_empty());
                if w > 0.0 && h > 0.0 {
                    let cmds = crate::shape_to_path::rectangle_to_path_cmds(w, h, node.json_field("cornerRadius"));
                    if has_fill {
                        out.push(Shape { contours: place(flatten(&cmds, TOLERANCE)), color: node_color(node) });
                    }
                    if let Some(s) = stroke_shape(node, &cmds, true) {
                        out.push(Shape { contours: place(s.contours), color: s.color });
                    }
                }
            }
            let own = scene.node_canvas_affine(id);
            let kids: Vec<String> = node
                .json_field("children")
                .and_then(|c| c.as_array())
                .map(|a| a.iter().filter_map(|c| c.as_str().map(String::from)).collect())
                .unwrap_or_default();
            for k in kids {
                let rel = match (own.as_ref().and_then(invert), scene.node_canvas_affine(&k)) {
                    (Some(inv), Some(kc)) => moda_scene::transform::compose(inv, kc),
                    _ => continue,
                };
                let a = match to_root {
                    Some(t) => moda_scene::transform::compose(*t, rel),
                    None => rel,
                };
                collect(scene, &k, Some(&a), out, depth + 1);
            }
        }
        "path" => {
            let cmds = crate::tess::path_cmds_for_node(node);
            out.push(Shape { contours: place(flatten(&cmds, TOLERANCE)), color: node_color(node) });
            if let Some(s) = stroke_shape(node, &cmds, node.bool("closed") == Some(true)) {
                out.push(Shape { contours: place(s.contours), color: s.color });
            }
        }
        "line" => {
            if let Some(s) = line_shape(node) {
                out.push(Shape { contours: place(s.contours), color: s.color });
            }
        }
        "richtext" => {
            let Ok(fills) = crate::text_geometry::node_solid_fills(scene, id) else { return };
            for f in fills.iter().filter(|f| f.kind == "glyphs" || f.kind == "shape") {
                let mut pts = Vec::new();
                for d in &f.paths {
                    crate::boolean_operands::parse_glyph_d(d, &mut pts);
                }
                let mut color = f.color;
                color[3] *= f.opacity;
                out.push(Shape { contours: place(flatten(&blend_to_cmds(&pts), TOLERANCE)), color });
            }
        }
        _ => {
            if let Some(pts) = crate::boolean_operands::undistorted_outline(scene, node) {
                let cmds = blend_to_cmds(&pts);
                out.push(Shape { contours: place(flatten(&cmds, TOLERANCE)), color: node_color(node) });
                if let Some(s) = stroke_shape(node, &cmds, true) {
                    out.push(Shape { contours: place(s.contours), color: s.color });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_and_arcs_flatten_within_tolerance() {
        let cmds = [
            PathCmd::Move { x: 0.0, y: 0.0 },
            PathCmd::Curve { x: 100.0, y: 0.0, cp1: (0.0, 100.0), cp2: (100.0, 100.0) },
            PathCmd::Line { x: 0.0, y: 0.0 },
        ];
        let c = flatten(&cmds, 0.25);
        assert_eq!(c.len(), 1);
        // The cubic's apex is at y = 75.
        let top = c[0].iter().map(|p| p[1]).fold(0.0f64, f64::max);
        assert!((top - 75.0).abs() < 0.3, "{top}");
        assert!(c[0].len() > 10 && c[0].len() < 80, "{}", c[0].len());
        let circle = crate::shape_to_path::ellipse_to_path_cmds(100.0, 100.0, None, None, None);
        let c = flatten(&circle, 0.25);
        assert_eq!(c.len(), 1);
        for p in &c[0] {
            let r = (p[0] - 50.0).hypot(p[1] - 50.0);
            assert!((r - 50.0).abs() < 0.3, "{r}");
        }
    }

    fn shapes_of(node: moda_scene::value::Value) -> Vec<Shape> {
        let id = node["id"].as_str().unwrap().to_string();
        let scene = Scene::load(&moda_scene::value::json!({
            "nodes": [node],
            "pages": [{"id": "p", "width": 800, "height": 800, "nodeIds": [id.clone()]}]
        })).unwrap();
        node_shapes(&scene, &id)
    }

    /// Nonzero winding of `contours` at `p`.
    fn inside(contours: &[Vec<[f64; 2]>], p: [f64; 2]) -> bool {
        let mut wn = 0i32;
        for c in contours {
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let cross = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]);
                if a[1] <= p[1] && b[1] > p[1] && cross > 0.0 {
                    wn += 1;
                } else if a[1] > p[1] && b[1] <= p[1] && cross < 0.0 {
                    wn -= 1;
                }
            }
        }
        wn != 0
    }

    fn stroke(color: &str, width: f64, align: Option<&str>) -> moda_scene::value::Value {
        let mut s = moda_scene::value::json!({"id": "stroke-0", "paint": {"type": "solid", "color": color}, "width": width});
        if let Some(a) = align {
            s["align"] = a.into();
        }
        moda_scene::value::json!([s])
    }

    #[test]
    fn a_stroked_line_is_a_bar_in_its_stroke_colour() {
        let shapes = shapes_of(moda_scene::value::json!({"id": "L", "type": "line", "x": 0, "y": 0,
            "points": [10, 50, 210, 50], "strokes": stroke("#ff0000", 20.0, None)}));
        assert_eq!(shapes.len(), 1, "{shapes:?}");
        assert_eq!(shapes[0].color, [1.0, 0.0, 0.0, 1.0]);
        let c = &shapes[0].contours;
        assert!(inside(c, [110.0, 50.0]) && inside(c, [110.0, 58.0]) && inside(c, [20.0, 42.0]));
        assert!(!inside(c, [110.0, 62.0]) && !inside(c, [110.0, 38.0]) && !inside(c, [5.0, 50.0]));
    }

    #[test]
    fn a_stroke_only_ellipse_is_a_ring() {
        let shapes = shapes_of(moda_scene::value::json!({"id": "E", "type": "ellipse", "x": 0, "y": 0, "width": 200, "height": 200,
            "fills": [], "strokes": stroke("#00ff00", 20.0, None)}));
        let ring: Vec<&Shape> = shapes.iter().filter(|s| s.color == [0.0, 1.0, 0.0, 1.0]).collect();
        assert_eq!(ring.len(), 1, "{shapes:?}");
        let c = &ring[0].contours;
        // On the centreline (r = 100), both edges of the band (r = 91, 109), not the middle nor outside.
        for (x, y) in [(200.0, 100.0), (100.0, 0.0), (191.0, 100.0), (100.0, 209.0), (170.71, 170.71)] {
            assert!(inside(c, [x, y]), "({x}, {y})");
        }
        for (x, y) in [(100.0, 100.0), (180.0, 100.0), (215.0, 100.0), (100.0, -15.0)] {
            assert!(!inside(c, [x, y]), "({x}, {y})");
        }
    }

    #[test]
    fn a_filled_and_stroked_rect_is_its_fill_then_its_stroke_aligned() {
        let fill = moda_scene::value::json!([{"id": "f", "paint": {"type": "solid", "color": "#0000ff"}}]);
        for (align, inner, outer) in [(None, 95.0, 105.0), (Some("inside"), 90.0, 100.0), (Some("outside"), 100.0, 110.0)] {
            let shapes = shapes_of(moda_scene::value::json!({"id": "R", "type": "rectangle", "x": 0, "y": 0, "width": 200,
                "height": 200, "fills": fill, "strokes": stroke("rgba(255, 0, 0, 0.5)", 10.0, align)}));
            assert_eq!(shapes.len(), 2, "{align:?}: {shapes:?}");
            assert_eq!(shapes[0].color, [0.0, 0.0, 1.0, 1.0]);
            assert_eq!(shapes[1].color, [1.0, 0.0, 0.0, 0.5]);
            // Measured from the centre along y = 100: the band spans [inner, outer].
            let c = &shapes[1].contours;
            let at = |d: f64| inside(c, [100.0 + d, 100.0]);
            assert!(at(inner + 1.0) && at(outer - 1.0), "{align:?}");
            assert!(!at(inner - 1.0) && !at(outer + 1.0) && !at(0.0), "{align:?}");
        }
    }

    #[test]
    fn an_unstroked_or_hidden_stroke_adds_no_part() {
        let mut s = stroke("#ff0000", 10.0, None);
        s[0]["visible"] = false.into();
        let fill = moda_scene::value::json!([{"id": "f", "paint": {"type": "solid", "color": "#0000ff"}}]);
        let shapes = shapes_of(moda_scene::value::json!({"id": "R", "type": "rectangle", "x": 0, "y": 0, "width": 50,
            "height": 50, "fills": fill, "strokes": s}));
        assert_eq!(shapes.len(), 1);
        let shapes = shapes_of(moda_scene::value::json!({"id": "R", "type": "rectangle", "x": 0, "y": 0, "width": 50,
            "height": 50, "fills": fill}));
        assert_eq!(shapes.len(), 1);
    }

    #[test]
    fn colors_parse() {
        assert_eq!(parse_color("#ff0000"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(parse_color("#0f08"), Some([0.0, 1.0, 0.0, 8.0 * 17.0 / 255.0]));
        assert_eq!(parse_color("rgba(255, 0, 0, 0.5)"), Some([1.0, 0.0, 0.0, 0.5]));
    }
}
