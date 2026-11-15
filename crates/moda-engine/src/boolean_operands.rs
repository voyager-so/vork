//! The engine half of live boolean groups (E7.1): the node-local outline of every operand
//! kind `moda_scene::boolean_node` cannot convert itself. Shapes come from the same
//! converters the tessellator draws them with (`shape_to_path`, `tess::shape_paths`); rich
//! text from the vector text layout (`text_geometry`), glyph outlines as drawn.
//!
//! [`install_scene_seams`] registers it with moda-scene. Every process that loads scenes
//! for drawing calls it once before its first load: the wasm module's start function and
//! `moda-render`'s `main`.

use crate::tess::moda_engine_path_types::PathCmd;
use moda_scene::blend::BlendPathPoint as P;
use moda_scene::types::RawNode;
use moda_scene::Scene;

/// Register the engine's seams with moda-scene (idempotent).
pub fn install_scene_seams() {
    moda_scene::boolean_node::install_operand_outliner(operand_outline);
    moda_scene::boolean_node::install_font_generation(crate::text_draw::text_host_generation);
    moda_scene::text_warp::install_text_outliner(crate::text_warp::text_warp_outlines);
    moda_scene::generator::install_surface_sink(generator_surface);
    moda_scene::expansion::install_table_text_measurer(crate::text_metrics::table_text_height);
    moda_assets::svg::install_fontdb_provider(Box::new(svg_image_fontdb));
}

/// The text engine's fonts for SVG images' `<text>`, rebuilt only when the engine changes (the wasm
/// editor installs a new one per font install). The wasm lane never builds an engine here: before
/// its first font install there are no fonts to lay text out with.
fn svg_image_fontdb() -> Option<std::sync::Arc<moda_assets::svg::FontDatabase>> {
    #[cfg(test)]
    crate::test_fonts();
    #[cfg(target_arch = "wasm32")]
    let engine = moda_text::degraded::installed_engine()?;
    #[cfg(not(target_arch = "wasm32"))]
    let engine = moda_text::degraded::engine().ok()?;
    static CACHE: std::sync::Mutex<Option<(u64, std::sync::Arc<moda_assets::svg::FontDatabase>)>> =
        std::sync::Mutex::new(None);
    let mut cache = CACHE.lock().ok()?;
    if let Some((id, db)) = cache.as_ref() {
        if *id == engine.id() {
            return Some(db.clone());
        }
    }
    let db = std::sync::Arc::new(engine.svg_font_database());
    *cache = Some((engine.id(), db.clone()));
    Some(db)
}

/// A generator's painted output (live-nodes §14.4) into the surface registry: its tiles replace the node's.
fn generator_surface(id: &str, w: u32, h: u32, tiles: &[(u32, u32, &[u8])]) {
    crate::surface::clear_tiles(id);
    let tiles: Vec<(u32, u32, Option<&[u8]>)> = tiles.iter().map(|&(x, y, t)| (x, y, Some(t))).collect();
    crate::surface::upload_tiles(id, w, h, &tiles);
}

fn cmds_to_points(cmds: &[PathCmd]) -> Vec<P> {
    cmds.iter()
        .map(|c| match *c {
            PathCmd::Move { x, y } => P::Move { x, y },
            PathCmd::Line { x, y } => P::Line { x, y },
            PathCmd::Curve { x, y, cp1, cp2 } => P::Curve { x, y, cp1, cp2 },
            PathCmd::Quadratic { x, y, cp } => P::Quadratic { x, y, cp },
            PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => P::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            },
        })
        .collect()
}

/// Node-local outline of a rectangle, ellipse, polygon, star or rich text node, as drawn:
/// a `distort` (perspective / skew / 3D tilt) warps it the way the renderer does (flattened
/// and projected, `tess::shape_paths::warp_path_cmds`).
pub fn operand_outline(scene: &Scene, node: &RawNode) -> Option<Vec<P>> {
    let outline = undistorted_outline(scene, node)?;
    let Some(distort) = crate::paint::node_distort(node) else {
        return Some(outline);
    };
    let cmds: Vec<PathCmd> = outline
        .iter()
        .map(crate::tess::path_cmd_from_blend_point)
        .collect();
    // A warp that fails (a degenerate outline) keeps the unwarped outline, as the renderer does.
    Some(
        crate::tess::shape_paths::warp_path_cmds(&cmds, &distort.local)
            .map(|warped| cmds_to_points(&warped))
            .unwrap_or(outline),
    )
}

pub(crate) fn undistorted_outline(scene: &Scene, node: &RawNode) -> Option<Vec<P>> {
    let w = node.f64("width").unwrap_or(0.0);
    let h = node.f64("height").unwrap_or(0.0);
    let radius = || {
        node.f64("cornerRadius")
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    };
    let cmds = match node.node_type() {
        // Styled corners (`cornerStyle`) as drawn; a round rectangle keeps its KAPPA corners.
        "rectangle" if moda_scene::rect_corners::is_styled_rect(node.json_field("cornerRadius"), node.json_field("cornerStyle")) => {
            crate::shape_to_path::rect_outline_cmds(w, h, node.json_field("cornerRadius"), None, node.json_field("cornerStyle"))
        }
        "rectangle" => crate::shape_to_path::rectangle_to_path_cmds(w, h, node.json_field("cornerRadius")),
        "ellipse" => crate::shape_to_path::ellipse_to_path_cmds(
            w,
            h,
            node.f64("startAngle"),
            node.f64("sweepAngle"),
            node.f64("innerRadiusRatio"),
        ),
        "polygon" => crate::tess::shape_paths::polygon_to_path_cmds(
            w,
            h,
            node.f64("sides").unwrap_or(f64::NAN),
            radius(),
        ),
        "star" => crate::tess::shape_paths::star_to_path_cmds(
            w,
            h,
            node.f64("numPoints").unwrap_or(f64::NAN),
            node.f64("innerRadiusRatio"),
            radius(),
        ),
        "richtext" => return text_outline(scene, node),
        _ => return None,
    };
    Some(cmds_to_points(&cmds))
}

/// The glyph outlines of a rich text node, node-local px, as the vector text engine draws
/// them. `None` when the text doesn't lay out through the vector lane (fonts not loaded yet,
/// a script it hands to the fallback shaper): the boolean retries on the next load.
fn text_outline(scene: &Scene, node: &RawNode) -> Option<Vec<P>> {
    let geometry = crate::text_geometry::node_text_geometry(scene, node.id()).ok()?;
    let mut out = Vec::new();
    for fill in geometry.fills.iter().filter(|f| f.kind == "glyphs") {
        for d in &fill.paths {
            parse_glyph_d(d, &mut out);
        }
    }
    Some(out)
}

/// The glyph path data `text_geometry` writes (its `GlyphPathData`): `M`/`m`, `L`/`l`, `H`/`h`, `V`/`v`,
/// `Q`/`q` and `Z`/`z`, a command's arguments repeating it (after a move, as lines), numbers such as `.5` and `1-2`
/// (a minus sign or a second point starts the next number). Points come out absolute.
pub(crate) fn parse_glyph_d(d: &str, out: &mut Vec<P>) {
    let bytes = d.as_bytes();
    let mut i = 0;
    // The next number, if one comes before the next command letter.
    let number = |i: &mut usize| -> Option<f64> {
        while *i < bytes.len() && matches!(bytes[*i], b' ' | b',' | b'\t' | b'\n' | b'\r') {
            *i += 1;
        }
        let start = *i;
        if *i < bytes.len() && matches!(bytes[*i], b'-' | b'+') {
            *i += 1;
        }
        let mut dot = false;
        while *i < bytes.len() && (bytes[*i].is_ascii_digit() || (bytes[*i] == b'.' && !dot)) {
            dot |= bytes[*i] == b'.';
            *i += 1;
        }
        if *i < bytes.len() && matches!(bytes[*i], b'e' | b'E') {
            *i += 1;
            if *i < bytes.len() && matches!(bytes[*i], b'-' | b'+') {
                *i += 1;
            }
            while *i < bytes.len() && bytes[*i].is_ascii_digit() {
                *i += 1;
            }
        }
        d.get(start..*i).and_then(|t| t.parse().ok())
    };
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    let mut cmd = 0u8;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() {
            cmd = bytes[i];
            i += 1;
            if cmd.eq_ignore_ascii_case(&b'z') {
                (x, y) = (sx, sy);
            }
            continue;
        }
        let rel = cmd.is_ascii_lowercase();
        let (ox, oy) = if rel { (x, y) } else { (0.0, 0.0) };
        let at = i;
        let point = match cmd.to_ascii_uppercase() {
            b'M' | b'L' => number(&mut i).zip(number(&mut i)).map(|(a, b)| {
                (x, y) = (ox + a, oy + b);
                if cmd.eq_ignore_ascii_case(&b'm') {
                    (sx, sy) = (x, y);
                    // Further pairs are lines.
                    cmd = if rel { b'l' } else { b'L' };
                    P::Move { x, y }
                } else {
                    P::Line { x, y }
                }
            }),
            b'H' => number(&mut i).map(|a| {
                x = ox + a;
                P::Line { x, y }
            }),
            b'V' => number(&mut i).map(|b| {
                y = oy + b;
                P::Line { x, y }
            }),
            b'Q' => match (number(&mut i), number(&mut i), number(&mut i), number(&mut i)) {
                (Some(cx), Some(cy), Some(ex), Some(ey)) => {
                    (x, y) = (ox + ex, oy + ey);
                    Some(P::Quadratic { x, y, cp: (ox + cx, oy + cy) })
                }
                _ => None,
            },
            _ => None,
        };
        match point {
            Some(p) => out.push(p),
            // Not a number this command takes: skip a character so the scan moves on.
            None if i == at => i += 1,
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    /// An SVG image's `<text>` draws with the text engine's fonts once the scene seams are installed
    /// (the editor's and the CLI's image decode), instead of rasterizing to nothing.
    #[test]
    fn svg_image_text_draws_with_the_engine_fonts() {
        install_scene_seams();
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60"><text x="100" y="40" text-anchor="middle" font-size="32" font-family="Noto Sans" fill="#222">WORDMARK</text></svg>"##;
        let img = moda_assets::decode_pattern_bytes("wordmark.svg", svg, None).unwrap();
        assert!(img.coverage_permille > 0, "the wordmark paints");
        assert_eq!(img.blank_cause, "");
    }

    fn area_of(points: &[P]) -> f64 {
        // The even-odd area (how the result is filled), sampled on a quarter-pixel grid over
        // the flattened subpaths.
        let mut rings: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut sub: Vec<P> = Vec::new();
        let mut flush = |sub: &mut Vec<P>| {
            if !sub.is_empty() {
                rings.push(moda_scene::blend::flatten_path_to_polyline(sub, true));
                sub.clear();
            }
        };
        for p in points {
            if matches!(p, P::Move { .. }) {
                flush(&mut sub);
            }
            sub.push(p.clone());
        }
        flush(&mut sub);
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(x, y) in rings.iter().flatten() {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let step = 0.25;
        let mut hits = 0usize;
        let mut y = y0 + step / 2.0;
        while y < y1 {
            let mut x = x0 + step / 2.0;
            while x < x1 {
                let mut inside = false;
                for ring in &rings {
                    for i in 0..ring.len() {
                        let (ax, ay) = ring[i];
                        let (bx, by) = ring[(i + 1) % ring.len()];
                        if (ay > y) != (by > y) && x < ax + (y - ay) / (by - ay) * (bx - ax) {
                            inside = !inside;
                        }
                    }
                }
                hits += inside as usize;
                x += step;
            }
            y += step;
        }
        hits as f64 * step * step
    }

    fn load(nodes: moda_scene::value::Value, root: &str) -> Scene {
        install_scene_seams();
        Scene::load(&json!({
            "nodes": nodes,
            "pages": [{"id": "p", "width": 800, "height": 800, "nodeIds": [root]}]
        })).unwrap()
    }

    fn result_points(scene: &Scene, id: &str) -> Vec<P> {
        scene
            .effective_node(id)
            .unwrap()
            .points()
            .unwrap()
            .iter()
            .filter_map(P::from_ref)
            .collect()
    }

    #[test]
    fn glyph_path_data_parses() {
        let mut out = Vec::new();
        parse_glyph_d("M1 2L3.5 -4Q5 6 7 8Z", &mut out);
        assert_eq!(out.len(), 3);
        assert!(matches!(out[1], P::Line { x, y } if x == 3.5 && y == -4.0));
        assert!(
            matches!(out[2], P::Quadratic { x, y, cp } if x == 7.0 && y == 8.0 && cp == (5.0, 6.0))
        );
    }

    /// The compact form `text_geometry` writes: relative commands, repeated arguments, `h` / `v`, `.5`, `1-2`.
    #[test]
    fn compact_relative_glyph_path_data_parses_to_absolute_points() {
        let mut out = Vec::new();
        parse_glyph_d("M1 2l2.5-6 1 1h-.5v2q1 1 2 0 1-1 2 0Zm1 1 2 2", &mut out);
        let pts: Vec<String> = out
            .iter()
            .map(|p| match p {
                P::Move { x, y } => format!("M{x},{y}"),
                P::Line { x, y } => format!("L{x},{y}"),
                P::Quadratic { x, y, cp } => format!("Q{},{} {x},{y}", cp.0, cp.1),
                _ => "?".into(),
            })
            .collect();
        assert_eq!(
            pts,
            [
                "M1,2", "L3.5,-4", "L4.5,-3", "L4,-3", "L4,-1", "Q5,0 6,-1", "Q7,-2 8,-1",
                // After Z the current point is the subpath's start (1, 2); a move's further pairs are lines.
                "M2,3", "L4,5",
            ]
        );
    }

    #[test]
    fn every_shape_kind_is_an_operand() {
        moda_scene::boolean_node::clear_cache();
        // A 100×100 square minus a circle of diameter 40 inside it: 10000 − π·20².
        let scene = load(
            json!([
                {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["sq", "hole"], "booleanOperation": "subtract"},
                {"id": "sq", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 100, "rotation": 0, "scale": 1},
                {"id": "hole", "type": "ellipse", "x": 30, "y": 30, "width": 40, "height": 40, "rotation": 0, "scale": 1},
            ]),
            "b",
        );
        let expected = 10_000.0 - std::f64::consts::PI * 400.0;
        let got = area_of(&result_points(&scene, "b"));
        assert!((got - expected).abs() < 15.0, "area {got}, want {expected}");

        // Polygon and star operands produce a result too.
        for (kind, extra) in [
            ("polygon", json!({"sides": 6})),
            ("star", json!({"numPoints": 5, "innerRadiusRatio": 0.5})),
        ] {
            moda_scene::boolean_node::clear_cache();
            let mut shape = json!({"id": "s", "type": kind, "x": 0, "y": 0, "width": 100, "height": 100, "rotation": 0, "scale": 1});
            for (k, v) in extra.as_object().unwrap() {
                shape[k] = v.clone();
            }
            let scene = load(
                json!([
                    {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                     "children": ["s", "r"], "booleanOperation": "intersect"},
                    shape,
                    {"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 50, "rotation": 0, "scale": 1},
                ]),
                "b",
            );
            assert!(area_of(&result_points(&scene, "b")) > 100.0, "{kind}");
        }
    }

    #[test]
    fn a_distorted_child_combines_as_drawn() {
        moda_scene::boolean_node::clear_cache();
        // A 100×100 square sheared right by its full width at the bottom (a parallelogram,
        // same area): the union's bounds reach x = 200.
        let scene = load(
            json!([
                {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["r"], "booleanOperation": "union"},
                {"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 100, "rotation": 0, "scale": 1,
                 "distort": {"type": "matrix", "matrix": [1, 1, 0, 0, 1, 0, 0, 0, 1]}},
            ]),
            "b",
        );
        let pts = result_points(&scene, "b");
        let max_x = pts
            .iter()
            .map(|p| match p {
                P::Move { x, .. }
                | P::Line { x, .. }
                | P::Curve { x, .. }
                | P::Quadratic { x, .. }
                | P::Arc { x, .. } => *x,
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max_x - 200.0).abs() < 0.5, "max x {max_x}");
        assert!(
            (area_of(&pts) - 10_000.0).abs() < 50.0,
            "area {}",
            area_of(&pts)
        );
    }

    #[test]
    fn a_rotated_child_is_rotated_in_the_result() {
        moda_scene::boolean_node::clear_cache();
        // A 100×100 square rotated 45° about its origin reaches x = -70.7: the union's
        // bounds must too.
        let scene = load(
            json!([
                {"id": "b", "type": "boolean", "x": 0, "y": 0, "rotation": 0, "scaleX": 1, "scaleY": 1,
                 "children": ["r"], "booleanOperation": "union"},
                {"id": "r", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 100, "rotation": 45, "scale": 1},
            ]),
            "b",
        );
        let min_x = result_points(&scene, "b")
            .iter()
            .map(|p| match p {
                P::Move { x, .. }
                | P::Line { x, .. }
                | P::Curve { x, .. }
                | P::Quadratic { x, .. }
                | P::Arc { x, .. } => *x,
            })
            .fold(f64::INFINITY, f64::min);
        assert!((min_x + 70.71).abs() < 0.5, "min x {min_x}");
    }
}
