//! Brush artwork (`ArtworkRef = { nodes: SceneNode[] }`, brushes.md §4.4) →
//! shapes in the artwork's own coordinate space, with their paints.
//!
//! The artwork is a small flat node list: roots are the nodes no group or
//! container lists as a child; groups compose their transform
//! (`x, y, rotation, scaleX/scaleY`) onto their children, like
//! `Scene::accumulated_affine`. Hidden nodes (`visible: false`) are skipped;
//! group opacity multiplies into its descendants.
//!
//! Per-node geometry comes from a callback, because the full shape → path
//! conversions (star, polygon, line bodies, ellipse arcs) live in moda-engine,
//! which depends on this crate. [`default_geometry`] covers `path` (with its
//! `cornerRadius`), `rectangle` (uniform radius) and full `ellipse`s; the
//! engine / WASM callers pass their own for everything else. A node the
//! callback declines is reported in `skipped`.

use std::collections::HashSet;

use crate::value::{Map, Value};

use super::colorize::{node_colors, Rgba};
use super::geom::{curve_bounds, subpaths_from_points, Bounds, Subpath};
use crate::blend::BlendPathPoint;
use crate::path_geom::round_corners::{round_path_corners_with, PathCorners};
use crate::path_geom::vec2::{v2, Cubic};
use crate::transform::{self, compose};
use crate::types::{Affine, Transform2D};

/// Most nodes an artwork may carry.
pub const MAX_ARTWORK_NODES: usize = 2_000;
/// Most segments across all artwork shapes.
pub const MAX_ARTWORK_SEGMENTS: usize = 20_000;
const MAX_DEPTH: usize = 32;

/// Keys that describe geometry, transform or brush-stroke objects; everything
/// else on a node (fill, stroke, opacity, blend, fillRule, …) is its paint and
/// is carried to the placed copy.
const GEOMETRY_KEYS: &[&str] = &[
    "id",
    "type",
    "name",
    "x",
    "y",
    "width",
    "height",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "points",
    "closed",
    "children",
    "parentId",
    "cornerRadius",
    "cornerStyle",
    "cornerSmoothing",
    "sides",
    "numPoints",
    "innerRadiusRatio",
    "startAngle",
    "sweepAngle",
    "locked",
    "visible",
    "inkChannels",
    "trimPathStart",
    "trimPathEnd",
    "trimPathOffset",
    "trimPathMode",
    "trimPathBy",
];

/// One drawable artwork shape in artwork space.
#[derive(Debug, Clone)]
pub struct ArtShape {
    pub source_id: String,
    pub subpaths: Vec<Subpath>,
    /// The node's paint fields (see `GEOMETRY_KEYS`), `opacity` already
    /// multiplied by its groups'.
    pub paint: Map,
    /// sqrt(|det|) of the node's accumulated transform: scales `strokeWidth`.
    pub transform_scale: f64,
}

/// A prepared artwork.
#[derive(Debug, Clone, Default)]
pub struct Artwork {
    pub shapes: Vec<ArtShape>,
    /// Ids of nodes that could not be turned into geometry (e.g. text).
    pub skipped: Vec<String>,
}

impl Artwork {
    /// Bounds of all shapes (curve samples; strokes not included).
    pub fn bounds(&self) -> Bounds {
        let mut b = Bounds::empty();
        for s in &self.shapes {
            for sp in &s.subpaths {
                b.union(&curve_bounds(&sp.segs));
            }
        }
        b
    }

    /// Bounds including stroke widths: a stroked shape (its top stroke) is
    /// offset by half its scaled width along the curve normals, with its line
    /// caps at the ends of open subpaths (`butt` adds nothing past the end,
    /// `round` a radius, `square` a half-width square). Miter joins are not
    /// included.
    pub fn visual_bounds(&self) -> Bounds {
        use crate::path_geom::vec2::v2;
        let mut b = Bounds::empty();
        for s in &self.shapes {
            let stroke = crate::paint::top_stroke_of(&s.paint);
            let w = stroke.as_ref().map_or(0.0, |st| st.width);
            let r = if w.is_finite() && w > 0.0 {
                w * s.transform_scale / 2.0
            } else {
                0.0
            };
            let cap = stroke
                .as_ref()
                .and_then(|st| st.cap.clone())
                .unwrap_or_else(|| "butt".to_string());
            let cap = cap.as_str();
            for sp in &s.subpaths {
                if r <= 0.0 {
                    b.union(&curve_bounds(&sp.segs));
                    continue;
                }
                for c in &sp.segs {
                    for i in 0..=16 {
                        let t = i as f64 / 16.0;
                        let p = c.eval(t);
                        match c.tangent(t) {
                            Some(tan) => {
                                let n = v2(-tan.y, tan.x) * r;
                                b.add(p + n);
                                b.add(p - n);
                            }
                            None => b.add(p),
                        }
                    }
                }
                if !sp.closed {
                    let ends = [
                        sp.segs
                            .first()
                            .and_then(|c| c.tangent(0.0).map(|t| (c.0[0], -t))),
                        sp.segs
                            .last()
                            .and_then(|c| c.tangent(1.0).map(|t| (c.0[3], t))),
                    ];
                    for (p, out) in ends.into_iter().flatten() {
                        let n = v2(-out.y, out.x) * r;
                        match cap {
                            "round" => {
                                b.add(p - v2(r, r));
                                b.add(p + v2(r, r));
                            }
                            "square" => {
                                b.add(p + out * r + n);
                                b.add(p + out * r - n);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        b
    }

    /// Paint colours per shape with a weight (bounds area, or length for
    /// hairlines), for picking the default key colour.
    pub fn weighted_colors(&self) -> Vec<(Vec<Rgba>, f64)> {
        self.shapes
            .iter()
            .map(|s| {
                let mut b = Bounds::empty();
                for sp in &s.subpaths {
                    b.union(&curve_bounds(&sp.segs));
                }
                let w = if b.is_empty() {
                    0.0
                } else {
                    (b.width() * b.height()).max(b.width().max(b.height()))
                };
                (node_colors(&s.paint), w)
            })
            .collect()
    }

    pub fn segment_count(&self) -> usize {
        self.shapes
            .iter()
            .flat_map(|s| s.subpaths.iter())
            .map(|sp| sp.segs.len())
            .sum()
    }

    /// Apply an affine to every shape (anchors and handles).
    pub fn transformed(&self, m: Affine) -> Artwork {
        let ap = |p: crate::path_geom::vec2::V2| {
            let (x, y) = transform::apply_point((p.x, p.y), m);
            v2(x, y)
        };
        Artwork {
            shapes: self
                .shapes
                .iter()
                .map(|s| ArtShape {
                    subpaths: s
                        .subpaths
                        .iter()
                        .map(|sp| Subpath {
                            segs: sp
                                .segs
                                .iter()
                                .map(|c| Cubic([ap(c.0[0]), ap(c.0[1]), ap(c.0[2]), ap(c.0[3])]))
                                .collect(),
                            closed: sp.closed,
                        })
                        .collect(),
                    ..s.clone()
                })
                .collect(),
            skipped: self.skipped.clone(),
        }
    }
}

/// Node-local geometry of one leaf node: `(points, closed)`, or `None` when
/// the node has no path geometry this callback understands.
pub type GeometryFn<'a> = &'a dyn Fn(&Map) -> Option<(Vec<BlendPathPoint>, bool)>;

fn f(node: &Map, k: &str) -> Option<f64> {
    node.get(k)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
}

/// `KAPPA` (`utils/path/constants.ts`).
const KAPPA: f64 = 0.5522847498;

/// Geometry for `path`, `rectangle` and full `ellipse` nodes.
pub fn default_geometry(node: &Map) -> Option<(Vec<BlendPathPoint>, bool)> {
    match node.get("type").and_then(Value::as_str)? {
        "path" => {
            let raw = node.get("points").and_then(Value::as_array)?;
            let pts: Vec<BlendPathPoint> =
                raw.iter().filter_map(BlendPathPoint::from_json).collect();
            let closed = node.get("closed").and_then(Value::as_bool).unwrap_or(false);
            let corners = PathCorners::read_json(node);
            let pts = if corners.rounds_anything() {
                round_path_corners_with(&pts, closed, &corners)
            } else {
                pts
            };
            Some((pts, closed))
        }
        "rectangle" => {
            let (w, h) = (f(node, "width")?, f(node, "height")?);
            let r = f(node, "cornerRadius").unwrap_or(0.0);
            let smoothing = f(node, "cornerSmoothing").unwrap_or(0.0);
            let style = node.get("cornerStyle");
            if crate::rect_corners::is_styled_rect(Some(&Value::from(r)), style) {
                let pts = crate::shape_outline::smoothed_or_plain_rectangle_outline(
                    w,
                    h,
                    Some(&Value::from(r)),
                    None,
                    style,
                );
                return Some((pts, true));
            }
            let json = if crate::corner_smoothing::is_smoothed([r; 4], smoothing) {
                crate::corner_smoothing::smoothed_rect_json_points(w, h, [r; 4], smoothing)
            } else {
                crate::rectangle_path::rectangle_to_path_json_points(0.0, 0.0, w, h, r, 0.0)
            };
            Some((
                json.iter().filter_map(BlendPathPoint::from_json).collect(),
                true,
            ))
        }
        "ellipse" => {
            let (w, h) = (f(node, "width")?, f(node, "height")?);
            let (rx, ry) = (w / 2.0, h / 2.0);
            let (cx, cy) = (rx, ry);
            let (kx, ky) = (rx * KAPPA, ry * KAPPA);
            let c = |x: f64, y: f64, c1: (f64, f64), c2: (f64, f64)| BlendPathPoint::Curve {
                x,
                y,
                cp1: c1,
                cp2: c2,
            };
            Some((
                vec![
                    BlendPathPoint::Move { x: cx + rx, y: cy },
                    c(cx, cy + ry, (cx + rx, cy + ky), (cx + kx, cy + ry)),
                    c(cx - rx, cy, (cx - kx, cy + ry), (cx - rx, cy + ky)),
                    c(cx, cy - ry, (cx - rx, cy - ky), (cx - kx, cy - ry)),
                    c(cx + rx, cy, (cx + kx, cy - ry), (cx + rx, cy - ky)),
                ],
                true,
            ))
        }
        _ => None,
    }
}

fn local_affine(node: &Map) -> Affine {
    let is_group = node.get("type").and_then(Value::as_str) == Some("group");
    let nz = |v: f64| if v == 0.0 { 1.0 } else { v };
    let (sx, sy) = if is_group || node.contains_key("scaleX") || node.contains_key("scaleY") {
        (
            nz(f(node, "scaleX").unwrap_or(1.0)),
            nz(f(node, "scaleY").unwrap_or(1.0)),
        )
    } else {
        let s = nz(f(node, "scale").unwrap_or(1.0));
        (s, s)
    };
    crate::transform::from_trsk(
        Transform2D {
            x: f(node, "x").unwrap_or(0.0),
            y: f(node, "y").unwrap_or(0.0),
            rotation: f(node, "rotation").unwrap_or(0.0),
            scale_x: sx,
            scale_y: sy,
        },
        f(node, "skewX").unwrap_or(0.0),
    )
}

fn is_parent(node: &Map) -> bool {
    matches!(
        node.get("type").and_then(Value::as_str),
        Some("group" | "container")
    ) && matches!(node.get("children"), Some(Value::Array(_)))
}

fn children(node: &Map) -> Vec<&str> {
    node.get("children")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// Segments a leaf will produce, estimated from its fields before building
/// any geometry (so an oversize node is refused without the work).
fn estimated_segments(node: &Map) -> usize {
    let count = |k: &str| {
        node.get(k)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .map_or(0.0, |v| v.clamp(0.0, 1e9)) as usize
    };
    match node.get("type").and_then(Value::as_str) {
        Some("path") => node
            .get("points")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        Some("polygon") => count("sides").min(4096) + 1,
        Some("star") => 2 * count("numPoints").min(4096) + 1,
        _ => 16,
    }
}

/// Walk an artwork node list into shapes. Never panics; oversize artwork is
/// truncated at `MAX_ARTWORK_NODES` / `MAX_ARTWORK_SEGMENTS`.
pub fn collect_artwork(nodes: &[Value], geometry: GeometryFn) -> Artwork {
    let nodes: Vec<&Map> = nodes
        .iter()
        .take(MAX_ARTWORK_NODES)
        .filter_map(Value::as_object)
        .filter(|n| n.get("id").and_then(Value::as_str).is_some())
        .collect();
    let by_id = |id: &str| {
        nodes
            .iter()
            .copied()
            .find(|n| n.get("id").and_then(Value::as_str) == Some(id))
    };
    let mut child_ids: HashSet<&str> = HashSet::new();
    for n in &nodes {
        if is_parent(n) {
            child_ids.extend(children(n));
        }
    }
    let mut out = Artwork::default();
    let mut segments = 0usize;
    let mut visited: HashSet<&str> = HashSet::new();
    struct Frame<'a> {
        node: &'a Map,
        affine: Affine,
        opacity: f64,
        depth: usize,
    }
    let mut stack: Vec<Frame> = nodes
        .iter()
        .rev()
        .filter(|n| !child_ids.contains(n.get("id").and_then(Value::as_str).unwrap_or("")))
        .map(|n| Frame {
            node: n,
            affine: transform::identity(),
            opacity: 1.0,
            depth: 0,
        })
        .collect();
    while let Some(Frame {
        node,
        affine,
        opacity,
        depth,
    }) = stack.pop()
    {
        let id = node.get("id").and_then(Value::as_str).unwrap_or("");
        if !visited.insert(id) || depth > MAX_DEPTH {
            continue;
        }
        if node.get("visible").and_then(Value::as_bool) == Some(false) {
            continue;
        }
        let m = compose(affine, local_affine(node));
        let op = opacity * f(node, "opacity").unwrap_or(1.0).clamp(0.0, 1.0);
        if is_parent(node) {
            for cid in children(node).into_iter().rev() {
                if let Some(child) = by_id(cid) {
                    stack.push(Frame {
                        node: child,
                        affine: m,
                        opacity: op,
                        depth: depth + 1,
                    });
                }
            }
            continue;
        }
        if segments + estimated_segments(node) > MAX_ARTWORK_SEGMENTS {
            out.skipped.push(id.to_string());
            continue;
        }
        let Some((points, closed)) = geometry(node) else {
            out.skipped.push(id.to_string());
            continue;
        };
        let pts: Vec<BlendPathPoint> = points.iter().map(|p| p.apply_affine(m)).collect();
        let subpaths = subpaths_from_points(&pts, closed);
        let count: usize = subpaths.iter().map(|s| s.segs.len()).sum();
        if count == 0 {
            continue;
        }
        if segments + count > MAX_ARTWORK_SEGMENTS {
            out.skipped.push(id.to_string());
            continue;
        }
        segments += count;
        let mut paint: Map = node
            .iter()
            .filter(|(k, _)| !GEOMETRY_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // A placed copy strokes plainly: the artwork's own width models and
        // brushes stay behind.
        crate::paint::strip_stroke_models(&mut paint);
        if op < 1.0 || paint.contains_key("opacity") {
            paint.insert("opacity".into(), Value::from(op));
        }

        let det = (m.a * m.d - m.b * m.c).abs().sqrt();
        out.shapes.push(ArtShape {
            source_id: id.to_string(),
            subpaths,
            paint,
            transform_scale: if det.is_finite() { det } else { 1.0 },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    #[test]
    fn groups_compose_and_hide() {
        let nodes = json!([
            {"id": "g", "type": "group", "x": 100, "y": 0, "scaleX": 2, "scaleY": 2, "opacity": 0.5,
             "children": ["r", "hidden", "t"]},
            {"id": "r", "type": "rectangle", "x": 1, "y": 1, "width": 2, "height": 3, "fill": "#f00", "opacity": 0.5},
            {"id": "hidden", "type": "rectangle", "x": 0, "y": 0, "width": 2, "height": 3, "visible": false},
            {"id": "t", "type": "text", "x": 0, "y": 0},
            {"id": "e", "type": "ellipse", "x": 0, "y": 0, "width": 10, "height": 4, "stroke": "#000", "strokeWidth": 2},
        ]);
        let a = collect_artwork(nodes.as_array().unwrap(), &default_geometry);
        assert_eq!(a.shapes.len(), 2);
        assert_eq!(a.skipped, vec!["t".to_string()]);
        let r = &a.shapes[0];
        assert_eq!(r.source_id, "r");
        let b = curve_bounds(&r.subpaths[0].segs);
        assert!((b.min.x - 102.0).abs() < 1e-9 && (b.max.x - 106.0).abs() < 1e-9);
        assert!((b.min.y - 2.0).abs() < 1e-9 && (b.max.y - 8.0).abs() < 1e-9);
        assert_eq!(r.paint["opacity"], json!(0.25));
        assert_eq!(r.paint["fill"], "#f00");
        assert!(!r.paint.contains_key("width"));
        assert!((r.transform_scale - 2.0).abs() < 1e-12);
        let e = &a.shapes[1];
        let eb = curve_bounds(&e.subpaths[0].segs);
        assert!((eb.width() - 10.0).abs() < 1e-3 && (eb.height() - 4.0).abs() < 1e-3);
    }

    #[test]
    fn cycles_and_garbage_are_harmless() {
        let nodes = json!([
            {"id": "a", "type": "group", "children": ["b"]},
            {"id": "b", "type": "group", "children": ["a"]},
            {"type": "rectangle"},
            42,
            {"id": "p", "type": "path", "points": "nope"},
        ]);
        let a = collect_artwork(nodes.as_array().unwrap(), &default_geometry);
        assert!(a.shapes.is_empty());
    }
}
