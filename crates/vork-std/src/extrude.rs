//! Polygon outline with holes → closed prism along Z, sharing text's cap tessellation.
use crate::text::{Mesh, contains, fill};
use anyhow::{Context, Result, ensure};
use lyon_path::math::{Point, Vector, point, vector};
use serde::Deserialize;

/// Blender 5.2.1 Shade Auto Smooth / Smooth by Angle default, as SMOOTH_ANGLE_DEGREES in
/// packages/ts/src/lib/constants.ts: wall corners turning further are sharp.
const SMOOTH_ANGLE_DEGREES: f32 = 30.;

fn default_name() -> String {
    "extrude".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Options {
    #[serde(default = "default_name")]
    name: String,
    #[serde(default)]
    outline: Option<Vec<[f32; 2]>>,
    #[serde(default)]
    d: Option<String>,
    #[serde(default)]
    holes: Vec<Vec<[f32; 2]>>,
    #[serde(default = "default_depth")]
    depth: f32,
    /// Inward rounded rim radius in metres; preserves outline bounds and total depth.
    #[serde(default)]
    bevel_depth: f32,
    /// Quarter-circle subdivisions (1..32).
    #[serde(default = "default_bevel_resolution")]
    bevel_resolution: u32,
}
fn default_bevel_resolution() -> u32 {
    4
}
fn default_depth() -> f32 {
    1.
}

pub(crate) fn evaluate(argument: &str) -> Result<String> {
    Ok(serde_json::to_string(&geometry(serde_json::from_str(
        argument,
    )?)?)?)
}

/// Twice the signed area in f64, relative to the first point so translation cannot cancel it.
fn signed_area(contour: &[Point]) -> f64 {
    let o = contour[0];
    let d = |p: Point| (f64::from(p.x - o.x), f64::from(p.y - o.y));
    (1..contour.len().saturating_sub(1))
        .map(|i| {
            let ((x0, y0), (x1, y1)) = (d(contour[i]), d(contour[i + 1]));
            x0 * y1 - x1 * y0
        })
        .sum()
}
/// Drop repeats (including a closing repeat of the first point) and points within `merge`
/// of the chord joining their neighbours, until none remain; the first point always stays.
fn clean(contour: &mut Vec<Point>, merge: f32) {
    let mut i = 1;
    let mut kept = 0;
    while contour.len() >= 3 && kept < contour.len() - 1 {
        let n = contour.len();
        let chord = [contour[i - 1], contour[(i + 1) % n]];
        if point_segment(contour[i], chord) <= f64::from(merge) {
            contour.remove(i);
            kept = 0;
        } else {
            i += 1;
            kept += 1;
        }
        if i >= contour.len() {
            i = 1;
        }
    }
    if contour.len() == 2 && contour[0] == contour[1] {
        contour.pop();
    }
}
fn point_segment(p: Point, [a, b]: [Point; 2]) -> f64 {
    let v = |p: Point| (f64::from(p.x), f64::from(p.y));
    let ((px, py), (ax, ay), (bx, by)) = (v(p), v(a), v(b));
    let (dx, dy) = (bx - ax, by - ay);
    let length = dx * dx + dy * dy;
    let t = if length > 0. {
        (((px - ax) * dx + (py - ay) * dy) / length).clamp(0., 1.)
    } else {
        0.
    };
    (px - ax - t * dx).hypot(py - ay - t * dy)
}
/// Closed-segment distance: zero when they cross, touch or overlap.
fn segment_distance(s: [Point; 2], t: [Point; 2]) -> f64 {
    let orient = |a: Point, b: Point, c: Point| {
        (f64::from(b.x) - f64::from(a.x)) * (f64::from(c.y) - f64::from(a.y))
            - (f64::from(b.y) - f64::from(a.y)) * (f64::from(c.x) - f64::from(a.x))
    };
    let (o1, o2) = (orient(s[0], s[1], t[0]), orient(s[0], s[1], t[1]));
    let (o3, o4) = (orient(t[0], t[1], s[0]), orient(t[0], t[1], s[1]));
    if o1 * o2 < 0. && o3 * o4 < 0. {
        return 0.;
    }
    point_segment(s[0], t)
        .min(point_segment(s[1], t))
        .min(point_segment(t[0], s))
        .min(point_segment(t[1], s))
}
/// Right-hand unit normal of the edge from `a` to `b`: outward on a counter-clockwise outline,
/// into the hole on a clockwise hole.
fn right(a: Point, b: Point) -> Vector {
    let d = (b - a).normalize();
    vector(d.y, -d.x)
}

fn geometry(mut o: Options) -> Result<serde_json::Value> {
    ensure!(
        !o.name.is_empty() && !o.name.contains('/'),
        "extrude name must be nonempty without /"
    );
    ensure!(
        o.depth.is_finite() && o.depth > 0.,
        "depth must be finite and positive"
    );
    ensure!(
        o.bevel_depth.is_finite() && o.bevel_depth >= 0.,
        "bevel_depth must be finite and nonnegative"
    );
    ensure!(
        (1..=32).contains(&o.bevel_resolution),
        "bevel_resolution must be in 1..32"
    );
    ensure!(
        o.outline.is_some() != o.d.is_some(),
        "extrude requires exactly one of outline or d"
    );
    if let Some(d) = &o.d {
        ensure!(
            o.holes.is_empty(),
            "holes cannot be combined with d; use SVG subpaths"
        );
        let (cmds, _) = crate::moda::path_commands(d)?;
        // Bound flattening error relative to the authored geometry, in object-space metres.
        let bounds = crate::moda::path_bounds(&cmds).context("d must enclose an area")?;
        let extent = (bounds[2] - bounds[0]).max(bounds[3] - bounds[1]);
        ensure!(extent.is_finite() && extent > 0., "d must enclose an area");
        // Moda's flattening for its own extrusion: every subpath closed, repeated points dropped.
        let mut contours = moda_engine::solid::outline::flatten(&cmds, extent * 0.0001);
        ensure!(!contours.is_empty(), "d must enclose an area");
        // SVG fill implicitly closes open subpaths. Extrude retains its existing
        // one-outline-with-holes contract, including intersection validation below.
        let points = |c: Vec<[f64; 2]>| c.into_iter().map(|[x, y]| [x as f32, y as f32]).collect::<Vec<_>>();
        o.outline = Some(points(contours.remove(0)));
        o.holes = contours.into_iter().map(points).collect();
    }
    let outline = o.outline.as_ref().unwrap();
    let mut contours: Vec<Vec<Point>> = std::iter::once(outline)
        .chain(&o.holes)
        .map(|c| c.iter().map(|&[x, y]| point(x, y)).collect())
        .collect();
    ensure!(
        contours
            .iter()
            .flatten()
            .all(|p| p.x.is_finite() && p.y.is_finite()),
        "outline and hole points must be finite [x, y]"
    );
    let (lo, hi) = contours[0].iter().fold(
        (
            point(f32::INFINITY, f32::INFINITY),
            point(f32::NEG_INFINITY, f32::NEG_INFINITY),
        ),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let extent = (hi - lo).x.max((hi - lo).y);
    ensure!(extent > 0., "outline must enclose an area");
    // Points closer than this to an edge they do not belong to are indistinguishable at
    // f32 output precision, and Lyon's fill tolerance would join them.
    let merge = extent * 8. * f32::EPSILON;
    for (k, contour) in contours.iter_mut().enumerate() {
        let what = if k == 0 { "outline" } else { "hole" };
        clean(contour, merge);
        ensure!(
            contour.len() >= 3 && signed_area(contour) != 0.,
            "{what} must enclose an area"
        );
        let area = signed_area(contour);
        // Outline counter-clockwise, holes clockwise: the non-zero fill then leaves holes
        // empty. The first point stays first: it anchors the wall UV seam.
        if (area > 0.) != (k == 0) {
            contour[1..].reverse();
        }
    }
    // Every pair of edges stays apart, except neighbours meeting at their shared point.
    let segments: Vec<_> = contours
        .iter()
        .enumerate()
        .flat_map(|(k, c)| (0..c.len()).map(move |i| (k, i, c.len(), [c[i], c[(i + 1) % c.len()]])))
        .collect();
    let mut clearance = f64::INFINITY;
    for (i, &(k, a, n, s)) in segments.iter().enumerate() {
        clearance = clearance.min(f64::from((s[1] - s[0]).length()));
        for &(l, b, _, t) in &segments[i + 1..] {
            let distance = if k != l || (b != a + 1 && !(a == 0 && b == n - 1)) {
                segment_distance(s, t)
            } else if b == a + 1 {
                point_segment(s[0], t).min(point_segment(t[1], s))
            } else {
                point_segment(s[1], t).min(point_segment(t[0], s))
            };
            if k != l || (b != a + 1 && !(a == 0 && b == n - 1)) {
                clearance = clearance.min(distance);
            }
            ensure!(
                distance >= f64::from(merge),
                "outline and holes must not cross or touch: edges {:?}-{:?} and {:?}-{:?}",
                s[0].to_array(),
                s[1].to_array(),
                t[0].to_array(),
                t[1].to_array()
            );
        }
    }
    for (k, contour) in contours.iter().enumerate().skip(1) {
        ensure!(
            contains(&contours[0], contour[0]),
            "hole {} must lie inside the outline",
            k - 1
        );
        for (j, other) in contours.iter().enumerate().take(k).skip(1) {
            ensure!(
                !contains(other, contour[0]) && !contains(contour, other[0]),
                "holes {} and {} must not nest",
                j - 1,
                k - 1
            );
        }
    }
    // Offset line intersections use the same contour orientation for outline and holes.
    // Bound every vertex's motion below half the shortest edge/non-neighbour clearance:
    // no edge can reverse and no moving contours can meet anywhere along the bevel.
    let miters: Vec<Vec<Vector>> = contours
        .iter()
        .map(|c| {
            (0..c.len())
                .map(|i| {
                    let a = right(c[(i + c.len() - 1) % c.len()], c[i]);
                    let b = right(c[i], c[(i + 1) % c.len()]);
                    (a + b) / (1. + a.dot(b))
                })
                .collect()
        })
        .collect();
    if o.bevel_depth > 0. {
        let longest = miters
            .iter()
            .flatten()
            .map(|m| m.length())
            .fold(0f32, f32::max);
        ensure!(
            miters
                .iter()
                .flatten()
                .all(|m| m.x.is_finite() && m.y.is_finite()),
            "profile must have finite inward rim intersections"
        );
        // Leave the merge tolerance between moving edges. Round down so the strict
        // clearance bound remains safe when converted back to output precision.
        let limit = (clearance - f64::from(merge)) / (2. * f64::from(longest));
        let rounded = limit as f32;
        let profile_limit = if f64::from(rounded) >= limit {
            rounded.next_down()
        } else {
            rounded
        };
        o.bevel_depth = crate::clamp_radii(
            &[o.bevel_depth],
            &[(o.depth, &[0, 0]), (profile_limit.max(0.), &[0])],
        )[0];
    }
    let cap_contours: Vec<Vec<Point>> = contours
        .iter()
        .zip(&miters)
        .map(|(c, m)| {
            c.iter()
                .zip(m)
                .map(|(p, m)| {
                    if o.bevel_depth > 0. {
                        *p - *m * o.bevel_depth
                    } else {
                        *p
                    }
                })
                .collect()
        })
        .collect();
    let caps = fill(&cap_contours, 2. * merge)?;
    let z = o.depth / 2.;
    let size = hi - lo;
    let mut mesh = Mesh::default();
    // Caps span the outline's bounds: u along +X (mirrored on the back, as seen from behind),
    // v down from the top.
    for sign in [1f32, -1.] {
        let base = mesh.p.len() as u32;
        for p in &caps.vertices {
            let u = (p.x - lo.x) / size.x;
            mesh.vertex_uv(
                [p.x, p.y, sign * z],
                [0., 0., sign],
                [if sign > 0. { u } else { 1. - u }, (hi.y - p.y) / size.y],
            );
        }
        for t in caps.indices.chunks_exact(3) {
            mesh.tri(base + t[0], base + t[1], base + t[2]);
        }
    }
    // Walls: u by length along each contour from its first point, v from front (0) to back (1).
    let smooth = SMOOTH_ANGLE_DEGREES.to_radians().cos();
    // (inset, z, outward normal component, cap normal component), front to back.
    let mut rings = Vec::new();
    if o.bevel_depth > 0. {
        let r = o.bevel_depth;
        for sign in [1f32, -1.] {
            for step in 0..=o.bevel_resolution {
                let k = if sign > 0. {
                    step
                } else {
                    o.bevel_resolution - step
                };
                let angle = k as f32 / o.bevel_resolution as f32 * std::f32::consts::FRAC_PI_2;
                let (inset, height, nxy, nz) = if k == 0 {
                    (r, sign * z, 0., sign)
                } else if k == o.bevel_resolution {
                    (0., sign * (z - r), 1., 0.)
                } else {
                    (
                        r * (1. - angle.sin()),
                        sign * (z - r + r * angle.cos()),
                        angle.sin(),
                        sign * angle.cos(),
                    )
                };
                if sign < 0. && step == 0 && r == z {
                    continue;
                }
                rings.push((inset, height, nxy, nz));
            }
        }
    } else {
        rings.extend([(0., z, 1., 0.), (0., -z, 1., 0.)]);
    }
    let rows = rings.len() as u32;
    for (contour, miter) in contours.iter().zip(&miters) {
        let n = contour.len();
        let mut along = vec![0f32];
        for i in 0..n {
            along.push(along[i] + (contour[(i + 1) % n] - contour[i]).length());
        }
        let normal: Vec<_> = (0..n)
            .map(|i| right(contour[i], contour[(i + 1) % n]))
            .collect();
        let mut column = |i: usize, normal: Vector, u: f32| {
            let front = mesh.p.len() as u32;
            for &(inset, height, nxy, nz) in &rings {
                let p = if inset > 0. {
                    contour[i] - miter[i] * inset
                } else {
                    contour[i]
                };
                mesh.vertex_uv(
                    [p.x, p.y, height],
                    [normal.x * nxy, normal.y * nxy, nz],
                    [u, (z - height) / o.depth],
                );
            }
            front
        };
        // starts[e] / ends[e]: the front vertex of edge e's columns at its first / last point.
        let (mut starts, mut ends) = (vec![0; n], vec![0; n]);
        for i in 0..n {
            let (incoming, outgoing) = (normal[(i + n - 1) % n], normal[i]);
            let shared =
                (incoming.dot(outgoing) >= smooth).then(|| (incoming + outgoing).normalize());
            let u = along[i] / along[n];
            let previous = (i + n - 1) % n;
            match shared {
                // The first point is also the seam at u = 1.
                Some(normal) if i > 0 => {
                    let c = column(i, normal, u);
                    (ends[previous], starts[i]) = (c, c);
                }
                _ => {
                    ends[previous] =
                        column(i, shared.unwrap_or(incoming), if i == 0 { 1. } else { u });
                    starts[i] = column(i, shared.unwrap_or(outgoing), u);
                }
            }
        }
        for e in 0..n {
            let (a, b) = (starts[e], ends[e]);
            for row in 0..rows - 1 {
                mesh.tri(a + row, b + row, b + row + 1);
                mesh.tri(a + row, b + row + 1, a + row + 1);
            }
        }
    }
    Ok(mesh.fragment(&o.name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::tests::watertight;
    use serde_json::{Value, json};

    fn extrude(options: Value) -> Result<Value> {
        geometry(serde_json::from_value(options)?)
    }
    fn data<T: serde::de::DeserializeOwned>(g: &Value, key: &str) -> T {
        serde_json::from_value(g["buffers"][format!("e_{key}")]["data"].clone()).unwrap()
    }
    const SQUARE: [[f32; 2]; 4] = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]];

    #[test]
    fn svg_and_outline_produce_the_same_mesh() {
        for bevel in [0., 0.03] {
            let outline = extrude(json!({"name":"e", "outline":SQUARE,
                "holes":[[[-0.3,-0.3],[-0.3,0.3],[0.3,0.3],[0.3,-0.3]]], "bevel_depth":bevel}))
            .unwrap();
            let svg = extrude(
                json!({"name":"e", "d":"M-1-1h2v2h-2z M -0.3 -0.3 v 0.6 h 0.6 v -0.6 z", "bevel_depth":bevel}),
            )
            .unwrap();
            assert_eq!(svg, outline);
            watertight(&svg, "e").unwrap();
        }
        for options in [
            json!({}),
            json!({"outline":SQUARE,"d":"M0 0L1 0L0 1Z"}),
            json!({"d":"M0 0h1v1z","holes":[SQUARE]}),
            json!({"d":"M0 0L"}),
        ] {
            assert!(extrude(options).is_err());
        }
    }

    #[test]
    fn svg_cleanup_is_independent_of_coordinate_units() {
        for size in [1e-10_f32, 1e-6, 1., 1e6] {
            let svg = extrude(json!({"name":"e", "d":format!("M0 0 L{size} 0 L0 {size} Z"), "depth":size})).unwrap();
            let array = extrude(json!({"name":"e", "outline":[[0.,0.],[size,0.],[0.,size]], "depth":size})).unwrap();
            assert_eq!(svg, array, "size {size}");
        }
        // A hole below the consumer's output precision must be refused, not
        // silently discarded by the path flattener (which would fill it in).
        assert!(extrude(json!({"name":"e", "d":"M0 0h1v1h-1z M.1 .1l1e-10 0l0 1e-10z"})).is_err());
    }

    fn assert_svg_builder_subdivision(d: &str, extent: f64) {
        // Compare the complete builder output to an array-outline reference at
        // the independently known extent, rather than testing the bounds helper.
        let (commands, _) = crate::moda::path_commands(d).unwrap();
        let contours = moda_engine::solid::outline::flatten(&commands, extent * 0.0001);
        assert_eq!(contours.len(), 1);
        let outline: Vec<_> = contours[0].iter().map(|&[x,y]| [x as f32,y as f32]).collect();
        let reference = extrude(json!({"name":"e", "outline":outline, "depth":0.25})).unwrap();
        let built = extrude(json!({"name":"e", "d":d, "depth":0.25})).unwrap();
        assert_eq!(data::<Vec<[f32;3]>>(&built, "position").len(),
            data::<Vec<[f32;3]>>(&reference, "position").len(), "subdivision: {d}");
        assert_eq!(built, reference, "mesh: {d}");
    }

    #[test]
    fn svg_builder_subdivision_uses_major_arc_extent() {
        // Near-complete arcs have diameter 2r despite endpoints one unit apart.
        for (d, extent) in [
            ("M0 0 A10 10 0 1 1 1 0 Z", 20.),
            ("M0 0 A1000 1000 0 1 1 1 0 Z", 2000.),
        ] {
            assert_svg_builder_subdivision(d, extent);
        }
    }

    #[test]
    fn svg_builder_subdivision_ordinary_curve_control() {
        // This quadratic stays inside its unit control box, so the copied
        // endpoint/control bounds and geometric bounds agree.
        assert_svg_builder_subdivision("M0 0 Q0 1 1 1 L1 0 Z", 1.);
    }

    #[test]
    fn svg_curves_extrude_watertight() {
        let mesh =
            extrude(json!({"name":"e", "d":"M1 0 A1 1 0 0 1 -1 0 A1 1 0 0 1 1 0Z", "depth":0.3}))
                .unwrap();
        watertight(&mesh, "e").unwrap();
    }

    #[test]
    fn plate_with_holes_is_closed_outward_and_exact() {
        // Either winding is accepted for the outline and for each hole.
        let hole = |x: f32| vec![[x, -0.2], [x, 0.2], [x + 0.4, 0.2], [x + 0.4, -0.2]];
        for (outline, holes) in [
            (SQUARE.to_vec(), vec![]),
            (
                SQUARE.iter().rev().copied().collect(),
                vec![hole(-0.6), hole(0.2)],
            ),
        ] {
            let g =
                extrude(json!({"name":"e","outline":outline,"holes":holes,"depth":0.5})).unwrap();
            watertight(&g, "e").unwrap();
            let p: Vec<[f64; 3]> = data(&g, "position");
            let i: Vec<usize> = data(&g, "indices");
            let volume: f64 = i
                .chunks_exact(3)
                .map(|t| {
                    let [a, b, c] = [p[t[0]], p[t[1]], p[t[2]]];
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                        + a[1] * (b[2] * c[0] - b[0] * c[2])
                        + a[2] * (b[0] * c[1] - b[1] * c[0])
                })
                .sum::<f64>()
                / 6.;
            let expected = (4. - 0.16 * holes.len() as f64) * 0.5;
            assert!((volume - expected).abs() < 1e-5, "{volume} != {expected}");
            assert!(p.iter().all(|p| p[2].abs() == 0.25));
        }
    }

    #[test]
    fn bevels_close_concave_profiles_and_holes_without_changing_bounds() {
        let star: Vec<[f32; 2]> = (0..10)
            .map(|i| {
                let a = i as f32 * std::f32::consts::PI / 5.;
                let r = if i % 2 == 0 { 1. } else { 0.45 };
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        for outline in [SQUARE.to_vec(), star] {
            for resolution in [1, 4, 32] {
                for depth in [0.2, 0.04] {
                    let g = extrude(json!({"name":"e", "outline":outline, "depth":depth,
                        "bevel_depth":0.02, "bevel_resolution":resolution}))
                    .unwrap();
                    watertight(&g, "e").unwrap_or_else(|e| {
                        panic!("{outline:?}, resolution {resolution}, depth {depth}: {e}")
                    });
                    let p: Vec<[f32; 3]> = data(&g, "position");
                    let n: Vec<[f32; 3]> = data(&g, "normal");
                    assert!(p.iter().all(|p| p[2].abs() <= depth / 2.));
                    assert!(n.iter().any(|n| n[2] > 0. && n[2] < 1.) || resolution == 1);
                    assert!(
                        n.iter()
                            .all(|n| (n.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5)
                    );
                    for axis in 0..2 {
                        for fold in [f32::min as fn(f32, f32) -> f32, f32::max] {
                            let expected = outline.iter().map(|p| p[axis]).reduce(fold).unwrap();
                            let actual = p.iter().map(|p| p[axis]).reduce(fold).unwrap();
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
        for reverse in [false, true] {
            let mut outline = SQUARE.to_vec();
            let mut hole = vec![[-0.3, -0.3], [-0.3, 0.3], [0.3, 0.3], [0.3, -0.3]];
            if reverse {
                outline.reverse();
                hole.reverse();
            }
            let g = extrude(
                json!({"name":"e","outline":outline,"holes":[hole],"depth":0.3,"bevel_depth":0.03}),
            )
            .unwrap();
            watertight(&g, "e").unwrap();
        }
        for options in [
            json!({"outline":SQUARE,"bevel_depth":-0.1}),
            json!({"outline":SQUARE,"bevel_resolution":0}),
            json!({"outline":SQUARE,"bevel_resolution":33}),
        ] {
            assert!(extrude(options).unwrap_err().to_string().contains("bevel"));
        }
    }

    #[test]
    fn oversized_bevel_matches_maximum_exactly() {
        for (outline, depth) in [
            (SQUARE.to_vec(), 0.1),
            (vec![[0., 0.], [2., 0.], [2., 0.01], [0., 0.01]], 1.),
        ] {
            let oversized =
                extrude(json!({"name":"e","outline":outline,"depth":depth,"bevel_depth":100.}))
                    .unwrap();
            // Exercise both the depth bound and the thin profile clearance bound.
            let maximum = if depth == 0.1 {
                depth / 2.
            } else {
                let merge = 2. * 8. * f32::EPSILON;
                let bound = (f64::from(0.01f32) - f64::from(merge)) / (2. * f64::from(2f32.sqrt()));
                let rounded = bound as f32;
                if f64::from(rounded) >= bound {
                    rounded.next_down()
                } else {
                    rounded
                }
            };
            assert_eq!(
                oversized,
                extrude(json!({"name":"e","outline":outline,"depth":depth,"bevel_depth":maximum}))
                    .unwrap()
            );
            watertight(&oversized, "e").unwrap();
        }
    }

    #[test]
    fn corners_split_above_the_smooth_angle() {
        // A 64-gon has 5.6° corners: smooth walls. A square's 90° corners: flat walls.
        let circle: Vec<[f32; 2]> = (0..64)
            .map(|i| {
                let a = i as f32 * std::f32::consts::TAU / 64.;
                [a.cos(), a.sin()]
            })
            .collect();
        for (outline, smooth) in [(circle, true), (SQUARE.to_vec(), false)] {
            let g = extrude(json!({"name":"e","outline":outline})).unwrap();
            watertight(&g, "e").unwrap();
            let p: Vec<[f32; 3]> = data(&g, "position");
            let n: Vec<[f32; 3]> = data(&g, "normal");
            let uv: Vec<[f32; 2]> = data(&g, "TEXCOORD_0");
            assert!(uv.iter().flatten().all(|c| (0. ..=1.).contains(c)));
            for (p, n) in p.iter().zip(&n) {
                assert!(((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.).abs() < 1e-5);
                if n[2] == 0. {
                    let radial = (n[0] * p[0] + n[1] * p[1]) / (p[0] * p[0] + p[1] * p[1]).sqrt();
                    if smooth {
                        assert!(radial > 0.9999, "{p:?} {n:?}");
                    } else {
                        assert!(
                            (n[0].abs() == 1. && n[1] == 0.) || (n[1].abs() == 1. && n[0] == 0.),
                            "{n:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uvs_follow_the_documented_axes() {
        let g = extrude(json!({"name":"e","outline":[[0,0],[2,0],[2,1],[0,1]]})).unwrap();
        let p: Vec<[f32; 3]> = data(&g, "position");
        let n: Vec<[f32; 3]> = data(&g, "normal");
        let uv: Vec<[f32; 2]> = data(&g, "TEXCOORD_0");
        for ((p, n), uv) in p.iter().zip(&n).zip(&uv) {
            if n[2] == 1. {
                assert_eq!(*uv, [p[0] / 2., 1. - p[1]]);
            } else if n[2] == -1. {
                assert_eq!(*uv, [1. - p[0] / 2., 1. - p[1]]);
            } else {
                assert_eq!(uv[1], if p[2] > 0. { 0. } else { 1. });
            }
        }
        // Wall u runs by length from the first point [0,0] around the perimeter of 6.
        let wall = |at: [f32; 2]| -> Vec<f32> {
            p.iter()
                .zip(&n)
                .zip(&uv)
                .filter(|((p, n), _)| n[2] == 0. && p[0] == at[0] && p[1] == at[1] && p[2] > 0.)
                .map(|(_, uv)| uv[0])
                .collect()
        };
        assert_eq!(wall([0., 0.]), [1., 0.]);
        assert_eq!(wall([2., 0.]), [2. / 6., 2. / 6.]);
        assert_eq!(wall([2., 1.]), [3. / 6., 3. / 6.]);
    }

    #[test]
    fn far_thin_and_reordered_outlines_stay_closed_and_outward() {
        for outline in [
            // Far from the origin: f32 products there would cancel the area.
            json!([[4096, 4096], [4097, 4096], [4097, 4097], [4096, 4097]]),
            // Thin but valid.
            json!([[0, 0], [1, 0], [1, 0.000_01], [0, 0.000_01]]),
            // The first point lies mid-edge, and the winding is clockwise.
            json!([[1, 0], [0, 0], [0, 1], [2, 1], [2, 0]]),
        ] {
            let g = extrude(json!({"name":"e","outline":outline,"depth":0.1})).unwrap();
            watertight(&g, "e").unwrap_or_else(|e| panic!("{outline}: {e}"));
        }
        // The supplied first point anchors the wall seam even when the winding is reversed.
        let g = extrude(json!({"name":"e","outline":[[1,0],[0,0],[0,1],[2,1],[2,0]]})).unwrap();
        let p: Vec<[f32; 3]> = data(&g, "position");
        let n: Vec<[f32; 3]> = data(&g, "normal");
        let uv: Vec<[f32; 2]> = data(&g, "TEXCOORD_0");
        let seam: Vec<f32> = p
            .iter()
            .zip(&n)
            .zip(&uv)
            .filter(|((p, n), _)| n[2] == 0. && p[0] == 1. && p[1] == 0. && p[2] > 0.)
            .map(|(_, uv)| uv[0])
            .collect();
        assert_eq!(seam, [1., 0.]);
    }

    /// Every malformed input is an ordinary error, never a panic.
    #[test]
    fn malformed_inputs_error_without_panicking() {
        let square = json!([[0, 0], [1, 0], [1, 1], [0, 1]]);
        let shapes = [
            json!([]),
            json!([[0, 0]]),
            json!([[0, 0], [0, 0], [0, 0]]),
            json!([[0, 0], [1, 0]]),
        ];
        let mut cases: Vec<Value> = shapes
            .iter()
            .map(|s| json!({"name":"e","outline":s}))
            .collect();
        cases.extend(
            shapes
                .iter()
                .map(|s| json!({"name":"e","outline":square,"holes":[s]})),
        );
        cases.extend([
            json!({"name":"e","outline":square,"holes":[[], []]}),
            json!({"name":"e","outline":square,"depth":-1}),
            json!({"name":"","outline":square}),
            json!({"name":"e"}),
            json!({"name":"e","outline":[[0]]}),
            json!({"name":"e","outline":"square"}),
        ]);
        for case in cases {
            let text = case.to_string();
            let outcome = std::panic::catch_unwind(|| evaluate(&text));
            assert!(
                outcome.expect(&format!("{text} panicked")).is_err(),
                "{text}"
            );
        }
    }

    #[test]
    fn invalid_outlines_fail_with_a_reason() {
        for (options, reason) in [
            (json!({"name":"e","outline":[[0,0],[1,0]]}), "area"),
            (json!({"name":"e","outline":[[0,0],[1,0],[2,0]]}), "area"),
            // Two holes sharing a corner.
            (
                json!({"name":"e","outline":[[-3,-3],[3,-3],[3,3],[-3,3]],
                    "holes":[[[-2,-1],[-1,-1],[-1,0],[-2,0]],[[-1,0],[0,0],[0,1],[-1,1]]]}),
                "touch",
            ),
            // A spike folding back along its own edge.
            (
                json!({"name":"e","outline":[[0,0],[2,0],[1,0],[1,1]]}),
                "touch",
            ),
            (
                json!({"name":"e","outline":[[0,0],[2,2],[2,0],[0,1]]}),
                "cross",
            ),
            (
                json!({"name":"e","outline":SQUARE,"holes":[[[2,2],[3,2],[3,3]]]}),
                "inside",
            ),
            (
                json!({"name":"e","outline":SQUARE,"holes":[[[-0.5,-0.5],[1.5,-0.5],[0,0.5]]]}),
                "cross",
            ),
            (json!({"name":"e","outline":SQUARE,"depth":0}), "depth"),
            (json!({"name":"e/x","outline":SQUARE}), "name"),
        ] {
            let error = extrude(options.clone()).unwrap_err().to_string();
            assert!(error.contains(reason), "{options}: {error}");
        }
        assert!(serde_json::from_str::<Options>(r#"{"name":"e","outline":[],"hole":[]}"#).is_err());
    }
}
