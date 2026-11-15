//! Property tests for standard-library shapes and text over generated options.
//! `PROPTEST_CASES` overrides the per-property case counts below.
use proptest::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use vork_std::evaluate;

type V3 = [f64; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

struct Mesh {
    positions: Vec<V3>,
    normals: Vec<V3>,
    uvs: Vec<[f64; 2]>,
    indices: Vec<usize>,
}

fn mesh(function: &str, options: Value) -> Mesh {
    let fragment: Value = serde_json::from_str(
        &evaluate(function, &options.to_string())
            .unwrap_or_else(|e| panic!("{function} {options}: {e}")),
    )
    .unwrap();
    let name = options["name"].as_str().unwrap();
    let buffer = |channel: &str| fragment["buffers"][format!("{name}_{channel}")]["data"].clone();
    let indices: Vec<u64> = serde_json::from_value(buffer("indices")).unwrap();
    Mesh {
        positions: serde_json::from_value(buffer("position")).unwrap(),
        normals: serde_json::from_value(buffer("normal")).unwrap(),
        uvs: serde_json::from_value(buffer("TEXCOORD_0")).unwrap(),
        indices: indices.into_iter().map(|i| i as usize).collect(),
    }
}

impl Mesh {
    /// Vertex ids merged by position, so UV and normal seams do not open the surface.
    fn welded(&self, tolerance: f64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.positions.len()).collect();
        order.sort_by(|&a, &b| self.positions[a][0].total_cmp(&self.positions[b][0]));
        let mut parent: Vec<usize> = (0..self.positions.len()).collect();
        fn root(parent: &[usize], mut i: usize) -> usize {
            while parent[i] != i {
                i = parent[i];
            }
            i
        }
        for (k, &a) in order.iter().enumerate() {
            for &b in &order[k + 1..] {
                if self.positions[b][0] - self.positions[a][0] > tolerance {
                    break;
                }
                let d = sub(self.positions[a], self.positions[b]);
                if dot(d, d) <= tolerance * tolerance {
                    let (ra, rb) = (root(&parent, a), root(&parent, b));
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        (0..parent.len()).map(|i| root(&parent, i)).collect()
    }
    fn lower(&self) -> V3 {
        self.positions.iter().fold([f64::INFINITY; 3], |m, p| {
            [m[0].min(p[0]), m[1].min(p[1]), m[2].min(p[2])]
        })
    }
    /// Twice-area geometric normal of triangle `t`, following its winding.
    fn face(&self, t: &[usize]) -> V3 {
        let p = |k: usize| self.positions[t[k]];
        cross(sub(p(1), p(0)), sub(p(2), p(0)))
    }
    fn area(&self) -> f64 {
        self.indices
            .chunks_exact(3)
            .map(|t| dot(self.face(t), self.face(t)).sqrt() / 2.)
            .sum()
    }
    fn signed_volume(&self) -> f64 {
        self.indices
            .chunks_exact(3)
            .map(|t| {
                dot(
                    self.positions[t[0]],
                    cross(self.positions[t[1]], self.positions[t[2]]),
                ) / 6.
            })
            .sum()
    }
}

/// A closed shape, its options and its exact volume (or an upper bound when not `exact`).
#[derive(Clone, Debug)]
struct Shape {
    function: &'static str,
    options: Value,
    volume: f64,
    exact: bool,
}

/// Exact volume of a closed `(radius, height)` profile polygon revolved through `n` straight
/// segments: each segment is a wedge of volume `sin(2pi/n)` times the profile's first moment.
fn revolved(profile: &[(f64, f64)], n: u32) -> f64 {
    let moment: f64 = (0..profile.len())
        .map(|i| {
            let ((x0, y0), (x1, y1)) = (profile[i], profile[(i + 1) % profile.len()]);
            (x0 + x1) * (x0 * y1 - x1 * y0) / 6.
        })
        .sum();
    n as f64 * (std::f64::consts::TAU / n as f64).sin() * moment.abs()
}

/// UV-sphere rows from the top pole to the bottom pole.
fn meridian(radius: f64, rings: u32) -> Vec<(f64, f64)> {
    (0..=rings)
        .map(|y| {
            let a = y as f64 * std::f64::consts::PI / rings as f64;
            (radius * a.sin(), radius * a.cos())
        })
        .collect()
}

fn shape() -> impl Strategy<Value = Shape> {
    let length = || 0.05f64..5.;
    let segments = || 3u32..40;
    prop_oneof![
        prop::array::uniform3(length()).prop_map(|size| Shape {
            function: "box",
            options: json!({"name": "s", "size": size}),
            volume: size.iter().product(),
            exact: true,
        }),
        (length(), segments(), 2u32..30).prop_map(|(radius, segments, rings)| Shape {
            function: "sphere",
            options: json!({"name": "s", "radius": radius, "segments": segments, "rings": rings}),
            volume: revolved(&meridian(radius, rings), segments),
            exact: true,
        }),
        (length(), length(), segments()).prop_map(|(radius, height, segments)| Shape {
            function: "cylinder",
            options: json!({"name": "s", "radius": radius, "height": height, "segments": segments}),
            volume: revolved(
                &[
                    (0., -height / 2.),
                    (radius, -height / 2.),
                    (radius, height / 2.),
                    (0., height / 2.)
                ],
                segments
            ),
            exact: true,
        }),
        (length(), length(), segments()).prop_map(|(radius, height, segments)| Shape {
            function: "cone",
            options: json!({"name": "s", "radius": radius, "height": height, "segments": segments}),
            volume: revolved(
                &[
                    (0., -height / 2.),
                    (radius, -height / 2.),
                    (0., height / 2.)
                ],
                segments
            ),
            exact: true,
        }),
        (length(), 0.05f64..0.95, segments(), 3u32..30).prop_map(
            |(radius, ratio, segments, rings)| {
                let tube = radius * ratio;
                let profile: Vec<_> = (0..rings)
                    .map(|y| {
                        let b = y as f64 * std::f64::consts::TAU / rings as f64;
                        (radius + tube * b.cos(), tube * b.sin())
                    })
                    .collect();
                Shape {
                    function: "torus",
                    options: json!({
                        "name": "s", "radius": radius, "tube": tube,
                        "segments": segments, "rings": rings,
                    }),
                    volume: revolved(&profile, segments),
                    exact: true,
                }
            }
        ),
        (length(), 2f64..6., segments(), 2u32..16).prop_map(|(radius, ratio, segments, rings)| {
            let height = radius * ratio;
            // The equator row repeats, offset to each end of the straight section.
            let mut profile = meridian(radius, 2 * rings);
            profile.insert(rings as usize + 1, profile[rings as usize]);
            for (y, p) in profile.iter_mut().enumerate() {
                p.1 += if y <= rings as usize { 1. } else { -1. } * (height / 2. - radius);
            }
            Shape {
                function: "capsule",
                options: json!({
                    "name": "s", "radius": radius, "height": height,
                    "segments": segments, "rings": rings,
                }),
                volume: revolved(&profile, segments),
                exact: true,
            }
        }),
        // A profile rising from the axis and back, listed upward or downward.
        (
            prop::collection::vec((0.05f64..5., 0.01f64..2.), 1..8),
            segments(),
            any::<bool>()
        )
            .prop_map(|(rows, segments, reverse)| {
                let mut y = 0.;
                let mut profile = vec![(0., 0.)];
                for (r, dy) in rows {
                    y += dy;
                    profile.push((r, y));
                }
                profile.push((0., y + 0.5));
                if reverse {
                    profile.reverse();
                }
                Shape {
                    function: "revolve",
                    options: json!({"name": "s", "profile": profile, "segments": segments}),
                    volume: revolved(&profile, segments),
                    exact: true,
                }
            }),
        // Closed profiles can meet or follow the axis at any cyclic offset.
        (length(), length(), segments(), 0usize..6, any::<bool>())
            .prop_map(|(radius, height, segments, shift, reverse)| {
                let mut profile = vec![(0., 0.), (radius, 0.), (radius, height),
                    (radius * 0.8, height), (radius * 0.8, height * 0.1), (0., height * 0.1)];
                profile.rotate_left(shift);
                if reverse { profile.reverse(); }
                profile.push(profile[0]);
                Shape { function: "revolve",
                    options: json!({"name":"s", "profile":profile, "segments":segments}),
                    volume: revolved(&profile, segments), exact:true }
            }),
        // A closed loop (a ring) star-shaped around its centre, either winding.
        (
            prop::collection::vec(0.3f64..1., 3..12),
            length(),
            segments(),
            any::<bool>()
        )
            .prop_map(|(radii, scale, segments, reverse)| {
                let mut profile: Vec<(f64, f64)> = radii
                    .iter()
                    .enumerate()
                    .map(|(k, rho)| {
                        let a = k as f64 * std::f64::consts::TAU / radii.len() as f64;
                        (scale * (1.5 + rho * a.cos()), scale * rho * a.sin())
                    })
                    .collect();
                if reverse {
                    profile.reverse();
                }
                let volume = revolved(&profile, segments);
                profile.push(profile[0]);
                Shape {
                    function: "revolve",
                    options: json!({"name": "s", "profile": profile, "segments": segments}),
                    volume,
                    exact: true,
                }
            }),
        (prop::array::uniform3(length()), 0.05f64..=1., 2u32..12).prop_map(
            |(size, ratio, segments)| Shape {
                function: "rounded_box",
                options: json!({
                    "name": "s", "size": size, "segments": segments,
                    "radius": size.iter().copied().fold(f64::INFINITY, f64::min) / 2. * ratio,
                }),
                volume: size.iter().product(),
                exact: false,
            }
        ),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Closed shapes weld into watertight, consistently wound, nondegenerate surfaces whose
    /// winding, unit normals and volume agree with the requested solid.
    #[test]
    fn closed_shapes_are_watertight_and_outward(shape in shape()) {
        let m = mesh(shape.function, shape.options.clone());
        let extent = m.positions.iter().flatten().fold(0f64, |e, c| e.max(c.abs()));
        let welded = m.welded(extent * 1e-9);
        prop_assert!(m.positions.iter().flatten().all(|c| c.is_finite()));
        prop_assert!(m.uvs.iter().flatten().all(|c| (0. ..=1.).contains(c)));
        prop_assert!(m.normals.iter().all(|n| (dot(*n, *n).sqrt() - 1.).abs() < 1e-9));
        prop_assert!(m.indices.len().is_multiple_of(3) && m.indices.iter().all(|&i| i < m.positions.len()));
        let mut edges: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [welded[t[0]], welded[t[1]], welded[t[2]]];
            prop_assert!(a != b && b != c && c != a, "degenerate triangle {:?}", t);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                edges.entry((u.min(v), u.max(v))).or_default().push((u, v));
            }
            // Front faces agree with every authored vertex normal.
            for &k in t {
                let facing = dot(m.face(t), m.normals[k]);
                prop_assert!(facing > 0., "triangle {:?} faces from {:?}", t, m.normals[k]);
            }
        }
        for (edge, uses) in &edges {
            prop_assert!(uses.len() == 2 && uses[0] != uses[1], "edge {:?} used {:?}", edge, uses);
        }
        let volume = m.signed_volume();
        prop_assert!(volume > 0.);
        if shape.exact {
            let error = (volume - shape.volume).abs();
            prop_assert!(error <= 1e-9 * shape.volume, "{} != {}", volume, shape.volume);
        } else {
            prop_assert!(volume <= shape.volume * (1. + 1e-9), "{} > {}", volume, shape.volume);
        }
    }

    /// Scaling every length option scales positions and keeps topology, normals and UVs.
    #[test]
    fn shapes_scale_with_their_lengths(shape in shape(), exponent in -4i32..=4) {
        let k = 2f64.powi(exponent);
        let mut scaled = shape.options.clone();
        for key in ["size", "radius", "height", "tube", "profile"] {
            match scaled.get_mut(key) {
                Some(Value::Array(points)) if points[0].is_array() => {
                    for p in points.iter_mut().flat_map(|p| p.as_array_mut().unwrap()) {
                        *p = json!(p.as_f64().unwrap() * k);
                    }
                }
                Some(Value::Array(size)) => {
                    for v in size {
                        *v = json!(v.as_f64().unwrap() * k);
                    }
                }
                Some(v) => *v = json!(v.as_f64().unwrap() * k),
                None => {}
            }
        }
        let (a, b) = (mesh(shape.function, shape.options), mesh(shape.function, scaled));
        prop_assert_eq!(&a.indices, &b.indices);
        prop_assert_eq!(&a.uvs, &b.uvs);
        let normals = a.normals.iter().zip(&b.normals);
        for ((p, q), (n, o)) in a.positions.iter().zip(&b.positions).zip(normals) {
            let d = sub([p[0] * k, p[1] * k, p[2] * k], *q);
            prop_assert!(dot(d, d).sqrt() <= 1e-12 * k, "{:?} * {} != {:?}", p, k, q);
            let d = sub(*n, *o);
            prop_assert!(dot(d, d).sqrt() <= 1e-12, "{:?} != {:?}", n, o);
        }
    }
}

/// Planar area must not depend on how coarsely the bevel is sampled.
#[test]
fn rounded_box_preserves_exact_flat_regions() {
    for size in [
        [0.8, 0.8, 0.02],
        [0.02, 0.8, 0.8],
        [0.8, 0.02, 0.8],
        [2., 3., 4.],
    ] {
        for radius in [
            size.iter().copied().fold(f64::INFINITY, f64::min) * 0.4,
            0.00001,
            size.iter().copied().fold(f64::INFINITY, f64::min) / 2.,
        ] {
            for segments in [2, 3, 8, 9] {
                let m = mesh(
                    "rounded_box",
                    json!({"name":"s", "size":size, "radius":radius, "segments":segments}),
                );
                assert!(m.indices.len() / 3 <= 12 * (segments + 2) * (segments + 2));
                assert!(boundary(&m).is_empty());
                let welded = m.welded(1e-12);
                for (i, &j) in welded.iter().enumerate() {
                    for k in 0..3 {
                        assert!((m.normals[i][k] - m.normals[j][k]).abs() < 1e-12);
                    }
                }
                for axis in 0..3 {
                    for sign in [-1., 1.] {
                        let mut area = 0.;
                        for t in m.indices.chunks_exact(3) {
                            if t.iter().all(|&i| {
                                (m.positions[i][axis] - sign * size[axis] / 2.).abs() < 1e-14
                            }) {
                                area += dot(m.face(t), m.face(t)).sqrt() / 2.;
                                let mut normal = [0.; 3];
                                normal[axis] = sign;
                                for &i in t {
                                    assert_eq!(m.normals[i], normal);
                                }
                            }
                        }
                        let exact = (size[(axis + 1) % 3] - 2. * radius)
                            * (size[(axis + 2) % 3] - 2. * radius);
                        assert!(
                            (area - exact).abs() < 1e-12,
                            "{size:?}, r={radius}, segments={segments}: {area} != {exact}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn maximum_radius_shapes_have_no_collapsed_triangles() {
    for (shape, options) in [
        (
            "capsule",
            json!({"name":"s", "radius":0.5, "height":1., "segments":8, "rings":3}),
        ),
        (
            "rounded_box",
            json!({"name":"s", "size":[1.,1.,1.], "radius":0.5, "segments":3}),
        ),
        (
            "rounded_box",
            json!({"name":"s", "size":[1.,2.,3.], "radius":0.5, "segments":2}),
        ),
    ] {
        let m = mesh(shape, options);
        assert!(boundary(&m).is_empty());
        for t in m.indices.chunks_exact(3) {
            for &i in t {
                assert!(dot(m.face(t), m.normals[i]) > 0.);
            }
        }
    }
}

/// Directed edges used by exactly one triangle, after welding: the open boundary.
fn boundary(m: &Mesh) -> Vec<(usize, usize)> {
    let welded = m.welded(1e-9);
    let mut edges: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    for t in m.indices.chunks_exact(3) {
        for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (u, v) = (welded[u], welded[v]);
            *edges.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    edges
        .into_iter()
        .filter(|&(_, n)| n == 1)
        .map(|(e, _)| e)
        .collect()
}

#[test]
fn grid_faces_up_with_documented_uvs() {
    let m = mesh(
        "grid",
        json!({"name": "s", "size": [4, 2], "segments": [4, 3]}),
    );
    assert_eq!(m.positions.len(), 5 * 4);
    assert_eq!(m.indices.len(), 4 * 3 * 6);
    for ((p, n), uv) in m.positions.iter().zip(&m.normals).zip(&m.uvs) {
        assert_eq!(*n, [0., 1., 0.]);
        assert_eq!(p[1], 0.);
        assert!(
            (uv[0] - (p[0] / 4. + 0.5)).abs() < 1e-12 && (uv[1] - (p[2] / 2. + 0.5)).abs() < 1e-12
        );
    }
    for t in m.indices.chunks_exact(3) {
        let f = m.face(t);
        assert!(f[1] > 0. && f[0] == 0. && f[2] == 0.);
    }
    assert!((m.area() - 8.).abs() < 1e-12);
    assert_eq!(boundary(&m).len(), 2 * (4 + 3));
    for bad in [
        json!({"name": "s", "segments": 4}),
        json!({"name": "s", "segments": [0, 1]}),
        json!({"name": "s", "size": [1, 1, 1]}),
    ] {
        assert!(evaluate("grid", &bad.to_string()).is_err(), "{bad}");
    }
}

#[test]
fn open_revolve_keeps_rims_and_splits_sharp_corners() {
    // An open funnel: a frustum tube; then a cup with a sharp foot and an open rim.
    let shade = mesh(
        "revolve",
        json!({"name": "s", "profile": [[1, 0], [0.5, 1]], "segments": 12}),
    );
    assert_eq!(boundary(&shade).len(), 2 * 12);
    let cup = mesh(
        "revolve",
        json!({"name": "s", "segments": 16, "profile": [[0, 0], [1, 0], [1, 1], [0.9, 1], [0.9, 0.1], [0, 0.1]]}),
    );
    assert!(boundary(&cup).is_empty());
    for m in [&shade, &cup] {
        for t in m.indices.chunks_exact(3) {
            for &k in t {
                assert!(dot(m.face(t), m.normals[k]) > 0.);
            }
        }
    }
    // Outward: the shade's normals point away from the axis; the cup's floor (inside) faces up.
    assert!(
        shade
            .positions
            .iter()
            .zip(&shade.normals)
            .all(|(p, n)| p[0] * n[0] + p[2] * n[2] > 0.)
    );
    assert!(
        cup.positions
            .iter()
            .zip(&cup.normals)
            .any(|(p, n)| (p[1] - 0.1).abs() < 1e-12 && n[1] == 1.)
    );
    // 90° corners split: the foot ring carries both the floor-down and the wall-out normal.
    let foot: Vec<_> = cup
        .positions
        .iter()
        .zip(&cup.normals)
        .filter(|(p, _)| p[1] == 0. && (p[0] - 1.).abs() < 1e-12)
        .map(|(_, n)| *n)
        .collect();
    assert!(foot.contains(&[0., -1., 0.]) && foot.contains(&[1., 0., 0.]));
    // v runs along the profile by length; u around from +X.
    assert!(
        cup.uvs
            .iter()
            .all(|uv| (0. ..=1.).contains(&uv[0]) && (0. ..=1.).contains(&uv[1]))
    );
    for bad in [
        json!({"name": "s", "profile": [[1, 0]]}),
        json!({"name": "s", "profile": [[1, 0], [1, 0]]}),
        json!({"name": "s", "profile": [[-1, 0], [1, 0]]}),
        json!({"name": "s", "profile": [[1, 0], [0, 1], [1, 2]]}),
        json!({"name": "s", "profile": [[0, 0], [0, 1]]}),
        json!({"name": "s", "profile": [[1, 0], [2, 0], [3, 0], [1, 0]]}),
    ] {
        assert!(evaluate("revolve", &bad.to_string()).is_err(), "{bad}");
    }
}

const FONT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fonts/LiberationSans-Regular.ttf"
);

fn body() -> impl Strategy<Value = String> {
    "[A-Za-z0-9 .,!?&@#%()]{1,12}(\n[A-Za-z0-9 .,]{0,8}){0,2}"
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Text meshes are finite with unit normals, stay within the extruded slab, and wind
    /// every triangle toward its normal.
    #[test]
    fn text_is_finite_and_front_facing(
        body in body(),
        size in 0.05f64..4.,
        extrude in prop_oneof![Just(0.), 0f64..0.5],
        bevel in prop_oneof![Just(0.), 0f64..0.05],
        resolution in 0u32..6,
        align_x in prop::sample::select(vec!["left", "center", "right"]),
        align_y in prop::sample::select(
            vec!["top_baseline", "top", "center", "bottom_baseline", "bottom"]
        ),
    ) {
        let options = json!({
            "name": "t", "body": body, "font": FONT, "size": size, "extrude": extrude * size,
            "bevel_depth": bevel * size, "bevel_resolution": resolution,
            "align_x": align_x, "align_y": align_y,
        });
        if body.chars().all(|c| c.is_whitespace()) {
            prop_assert!(evaluate("text", &options.to_string()).is_err());
            return Ok(());
        }
        let m = mesh("text", options);
        let depth = (extrude + bevel) * size;
        prop_assert!(!m.indices.is_empty() && m.indices.len().is_multiple_of(3));
        prop_assert!(m.indices.iter().all(|&i| i < m.positions.len()));
        prop_assert!(m.positions.iter().flatten().all(|c| c.is_finite()));
        prop_assert!(m.positions.iter().all(|p| p[2].abs() <= depth * (1. + 1e-6)));
        prop_assert!(m.normals.iter().all(|n| (dot(*n, *n).sqrt() - 1.).abs() < 1e-5));
        for t in m.indices.chunks_exact(3) {
            prop_assert!(dot(m.face(t), m.normals[t[0]]) >= 0.);
        }
    }

    /// Horizontal and vertical alignment only translate a single line of text. Cap
    /// triangulation may differ by rounding, so compare vertices and total area.
    #[test]
    fn alignment_translates_text(
        body in "[A-Za-z0-9.,!?&]{1,12}",
        size in 0.05f64..4.,
        align_x in prop::sample::select(vec!["center", "right"]),
        align_y in prop::sample::select(vec!["top", "center", "bottom_baseline", "bottom"]),
    ) {
        let text = |x: &str, y: &str| {
            let options = json!({
                "name": "t", "body": body, "font": FONT, "size": size, "extrude": 0.1 * size,
                "align_x": x, "align_y": y,
            });
            mesh("text", options)
        };
        let base = text("left", "top_baseline");
        let moved = text(align_x, align_y);
        prop_assert_eq!(base.indices.len(), moved.indices.len());
        let (a, b) = (base.area(), moved.area());
        prop_assert!((a - b).abs() <= 1e-4 * a, "area {} != {}", a, b);
        prop_assert_eq!(base.positions.len(), moved.positions.len());
        let shift = sub(moved.lower(), base.lower());
        prop_assert!(shift[2] == 0.);
        // Every aligned vertex must coincide with a translated unaligned vertex.
        let n = base.positions.len();
        let translated = base.positions.iter().map(|p| [p[0] + shift[0], p[1] + shift[1], p[2]]);
        let both = Mesh {
            positions: translated.chain(moved.positions).collect(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        };
        let welded = both.welded(1e-5 * size);
        prop_assert!(welded[n..].iter().all(|&root| root < n), "alignment is not a translation");
    }
}
