//! Property and no-panic coverage for planarisation and the face walk.
//!
//! Like `path_bool/prop_tests.rs`, this uses a fixed-seed xorshift rather than
//! a property-testing crate: failures reproduce from the seed in the message,
//! and nothing depends on the host or the clock.

use super::tests::v;
use super::*;

const TOL: f64 = DEFAULT_TOLERANCE;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A point on the boundary of `[0, w] × [0, h]`.
fn on_frame(rng: &mut Rng, w: f64, h: f64) -> Vector {
    match rng.below(4) {
        0 => v(rng.f(0.0, w), 0.0),
        1 => v(w, rng.f(0.0, h)),
        2 => v(rng.f(0.0, w), h),
        _ => v(0.0, rng.f(0.0, h)),
    }
}

fn frame(w: f64, h: f64) -> VectorNetwork {
    let mut n = VectorNetwork::new();
    let c = [v(0.0, 0.0), v(w, 0.0), v(w, h), v(0.0, h)];
    let ids: Vec<usize> = c.iter().map(|&p| n.add_vertex(p)).collect();
    for i in 0..4 {
        n.add_line(ids[i], ids[(i + 1) % 4]).unwrap();
    }
    n
}

/// Chords across a rectangle (straight, or cubic with control points inside
/// it, so they stay inside) cut it into faces that tile it exactly.
#[test]
fn faces_of_a_cut_rectangle_tile_it() {
    for seed in 0..150u64 {
        let mut rng = Rng::new(seed);
        let (w, h) = (rng.f(20.0, 200.0), rng.f(20.0, 200.0));
        let mut n = frame(w, h);
        let chords = 1 + rng.below(12);
        for _ in 0..chords {
            let a = on_frame(&mut rng, w, h);
            let b = on_frame(&mut rng, w, h);
            if a.dist(b) < 1.0 {
                continue;
            }
            let ia = n.add_vertex(a);
            let ib = n.add_vertex(b);
            if rng.below(2) == 0 {
                n.add_line(ia, ib).unwrap();
            } else {
                let c1 = v(rng.f(0.0, w), rng.f(0.0, h));
                let c2 = v(rng.f(0.0, w), rng.f(0.0, h));
                n.add_segment(ia, ib, c1 - a, c2 - b).unwrap();
            }
        }
        n.planarize(TOL);
        assert!(n.is_planar(TOL), "seed {seed}: not planar");
        n.validate().unwrap();
        let faces = n.faces(TOL);
        assert!(!faces.is_empty(), "seed {seed}");
        let sum: f64 = faces.iter().map(|f| f.area).sum();
        let want = w * h;
        assert!(
            (sum - want).abs() <= 1e-6 * want,
            "seed {seed}: faces sum to {sum}, rectangle is {want} ({} faces)",
            faces.len()
        );
        for (i, f) in faces.iter().enumerate() {
            assert!(f.area > 0.0, "seed {seed}: face {i} area {}", f.area);
            // A face's interior point is in that face and no other.
            if f.area < 1e-2 {
                continue;
            }
            let Some(p) = f.interior_point(&n, TOL) else {
                continue;
            };
            let owners: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, g)| g.contains(&n, p, TOL))
                .map(|(k, _)| k)
                .collect();
            assert_eq!(owners, vec![i], "seed {seed}: point {p:?}");
        }
        // Planarising again changes nothing.
        let again = n.clone().planarize(TOL);
        assert!(!again.changed(), "seed {seed}: second pass {again:?}");
    }
}

/// A star-shaped polygon fanned from its centre: the triangles' areas sum to
/// the polygon's, and every spoke is interior.
#[test]
fn fan_triangulation_areas_sum_to_the_polygon() {
    for seed in 0..150u64 {
        let mut rng = Rng::new(1000 + seed);
        let k = 3 + rng.below(10) as usize;
        let mut angles: Vec<f64> = (0..k).map(|_| rng.f(0.0, std::f64::consts::TAU)).collect();
        angles.sort_by(f64::total_cmp);
        angles.dedup_by(|a, b| (*a - *b).abs() < 0.05);
        // Every angular gap under π keeps the origin inside and the polygon
        // star-shaped about it, so every spoke is interior.
        let gaps_ok = angles.len() >= 3
            && (0..angles.len()).all(|i| {
                let next = if i + 1 < angles.len() {
                    angles[i + 1]
                } else {
                    angles[0] + std::f64::consts::TAU
                };
                next - angles[i] < std::f64::consts::PI - 0.05
            });
        if !gaps_ok {
            continue;
        }
        let pts: Vec<Vector> = angles
            .iter()
            .map(|&a| {
                let r = rng.f(5.0, 50.0);
                v(r * a.cos(), r * a.sin())
            })
            .collect();
        let poly_area: f64 = (0..pts.len())
            .map(|i| pts[i].cross(pts[(i + 1) % pts.len()]) * 0.5)
            .sum();
        if poly_area <= 1.0 {
            continue;
        }
        let mut n = VectorNetwork::new();
        let ids: Vec<usize> = pts.iter().map(|&p| n.add_vertex(p)).collect();
        for i in 0..ids.len() {
            n.add_line(ids[i], ids[(i + 1) % ids.len()]).unwrap();
        }
        let hub = n.add_vertex(v(0.0, 0.0));
        let mut spokes = 0;
        for &id in &ids {
            if rng.below(3) != 0 {
                n.add_line(hub, id).unwrap();
                spokes += 1;
            }
        }
        n.planarize(TOL);
        let faces = n.faces(TOL);
        let sum: f64 = faces.iter().map(|f| f.area).sum();
        assert!(
            (sum - poly_area).abs() <= 1e-9 * poly_area.max(1.0),
            "seed {seed}: {sum} vs {poly_area}"
        );
        // With two or more spokes the fan has that many faces (sectors); a
        // single spoke is a dangling edge inside one face.
        let want = if spokes >= 2 { spokes } else { 1 };
        // (A sector wider than π can make the hub reflex, which is still one
        // face per sector.)
        assert_eq!(faces.len(), want, "seed {seed}");
    }
}

/// Messy input (coincident points, duplicate and zero-length segments,
/// self-loops, wild tangents) never panics, always validates, and comes out
/// planar; toggling fills and converting to paths never panics either.
#[test]
fn messy_networks_do_not_panic_and_end_planar() {
    for seed in 0..250u64 {
        let mut rng = Rng::new(5000 + seed);
        let mut n = VectorNetwork::new();
        let nv = 2 + rng.below(14) as usize;
        for _ in 0..nv {
            // A coarse grid makes coincidences likely.
            let p = if rng.below(3) == 0 {
                v(rng.below(4) as f64 * 10.0, rng.below(4) as f64 * 10.0)
            } else {
                v(rng.f(-5.0, 35.0), rng.f(-5.0, 35.0))
            };
            n.add_vertex(p);
        }
        let ns = rng.below(20) as usize;
        for _ in 0..ns {
            let a = rng.below(nv as u64) as usize;
            let b = rng.below(nv as u64) as usize;
            let (ts, te) = match rng.below(4) {
                0 | 1 => (Vector::ZERO, Vector::ZERO),
                2 => (
                    v(rng.f(-30.0, 30.0), rng.f(-30.0, 30.0)),
                    v(rng.f(-30.0, 30.0), rng.f(-30.0, 30.0)),
                ),
                _ => (v(rng.f(-1e-4, 1e-4), 0.0), Vector::ZERO),
            };
            n.add_segment(a, b, ts, te).unwrap();
            if rng.below(6) == 0 {
                // Exact duplicate.
                n.add_segment(a, b, ts, te).unwrap();
            }
        }
        if !n.segments.is_empty() && rng.below(2) == 0 {
            let lp: Vec<usize> = (0..n.segments.len())
                .filter(|_| rng.below(2) == 0)
                .collect();
            if !lp.is_empty() {
                n.regions
                    .push(VectorRegion::new(WindingRule::NonZero, vec![lp]));
            }
        }
        n.planarize(TOL);
        n.validate().unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(n.is_planar(TOL), "seed {seed}: not planar: {}", n.to_json());
        let faces = n.faces(TOL);
        for (i, f) in faces.iter().enumerate() {
            assert!(f.area > 0.0 && f.area.is_finite(), "seed {seed}");
            // Faces are disjoint: a face's interior point is in no other face.
            if f.area < 0.5 {
                continue;
            }
            if let Some(p) = f.interior_point(&n, TOL) {
                let owners = faces.iter().filter(|g| g.contains(&n, p, TOL)).count();
                assert_eq!(owners, 1, "seed {seed}: face {i} point {p:?}");
            }
        }
        let _ = n.to_paths();
        let _ = n.stroke_runs();
        for _ in 0..3 {
            let p = v(rng.f(-5.0, 35.0), rng.f(-5.0, 35.0));
            let _ = n.toggle_fill_at(p, TOL);
            n.validate().unwrap();
        }
    }
}
