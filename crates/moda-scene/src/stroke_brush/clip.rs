//! Half-plane clipping of cubic subpaths against a vertical line `x = c`, in
//! the placers' art / leg coordinates: art brush trim / dash windows (clipped
//! in art space before warping) and the pattern brush's sliced corners (leg
//! coordinates, where the miter is the line `x = leg length`).
//!
//! Two flavours: [`clip_x`] for fills (a closed subpath stays closed, joined
//! along the cut) and [`clip_x_open`] for strokes (only the original outline
//! survives, as open runs: a cut edge is never stroked).
//!
//! Each cubic is split where it crosses the line; pieces on the kept side
//! survive. A closed subpath stays closed: consecutive kept runs are joined
//! by straight segments along the clip line (exact for one half-plane; the
//! joins lie on the boundary, so fills are unchanged under nonzero and
//! evenodd). An open subpath breaks into one open subpath per kept run.

use super::geom::{sub, Subpath};
use super::warp::crossings;
use crate::path_geom::vec2::Cubic;

/// Keep the part of `subpaths` with `x ≤ c` (`keep_below`) or `x ≥ c`.
pub fn clip_x(subpaths: &[Subpath], c: f64, keep_below: bool) -> Vec<Subpath> {
    let keep = |x: f64| if keep_below { x <= c } else { x >= c };
    let mut out = Vec::new();
    for sp in subpaths {
        // Pieces in order, each tagged kept or not.
        let mut pieces: Vec<(Cubic, bool)> = Vec::new();
        for seg in &sp.segs {
            let ts = crossings(seg, |p| p.x, &[c], 8);
            let mut t0 = 0.0;
            for t1 in ts.into_iter().chain(std::iter::once(1.0)) {
                if t1 <= t0 + 1e-12 {
                    continue;
                }
                let piece = sub(seg, t0, t1);
                let mid = piece.eval(0.5).x;
                pieces.push((piece, keep(mid)));
                t0 = t1;
            }
        }
        if pieces.iter().all(|p| p.1) {
            out.push(sp.clone());
            continue;
        }
        if !pieces.iter().any(|p| p.1) {
            continue;
        }
        if sp.closed {
            // Start the walk right after a dropped piece so runs are contiguous.
            let first_drop = pieces.iter().position(|p| !p.1).unwrap_or(0);
            pieces.rotate_left(first_drop);
            let mut segs: Vec<Cubic> = Vec::new();
            for (piece, kept) in &pieces {
                if !*kept {
                    continue;
                }
                if let Some(last) = segs.last() {
                    let (a, b) = (last.0[3], piece.0[0]);
                    if a.dist(b) > 1e-9 {
                        segs.push(Cubic::line(a, b));
                    }
                }
                segs.push(*piece);
            }
            if let (Some(first), Some(last)) = (segs.first().copied(), segs.last().copied()) {
                if last.0[3].dist(first.0[0]) > 1e-9 {
                    segs.push(Cubic::line(last.0[3], first.0[0]));
                }
            }
            out.push(Subpath { segs, closed: true });
        } else {
            let mut run: Vec<Cubic> = Vec::new();
            for (piece, kept) in pieces {
                if kept {
                    run.push(piece);
                } else if !run.is_empty() {
                    out.push(Subpath {
                        segs: std::mem::take(&mut run),
                        closed: false,
                    });
                }
            }
            if !run.is_empty() {
                out.push(Subpath {
                    segs: run,
                    closed: false,
                });
            }
        }
    }
    out
}

/// The part of `subpaths` on the kept side as OPEN runs of the original
/// outline: for stroking a clipped shape, so the cut itself isn't stroked. A
/// closed subpath entirely on the kept side stays closed.
pub fn clip_x_open(subpaths: &[Subpath], c: f64, keep_below: bool) -> Vec<Subpath> {
    let keep = |x: f64| if keep_below { x <= c } else { x >= c };
    let mut out = Vec::new();
    for sp in subpaths {
        let mut pieces: Vec<(Cubic, bool)> = Vec::new();
        for seg in &sp.segs {
            let ts = crossings(seg, |p| p.x, &[c], 8);
            let mut t0 = 0.0;
            for t1 in ts.into_iter().chain(std::iter::once(1.0)) {
                if t1 <= t0 + 1e-12 {
                    continue;
                }
                let piece = sub(seg, t0, t1);
                let mid = piece.eval(0.5).x;
                pieces.push((piece, keep(mid)));
                t0 = t1;
            }
        }
        if pieces.iter().all(|p| p.1) {
            out.push(sp.clone());
            continue;
        }
        if sp.closed {
            // Start after a dropped piece so a run through the seam stays whole.
            let first_drop = pieces.iter().position(|p| !p.1).unwrap_or(0);
            pieces.rotate_left(first_drop);
        }
        let mut run: Vec<Cubic> = Vec::new();
        for (piece, kept) in pieces {
            if kept {
                run.push(piece);
            } else if !run.is_empty() {
                out.push(Subpath {
                    segs: std::mem::take(&mut run),
                    closed: false,
                });
            }
        }
        if !run.is_empty() {
            out.push(Subpath {
                segs: run,
                closed: false,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_geom::vec2::v2;

    fn square(x0: f64, x1: f64) -> Subpath {
        let p = [v2(x0, 0.0), v2(x1, 0.0), v2(x1, 10.0), v2(x0, 10.0)];
        Subpath {
            segs: (0..4).map(|i| Cubic::line(p[i], p[(i + 1) % 4])).collect(),
            closed: true,
        }
    }

    fn area(sp: &Subpath) -> f64 {
        let mut a = 0.0;
        for c in &sp.segs {
            for i in 0..16 {
                let (p, q) = (c.eval(i as f64 / 16.0), c.eval((i + 1) as f64 / 16.0));
                a += p.x * q.y - q.x * p.y;
            }
        }
        a / 2.0
    }

    #[test]
    fn closed_square_is_cut_and_closed() {
        let got = clip_x(&[square(0.0, 10.0)], 4.0, true);
        assert_eq!(got.len(), 1);
        assert!((area(&got[0]) - 40.0).abs() < 1e-9, "{}", area(&got[0]));
        let got = clip_x(&[square(0.0, 10.0)], 4.0, false);
        assert!((area(&got[0]) - 60.0).abs() < 1e-9);
        let segs = &got[0].segs;
        for w in segs.windows(2) {
            assert!(w[0].0[3].dist(w[1].0[0]) < 1e-9);
        }
        assert!(clip_x(&[square(0.0, 10.0)], 20.0, false).is_empty());
        assert_eq!(
            clip_x(&[square(0.0, 10.0)], 20.0, true)[0],
            square(0.0, 10.0)
        );
    }

    #[test]
    fn open_runs_never_include_the_cut() {
        // A square cut at x = 4: the stroke keeps the three original edges on
        // the left (top, left, bottom) as one open run, not the cut edge.
        let got = clip_x_open(&[square(0.0, 10.0)], 4.0, true);
        assert_eq!(got.len(), 1);
        assert!(!got[0].closed);
        for c in &got[0].segs {
            let mid = c.eval(0.5);
            assert!(
                mid.x < 4.0 - 1e-9 || (mid.y.abs() < 1e-9 || (mid.y - 10.0).abs() < 1e-9),
                "cut edge stroked: {mid:?}"
            );
        }
        assert!(clip_x_open(&[square(0.0, 10.0)], 20.0, true)[0].closed);
    }

    #[test]
    fn open_paths_split_into_runs() {
        let zig = Subpath {
            segs: vec![
                Cubic::line(v2(0.0, 0.0), v2(10.0, 0.0)),
                Cubic::line(v2(10.0, 0.0), v2(0.0, 5.0)),
                Cubic::line(v2(0.0, 5.0), v2(10.0, 10.0)),
            ],
            closed: false,
        };
        let got = clip_x(&[zig], 5.0, true);
        assert_eq!(got.len(), 2);
    }
}
