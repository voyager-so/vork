//! Aligned dashes (`dashAlignment: "corners"`, Illustrator's "Align dashes to
//! corners and path ends, adjusting lengths to fit"): the interval geometry in
//! `dash.rs`.

use super::*;

fn line(p0: Pt, p1: Pt) -> PathSegment {
    PathSegment {
        seg: Seg::Line { p0, p1 },
        arc_length: (p1[0] - p0[0]).hypot(p1[1] - p0[1]),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

/// Every interval is `(a, b)` with a < b, sorted and apart.
fn well_formed(iv: &[(f64, f64)]) {
    for w in iv.windows(2) {
        assert!(w[0].1 < w[1].0, "{iv:?}");
    }
    assert!(iv.iter().all(|(a, b)| a < b), "{iv:?}");
}

#[test]
fn a_straight_open_path_gets_half_dashes_centred_on_its_ends() {
    // 100 long, 10 on / 10 off: five whole repeats fit, so nothing stretches.
    let iv = aligned_dash_intervals(100.0, &[], false, &[10.0, 10.0]);
    well_formed(&iv);
    assert_eq!(iv.len(), 6, "{iv:?}");
    assert!(close(iv[0].0, 0.0) && close(iv[0].1, 5.0), "{iv:?}");
    assert!(close(iv[1].0, 15.0) && close(iv[1].1, 25.0));
    assert!(close(iv[5].0, 95.0) && close(iv[5].1, 100.0), "{iv:?}");
}

#[test]
fn lengths_stretch_or_squeeze_to_a_whole_number_of_repeats() {
    // Period 20: 110 is 5.5 repeats, rounded to 6 (squeezed); 90 is 4.5, rounded to 5; 7 is at least 1 (stretched).
    for (len, repeats) in [(110.0, 6.0), (90.0, 5.0), (7.0, 1.0)] {
        let iv = aligned_dash_intervals(len, &[], false, &[10.0, 10.0]);
        well_formed(&iv);
        let k = len / (repeats * 20.0);
        assert!(close(iv[0].1, 5.0 * k), "{len}: {iv:?}");
        assert!(close(iv[iv.len() - 1].0, len - 5.0 * k), "{len}: {iv:?}");
        assert_eq!(iv.len(), repeats as usize + 1, "{len}: {iv:?}");
        // Interior dashes keep the pattern's proportions.
        if iv.len() > 2 {
            assert!(close(iv[1].1 - iv[1].0, 10.0 * k));
        }
    }
}

#[test]
fn a_closed_square_gets_one_dash_centred_on_every_corner() {
    // Perimeter 400, corners at 0 / 100 / 200 / 300; 20 on / 10 off: 3 repeats per side (k = 100/90).
    let iv = aligned_dash_intervals(400.0, &[0.0, 100.0, 200.0, 300.0], true, &[20.0, 10.0]);
    well_formed(&iv);
    let half = 10.0 * 100.0 / 90.0;
    // The start corner's dash wraps the seam: its halves are the first and last intervals.
    assert!(close(iv[0].0, 0.0) && close(iv[0].1, half), "{iv:?}");
    assert!(close(iv[iv.len() - 1].0, 400.0 - half) && close(iv[iv.len() - 1].1, 400.0));
    for c in [100.0, 200.0, 300.0] {
        let d = iv.iter().find(|(a, b)| *a < c && c < *b).expect("a dash on the corner");
        assert!(close(c - d.0, half) && close(d.1 - c, half), "centred on {c}: {d:?}");
    }
    // Per side, 2 whole dashes between the corner dashes: 4 corner dashes + 8 = 12, the seam one in two halves.
    assert_eq!(iv.len(), 13, "{iv:?}");
}

#[test]
fn a_closed_path_without_corners_is_one_span_from_its_start() {
    let iv = aligned_dash_intervals(314.0, &[], true, &[20.0, 10.0]);
    well_formed(&iv);
    assert!(close(iv[0].0, 0.0));
    assert!(close(iv[iv.len() - 1].1, 314.0), "the dash on the start wraps the seam: {iv:?}");
}

#[test]
fn a_closed_path_whose_start_is_not_a_corner_wraps_the_last_span() {
    // Corners at 50 and 250 of 400: the span 250 → 450 crosses the seam and is split there.
    let iv = aligned_dash_intervals(400.0, &[50.0, 250.0], true, &[20.0, 10.0]);
    well_formed(&iv);
    for c in [50.0, 250.0] {
        let d = iv.iter().find(|(a, b)| *a < c && c < *b).expect("a dash on the corner");
        assert!(close(c - d.0, d.1 - c), "centred on {c}: {d:?}");
    }
    assert!(iv.iter().all(|(a, b)| *a >= 0.0 && *b <= 400.0));
}

#[test]
fn corners_are_where_the_direction_turns() {
    // An L (corner at 50), then a cubic leaving tangentially (smooth: no corner).
    let c = ([50.0, 50.0], [50.0, 70.0], [70.0, 90.0], [90.0, 90.0]);
    let segs = vec![
        line([0.0, 0.0], [50.0, 0.0]),
        line([50.0, 0.0], [50.0, 50.0]),
        PathSegment {
            seg: Seg::Cubic { p0: c.0, cp1: c.1, cp2: c.2, p1: c.3 },
            arc_length: cubic_arc_length(c.0, c.1, c.2, c.3),
        },
    ];
    assert_eq!(segment_corner_positions(&segs, false), vec![50.0]);
    // Closed square: the start is a corner too.
    let sq = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
    let segs: Vec<PathSegment> = (0..4).map(|i| line(sq[i], sq[(i + 1) % 4])).collect();
    assert_eq!(segment_corner_positions(&segs, true), vec![0.0, 100.0, 200.0, 300.0]);
}

#[test]
fn intervals_become_a_walk_that_draws_exactly_them() {
    let segs = vec![line([0.0, 0.0], [100.0, 0.0])];
    let iv = aligned_dash_intervals(100.0, &[], false, &[10.0, 10.0]);
    let walked = walk_path_by_arc_length(&segs, &intervals_as_walk_pattern(&iv, 100.0));
    let pieces = dash_pieces(&walked, false, [0.0, 0.0], [100.0, 0.0]);
    assert_eq!(pieces.len(), iv.len());
    for (p, (a, b)) in pieces.iter().zip(&iv) {
        let last = p.points[p.points.len() - 1];
        assert!(close(p.points[0][0], *a) && close(last[0], *b), "{p:?} vs {a}..{b}");
    }
    // No intervals draw nothing.
    assert!(walk_path_by_arc_length(&segs, &intervals_as_walk_pattern(&[], 100.0)).is_empty());
}
