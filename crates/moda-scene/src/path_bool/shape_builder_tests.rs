//! Shape Builder: the regions overlapping shapes make (count, area, cover,
//! style source, an interior sample) and merging / removing picked regions.

use crate::value::Value;

use super::tests::{ellipse, even_odd_area, rect};
use super::*;

const SQ: f64 = 200.0 * 200.0;
const OVER: f64 = 100.0 * 100.0;

fn op(points: Vec<PathPoint>) -> BooleanOperand {
    BooleanOperand {
        points,
        fill_rule: FillRule::NonZero,
    }
}

/// Back square at (0, 0), front square at (100, 100), both 200 × 200.
fn squares() -> Vec<BooleanOperand> {
    vec![
        op(rect(0.0, 0.0, 200.0, 200.0)),
        op(rect(100.0, 100.0, 200.0, 200.0)),
    ]
}

fn regions(operands: &[BooleanOperand], painted: &[bool]) -> Vec<ShapeBuilderRegion> {
    shape_builder_regions(operands, painted, None).expect("regions")
}

fn build(
    operands: &[BooleanOperand],
    painted: &[bool],
    mode: BuildMode,
    picks: &[(f64, f64)],
) -> BuildResult {
    shape_builder_build(operands, painted, None, mode, picks)
        .expect("build")
        .expect("a pick hit a region")
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-3 * b.abs().max(1.0)
}

fn changed_area(r: &OperandResult) -> Option<f64> {
    match r {
        OperandResult::Changed(s) => Some(even_odd_area(&s.points)),
        _ => None,
    }
}

/// The region `p` falls in, by its cover.
fn region_at<'a>(rs: &'a [ShapeBuilderRegion], cover: &[usize]) -> &'a ShapeBuilderRegion {
    rs.iter().find(|r| r.cover == cover).unwrap_or_else(|| {
        panic!(
            "no region with cover {cover:?}: {:?}",
            rs.iter().map(|r| &r.cover).collect::<Vec<_>>()
        )
    })
}

#[test]
fn two_squares_make_three_regions_with_cover_source_and_samples() {
    let rs = regions(&squares(), &[true, true]);
    assert_eq!(rs.len(), 3);
    let back = region_at(&rs, &[0]);
    let lens = region_at(&rs, &[0, 1]);
    let front = region_at(&rs, &[1]);
    assert!(close(even_odd_area(&back.points), SQ - OVER));
    assert!(close(even_odd_area(&lens.points), OVER));
    assert!(close(even_odd_area(&front.points), SQ - OVER));
    // Style: the topmost filled shape over the region.
    assert_eq!((back.source, lens.source, front.source), (0, 1, 1));
    // Samples sit inside their region, not on an edge.
    let (x, y) = lens.sample;
    assert!(
        x > 101.0 && x < 199.0 && y > 101.0 && y < 199.0,
        "lens sample {x},{y}"
    );
    let (x, y) = back.sample;
    assert!(
        x > 0.0 && y > 0.0 && x < 200.0 && y < 200.0 && !(x > 100.0 && y > 100.0),
        "back sample {x},{y}"
    );
}

#[test]
fn an_unfilled_shape_still_bounds_regions_but_does_not_style_them() {
    let rs = regions(&squares(), &[true, false]);
    assert_eq!(
        rs.len(),
        3,
        "the unfilled front square's own region is kept"
    );
    assert_eq!(
        region_at(&rs, &[0, 1]).source,
        0,
        "the lens wears the filled back square"
    );
    assert_eq!(
        region_at(&rs, &[1]).source,
        1,
        "no filled shape covers it: the topmost covering one"
    );
}

#[test]
fn a_hole_region_is_sampled_in_its_ring_not_its_hole() {
    let rs = regions(
        &[
            op(rect(0.0, 0.0, 300.0, 300.0)),
            op(rect(100.0, 100.0, 100.0, 100.0)),
        ],
        &[true, true],
    );
    assert_eq!(rs.len(), 2);
    let ring = region_at(&rs, &[0]);
    assert!(close(even_odd_area(&ring.points), 300.0 * 300.0 - OVER));
    let (x, y) = ring.sample;
    assert!(
        !(x >= 100.0 && x <= 200.0 && y >= 100.0 && y <= 200.0),
        "ring sample {x},{y} is in the hole"
    );
}

#[test]
fn venn_of_three_circles_has_seven_regions() {
    let operands = vec![
        op(ellipse(140.0, 140.0, 100.0, 100.0)),
        op(ellipse(240.0, 140.0, 100.0, 100.0)),
        op(ellipse(190.0, 227.0, 100.0, 100.0)),
    ];
    let rs = regions(&operands, &[true, true, true]);
    assert_eq!(rs.len(), 7);
    let mut covers: Vec<Vec<usize>> = rs.iter().map(|r| r.cover.clone()).collect();
    covers.sort();
    assert_eq!(
        covers,
        vec![
            vec![0],
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 2],
            vec![1],
            vec![1, 2],
            vec![2]
        ]
    );
    // The regions tile the union: their areas add up to it.
    let total: f64 = rs.iter().map(|r| even_odd_area(&r.points)).sum();
    let union = match boolean_paths(&BooleanRequest {
        op: BooleanOp::Union,
        operands,
        accuracy: None,
    })
    .unwrap()
    {
        BooleanOutcome::Shape { points, .. } => even_odd_area(&points),
        BooleanOutcome::Empty => panic!("empty union"),
    };
    assert!(close(total, union), "regions {total} vs union {union}");
}

#[test]
fn merging_back_and_lens_restyles_from_the_first_pick_and_cuts_the_operands() {
    // Drag from the back-only region into the lens.
    let r = build(
        &squares(),
        &[true, true],
        BuildMode::Merge,
        &[(50.0, 50.0), (150.0, 150.0)],
    );
    assert_eq!(r.picked.len(), 2);
    let (merged, source) = r.merged.expect("merged");
    assert_eq!(source, 0, "the region the drag started on");
    assert_eq!(
        merged.subpath_count, 1,
        "adjacent regions merge into one outline"
    );
    assert!(close(even_odd_area(&merged.points), SQ));
    // The back square is entirely taken; the front loses the lens.
    assert!(
        matches!(r.operands[0], OperandResult::Removed),
        "{:?}",
        r.operands[0]
    );
    assert!(close(
        changed_area(&r.operands[1]).expect("front changed"),
        SQ - OVER
    ));
}

#[test]
fn a_single_pick_extracts_the_region() {
    let r = build(
        &squares(),
        &[true, true],
        BuildMode::Merge,
        &[(150.0, 150.0)],
    );
    let (merged, source) = r.merged.expect("merged");
    assert_eq!(source, 1);
    assert!(close(even_odd_area(&merged.points), OVER));
    for o in &r.operands {
        assert!(close(changed_area(o).expect("changed"), SQ - OVER));
    }
}

#[test]
fn remove_deletes_regions_and_leaves_the_rest() {
    let r = build(
        &squares(),
        &[true, true],
        BuildMode::Remove,
        &[(150.0, 150.0)],
    );
    assert!(r.merged.is_none());
    for o in &r.operands {
        assert!(close(changed_area(o).expect("changed"), SQ - OVER));
    }
    // Removing the front-only region leaves the back square alone.
    let r = build(
        &squares(),
        &[true, true],
        BuildMode::Remove,
        &[(250.0, 250.0)],
    );
    assert!(matches!(r.operands[0], OperandResult::Unchanged));
    assert!(close(
        changed_area(&r.operands[1]).expect("front changed"),
        OVER
    ));
}

#[test]
fn merging_every_region_of_two_circles_is_their_union() {
    let operands = vec![
        op(ellipse(140.0, 140.0, 100.0, 100.0)),
        op(ellipse(240.0, 140.0, 100.0, 100.0)),
    ];
    let r = build(
        &operands,
        &[true, true],
        BuildMode::Merge,
        &[(80.0, 140.0), (190.0, 140.0), (300.0, 140.0)],
    );
    let (merged, source) = r.merged.expect("merged");
    assert_eq!(source, 0);
    assert_eq!(merged.subpath_count, 1);
    let circle = std::f64::consts::PI * 100.0 * 100.0;
    // Lens of two r=100 circles 100 apart: 2r²acos(d/2r) − (d/2)√(4r² − d²).
    let lens = 2.0 * 100.0f64.powi(2) * (0.5f64).acos()
        - 50.0 * (4.0 * 100.0f64.powi(2) - 100.0f64.powi(2)).sqrt();
    let area = even_odd_area(&merged.points);
    assert!(
        (area - (2.0 * circle - lens)).abs() / area < 2e-3,
        "union area {area}"
    );
    assert!(r
        .operands
        .iter()
        .all(|o| matches!(o, OperandResult::Removed)));
}

#[test]
fn picks_outside_every_region_and_repeats_are_ignored() {
    assert!(shape_builder_build(
        &squares(),
        &[true, true],
        None,
        BuildMode::Merge,
        &[(500.0, 500.0)]
    )
    .unwrap()
    .is_none());
    let r = build(
        &squares(),
        &[true, true],
        BuildMode::Merge,
        &[
            (500.0, 500.0),
            (150.0, 150.0),
            (160.0, 160.0),
            (f64::NAN, 1.0),
        ],
    );
    assert_eq!(r.picked.len(), 1);
}

#[test]
fn json_envelope_round_trips() {
    let operands = r#"[
        {"points":[{"type":"move","x":0,"y":0},{"type":"line","x":200,"y":0},{"type":"line","x":200,"y":200},{"type":"line","x":0,"y":200}],"fillRule":"nonzero"},
        {"points":[{"type":"move","x":100,"y":100},{"type":"line","x":300,"y":100},{"type":"line","x":300,"y":300},{"type":"line","x":100,"y":300}],"fillRule":"nonzero","painted":false}
    ]"#;
    let regions: Value = crate::value::from_str(&shape_builder_json(&format!(
        r#"{{"op":"regions","operands":{operands}}}"#
    )))
    .unwrap();
    assert_eq!(regions["status"], "ok");
    assert_eq!(regions["regions"].as_array().unwrap().len(), 3);
    for r in regions["regions"].as_array().unwrap() {
        assert!(r["sample"]["x"].is_number() && r["cover"].is_array() && r["source"].is_number());
    }

    let merged: Value = crate::value::from_str(&shape_builder_json(&format!(
        r#"{{"op":"merge","operands":{operands},"picks":[[150,150],[250,250]]}}"#
    )))
    .unwrap();
    assert_eq!(merged["status"], "ok");
    assert_eq!(
        merged["merged"]["source"], 0,
        "the lens wears the filled back square"
    );
    assert_eq!(merged["operands"][0]["status"], "changed");
    assert_eq!(merged["operands"][1]["status"], "removed");

    let removed: Value = crate::value::from_str(&shape_builder_json(&format!(
        r#"{{"op":"remove","operands":{operands},"picks":[[50,50]]}}"#
    )))
    .unwrap();
    assert!(removed["merged"].is_null());
    assert_eq!(removed["operands"][1]["status"], "unchanged");

    let miss: Value = crate::value::from_str(&shape_builder_json(&format!(
        r#"{{"op":"merge","operands":{operands},"picks":[[900,900]]}}"#
    )))
    .unwrap();
    assert_eq!(miss["status"], "empty");

    let bad: Value = crate::value::from_str(&shape_builder_json(
        r#"{"op":"merge","operands":[],"picks":[1]}"#,
    ))
    .unwrap();
    assert_eq!(bad["status"], "error");
    let bad: Value =
        crate::value::from_str(&shape_builder_json(r#"{"op":"nope","operands":[]}"#)).unwrap();
    assert_eq!(bad["code"], "invalid_request");
}
