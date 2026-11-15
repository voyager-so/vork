//! Visual gallery (test-only): `cargo test -p moda-scene --lib
//! stroke_outline::gallery -- --ignored` writes PNGs of filled outlines to
//! `native/target/stroke-outline-gallery/` (git-ignored). Dark = the outline
//! filled nonzero; red = the centerline (itself outlined by this module).

use super::raster::{coverage_aa, polygons, write_rgb_png};
use super::tests::node;
use super::*;
use crate::path_geom::vec2::V2;
use crate::value::json;

fn translate(points: &Value, dx: f64, dy: f64) -> Value {
    let mut v = points.clone();
    for p in v.as_array_mut().unwrap() {
        for k in ["", "cp1", "cp2", "cp"] {
            let o = if k.is_empty() {
                Some(&mut *p)
            } else {
                p.get_mut(k)
            };
            if let Some(o) = o {
                if let (Some(x), Some(y)) = (
                    o.get("x").and_then(Value::as_f64),
                    o.get("y").and_then(Value::as_f64),
                ) {
                    o["x"] = json!(x + dx);
                    o["y"] = json!(y + dy);
                }
            }
        }
    }
    v
}

struct Sheet {
    w: usize,
    h: usize,
    ink: Vec<Vec<V2>>,
    center: Vec<Vec<V2>>,
}

impl Sheet {
    fn new(w: usize, h: usize) -> Sheet {
        Sheet {
            w,
            h,
            ink: Vec::new(),
            center: Vec::new(),
        }
    }
    fn add(&mut self, mut v: Value, brush: Option<Value>, dx: f64, dy: f64) {
        v["points"] = translate(&v["points"], dx, dy);
        let o =
            stroke_outline(&node(v.clone()), brush.as_ref(), &OutlineOptions::default()).unwrap();
        if let Outline::Shape { points, .. } = &o {
            self.ink.extend(polygons(points));
        }
        let hair = json!({"points": v["points"], "closed": v["closed"], "cornerRadius": v["cornerRadius"], "strokeWidth": 0.8, "lineJoin": "round"});
        if let Outline::Shape { points, .. } =
            stroke_outline(&node(hair), None, &OutlineOptions::default()).unwrap()
        {
            self.center.extend(polygons(&points));
        }
    }
    fn write(&self, name: &str) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/stroke-outline-gallery");
        std::fs::create_dir_all(&dir).unwrap();
        let ink = coverage_aa(&self.ink, 0.0, 0.0, 1.0, self.w, self.h, 3);
        let center = coverage_aa(&self.center, 0.0, 0.0, 1.0, self.w, self.h, 3);
        let path = dir.join(name);
        write_rgb_png(
            &path,
            self.w,
            self.h,
            &[(&ink, [29, 39, 51]), (&center, [230, 57, 70])],
        );
        eprintln!("wrote {}", path.display());
    }
}

fn s_curve() -> Value {
    json!([{"type": "move", "x": 0, "y": 30}, {"type": "curve", "x": 220, "y": 30, "cp1": {"x": 70, "y": -30}, "cp2": {"x": 150, "y": 90}}])
}

#[test]
#[ignore]
fn gallery() {
    // 1. Presets (per subpath), each on the same S curve.
    let mut sheet = Sheet::new(520, 330);
    for (i, preset) in [
        "uniform",
        "wedge",
        "taper",
        "quarter-taper",
        "eye",
        "mirrored-taper",
    ]
    .iter()
    .enumerate()
    {
        let (col, row) = (i % 2, i / 2);
        let v = json!({"points": s_curve(), "strokeWidth": 22, "lineCap": "round", "strokeProfile": {"preset": preset}});
        sheet.add(
            v,
            None,
            30.0 + col as f64 * 250.0,
            25.0 + row as f64 * 105.0,
        );
    }
    sheet.write("1-presets.png");

    // 2. Markers, taper, wave, dynamic.
    let mut sheet = Sheet::new(520, 330);
    let markers = json!({"markers": [
        {"id": "a", "t": 0, "left": 0.2, "right": 0.2},
        {"id": "b", "t": 0.3, "left": 1.6, "right": 0.4, "corner": true},
        {"id": "c", "t": 0.55, "left": 0.6, "right": 1.8},
        {"id": "d", "t": 0.55, "left": 1.4, "right": 1.4},
        {"id": "e", "t": 1, "left": 0.3, "right": 0.3}]});
    sheet.add(
        json!({"points": s_curve(), "strokeWidth": 22, "strokeProfile": markers}),
        None,
        30.0,
        25.0,
    );
    sheet.add(
        json!({"points": s_curve(), "strokeWidth": 22, "lineCap": "round",
        "strokeTaper": {"startLength": 35, "endLength": 35, "startEase": -1, "endEase": 1}}),
        None,
        280.0,
        25.0,
    );
    sheet.add(json!({"points": s_curve(), "strokeWidth": 16, "strokeWave": {"amount": 0.6, "units": "px", "wavelength": 28}}), None, 30.0, 150.0);
    sheet.add(json!({"points": s_curve(), "strokeWidth": 6, "lineCap": "round", "strokeDynamic": {"frequency": 4, "wiggle": 4, "smoothen": 0.3, "seed": 7}}), None, 280.0, 150.0);
    let ring = json!([{"type": "move", "x": 50, "y": 0}, {"type": "arc", "x": -50, "y": 0, "rx": 50, "ry": 50, "largeArc": false, "sweep": true},
                      {"type": "arc", "x": 50, "y": 0, "rx": 50, "ry": 50, "largeArc": false, "sweep": true}]);
    sheet.add(json!({"points": ring, "closed": true, "strokeWidth": 10, "strokeWave": {"amount": 0.5, "units": "cycles", "wavelength": 9}}), None, 140.0, 270.0);
    sheet.add(json!({"points": ring, "closed": true, "strokeWidth": 4, "strokeDynamic": {"frequency": 3, "wiggle": 5, "seed": 2}}), None, 390.0, 270.0);
    sheet.write("2-markers-taper-wave-dynamic.png");

    // 3. Calligraphic nib: flat 30° nib with pressure along a loopy script line.
    let mut sheet = Sheet::new(520, 220);
    let script = json!([{"type": "move", "x": 0, "y": 100},
        {"type": "curve", "x": 80, "y": 40, "cp1": {"x": 30, "y": 20}, "cp2": {"x": 60, "y": 0}},
        {"type": "curve", "x": 110, "y": 110, "cp1": {"x": 100, "y": 80}, "cp2": {"x": 60, "y": 140}},
        {"type": "curve", "x": 190, "y": 50, "cp1": {"x": 150, "y": 90}, "cp2": {"x": 170, "y": 20}},
        {"type": "curve", "x": 260, "y": 100, "cp1": {"x": 210, "y": 80}, "cp2": {"x": 230, "y": 120}},
        {"type": "curve", "x": 330, "y": 40, "cp1": {"x": 300, "y": 80}, "cp2": {"x": 260, "y": -10}},
        {"type": "curve", "x": 440, "y": 90, "cp1": {"x": 380, "y": 100}, "cp2": {"x": 400, "y": 110}}]);
    let pressure: Vec<f64> = (0..=40)
        .map(|i| 0.35 + 0.65 * ((i as f64) / 40.0 * std::f64::consts::PI).sin())
        .collect();
    let ts: Vec<f64> = (0..=40).map(|i| i as f64 / 40.0).collect();
    let brush = json!({"id": "nib", "type": "calligraphic", "angle": 30, "roundness": 0.15,
        "size": {"value": 14, "vary": "pressure", "min": 5, "max": 16}});
    sheet.add(json!({"points": script, "strokeWidth": 1, "lineCap": "round", "strokeBrush": {"brushId": "nib"},
        "inkChannels": {"t": ts, "pressure": pressure}}), Some(brush), 40.0, 40.0);
    sheet.write("3-calligraphic-pressure.png");

    // 4. Hard cases: widths far beyond curvature, hairpins, cusps, polylines.
    let mut sheet = Sheet::new(520, 330);
    let uturn = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 90, "y": 0},
        {"type": "curve", "x": 90, "y": 12, "cp1": {"x": 98, "y": 0}, "cp2": {"x": 98, "y": 12}}, {"type": "line", "x": 0, "y": 12}]);
    sheet.add(
        json!({"points": uturn, "strokeWidth": 40, "lineCap": "round", "lineJoin": "round"}),
        None,
        40.0,
        50.0,
    );
    let cusp = json!([{"type": "move", "x": 0, "y": 60}, {"type": "curve", "x": 120, "y": 60, "cp1": {"x": 170, "y": -20}, "cp2": {"x": -50, "y": -20}}]);
    sheet.add(
        json!({"points": cusp, "strokeWidth": 24, "strokeProfile": {"preset": "eye"}}),
        None,
        320.0,
        30.0,
    );
    let zig = json!([{"type": "move", "x": 0, "y": 40}, {"type": "line", "x": 30, "y": 0}, {"type": "line", "x": 60, "y": 40},
        {"type": "line", "x": 90, "y": 0}, {"type": "line", "x": 120, "y": 40}, {"type": "line", "x": 150, "y": 0}, {"type": "line", "x": 180, "y": 40}]);
    sheet.add(json!({"points": zig, "strokeWidth": 14, "lineJoin": "miter", "strokeProfile": {"preset": "mirrored-taper"}}), None, 30.0, 170.0);
    let small = json!([{"type": "move", "x": 12, "y": 0}, {"type": "arc", "x": -12, "y": 0, "rx": 12, "ry": 12, "largeArc": false, "sweep": true},
                       {"type": "arc", "x": 12, "y": 0, "rx": 12, "ry": 12, "largeArc": false, "sweep": true}]);
    sheet.add(json!({"points": small, "closed": true, "strokeWidth": 50, "strokeProfile": {"markers": [
        {"id": "a", "t": 0, "left": 0.4, "right": 1.2}, {"id": "b", "t": 0.5, "left": 1.3, "right": 0.2}, {"id": "c", "t": 1, "left": 0.4, "right": 1.2}]}}), None, 290.0, 230.0);
    let hairpin = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 110, "y": 12}, {"type": "line", "x": 0, "y": 24}]);
    sheet.add(
        json!({"points": hairpin, "strokeWidth": 12, "lineJoin": "round", "lineCap": "round"}),
        None,
        380.0,
        200.0,
    );
    sheet.write("4-hard-cases.png");
}
