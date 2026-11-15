//! Visual gallery (test-only): `cargo test -p moda-scene --lib
//! stroke_brush::gallery -- --ignored` writes `docs/roadmap/brushes/gallery/
//! art-*.png`. Placed artwork is drawn with its own paints; the grey hairline
//! is the stroke's centerline.

use crate::value::{json, Value};

use super::art::{art_brush_for_node, parse_art_brush, ArtOptions};
use super::artwork::default_geometry;
use super::raster::{polygons, Canvas};
use super::tests_art::{arrow_art, brush, line_node};

fn out_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/roadmap/brushes/gallery");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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
    canvas: Canvas,
}

impl Sheet {
    fn new(w: usize, h: usize) -> Sheet {
        Sheet {
            canvas: Canvas::new(w, h, 0.0, 0.0, 1.0),
        }
    }

    fn add(&mut self, points: &Value, node_extra: Value, brush_def: &Value, dx: f64, dy: f64) {
        let pts = translate(points, dx, dy);
        let node = line_node(pts.clone(), node_extra);
        let b = parse_art_brush(brush_def).unwrap();
        let placed = art_brush_for_node(
            node.as_object().unwrap(),
            &b,
            &default_geometry,
            &ArtOptions::default(),
        );
        self.canvas.draw(&placed.paths);
        let closed = node["closed"].as_bool().unwrap_or(false);
        let center: Vec<_> = pts
            .as_array()
            .unwrap()
            .iter()
            .filter_map(crate::blend::BlendPathPoint::from_json)
            .collect();
        self.canvas
            .hairline(&polygons(&center), closed, [150.0, 150.0, 150.0]);
    }

    fn save(&self, name: &str) {
        self.canvas.write_png(&out_dir().join(name));
    }
}

fn s_curve() -> Value {
    json!([{"type": "move", "x": 0, "y": 40},
        {"type": "curve", "x": 150, "y": 40, "cp1": {"x": 40, "y": -30}, "cp2": {"x": 110, "y": -30}},
        {"type": "curve", "x": 300, "y": 40, "cp1": {"x": 190, "y": 110}, "cp2": {"x": 260, "y": 110}}])
}

fn leaf() -> Value {
    json!([
        {"id": "leaf", "type": "path", "closed": true, "fill": "#2a9d4a",
         "points": [{"type": "move", "x": 0, "y": 10},
            {"type": "curve", "x": 100, "y": 10, "cp1": {"x": 30, "y": -8}, "cp2": {"x": 75, "y": -4}},
            {"type": "curve", "x": 0, "y": 10, "cp1": {"x": 75, "y": 24}, "cp2": {"x": 30, "y": 28}}]},
        {"id": "rib", "type": "path", "closed": false, "stroke": "#14532d", "strokeWidth": 1.2,
         "points": [{"type": "move", "x": 2, "y": 10}, {"type": "line", "x": 96, "y": 10}]},
    ])
}

fn ink() -> Value {
    json!([{"id": "ink", "type": "path", "closed": true, "fill": "#1b1b1b",
        "points": [{"type": "move", "x": 0, "y": 10},
            {"type": "curve", "x": 60, "y": 2, "cp1": {"x": 10, "y": 2}, "cp2": {"x": 30, "y": 0}},
            {"type": "curve", "x": 100, "y": 9, "cp1": {"x": 85, "y": 4}, "cp2": {"x": 98, "y": 5}},
            {"type": "curve", "x": 55, "y": 18, "cp1": {"x": 96, "y": 14}, "cp2": {"x": 80, "y": 17}},
            {"type": "curve", "x": 0, "y": 10, "cp1": {"x": 25, "y": 19}, "cp2": {"x": 5, "y": 16}}]}])
}

fn banner() -> Value {
    json!([
        {"id": "body", "type": "rectangle", "x": 12, "y": 6, "width": 76, "height": 8, "fill": "#3366cc"},
        {"id": "l", "type": "ellipse", "x": 0, "y": 0, "width": 20, "height": 20, "fill": "#d03a2f"},
        {"id": "r", "type": "path", "closed": true, "fill": "#d0b02f",
         "points": [{"type": "move", "x": 80, "y": 0}, {"type": "line", "x": 100, "y": 10},
                    {"type": "line", "x": 80, "y": 20}]},
    ])
}

fn stripes() -> Value {
    let colors = [
        "#000000", "#d03a2f", "#e98b2a", "#7a1f1a", "#999999", "#ffffff", "#2f6fd0",
    ];
    let mut v = Vec::new();
    for (i, c) in colors.iter().enumerate() {
        v.push(
            json!({"id": format!("s{i}"), "type": "rectangle", "x": i as f64 * 100.0 / 7.0, "y": 0,
            "width": 100.0 / 7.0, "height": 20, "fill": c}),
        );
    }
    v.push(
        json!({"id": "outline", "type": "rectangle", "x": 0, "y": 0, "width": 100, "height": 20,
        "stroke": "#000000", "strokeWidth": 1}),
    );
    Value::Array(v)
}

#[test]
#[ignore]
fn gallery_art() {
    // 1. Artwork along an S-curve.
    let mut s = Sheet::new(340, 340);
    s.add(
        &s_curve(),
        json!({"strokeWidth": 1.2}),
        &brush(leaf(), json!({})),
        20.0,
        25.0,
    );
    s.add(
        &s_curve(),
        json!({"strokeWidth": 0.9}),
        &brush(ink(), json!({})),
        20.0,
        130.0,
    );
    s.add(
        &s_curve(),
        json!({"strokeWidth": 0.8}),
        &brush(arrow_art(), json!({})),
        20.0,
        230.0,
    );
    s.save("art-stretch.png");

    // 2. Fits: stretch, proportional, guides (banner with protected ends).
    let short = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 300, "y": 0, "cp1": {"x": 80, "y": -40}, "cp2": {"x": 220, "y": 40}}]);
    let mut s = Sheet::new(340, 260);
    s.add(
        &short,
        json!({"strokeWidth": 1}),
        &brush(banner(), json!({"fit": "stretch"})),
        20.0,
        40.0,
    );
    s.add(
        &short,
        json!({"strokeWidth": 1}),
        &brush(banner(), json!({"fit": "proportional", "scale": 0.8})),
        20.0,
        125.0,
    );
    s.add(
        &short,
        json!({"strokeWidth": 1}),
        &brush(
            banner(),
            json!({"fit": "guides", "guides": {"start": 0.2, "end": 0.8}}),
        ),
        20.0,
        215.0,
    );
    s.save("art-fit.png");

    // 3. Direction and flips.
    let line = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 140, "y": 0}]);
    let mut s = Sheet::new(340, 170);
    for (i, extra) in [
        json!({}),
        json!({"direction": "left"}),
        json!({"flipAlong": true}),
        json!({"flipAcross": true}),
        json!({"direction": "up"}),
        json!({"direction": "down"}),
    ]
    .into_iter()
    .enumerate()
    {
        let art = if i >= 4 {
            // The banner turned upright, so up/down read like right/left.
            json!([{"id": "u", "type": "path", "closed": true, "fill": "#d03a2f",
                "points": [{"type": "move", "x": 0, "y": 100}, {"type": "line", "x": 10, "y": 0},
                           {"type": "line", "x": 20, "y": 100}]},
                   {"id": "d", "type": "ellipse", "x": 0, "y": 80, "width": 8, "height": 8, "fill": "#1b1b1b"}])
        } else {
            let mut a = arrow_art();
            a.as_array_mut().unwrap().push(json!({"id": "dot", "type": "ellipse", "x": 5, "y": 2, "width": 8, "height": 8, "fill": "#1b1b1b"}));
            a
        };
        let (col, row) = (i % 2, i / 2);
        s.add(
            &line,
            json!({"strokeWidth": 1.5}),
            &brush(art, extra),
            20.0 + col as f64 * 170.0,
            30.0 + row as f64 * 55.0,
        );
    }
    s.save("art-direction.png");

    // 4. Overlap at corners: none vs adjust.
    let zig = json!([{"type": "move", "x": 0, "y": 80}, {"type": "line", "x": 50, "y": 0},
        {"type": "line", "x": 100, "y": 80}, {"type": "line", "x": 150, "y": 10}]);
    // Translucent fill + outline, so folds read as darker overlaps and crossing outlines.
    let band = json!([{"id": "band", "type": "rectangle", "width": 100, "height": 20,
        "fill": "rgba(208, 58, 47, 0.45)", "stroke": "#1b1b1b", "strokeWidth": 1.5},
        {"id": "mid", "type": "path", "stroke": "#1b1b1b", "strokeWidth": 1,
         "points": [{"type": "move", "x": 0, "y": 5}, {"type": "line", "x": 100, "y": 5}]}]);
    let mut s = Sheet::new(360, 140);
    s.add(
        &zig,
        json!({"strokeWidth": 1.7}),
        &brush(band.clone(), json!({"overlap": "none"})),
        15.0,
        30.0,
    );
    s.add(
        &zig,
        json!({"strokeWidth": 1.7}),
        &brush(band, json!({"overlap": "adjust"})),
        195.0,
        30.0,
    );
    s.save("art-overlap.png");

    // 5. Colorize: none, tints, tints-shades, hue-shift (stroke #2a9d4a, key red).
    let small = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 140, "y": 0, "cp1": {"x": 40, "y": -20}, "cp2": {"x": 100, "y": 20}}]);
    let mut s = Sheet::new(340, 150);
    for (i, method) in ["none", "tints", "tints-shades", "hue-shift"]
        .iter()
        .enumerate()
    {
        let b = brush(
            stripes(),
            json!({"colorize": {"method": method, "keyColor": "#d03a2f"}}),
        );
        let (col, row) = (i % 2, i / 2);
        s.add(
            &small,
            json!({"strokeWidth": 1.1, "stroke": "#2a9d4a"}),
            &b,
            20.0 + col as f64 * 165.0,
            40.0 + row as f64 * 70.0,
        );
    }
    s.save("art-colorize.png");

    // 6. Width: pressure-driven scale, and the stroke's width profile (widthFrom).
    let mut s = Sheet::new(340, 210);
    s.add(&s_curve(), json!({"strokeWidth": 1.3, "inkChannels": {"t": [0, 0.3, 0.7, 1], "pressure": [0.1, 1, 0.6, 0]}}),
        &brush(ink(), json!({"scale": {"value": 1, "vary": "pressure", "min": 0.15, "max": 1.2}})), 20.0, 20.0);
    s.add(
        &s_curve(),
        json!({"strokeWidth": 1.3, "strokeProfile": {"preset": "eye"},
            "strokeBrush": {"brushId": "b", "overrides": {"widthFrom": "profile"}}}),
        &brush(leaf(), json!({})),
        20.0,
        105.0,
    );
    s.save("art-width.png");
}

fn flower() -> Value {
    let mut v = Vec::new();
    for i in 0..5 {
        let a = i as f64 / 5.0 * std::f64::consts::TAU;
        v.push(
            json!({"id": format!("p{i}"), "type": "ellipse", "x": 10.0 + 6.0 * a.cos() - 5.0,
            "y": 10.0 + 6.0 * a.sin() - 5.0, "width": 10, "height": 10, "fill": "#e98b2a"}),
        );
    }
    v.push(json!({"id": "c", "type": "ellipse", "x": 6, "y": 6, "width": 8, "height": 8, "fill": "#7a1f1a"}));
    Value::Array(v)
}

fn place_scatter(
    sheet: &mut Sheet,
    points: &Value,
    node_extra: Value,
    brush_def: &Value,
    dx: f64,
    dy: f64,
) {
    use super::scatter::{parse_scatter_brush, scatter_brush_for_node};
    let pts = translate(points, dx, dy);
    let node = line_node(pts.clone(), node_extra);
    let b = parse_scatter_brush(brush_def).unwrap();
    let placed = scatter_brush_for_node(node.as_object().unwrap(), &b, &default_geometry);
    sheet.canvas.draw(&placed.paths);
    let center: Vec<_> = pts
        .as_array()
        .unwrap()
        .iter()
        .filter_map(crate::blend::BlendPathPoint::from_json)
        .collect();
    let closed = node["closed"].as_bool().unwrap_or(false);
    sheet
        .canvas
        .hairline(&polygons(&center), closed, [150.0, 150.0, 150.0]);
}

#[test]
#[ignore]
fn gallery_scatter() {
    use super::tests_scatter::scatter;
    // 1. Leaves along an S-curve (relative to path), flowers with random size/scatter/rotation.
    let mut s = Sheet::new(340, 270);
    place_scatter(
        &mut s,
        &s_curve(),
        json!({"strokeWidth": 0.6}),
        &scatter(leaf(), json!({"spacing": 1.2})),
        20.0,
        20.0,
    );
    place_scatter(
        &mut s,
        &s_curve(),
        json!({"strokeWidth": 1, "strokeBrush": {"brushId": "s", "seed": 4}}),
        &scatter(
            flower(),
            json!({
            "size": {"value": 1, "vary": "random", "min": 0.5, "max": 1.3},
            "spacing": {"value": 1, "vary": "random", "min": 0.9, "max": 1.6},
            "scatter": {"value": 0, "vary": "random", "min": -1.2, "max": 1.2},
            "rotation": {"value": 0, "vary": "random", "min": 0, "max": 360}}),
        ),
        20.0,
        130.0,
    );
    s.save("scatter-along.png");

    // 2. Options: rotation relative to path vs page, spacing, pressure size, tints.
    let arc = json!([{"type": "move", "x": 0, "y": 30},
        {"type": "curve", "x": 140, "y": 30, "cp1": {"x": 30, "y": -20}, "cp2": {"x": 110, "y": -20}}]);
    let dart = json!([{"id": "a", "type": "path", "closed": true, "fill": "#d03a2f",
        "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 24, "y": 6},
                   {"type": "line", "x": 0, "y": 12}, {"type": "line", "x": 5, "y": 6}]}]);
    let mut s = Sheet::new(340, 190);
    place_scatter(
        &mut s,
        &arc,
        json!({"strokeWidth": 1}),
        &scatter(dart.clone(), json!({"spacing": 1.3})),
        20.0,
        25.0,
    );
    place_scatter(
        &mut s,
        &arc,
        json!({"strokeWidth": 1}),
        &scatter(dart, json!({"spacing": 1.3, "rotationRelativeTo": "page"})),
        180.0,
        25.0,
    );
    place_scatter(
        &mut s,
        &arc,
        json!({"strokeWidth": 1.2, "inkChannels": {"t": [0, 0.5, 1], "pressure": [0.1, 1, 0.1]}}),
        &scatter(
            flower(),
            json!({"size": {"value": 1, "vary": "pressure", "min": 0.2, "max": 1}, "spacing": 0.8}),
        ),
        20.0,
        115.0,
    );
    place_scatter(
        &mut s,
        &arc,
        json!({"strokeWidth": 0.9, "stroke": "#3366cc"}),
        &scatter(
            flower(),
            json!({"colorize": {"method": "hue-shift"}, "spacing": 1.1}),
        ),
        180.0,
        115.0,
    );
    s.save("scatter-options.png");
}

fn braid() -> Value {
    json!({"nodes": [
        {"id": "band", "type": "rectangle", "x": 0, "y": 0, "width": 20, "height": 10, "fill": "#e9d8a6"},
        {"id": "stripe", "type": "path", "closed": true, "fill": "#bb3e03",
         "points": [{"type": "move", "x": 0, "y": 10}, {"type": "line", "x": 8, "y": 0},
                    {"type": "line", "x": 14, "y": 0}, {"type": "line", "x": 6, "y": 10}]},
        {"id": "bead", "type": "ellipse", "x": 13, "y": 3, "width": 4, "height": 4, "fill": "#005f73"},
    ]})
}

fn rosette() -> Value {
    json!({"nodes": [
        {"id": "sq", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10, "fill": "#005f73"},
        {"id": "ro", "type": "ellipse", "x": 2, "y": 2, "width": 6, "height": 6, "fill": "#ee9b00"},
    ]})
}

fn cap(color: &str) -> Value {
    json!({"nodes": [{"id": format!("cap{color}"), "type": "path", "closed": true, "fill": color,
        "points": [{"type": "move", "x": 0, "y": 5}, {"type": "line", "x": 10, "y": 0},
                   {"type": "line", "x": 10, "y": 10}]}]})
}

fn place_pattern(
    sheet: &mut Sheet,
    points: &Value,
    node_extra: Value,
    brush_def: &Value,
    dx: f64,
    dy: f64,
) {
    place_pattern_windows(sheet, points, node_extra, brush_def, dx, dy, None);
}

fn place_pattern_windows(
    sheet: &mut Sheet,
    points: &Value,
    node_extra: Value,
    brush_def: &Value,
    dx: f64,
    dy: f64,
    windows: Option<&[super::art::ArcWindow]>,
) {
    use super::pattern::{parse_pattern_brush, pattern_brush_for_node_with, PatternOptions};
    let pts = translate(points, dx, dy);
    let node = line_node(pts.clone(), node_extra);
    let b = parse_pattern_brush(brush_def).unwrap();
    let placed = pattern_brush_for_node_with(
        node.as_object().unwrap(),
        &b,
        &default_geometry,
        &PatternOptions {
            tolerance: 0.1,
            windows,
        },
    );
    sheet.canvas.draw(&placed.paths);
    let center: Vec<_> = pts
        .as_array()
        .unwrap()
        .iter()
        .filter_map(crate::blend::BlendPathPoint::from_json)
        .collect();
    let closed = node["closed"].as_bool().unwrap_or(false);
    sheet
        .canvas
        .hairline(&polygons(&center), closed, [150.0, 150.0, 150.0]);
}

#[test]
#[ignore]
fn gallery_pattern() {
    use super::tests_pattern::pattern;
    // 1. Frames: a rectangle with rosette corners; a pentagon (sliced) and a zigzag with both kinds.
    let rect = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 120, "y": 0},
        {"type": "line", "x": 120, "y": 80}, {"type": "line", "x": 0, "y": 80}]);
    let mut pent = Vec::new();
    for i in 0..5 {
        let a = -std::f64::consts::FRAC_PI_2 + i as f64 * std::f64::consts::TAU / 5.0;
        pent.push(json!({"type": if i == 0 { "move" } else { "line" }, "x": 50.0 + 45.0 * a.cos(), "y": 48.0 + 45.0 * a.sin()}));
    }
    let mut s = Sheet::new(340, 130);
    place_pattern(
        &mut s,
        &rect,
        json!({"closed": true, "strokeWidth": 1.4}),
        &pattern(
            json!({"side": braid(), "outerCorner": rosette()}),
            json!({}),
        ),
        20.0,
        25.0,
    );
    place_pattern(
        &mut s,
        &Value::Array(pent),
        json!({"closed": true, "strokeWidth": 1.2}),
        &pattern(json!({"side": braid(), "outerCorner": "sliced"}), json!({})),
        200.0,
        10.0,
    );
    s.save("pattern-frame.png");

    // 2. Corners on a zigzag: centered, between, sliced, overlap (outer and inner alike).
    let zig = json!([{"type": "move", "x": 0, "y": 60}, {"type": "line", "x": 40, "y": 0},
        {"type": "line", "x": 80, "y": 60}, {"type": "line", "x": 120, "y": 0}]);
    let mut s = Sheet::new(340, 200);
    for (i, mode) in ["centered", "between", "sliced", "overlap"]
        .iter()
        .enumerate()
    {
        place_pattern(
            &mut s,
            &zig,
            json!({"strokeWidth": 1.4}),
            &pattern(
                json!({"side": braid(), "outerCorner": mode, "innerCorner": mode}),
                json!({}),
            ),
            25.0 + (i % 2) as f64 * 165.0,
            15.0 + (i / 2) as f64 * 95.0,
        );
    }
    s.save("pattern-corners.png");

    // 3. Fit on a curve with start/end tiles: stretch, add-space, approximate-path.
    let wave = json!([{"type": "move", "x": 0, "y": 20},
        {"type": "curve", "x": 290, "y": 20, "cp1": {"x": 90, "y": -25}, "cp2": {"x": 200, "y": 65}}]);
    let mut s = Sheet::new(340, 170);
    for (i, fit) in ["stretch", "add-space", "approximate-path"]
        .iter()
        .enumerate()
    {
        place_pattern(
            &mut s,
            &wave,
            json!({"strokeWidth": 1.4}),
            &pattern(
                json!({"side": braid(), "start": cap("#0a9396"), "end": cap("#9b2226")}),
                json!({"fit": fit, "spacing": 0.3}),
            ),
            25.0,
            15.0 + i as f64 * 50.0,
        );
    }
    s.save("pattern-fit.png");

    // 4. Trim (write-on): the rectangle frame revealed to 20 / 45 / 70 / 100 % from mid-top.
    // Side tiles are cut at the window end; corner tiles show whole once their joint is reached.
    let frame = json!([{"type": "move", "x": 60, "y": 0}, {"type": "line", "x": 120, "y": 0},
        {"type": "line", "x": 120, "y": 80}, {"type": "line", "x": 0, "y": 80},
        {"type": "line", "x": 0, "y": 0}]);
    let len = 400.0;
    let mut s = Sheet::new(340, 250);
    for (i, t) in [0.2, 0.45, 0.7, 1.0].iter().enumerate() {
        let w = [super::art::ArcWindow {
            subpath: 0,
            start: 0.0,
            end: t * len,
            taper_extent: None,
        }];
        place_pattern_windows(
            &mut s,
            &frame,
            json!({"closed": true, "strokeWidth": 1.4}),
            &pattern(
                json!({"side": braid(), "outerCorner": rosette()}),
                json!({}),
            ),
            25.0 + (i % 2) as f64 * 165.0,
            20.0 + (i / 2) as f64 * 115.0,
            Some(&w),
        );
    }
    s.save("pattern-trim.png");
}
