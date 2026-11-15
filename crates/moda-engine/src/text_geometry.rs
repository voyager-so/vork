//! Read-only richtext → vector geometry, for exporters that write vectors (SVG export, F8.1).
//!
//! Runs the SAME vector layout and draw-op builder the renderer draws with
//! (`text_draw::node_to_text_spec` → `moda_text::vector::build_vector_node_ops`, `VectorParity`
//! placement) and turns its ops into path data in node-local px:
//!
//! - analytic glyph ops → the placed glyphs' quadratic outlines (the curve buffer the GPU
//!   coverage shader reads, placed with the shader's own `pen + lin·(em·size − C) + C + off`
//!   math, synthetic oblique included), one path per op, nonzero;
//! - tess ops (list bullets, underline / strike-through rects) → their
//!   triangles as one nonzero path per op (one path, so no anti-aliasing seams).
//!
//! Outlined text therefore lands on the pixels the canvas draws. The runs (text, family,
//! weight, style, size, baseline, colour, per-glyph pen x) come from the text-layer extractor
//! over the same layout, for exporters that keep text live.
//!
//! Declines (returns `Err(reason)`) when the node would not draw through the vector lane, or
//! when its ops carry passes a flat fill can't express (shadow / glow blur, text stroke,
//! additive glow, gradient or shader fills) — the exporter rasterizes those nodes instead.

use moda_scene::Scene;
use moda_text::node_layout::RichTextNodeSpec;
use moda_text::vector::extract::{extract_vector_text_runs, ExtractedRun};
use moda_text::vector::{VectorEngineDecision, VectorOpPipeline, VectorPlacement};

/// One filled path of a text node, node-local px.
#[derive(Debug, Clone)]
pub struct TextGeometryFill {
    /// "glyphs" (outlined glyphs) or "shape" (bullets, decorations); `node_fill_outlines` also writes "effect" (an
    /// echo copy or extrude wall, in its own colour).
    pub kind: &'static str,
    /// Path data: one entry per glyph for "glyphs" (a line of glyphs per fill), one for "shape".
    pub paths: Vec<String>,
    /// Straight RGBA 0..1.
    pub color: [f32; 4],
    pub opacity: f32,
    /// A text range's gradient (`--nf-text-fill`) painting these glyphs instead of `color`, as a Moda paint over
    /// the node box (JSON: `{ type: 'linear' | 'radial', … }`).
    pub gradient: Option<moda_scene::value::Value>,
}

#[derive(Debug, Clone)]
pub struct TextGeometry {
    pub node_id: String,
    pub fills: Vec<TextGeometryFill>,
    /// `None` when the extractor could not map the node's glyphs back to text.
    pub runs: Option<Vec<ExtractedRun>>,
}

fn fmt(v: f32) -> String {
    let r = (v as f64 * 1000.0).round() / 1000.0;
    if r == 0.0 {
        "0".to_string()
    } else {
        format!("{r}")
    }
}

/// Triangles `[x0,y0,x1,y1,x2,y2, …]` → path data.
fn triangles_to_d(tris: &[f32]) -> String {
    let mut d = String::with_capacity(tris.len() * 6);
    for t in tris.chunks_exact(6) {
        d.push_str(&format!(
            "M{} {}L{} {}L{} {}Z",
            fmt(t[0]),
            fmt(t[1]),
            fmt(t[2]),
            fmt(t[3]),
            fmt(t[4]),
            fmt(t[5])
        ));
    }
    d
}

/// Compact SVG path data for glyph outlines, at the same 0.001 px precision as [`fmt`]: the first move absolute,
/// every later point relative to the one before, a command letter only where it changes (`h` / `v` for axis-aligned
/// lines), no leading zero and no separator before a minus sign. Coordinates are rounded to thousandths first and the
/// deltas taken between the rounded values, so the points read back are exactly the rounded absolute ones (no error
/// accumulates along a path). A text-heavy page's outlines take about 60% of the characters absolute data took.
/// [`crate::boolean_operands::parse_glyph_d`] reads it back.
struct GlyphPathData {
    d: String,
    /// Current point and subpath start, in thousandths of a px.
    at: (i64, i64),
    start: (i64, i64),
    /// The last command letter written (0: none yet).
    cmd: u8,
}

impl GlyphPathData {
    fn new() -> Self {
        Self { d: String::new(), at: (0, 0), start: (0, 0), cmd: 0 }
    }

    fn milli(v: f32) -> i64 {
        (v as f64 * 1000.0).round() as i64
    }

    /// `cmd` unless it is the last one written (then its arguments continue it).
    fn command(&mut self, cmd: u8) {
        if self.cmd != cmd {
            self.d.push(cmd as char);
        }
        self.cmd = cmd;
    }

    /// One number in thousandths, after a separator when it follows another number and doesn't start with `-`.
    fn num(&mut self, k: i64, first: bool) {
        use std::fmt::Write as _;
        if k < 0 {
            self.d.push('-');
        } else if !first {
            self.d.push(' ');
        }
        let a = k.unsigned_abs();
        let (int, frac) = (a / 1000, a % 1000);
        if frac == 0 {
            let _ = write!(self.d, "{int}");
            return;
        }
        if int > 0 {
            let _ = write!(self.d, "{int}");
        }
        let (mut frac, mut digits) = (frac, 3);
        while frac % 10 == 0 {
            frac /= 10;
            digits -= 1;
        }
        let _ = write!(self.d, ".{frac:0digits$}");
    }

    /// The arguments of the command just written: the first needs a separator only when the command letter was
    /// left out (it continues the previous command's numbers).
    fn args(&mut self, continued: bool, nums: &[i64]) {
        for (i, &k) in nums.iter().enumerate() {
            self.num(k, i == 0 && !continued);
        }
    }

    fn move_to(&mut self, x: f32, y: f32) {
        let p = (Self::milli(x), Self::milli(y));
        if self.d.is_empty() {
            // Absolute, and a space even before a minus: readers of a glyph's first point split on letters and spaces.
            self.d.push('M');
            self.num(p.0, true);
            self.d.push(' ');
            self.num(p.1, true);
        } else {
            self.d.push('m');
            self.args(false, &[p.0 - self.at.0, p.1 - self.at.1]);
        }
        // After a move, the next command always writes its letter (implicit arguments of `m` are lines).
        self.cmd = b'm';
        self.at = p;
        self.start = p;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = (Self::milli(x), Self::milli(y));
        let (dx, dy) = (p.0 - self.at.0, p.1 - self.at.1);
        let (cmd, nums): (u8, &[i64]) = match (dx, dy) {
            (0, 0) => (b'l', &[0, 0]),
            (_, 0) => (b'h', &[dx]),
            (0, _) => (b'v', &[dy]),
            _ => (b'l', &[dx, dy]),
        };
        let continued = self.cmd == cmd;
        self.command(cmd);
        self.args(continued, nums);
        self.at = p;
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (c, p) = ((Self::milli(cx), Self::milli(cy)), (Self::milli(x), Self::milli(y)));
        let continued = self.cmd == b'q';
        self.command(b'q');
        self.args(continued, &[c.0 - self.at.0, c.1 - self.at.1, p.0 - self.at.0, p.1 - self.at.1]);
        self.at = p;
    }

    fn close(&mut self) {
        self.d.push('Z');
        self.cmd = b'Z';
        self.at = self.start;
    }
}

/// Glyph records of an analytic op → path data (quadratics, contiguous segments joined; [`GlyphPathData`]).
fn glyphs_to_d(records: &[moda_text::vector::ops::VectorAnalyticRecord], curves: &[f32]) -> String {
    let mut d = GlyphPathData::new();
    for r in records {
        let size = r.font_size_px;
        let cx = (r.em_box[0] + r.em_box[2]) * 0.5 * size;
        let cy = (r.em_box[1] + r.em_box[3]) * 0.5 * size;
        let [a, c, b, dd] = r.glyph_lin;
        // Vertex shader placement: pen + lin·(em·size − C) + C + off.
        let place = |ex: f32, ey: f32| -> (f32, f32) {
            let (px, py) = (ex * size - cx, ey * size - cy);
            (
                r.pen_x + a * px + c * py + cx + r.glyph_off[0],
                r.baseline_y + b * px + dd * py + cy + r.glyph_off[1],
            )
        };
        let start = r.first_curve as usize * 6;
        let end = start + r.curve_count as usize * 6;
        let Some(segs) = curves.get(start..end) else {
            continue;
        };
        let mut open: Option<((f32, f32), (f32, f32))> = None; // (subpath start, current point)
        let close_eps = 1e-3;
        for s in segs.chunks_exact(6) {
            let p0 = place(s[0], s[1]);
            let q = place(s[2], s[3]);
            let p1 = place(s[4], s[5]);
            let continues = matches!(open, Some((_, cur)) if (cur.0 - p0.0).abs() < close_eps && (cur.1 - p0.1).abs() < close_eps);
            if !continues {
                if let Some((st, cur)) = open {
                    if (st.0 - cur.0).abs() < close_eps && (st.1 - cur.1).abs() < close_eps {
                        d.close();
                    }
                }
                d.move_to(p0.0, p0.1);
                open = Some((p0, p0));
            }
            // Lines are stored as quadratics with the control point at the midpoint.
            let mid = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
            if (mid.0 - q.0).abs() < 1e-3 && (mid.1 - q.1).abs() < 1e-3 {
                d.line_to(p1.0, p1.1);
            } else {
                d.quad_to(q.0, q.1, p1.0, p1.1);
            }
            if let Some((st, _)) = open {
                open = Some((st, p1));
            }
        }
        if let Some((st, cur)) = open {
            if (st.0 - cur.0).abs() < close_eps && (st.1 - cur.1).abs() < close_eps {
                d.close();
            }
        }
    }
    d.d
}

/// The family name a font file calls itself (name table: typographic family, else family; Windows or Mac
/// platform). The catalog can map a document family to a file named otherwise (the bundled "Inter" is the
/// static "Inter 24pt" instance), so live `<text>` lists both.
pub fn face_family_name(data: &[u8]) -> Option<String> {
    let u16_at =
        |o: usize| -> Option<u16> { Some(u16::from_be_bytes([*data.get(o)?, *data.get(o + 1)?])) };
    let u32_at = |o: usize| -> Option<u32> {
        Some(u32::from_be_bytes(data.get(o..o + 4)?.try_into().ok()?))
    };
    let tables = u16_at(4)? as usize;
    let name = (0..tables).find_map(|i| {
        let rec = 12 + 16 * i;
        (data.get(rec..rec + 4)? == b"name")
            .then(|| u32_at(rec + 8))
            .flatten()
    })? as usize;
    let count = u16_at(name + 2)? as usize;
    let strings = name + u16_at(name + 4)? as usize;
    let mut best: Option<(u8, String)> = None;
    for j in 0..count {
        let rec = name + 6 + 12 * j;
        let (platform, name_id) = (u16_at(rec)?, u16_at(rec + 6)?);
        let rank = match name_id {
            16 => 2,
            1 => 1,
            _ => continue,
        };
        let (len, off) = (u16_at(rec + 8)? as usize, u16_at(rec + 10)? as usize);
        let raw = data.get(strings + off..strings + off + len)?;
        let text = match platform {
            0 | 3 => String::from_utf16_lossy(
                &raw.chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect::<Vec<_>>(),
            ),
            1 => raw.iter().map(|&b| b as char).collect(),
            _ => continue,
        };
        if !text.is_empty() && best.as_ref().is_none_or(|(r, _)| rank > *r) {
            best = Some((rank, text));
        }
    }
    best.map(|(_, t)| t)
}

/// A text gradient's shader uniforms as the Moda paint they came from (node 0-100 box space; stop colours `rgba()`):
/// `{ type: 'linear', start, end, stops }` or `{ type: 'radial', center, radiusX, radiusY, rotation, stops }`.
fn gradient_paint_json(g: &moda_text::ops::TextGradientUniforms) -> moda_scene::value::Value {
    use moda_scene::value::json;
    let count = (g.config[0].max(0.0) as usize).min(g.stops.len());
    let stops: Vec<moda_scene::value::Value> = g.stops[..count]
        .iter()
        .map(|&[offset, r, gr, packed]| {
            // `floor(b·255) + a·0.99999` (packTextGradientUniformFloats).
            let b = packed.floor();
            let a = ((packed - b) / 0.99999).clamp(0.0, 1.0);
            let a = (a * 1000.0).round() / 1000.0;
            let c = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
            json!({
                "offset": offset,
                "color": format!("rgba({}, {}, {}, {})", c(r), c(gr), b.clamp(0.0, 255.0) as u8, a),
            })
        })
        .collect();
    let [p0, p1, p2, p3] = g.params;
    if g.config[1] == 2.0 {
        json!({
            "type": "radial",
            "center": { "x": p0, "y": p1 },
            "radiusX": p2,
            "radiusY": p3,
            "rotation": g.config[2].to_degrees(),
            "stops": stops,
        })
    } else {
        json!({ "type": "linear", "start": { "x": p0, "y": p1 }, "end": { "x": p2, "y": p3 }, "stops": stops })
    }
}

/// The vector geometry of one richtext node, or why it can't be expressed as flat fills.
pub fn node_text_geometry(scene: &Scene, node_id: &str) -> Result<TextGeometry, String> {
    let node = scene
        .effective_node(node_id)
        .ok_or_else(|| format!("no node {node_id}"))?;
    if node.node_type() != "richtext" {
        return Err(format!("{node_id} is a {}, not richtext", node.node_type()));
    }
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    crate::text_area::apply_to_spec(scene, node, &mut spec);
    if spec.gradient.is_some() {
        return Err("gradient text".into());
    }
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    let ops = match moda_text::vector::build_vector_node_ops(
        &spec,
        &store,
        &host.catalog,
        1.0,
        VectorPlacement::VectorParity,
    ) {
        VectorEngineDecision::Vector(ops) => ops,
        VectorEngineDecision::Fallback { reason, .. } => {
            return Err(format!("not vector text: {reason}"))
        }
    };
    if !ops.emojis.is_empty() {
        // Colour emoji are images, not outlines: the exporter rasterizes the node.
        return Err("colour emoji".into());
    }
    let fills = fills_from_ops(&ops, FillMode::Strict)?;
    // Type on a path (F7.2): the runs' straight positions don't say where the glyphs are; exporters that keep text
    // live write the path's own text instead (SVG `<textPath>`), or the outlines.
    let runs = if spec.text_path.is_some() {
        None
    } else {
        extract_vector_text_runs(
            &RichTextNodeSpec {
                node_id: node.id().to_string(),
                html_content: spec.html.to_string(),
                width: spec.width,
                height: spec.height,
                paragraph_spacing: spec.paragraph_spacing,
                line_height_policy: spec.line_height_policy,
                vertical_align: spec.vertical_align.clone(),
                truncation: spec.truncation,
                auto_width: spec.auto_width,
                area: spec.area.clone(),
            },
            &store,
            &host.catalog,
        )
    };
    Ok(TextGeometry {
        node_id: node_id.to_string(),
        fills,
        runs,
    })
}

/// One outlined glyph of a text node at rest, for per-glyph motion (`text.units`, vector motion export): its
/// code-point unit (the `unitTransforms` index), its path, its fill, the point a unit transform turns and scales it
/// about (the renderer's `pen + C`), and its outline's box at rest (`[x0, y0, x1, y1]`, node-local px): a unit of
/// several glyphs (a `unitTransforms` entry's `unit`) turns and scales about the centre of its glyphs' boxes instead.
#[derive(Debug, Clone)]
pub struct UnitGlyph {
    pub unit: usize,
    pub d: String,
    pub cx: f32,
    pub cy: f32,
    pub bounds: [f32; 4],
    pub color: [f32; 4],
    pub opacity: f32,
}

/// A text node's glyphs by unit, and what isn't a glyph (bullets, decorations: they stay put).
#[derive(Debug, Clone)]
pub struct TextUnits {
    pub count: usize,
    pub glyphs: Vec<UnitGlyph>,
    pub shapes: Vec<TextGeometryFill>,
}

/// The node's glyphs at rest with the unit each belongs to. The vector lane maps a glyph to its code point through
/// the shaper's clusters; this reads that mapping back by drawing with a `unitTransforms` whose only content is an
/// opacity that encodes each unit's index (geometry untouched), then decoding it from each glyph's record.
pub fn node_text_units(scene: &Scene, node_id: &str) -> Result<TextUnits, String> {
    use moda_text::vector::ops::{GlyphXform, UNIT_TRANSFORMS_MAX_LENGTH};
    let node = scene
        .effective_node(node_id)
        .ok_or_else(|| format!("no node {node_id}"))?;
    if node.node_type() != "richtext" {
        return Err(format!("{node_id} is a {}, not richtext", node.node_type()));
    }
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    crate::text_area::apply_to_spec(scene, node, &mut spec);
    if spec.gradient.is_some() {
        return Err("gradient text".into());
    }
    if spec.text_path.is_some() {
        return Err("type on a path".into());
    }
    let count = moda_text::parser::source_code_point_count(&spec.html);
    if count == 0 || count > UNIT_TRANSFORMS_MAX_LENGTH {
        return Err(format!("{count} units"));
    }
    let code = |i: usize| (i + 1) as f64 / (count + 2) as f64;
    spec.unit_transforms = ((0..count)
        .map(|i| GlyphXform::resolve(0.0, 0.0, 1.0, 0.0, code(i)))
        .collect::<Vec<_>>()).into();
    // The opacity carries the code; a unit's fill amount would scale it.
    spec.unit_fill = (Vec::new()).into();
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    let ops = match moda_text::vector::build_vector_node_ops(
        &spec,
        &store,
        &host.catalog,
        1.0,
        VectorPlacement::VectorParity,
    ) {
        VectorEngineDecision::Vector(ops) => ops,
        VectorEngineDecision::Fallback { reason, .. } => {
            return Err(format!("not vector text: {reason}"))
        }
    };
    if !ops.emojis.is_empty() {
        return Err("colour emoji".into());
    }
    let mut glyphs = Vec::new();
    let mut shapes = Vec::new();
    for op in &ops.ops {
        if op.blur_px > 0.0 || op.stroke_px > 0.0 || op.additive {
            return Err("text effect pass".into());
        }
        match op.pipeline {
            VectorOpPipeline::Tess => {
                if !op.tris.is_empty() {
                    shapes.push(TextGeometryFill {
                        kind: "shape",
                        paths: vec![triangles_to_d(&op.tris)],
                        color: op.color,
                        opacity: op.opacity,
                        gradient: None,
                    });
                }
            }
            VectorOpPipeline::Analytic => {
                for r in &op.records {
                    let unit = ((f64::from(r.glyph_opacity) * (count + 2) as f64).round() as usize)
                        .saturating_sub(1);
                    let mut rest = *r;
                    rest.glyph_opacity = 1.0;
                    let d = glyphs_to_d(std::slice::from_ref(&rest), &ops.curve_points);
                    if d.is_empty() {
                        continue;
                    }
                    let size = r.font_size_px;
                    let start = r.first_curve as usize * 6;
                    let end = start + r.curve_count as usize * 6;
                    let mut b = [
                        f32::INFINITY,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                    ];
                    for p in ops
                        .curve_points
                        .get(start..end)
                        .unwrap_or(&[])
                        .chunks_exact(2)
                    {
                        b = [
                            b[0].min(p[0]),
                            b[1].min(p[1]),
                            b[2].max(p[0]),
                            b[3].max(p[1]),
                        ];
                    }
                    glyphs.push(UnitGlyph {
                        unit: unit.min(count - 1),
                        bounds: [
                            r.pen_x + b[0] * size,
                            r.baseline_y + b[1] * size,
                            r.pen_x + b[2] * size,
                            r.baseline_y + b[3] * size,
                        ],
                        d,
                        cx: r.pen_x + (r.em_box[0] + r.em_box[2]) * 0.5 * size,
                        cy: r.baseline_y + (r.em_box[1] + r.em_box[3]) * 0.5 * size,
                        color: op.color,
                        opacity: op.opacity,
                    });
                }
            }
            _ => return Err("gradient or shader text fill".into()),
        }
    }
    Ok(TextUnits {
        count,
        glyphs,
        shapes,
    })
}

/// How [`fills_from_ops`] treats what flat fills can't express.
#[derive(Clone, Copy, PartialEq)]
enum FillMode {
    /// Decline effect passes, shader fills and per-glyph opacity.
    Strict,
    /// Leave the effect passes out and draw the rest as flat fills, ignoring per-glyph opacity.
    Lenient,
    /// As `Lenient`, but a glyph's own opacity (a `text.units` unit's) splits the line's fill and scales its
    /// opacity; a glyph at opacity 0 is left out.
    GlyphOpacity,
}

/// The ops' fills as path data, per [`FillMode`].
fn fills_from_ops(
    ops: &moda_text::vector::ops::VectorNodeOps,
    mode: FillMode,
) -> Result<Vec<TextGeometryFill>, String> {
    let strict = mode == FillMode::Strict;
    let mut fills = Vec::new();
    for op in &ops.ops {
        if op.blur_px > 0.0 || op.stroke_px > 0.0 || op.additive {
            if strict {
                return Err("text effect pass".into());
            }
            continue;
        }
        match op.pipeline {
            VectorOpPipeline::Tess => {
                if op.tris.is_empty() {
                    continue;
                }
                fills.push(TextGeometryFill {
                    kind: "shape",
                    paths: vec![triangles_to_d(&op.tris)],
                    color: op.color,
                    opacity: op.opacity,
                    gradient: None,
                });
            }
            VectorOpPipeline::AnalyticShaderFill if strict => {
                return Err("shader text fill".into());
            }
            VectorOpPipeline::Analytic
            | VectorOpPipeline::AnalyticGradient
            | VectorOpPipeline::AnalyticShaderFill => {
                // A gradient op here is a text range's own gradient (a node gradient declined above).
                let gradient = match (op.pipeline, &op.gradient) {
                    (VectorOpPipeline::AnalyticGradient, Some(g)) => Some(gradient_paint_json(g)),
                    (VectorOpPipeline::AnalyticGradient, None) => {
                        return Err("gradient text fill".into())
                    }
                    _ => None,
                };
                if strict && op.records.iter().any(|r| r.glyph_opacity < 1.0) {
                    return Err("per-glyph opacity".into());
                }
                // A path per glyph, a fill per line (like Illustrator's Create Outlines): one compound path for a
                // whole paragraph is heavy to edit and to re-import.
                let by_opacity = mode == FillMode::GlyphOpacity;
                for line in op.records.chunk_by(|a, b| {
                    a.baseline_y == b.baseline_y && (!by_opacity || a.glyph_opacity == b.glyph_opacity)
                }) {
                    let glyph_opacity = if by_opacity { line[0].glyph_opacity.clamp(0.0, 1.0) } else { 1.0 };
                    if glyph_opacity <= 0.0 {
                        continue;
                    }
                    let paths: Vec<String> = line
                        .iter()
                        .map(|r| glyphs_to_d(std::slice::from_ref(r), &ops.curve_points))
                        .filter(|d| !d.is_empty())
                        .collect();
                    if paths.is_empty() {
                        continue;
                    }
                    fills.push(TextGeometryFill {
                        kind: "glyphs",
                        paths,
                        color: op.color,
                        opacity: op.opacity * glyph_opacity,
                        gradient: gradient.clone(),
                    });
                }
            }
        }
    }
    Ok(fills)
}

/// The glyph fills a solid sweeps (`solid::outline`, `shape: 'extrude'`): node-local px, each glyph where its
/// `text.units` transform puts it, a fill per line and glyph opacity (a unit fading in extrudes at its opacity; one
/// at opacity 0 is left out). Colour emoji (images) and the effect passes (shadows, glows, strokes) have no outline
/// to sweep and are left out, so the rest of the text still extrudes; a gradient fill declines.
pub fn node_solid_fills(scene: &Scene, node_id: &str) -> Result<Vec<TextGeometryFill>, String> {
    let node = scene
        .effective_node(node_id)
        .ok_or_else(|| format!("no node {node_id}"))?;
    if node.node_type() != "richtext" {
        return Err(format!("{node_id} is a {}, not richtext", node.node_type()));
    }
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    crate::text_area::apply_to_spec(scene, node, &mut spec);
    if spec.gradient.is_some() {
        return Err("gradient text".into());
    }
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    match moda_text::vector::build_vector_node_ops(&spec, &store, &host.catalog, 1.0, VectorPlacement::VectorParity) {
        VectorEngineDecision::Vector(ops) => fills_from_ops(&ops, FillMode::GlyphOpacity),
        VectorEngineDecision::Fallback { reason, .. } => Err(format!("not vector text: {reason}")),
    }
}

/// The glyph fills of rich text `node` for a live geometry effect (`moda_scene::text_warp`): node-local px, laid
/// out from `node` itself (the scene's record for its id may already be the warped path). The node's opacity,
/// strokes, drop shadows and glows are left out (the path it draws as applies them); echo copies and extrude walls
/// are outlined too, behind the glyphs. A `textFills` gradient is left to the caller (glyphs come in their run
/// colours).
pub fn node_fill_outlines(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
) -> Result<Vec<TextGeometryFill>, String> {
    if node.node_type() != "richtext" {
        return Err(format!("{} is a {}, not richtext", node.id(), node.node_type()));
    }
    let mut spec = crate::text_draw::node_to_text_spec_in(scene, node);
    // Outside a document (`text_warp::resolve_standalone`) the caller hands the area input in.
    match node.json_field(moda_scene::text_warp::TEXT_AREA_KEY) {
        Some(area) => spec.area = moda_text::area::AreaSpec::from_json(&area.to_json()),
        None => crate::text_area::apply_to_spec(scene, node, &mut spec),
    }
    spec.gradient = None;
    spec.opacity = 1.0;
    spec.stroke_color = None;
    spec.stroke_width = None;
    // Echo and extrude are more glyph geometry (offset copies, swept walls): they are outlined with the glyphs and
    // bend with them. Drop shadows and glows are drawn by the path's own effect lanes.
    spec.effects.retain(|e| {
        matches!(
            e,
            moda_text::ops::TextEffectSpec::Echo { .. } | moda_text::ops::TextEffectSpec::Extrude { .. }
        )
    });
    spec.distort = None;
    let (store, host) = crate::text_draw::vector_store_and_host().map_err(|e| format!("{e:#}"))?;
    let build = |spec: &moda_text::TextNodeSpec| -> Result<Vec<TextGeometryFill>, String> {
        match moda_text::vector::build_vector_node_ops(
            spec,
            &store,
            &host.catalog,
            1.0,
            VectorPlacement::VectorParity,
        ) {
            VectorEngineDecision::Vector(ops) => fills_from_ops(&ops, FillMode::Lenient),
            VectorEngineDecision::Fallback { reason, .. } => Err(format!("not vector text: {reason}")),
        }
    };
    let mut fills = build(&spec)?;
    if !spec.effects.is_empty() {
        // The effect passes' fills come first (back to front); they keep their own colours whatever paints the
        // glyphs, so mark them. Their count is what the plain layout lacks.
        let mut plain = spec.clone();
        plain.effects.clear();
        let effect_fills = fills.len().saturating_sub(build(&plain)?.len());
        for f in fills.iter_mut().take(effect_fills) {
            f.kind = "effect";
        }
    }
    Ok(fills)
}

/// JSON for `moda-render text-geometry`: `{ "<id>": { fills, runs } | { error } }`.
pub fn text_geometry_json(scene: &Scene, node_ids: &[String]) -> moda_scene::value::Value {
    text_geometry_json_with(scene, node_ids, false)
}

/// [`text_geometry_json`], with each node's `units` ([`node_text_units`]) when `units` is set: `{ count, glyphs: [{
/// unit, d, cx, cy, color, opacity }], shapes: [{ d, color, opacity }] }`, or `{ error }`.
pub fn text_geometry_json_with(scene: &Scene, node_ids: &[String], units: bool) -> moda_scene::value::Value {
    use moda_scene::value::{json, Map, Value};
    let mut out = Map::new();
    for id in node_ids {
        let value = match node_text_geometry(scene, id) {
            Ok(g) => {
                let fills: Vec<Value> = g
                    .fills
                    .iter()
                    .map(|f| {
                        let mut v = json!({ "kind": f.kind, "paths": f.paths, "color": f.color, "opacity": f.opacity });
                        if let Some(g) = &f.gradient {
                            v["gradient"] = g.clone();
                        }
                        v
                    })
                    .collect();
                let runs = g.runs.map(|runs| {
                    runs.iter()
                        .map(|r| {
                            let mut run = json!({
                                "text": r.text,
                                "family": r.family,
                                "weight": r.weight,
                                "style": r.style,
                                "fontSize": r.font_size_px,
                                "baselineY": r.baseline_y_px,
                                "endX": r.end_x_px,
                                "letterSpacing": (r.letter_spacing_px != 0.0).then_some(r.letter_spacing_px),
                                "color": r.color,
                                // not-a-node-key: a text-geometry run's gradient paint (text range gradients, #629)
                                "fill": r.fill.as_ref().map(|g| gradient_paint_json(
                                    &moda_text::run_fill::gradient_uniforms(g, 0.0, 0.0),
                                )),
                                "fontFeatures": (!r.font_features.is_empty()).then_some(&r.font_features),
                                "fontVariations": (!r.font_variations.is_empty()).then_some(&r.font_variations),
                                "faceFamily": crate::text_extract::vector_font_bytes(&r.family, r.weight, &r.style)
                                    .and_then(|b| face_family_name(&b)),
                                "glyphs": r.glyphs.iter().map(|g| {
                                    let mut v = json!({
                                        "x": g.x_px,
                                        "start": g.text_range.start,
                                        "end": g.text_range.end,
                                    });
                                    // E8.22: a shifted or turned glyph says where live text writes it: `y` is its
                                    // baseline, `ox` its origin once turned about its centre (SVG `rotate` turns
                                    // about the origin).
                                    let (ox, oy) = g.rotation_origin_px;
                                    if r.character.baseline_shift != 0.0 || r.character.rotation != 0.0 {
                                        v["y"] = json!(r.baseline_y_px - g.y_offset_px + oy);
                                    }
                                    if r.character.rotation != 0.0 {
                                        v["ox"] = json!(g.x_px + ox);
                                    }
                                    v
                                }).collect::<Vec<_>>(),
                            });
                            // E8.22: the run's character attributes, the non-default ones (`moda_text::character`).
                            let ch = &r.character;
                            if ch.baseline_shift != 0.0 {
                                run["baselineShift"] = json!(ch.baseline_shift);
                            }
                            if ch.horizontal_scale != 1.0 {
                                run["horizontalScale"] = json!(ch.horizontal_scale);
                            }
                            if ch.vertical_scale != 1.0 {
                                run["verticalScale"] = json!(ch.vertical_scale);
                            }
                            if ch.rotation != 0.0 {
                                run["rotation"] = json!(ch.rotation);
                            }
                            if ch.kerning != moda_text::character::Kerning::Metrics {
                                run["kerning"] = json!(ch.kerning.css());
                            }
                            run
                        })
                        .collect::<Vec<_>>()
                });
                let mut v = json!({ "nodeId": g.node_id, "fills": fills, "runs": runs });
                if units {
                    v["units"] = match node_text_units(scene, id) {
                        Ok(u) => json!({
                            "count": u.count,
                            "glyphs": u.glyphs.iter().map(|g| json!({
                                "unit": g.unit, "d": g.d, "cx": g.cx, "cy": g.cy, "bounds": g.bounds, "color": g.color, "opacity": g.opacity,
                            })).collect::<Vec<_>>(),
                            "shapes": u.shapes.iter().map(|f| json!({
                                "d": f.paths.first(), "color": f.color, "opacity": f.opacity,
                            })).collect::<Vec<_>>(),
                        }),
                        Err(reason) => json!({ "error": reason }),
                    };
                }
                v
            }
            Err(reason) => json!({ "error": reason }),
        };
        out.insert(id.clone().into(), value);
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn scene(nodes: moda_scene::value::Value, ids: Vec<&str>) -> Scene {
        Scene::load(&json!({
            "nodes": nodes,
            "pages": [{"id":"p1","name":"P","width":800.0,"height":600.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],"nodeIds": ids}]
        })).unwrap()
    }

    #[test]
    fn units_map_glyphs_to_code_points_at_rest() {
        let s = scene(
            json!([{"id":"t1","type":"richtext","x":40.0,"y":30.0,"width":300.0,"height":100.0,
                "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px; color: #112233;\">Hi yo</span></p>"}]),
            vec!["t1"],
        );
        let u = node_text_units(&s, "t1").expect("vector text");
        assert_eq!(u.count, 5);
        let units: Vec<usize> = u.glyphs.iter().map(|g| g.unit).collect();
        assert_eq!(units, vec![0, 1, 3, 4], "the space has no outline");
        assert!(u.glyphs.windows(2).all(|w| w[1].cx > w[0].cx));
        // Each glyph's outline box at rest holds its centre (the pivot of a unit of its own); a word turns about its
        // glyphs' boxes together.
        for g in &u.glyphs {
            let [x0, y0, x1, y1] = g.bounds;
            assert!(
                x0 < x1
                    && y0 < y1
                    && (x0..=x1).contains(&g.cx)
                    && (y0 - 2.0..=y1 + 2.0).contains(&g.cy),
                "{g:?}"
            );
        }
        // At rest: the same outlines the plain geometry draws.
        let g = node_text_geometry(&s, "t1").unwrap();
        let plain: Vec<&String> = g.fills.iter().flat_map(|f| f.paths.iter()).collect();
        assert_eq!(plain, u.glyphs.iter().map(|g| &g.d).collect::<Vec<_>>());
    }

    /// A solid (extrusion) sweeps what a unit shows: a unit fading in extrudes at its opacity, a hidden one is left
    /// out, and emoji (images, no outline) are skipped while the rest of the text still extrudes. Before, any unit
    /// under opacity 1 or any emoji made the whole layer's extrusion draw nothing.
    #[test]
    fn solid_fills_follow_unit_opacity_and_skip_emoji() {
        let text = |html: &str, units: moda_scene::value::Value| {
            scene(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":100.0,
                    "htmlContent": html, "unitTransforms": units}]),
                vec!["t"],
            )
        };
        let html = "<p><span style=\"font-family: Inter; font-size: 40px; color: #ff8800;\">OPAC</span></p>";
        let s = text(html, json!([{"opacity": 0.5}, {"opacity": 0.0}, {}, {}]));
        assert!(node_text_geometry(&s, "t").is_err(), "flat outlines still decline per-glyph opacity");
        let fills = node_solid_fills(&s, "t").expect("a solid takes per-glyph opacity");
        let glyphs: Vec<(usize, f32)> = fills.iter().filter(|f| f.kind == "glyphs").map(|f| (f.paths.len(), f.opacity)).collect();
        assert_eq!(glyphs, vec![(1, 0.5), (2, 1.0)], "O at half, P left out, AC whole");
        let at_rest = node_solid_fills(&text(html, json!([])), "t").unwrap();
        assert_eq!(at_rest.iter().map(|f| f.paths.len()).sum::<usize>(), 4);

        let emoji = "<p><span style=\"font-family: Inter; font-size: 40px; color: #ff8800;\">GO 🚀</span></p>";
        let s = text(emoji, json!([]));
        assert!(node_text_geometry(&s, "t").is_err(), "flat outlines still decline emoji");
        let fills = node_solid_fills(&s, "t").expect("a solid skips emoji");
        assert_eq!(fills.iter().map(|f| f.paths.len()).sum::<usize>(), 2, "G and O");
    }

    /// The SVG / PDF exports' decorations come from here: a styled underline is the shape its style draws (a wave,
    /// dots, dashes), in its own colour, not a plain bar in the text's.
    #[test]
    fn decoration_styles_reach_the_shape_fills() {
        let fill_for = |css: &str| {
            let html = format!(
                "<p><span style=\"font-family: Inter; font-size: 24px; color: #112233; {css}\">Hello world</span></p>"
            );
            let s = scene(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":60.0,"htmlContent": html}]),
                vec!["t"],
            );
            let g = node_text_geometry(&s, "t").expect("vector text");
            let shapes: Vec<_> = g.fills.into_iter().filter(|f| f.kind == "shape").collect();
            assert_eq!(shapes.len(), 1, "{css}: one decoration colour");
            shapes.into_iter().next().unwrap()
        };
        let subpaths = |d: &str| d.matches('M').count();
        let plain = fill_for("text-decoration: underline");
        assert_eq!(subpaths(&plain.paths[0]), 2, "a solid underline is one quad (two triangles)");
        assert_eq!(plain.color, [0x11 as f32 / 255.0, 0x22 as f32 / 255.0, 0x33 as f32 / 255.0, 1.0]);
        let wavy = fill_for("text-decoration: underline wavy #ff0000 2px");
        assert_eq!(wavy.color, [1.0, 0.0, 0.0, 1.0]);
        assert!(subpaths(&wavy.paths[0]) > 40, "a wave is a strip of many quads");
        let dotted = fill_for("text-decoration: underline dotted");
        assert_eq!(
            subpaths(&dotted.paths[0]) % moda_text::decorations::DOT_SEGMENTS,
            0,
            "dots are fans of {} triangles",
            moda_text::decorations::DOT_SEGMENTS
        );
        let dashed = fill_for("text-decoration: line-through dashed");
        assert!(subpaths(&dashed.paths[0]) >= 2 * 8, "dashes: a quad each");
    }

    /// Fixture-backed (the vendored oracle fonts, like text_extract's tests).
    /// Every point of glyph path data (moves, line and curve ends, control points), absolute.
    fn path_points(d: &str) -> Vec<(f32, f32)> {
        use moda_scene::blend::BlendPathPoint as P;
        let mut out = Vec::new();
        crate::boolean_operands::parse_glyph_d(d, &mut out);
        out.iter()
            .flat_map(|p| match *p {
                P::Move { x, y } | P::Line { x, y } => vec![(x as f32, y as f32)],
                P::Quadratic { x, y, cp } => vec![(cp.0 as f32, cp.1 as f32), (x as f32, y as f32)],
                _ => vec![],
            })
            .collect()
    }

    #[test]
    fn compact_glyph_path_data_form() {
        let mut d = GlyphPathData::new();
        d.move_to(1.0, 2.0);
        d.line_to(3.5, -4.0);
        d.line_to(4.5, -3.0);
        d.line_to(4.0, -3.0);
        d.line_to(4.0, -1.0);
        d.quad_to(5.0, 0.0, 6.0, -1.0);
        d.quad_to(7.0, -2.0, 8.0, -1.0);
        d.close();
        d.move_to(2.0, 3.0);
        d.line_to(4.0, 5.0);
        d.move_to(-0.0004, -2.25);
        assert_eq!(d.d, "M1 2l2.5-6 1 1h-.5v2q1 1 2 0 1-1 2 0Zm1 1l2 2m-4-7.25");
        let mut first = GlyphPathData::new();
        first.move_to(0.25, -1.5);
        assert_eq!(first.d, "M.25 -1.5");
    }

    /// Compact glyph path data reads back as exactly the absolute points rounded to 0.001 px, however long the path.
    #[test]
    fn compact_glyph_path_data_round_trips_to_the_rounded_points() {
        use moda_scene::blend::BlendPathPoint as P;
        let mut d = GlyphPathData::new();
        let mut want = Vec::new();
        let r = |v: f32| (v as f64 * 1000.0).round() / 1000.0;
        let mut x = 12.3456f32;
        for k in 0..400 {
            let (px, py) = (x, 40.0 - (k % 7) as f32 * 0.37);
            if k % 50 == 0 {
                if k > 0 {
                    d.close();
                }
                d.move_to(px, py);
                want.push(format!("M{},{}", r(px), r(py)));
            } else if k % 3 == 0 {
                d.quad_to(px - 0.4, py + 1.0, px, py);
                want.push(format!("Q{},{} {},{}", r(px - 0.4), r(py + 1.0), r(px), r(py)));
            } else {
                d.line_to(px, if k % 5 == 0 { 40.0 } else { py });
                want.push(format!("L{},{}", r(px), r(if k % 5 == 0 { 40.0 } else { py })));
            }
            x += if k % 4 == 0 { 0.0 } else { 0.1234 };
        }
        d.close();
        assert!(d.d.starts_with("M12.346 40"));
        let mut out = Vec::new();
        crate::boolean_operands::parse_glyph_d(&d.d, &mut out);
        let got: Vec<String> = out
            .iter()
            .map(|p| {
                let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
                match p {
                    P::Move { x, y } => format!("M{},{}", r3(*x), r3(*y)),
                    P::Line { x, y } => format!("L{},{}", r3(*x), r3(*y)),
                    P::Quadratic { x, y, cp } => format!("Q{},{} {},{}", r3(cp.0), r3(cp.1), r3(*x), r3(*y)),
                    _ => "?".into(),
                }
            })
            .collect();
        assert_eq!(got, want);
        // Every point within float noise of its rounded value: no drift along the path.
        for p in &out {
            if let P::Line { x, y } | P::Move { x, y } = p {
                assert!((x * 1000.0 - (x * 1000.0).round()).abs() < 1e-6 && (y * 1000.0 - (y * 1000.0).round()).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn outlines_and_runs_for_plain_richtext() {
        let s = scene(
            json!([{"id":"t1","type":"richtext","x":40.0,"y":30.0,"width":300.0,"height":100.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px; color: #112233;\">Hello</span></p>"}]),
            vec!["t1"],
        );
        let g = node_text_geometry(&s, "t1").expect("vector text");
        let glyphs: Vec<_> = g.fills.iter().filter(|f| f.kind == "glyphs").collect();
        assert_eq!(glyphs.len(), 1, "one line");
        assert_eq!(glyphs[0].paths.len(), 5, "a path per glyph of Hello");
        assert!(glyphs[0]
            .paths
            .iter()
            .all(|d| d.starts_with('M') && d.contains('Z')));
        assert!(
            !g.fills.iter().any(|f| f.kind == "shape"),
            "plain text (no bullets/decorations) has no shape fills"
        );
        let runs = g.runs.expect("runs");
        assert_eq!(runs[0].color, "#112233");
        let bytes =
            crate::text_extract::vector_font_bytes(&runs[0].family, runs[0].weight, &runs[0].style)
                .expect("face");
        assert!(face_family_name(&bytes).is_some_and(|f| f.starts_with("Inter")));
        assert!(runs[0].text.contains("Hello"));
        // Glyphs sit inside the box: every coordinate between the node's 0 and its width.
        let nums: Vec<f32> = glyphs[0]
            .paths
            .iter()
            .flat_map(|d| path_points(d))
            .flat_map(|(x, y)| [x, y])
            .collect();
        assert!(nums.len() > 20 && nums.iter().all(|v| (-1.0..=300.0).contains(v)), "{nums:?}");
    }

    /// E8.22: the outlines carry each run's character attributes (shifted up, turned), and its runs say so for live
    /// text: the non-default attributes, each shifted or turned glyph's own baseline `y`, and a turned glyph's origin.
    #[test]
    fn character_attributes_reach_outlines_and_runs() {
        let html = |style: &str| {
            format!(
                "<p><span style=\"font-family: Inter; font-size: 40px;{style}\">HH</span></p>"
            )
        };
        let geometry = |style: &str| {
            let s = scene(
                json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":100.0,
                    "rotation":0,"scale":1,"htmlContent": html(style)}]),
                vec!["t"],
            );
            text_geometry_json(&s, &["t".to_string()])["t"].clone()
        };
        let ys = |g: &moda_scene::value::Value| -> Vec<f32> {
            path_points(g["fills"][0]["paths"][0].as_str().unwrap()).into_iter().map(|p| p.1).collect()
        };
        let plain = geometry("");
        let shifted = geometry(" --nf-baseline-shift: 10px");
        let (p_min, s_min) = (
            ys(&plain).into_iter().fold(f32::INFINITY, f32::min),
            ys(&shifted).into_iter().fold(f32::INFINITY, f32::min),
        );
        assert!((p_min - s_min - 10.0).abs() < 0.05, "{p_min} vs {s_min}");
        let run = &shifted["runs"][0];
        assert_eq!(run["baselineShift"], json!(10.0));
        let baseline = run["baselineY"].as_f64().unwrap();
        assert!((run["glyphs"][0]["y"].as_f64().unwrap() - (baseline - 10.0)).abs() < 1e-6);
        assert!(plain["runs"][0].get("baselineShift").is_none() && plain["runs"][0]["glyphs"][0].get("y").is_none());

        let turned = geometry(" --nf-character-rotation: 90deg; --nf-horizontal-scale: 50%; --nf-kerning: none");
        let run = &turned["runs"][0];
        assert_eq!(run["rotation"], json!(90.0));
        assert_eq!(run["horizontalScale"], json!(0.5));
        assert_eq!(run["kerning"], json!("none"));
        assert!(run["glyphs"][0]["ox"].is_number());
        // Turned a quarter: the H is wider than tall (at half width, still: 40 px tall becomes 40 px wide).
        let (xs, ys): (Vec<f32>, Vec<f32>) = path_points(turned["fills"][0]["paths"][0].as_str().unwrap()).into_iter().unzip();
        let span = |v: &[f32]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max) - v.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(span(&xs) > span(&ys), "turned: {} × {}", span(&xs), span(&ys));
    }

    /// A run's letter-spacing rides along (the pens already include it) so live SVG text can write `letter-spacing`;
    /// a run without any leaves it out.
    #[test]
    fn runs_carry_letter_spacing() {
        let s = scene(
            json!([{"id":"t1","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":100.0,
                "rotation":0,"scale":1,
                "htmlContent":"<p><span style=\"font-family: Inter; font-size: 20px; letter-spacing: 3px;\">Wide</span> <span style=\"font-family: Inter; font-size: 20px;\">tight</span></p>"}]),
            vec!["t1"],
        );
        let json = text_geometry_json(&s, &["t1".to_string()]);
        let runs = json["t1"]["runs"].as_array().expect("runs");
        let run = |text: &str| runs.iter().find(|r| r["text"].as_str().unwrap().contains(text)).unwrap();
        assert_eq!(run("Wide")["letterSpacing"], json!(3.0));
        assert!(run("tight")["letterSpacing"].is_null());
        // The pens step by advance + spacing.
        let xs: Vec<f64> = run("Wide")["glyphs"].as_array().unwrap().iter().map(|g| g["x"].as_f64().unwrap()).collect();
        assert!(xs.windows(2).all(|w| w[1] - w[0] > 3.0), "{xs:?}");
    }

    /// A gradient on one range (`--nf-text-fill`) stays vector: its glyphs carry the gradient as a Moda paint over the
    /// node box, and its run says so for live text; the rest of the text keeps its colour.
    #[test]
    fn a_range_gradient_is_a_gradient_fill() {
        let s = scene(
            json!([{"id":"t","type":"richtext","x":0.0,"y":0.0,"width":400.0,"height":96.0,
                "htmlContent":"<p style=\"font-family: Inter; font-size: 24px; color: rgb(255, 255, 255)\">Plan <span style=\"color: rgb(167, 139, 250); --nf-text-fill: radial(50% 50%, 40% 60%, 90deg, rgb(167, 139, 250) 0%, rgba(244, 114, 182, 0.5) 100%)\">one</span></p>"}]),
            vec!["t"],
        );
        let json = text_geometry_json(&s, &["t".to_string()]);
        let fills = json["t"]["fills"].as_array().expect("vector text");
        let (solid, grad): (Vec<_>, Vec<_>) = fills.iter().partition(|f| f.get("gradient").is_none());
        assert_eq!(solid.len(), 1);
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0]["paths"].as_array().unwrap().len(), 3, "o n e");
        let g = &grad[0]["gradient"];
        assert_eq!(g["type"], "radial");
        assert_eq!(g["center"], json!({ "x": 50.0, "y": 50.0 }));
        assert_eq!([g["radiusX"].as_f64().unwrap(), g["radiusY"].as_f64().unwrap()], [40.0, 60.0]);
        assert!((g["rotation"].as_f64().unwrap() - 90.0).abs() < 1e-3);
        assert_eq!(g["stops"][0]["color"], "rgba(167, 139, 250, 1)");
        assert_eq!(g["stops"][1]["color"], "rgba(244, 114, 182, 0.5)");
        let runs = json["t"]["runs"].as_array().unwrap();
        let one = runs.iter().find(|r| r["text"].as_str().unwrap().contains("one")).unwrap();
        assert_eq!(one["fill"]["type"], "radial");
        let plan = runs.iter().find(|r| r["text"].as_str().unwrap().contains("Plan")).unwrap();
        assert!(plan["fill"].is_null());
    }

    /// F7.2 type on a path: the glyphs run along the path (here a closed circle), not along one baseline in the box.
    #[test]
    fn path_text_glyphs_follow_the_path() {
        let k = 0.552_284_75 * 90.0;
        let pts = json!([
            {"x":180.0,"y":90.0,"type":"move"},
            {"x":90.0,"y":180.0,"type":"curve","cp1":{"x":180.0,"y":90.0+k},"cp2":{"x":90.0+k,"y":180.0}},
            {"x":0.0,"y":90.0,"type":"curve","cp1":{"x":90.0-k,"y":180.0},"cp2":{"x":0.0,"y":90.0+k}},
            {"x":90.0,"y":0.0,"type":"curve","cp1":{"x":0.0,"y":90.0-k},"cp2":{"x":90.0-k,"y":0.0}},
            {"x":180.0,"y":90.0,"type":"curve","cp1":{"x":90.0+k,"y":0.0},"cp2":{"x":180.0,"y":90.0-k}}
        ]);
        let s = scene(
            json!([{"id":"t","type":"richtext","x":40.0,"y":30.0,"width":180.0,"height":180.0,
                "rotation":0,"scale":1,"resizeMode":"no-resize",
                "textPath":{"points":pts,"closed":true,"boxWidth":180.0,"boxHeight":180.0},
                "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px; color: #111111;\">Hello design world</span></p>"}]),
            vec!["t"],
        );
        let g = node_text_geometry(&s, "t").expect("vector text");
        let glyphs: Vec<&String> = g
            .fills
            .iter()
            .filter(|f| f.kind == "glyphs")
            .flat_map(|f| f.paths.iter())
            .collect();
        assert_eq!(glyphs.len(), 16, "a path per visible glyph");
        // Each glyph's first point: along a circle they spread over the box, not along one line of text.
        let first = |d: &str| -> (f32, f32) {
            let n: Vec<f32> = d[1..]
                .split(|c: char| c.is_ascii_alphabetic() || c == ' ')
                .filter_map(|t| t.parse().ok())
                .take(2)
                .collect();
            (n[0], n[1])
        };
        let ys: Vec<f32> = glyphs.iter().map(|d| first(d).1).collect();
        let span = ys.iter().cloned().fold(f32::MIN, f32::max)
            - ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!(span > 60.0, "glyphs on one baseline: {ys:?}");
        assert!(g.runs.is_none(), "path text has no straight runs");
    }

    #[test]
    fn declines_non_text_and_text_effects() {
        let s = scene(
            json!([
                {"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0},
                {"id":"t","type":"richtext","x":0.0,"y":0.0,"width":300.0,"height":100.0,
                 "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":8}],
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">Hi</span></p>"}
            ]),
            vec!["r", "t"],
        );
        assert!(node_text_geometry(&s, "r").is_err());
        assert!(node_text_geometry(&s, "t").is_err());
        let json = text_geometry_json(&s, &["r".to_string()]);
        assert!(json["r"]["error"].is_string());
    }
}
