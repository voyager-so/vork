//! Font outlines are authoring-time geometry. The renderer never sees a font or glyph.
//! Moda's text engine (crate `moda-text`, the 2D canvas's text) lays the text out and gives each
//! glyph's outline; this builder fills and extrudes them.
use crate::moda;
use anyhow::{Context, Result, ensure};
use lyon_path::{
    Path,
    geom::LineSegment,
    math::{Point, point},
};
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use serde::Deserialize;
use serde_json::{Value, json};

fn default_name() -> String {
    "text".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Options {
    #[serde(default = "default_name")]
    name: String,
    body: String,
    font: String,
    #[serde(default = "default_size")]
    size: f32,
    #[serde(default = "default_extrude")]
    extrude: f32,
    #[serde(default = "default_bevel_depth")]
    bevel_depth: f32,
    #[serde(default = "default_bevel_resolution")]
    bevel_resolution: u32,
    #[serde(default = "default_align_x")]
    align_x: String,
    #[serde(default = "default_align_y")]
    align_y: String,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            name: "text".into(),
            body: String::new(),
            font: String::new(),
            size: 1.,
            extrude: 0.,
            bevel_depth: 0.,
            bevel_resolution: 4,
            align_x: "left".into(),
            align_y: "top_baseline".into(),
        }
    }
}
const ALIGN_X: &[&str] = &["left", "center", "right"];
const ALIGN_Y: &[&str] = &["top_baseline", "top", "center", "bottom_baseline", "bottom"];

/// Builder facts generated from the same defaults used by serde.
pub fn input_help() -> String {
    let d = Options::default();
    format!(
        "std/text.name: string; default {name}; standalone prefix only.\n\
std/text.body: string; required; text to lay out.\n\
std/text.font: string; required; TrueType/OpenType file.\n\
std/text.size: number; default {size}; metres per font em; > 0.\n\
std/text.extrude: number; default {extrude}; metres; half-depth; >= 0.\n\
std/text.bevel_depth: number; default {bevel}; metres; >= 0.\n\
std/text.bevel_resolution: uint; default {resolution}; range [0,32]; subdivisions.\n\
std/text.align_x: enum {x_tokens}; default {x}.\n\
std/text.align_y: enum {y_tokens}; default {y}.\n",
        name = d.name,
        size = d.size,
        extrude = d.extrude,
        bevel = d.bevel_depth,
        resolution = d.bevel_resolution,
        x = d.align_x,
        y = d.align_y,
        x_tokens = ALIGN_X.join("|"),
        y_tokens = ALIGN_Y.join("|")
    )
}

fn default_size() -> f32 {
    Options::default().size
}
fn default_extrude() -> f32 {
    Options::default().extrude
}
fn default_bevel_depth() -> f32 {
    Options::default().bevel_depth
}
fn default_bevel_resolution() -> u32 {
    Options::default().bevel_resolution
}
fn default_align_x() -> String {
    Options::default().align_x
}
fn default_align_y() -> String {
    Options::default().align_y
}
#[cfg(test)]
struct Outline(lyon_path::path::Builder);
#[cfg(test)]
impl ttf_parser::OutlineBuilder for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.begin(point(x, y));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(point(x, y));
    }
    fn quad_to(&mut self, x: f32, y: f32, ex: f32, ey: f32) {
        self.0.quadratic_bezier_to(point(x, y), point(ex, ey));
    }
    fn curve_to(&mut self, x: f32, y: f32, x2: f32, y2: f32, ex: f32, ey: f32) {
        self.0
            .cubic_bezier_to(point(x, y), point(x2, y2), point(ex, ey));
    }
    fn close(&mut self) {
        self.0.end(true);
    }
}
#[derive(Default)]
pub(crate) struct Mesh {
    pub(crate) p: Vec<[f32; 3]>,
    pub(crate) n: Vec<[f32; 3]>,
    pub(crate) uv: Vec<[f32; 2]>,
    pub(crate) i: Vec<u32>,
}
impl Mesh {
    fn vertex(&mut self, p: [f32; 3], n: [f32; 3]) -> u32 {
        self.vertex_uv(p, n, [p[0], p[1]])
    }
    pub(crate) fn vertex_uv(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> u32 {
        let id = self.p.len() as u32;
        self.p.push(p);
        self.n.push(n);
        self.uv.push(uv);
        id
    }
    /// Wind the triangle so its front faces the first vertex's normal.
    pub(crate) fn tri(&mut self, a: u32, b: u32, c: u32) {
        let p = self.p[a as usize];
        let q = self.p[b as usize];
        let r = self.p[c as usize];
        let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let n = self.n[a as usize];
        if cross.iter().zip(n).map(|(a, b)| a * b).sum::<f32>() >= 0. {
            self.i.extend([a, b, c]);
        } else {
            self.i.extend([a, c, b]);
        }
    }
}
pub(crate) fn contains(contour: &[Point], p: Point) -> bool {
    let mut inside = false;
    for (a, b) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
    {
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
    }
    inside
}
/// Drop repeated and collinear points: those within `threshold` of the segment joining
/// their neighbours, until none remain.
pub(crate) fn simplify(contour: &mut Vec<Point>, threshold: f32) {
    let mut i = 0;
    let mut kept = 0;
    while contour.len() >= 3 && kept < contour.len() {
        let n = contour.len();
        let chord = LineSegment {
            from: contour[(i + n - 1) % n],
            to: contour[(i + 1) % n],
        };
        if chord.from == chord.to || chord.distance_to_point(contour[i]) <= threshold {
            contour.remove(i);
            kept = 0;
        } else {
            i += 1;
            kept += 1;
        }
        i %= contour.len();
    }
}
/// Lyon joins vertices that share a sweep line with zero-area triangles, leaving the
/// neighbouring triangle's long edge across them. Drop those triangles and split the
/// long edge at their middle vertex, so the caps meet the walls edge for edge.
fn conform(caps: &mut VertexBuffers<Point, u32>) {
    type Splits = std::collections::HashMap<(u32, u32), u32>;
    // Corners from `a` towards `b`, excluding `b`, following nested splits.
    fn edge(splits: &Splits, a: u32, b: u32, out: &mut Vec<u32>) {
        if let Some(&m) = splits.get(&(a.min(b), a.max(b))) {
            edge(splits, a, m, out);
            edge(splits, m, b, out);
        } else {
            out.push(a);
        }
    }
    let vertices = &caps.vertices;
    let p = |i: u32| vertices[i as usize];
    let mut splits = Splits::new();
    let mut triangles = Vec::new();
    for t in caps.indices.chunks_exact(3) {
        let mut t = [t[0], t[1], t[2]];
        if (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])) != 0. {
            triangles.push(t);
        } else {
            t.sort_by(|&a, &b| p(a).x.total_cmp(&p(b).x).then(p(a).y.total_cmp(&p(b).y)));
            splits.insert((t[0].min(t[2]), t[0].max(t[2])), t[1]);
        }
    }
    let mut indices = Vec::new();
    let mut centres = Vec::new();
    for t in triangles {
        let sides = [0, 1, 2].map(|i| {
            let mut corners = Vec::new();
            edge(&splits, t[i], t[(i + 1) % 3], &mut corners);
            corners
        });
        let split: Vec<_> = (0..3).filter(|&i| sides[i].len() > 1).collect();
        if split.is_empty() {
            indices.extend(t);
            continue;
        }
        // Fan from the corner opposite a single split side, else from the centroid.
        let apex = if split.len() == 1 {
            t[(split[0] + 2) % 3]
        } else {
            let [a, b, c] = t.map(p);
            centres.push(point((a.x + b.x + c.x) / 3., (a.y + b.y + c.y) / 3.));
            (vertices.len() + centres.len() - 1) as u32
        };
        let ring = sides.concat();
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if a != apex && b != apex {
                indices.extend([apex, a, b]);
            }
        }
    }
    caps.vertices.extend(centres);
    caps.indices = indices;
}
/// Non-zero fill of closed polyline contours into cap triangles whose edges meet the
/// contours edge for edge (see `conform`).
pub(crate) fn fill(contours: &[Vec<Point>], tolerance: f32) -> Result<VertexBuffers<Point, u32>> {
    let mut builder = Path::builder();
    for contour in contours {
        builder.begin(contour[0]);
        for p in &contour[1..] {
            builder.line_to(*p);
        }
        builder.end(true);
    }
    let path = builder.build();
    let mut caps: VertexBuffers<Point, u32> = VertexBuffers::new();
    FillTessellator::new().tessellate_path(
        &path,
        &FillOptions::default()
            .with_fill_rule(lyon_path::FillRule::NonZero)
            .with_tolerance(tolerance)
            // Lyon's vertical sweep misjoins some vertices sharing a scanline
            // (U+0593); the horizontal sweep tessellates every bundled glyph.
            .with_sweep_orientation(lyon_tessellation::Orientation::Horizontal),
        &mut BuffersBuilder::new(&mut caps, |v: FillVertex| v.position()),
    )?;
    conform(&mut caps);
    Ok(caps)
}
fn validate_options(o: &Options) -> Result<()> {
    ensure!(
        !o.name.is_empty() && !o.name.contains('/'),
        "text name must be nonempty without /"
    );
    ensure!(
        o.size.is_finite()
            && o.size > 0.
            && o.extrude.is_finite()
            && o.extrude >= 0.
            && o.bevel_depth.is_finite()
            && o.bevel_depth >= 0.
            && o.bevel_resolution <= 32,
        "invalid text size/extrude/bevel"
    );
    ensure!(
        !matches!(o.align_x.as_str(), "justify" | "flush"),
        "std/text.align_x {} was removed; std/text does not wrap: write left, center or right, or use wrapped canvas text for justification; see vork help upgrade",
        o.align_x
    );
    ensure!(ALIGN_X.contains(&o.align_x.as_str()), "invalid align_x");
    ensure!(ALIGN_Y.contains(&o.align_y.as_str()), "invalid align_y");
    Ok(())
}
pub(crate) fn evaluate(argument: &str) -> Result<String> {
    let options: Options = serde_json::from_str(argument)?;
    let metadata = std::fs::metadata(&options.font)
        .with_context(|| format!("read font {}; restore the font file or correct its path", options.font))?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
            format!("font {} is not a regular file; restore a readable font file or correct its path", options.font)).into());
    }
    let bytes =
        std::fs::read(&options.font).with_context(|| format!("read font {}", options.font))?;
    let label = options.font.clone();
    Ok(serde_json::to_string(&geometry(options, &label, &bytes)?)?)
}
/// Moda's text host, vector font store and degraded tier for one font, kept per thread: std/text lays
/// out many strings in few fonts, and the store keeps each glyph's outline once parsed.
struct Hosted {
    host: moda_text::TextHost,
    store: moda_text::vector::VectorFontStore,
    degraded: std::cell::OnceCell<moda_text::degraded::DegradedEngine>,
    font_bytes: usize,
}
impl Hosted {
    /// Moda's degraded tier, the lane that shapes right-to-left and complex scripts, on the same font.
    fn degraded(&self) -> &moda_text::degraded::DegradedEngine {
        self.degraded.get_or_init(|| self.host.degraded_engine())
    }
}
thread_local! {
    static HOSTS: std::cell::RefCell<std::collections::VecDeque<([u8; 32], std::rc::Rc<Hosted>)>> = Default::default();
}
fn hosted(label: &str, bytes: &[u8]) -> Result<std::rc::Rc<Hosted>> {
    let key = *blake3::hash(bytes).as_bytes();
    if let Some(hosted) = HOSTS.with(|hosts| {
        let mut hosts = hosts.borrow_mut();
        let index = hosts.iter().position(|(k, _)| *k == key)?;
        let entry = hosts.remove(index)?;
        let host = entry.1.clone();
        hosts.push_back(entry);
        Some(host)
    }) {
        return Ok(hosted);
    }
    let host = moda::host(&[vec![moda::Face { label: label.into(), bytes: bytes.to_vec() }]])?;
    let store = host.vector_font_store_lenient();
    let hosted = std::rc::Rc::new(Hosted { host, store, degraded: Default::default(), font_bytes: bytes.len() });
    HOSTS.with(|hosts| {
        let mut hosts = hosts.borrow_mut();
        hosts.push_back((key, hosted.clone()));
        // Entry and byte ceilings bound both small parsed stores and large font binaries.
        while hosts.len() > 8 || hosts.iter().map(|(_, h)| h.font_bytes).sum::<usize>() > 64 * 1024 * 1024 {
            hosts.pop_front();
        }
    });
    Ok(hosted)
}
/// One placed line of laid-out text (px, y down): its top, height and baseline, and where its advance
/// starts and how wide it is.
struct Line {
    top: f64,
    height: f64,
    baseline: f64,
    start: f64,
    width: f64,
}
fn lines_of(layout: &Value) -> Vec<Line> {
    let at = |line: &Value, key: &str| line[key].as_f64().unwrap_or(0.);
    layout["paragraphs"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|paragraph| paragraph["lines"].as_array().into_iter().flatten())
        .map(|line| Line {
            top: at(line, "lineTopY"),
            height: at(line, "lineHeightPx"),
            baseline: at(line, "baselineY"),
            start: at(line, "lineStartX"),
            width: at(line, "totalWidth"),
        })
        .collect()
}
/// The text's glyph outlines as closed contours in font units (x right, y down from the node's top-left),
/// aligned in the box (the widest line), with its lines.
struct Laid {
    contours: Vec<Vec<Point>>,
    lines: Vec<Line>,
    box_width: f64,
}
/// Moda's vector lane: the text laid out at one px per font unit, aligned in the widest line by CSS
/// text-align, each placed glyph's outline Moda's, flattened as its extrusion flattens it.
fn vector_laid(o: &Options, label: &str, hosted: &Hosted, style: &str, upem: f64) -> Result<Option<Laid>> {
    let html = paragraphs(&o.body, &format!("text-align: {}; {style}; font-size: {upem}px", o.align_x));
    let mut spec = moda_text::RichTextNodeSpec { node_id: o.name.clone(), html_content: html, auto_width: true, ..Default::default() };
    let layout = moda_text::vector::layout_rich_text_node_vector(&spec, &hosted.store, &hosted.host.catalog);
    if layout["engine"].as_str() != Some("vector") {
        return Ok(None);
    }
    let box_width = moda::widest_line(&layout);
    spec.width = Some(box_width);
    let layout = moda_text::vector::layout_rich_text_node_vector(&spec, &hosted.store, &hosted.host.catalog);
    let Some(runs) = moda_text::vector::extract::extract_vector_text_runs(&spec, &hosted.store, &hosted.host.catalog) else {
        return Ok(None);
    };
    let mut contours = Vec::new();
    for run in &runs {
        let variations = moda_text::font_variations::parse_font_variation_settings(&run.font_variations);
        let instance = moda_text::vector::measure::Instance { size_px: Some(run.font_size_px), variations: &variations };
        let font = hosted
            .store
            .font_for_variant(&hosted.host.catalog, &run.family, run.weight, &run.style, instance)
            .with_context(|| format!("font {label}: Moda's vector text lane cannot read it"))?;
        for glyph in &run.glyphs {
            // Moda's outline in em, y down from the glyph's origin.
            let rings = font
                .glyph_em_contours(glyph.glyph_id)
                .map_err(|error| anyhow::anyhow!("font {label}: glyph {}: {}", glyph.glyph_id, error.message()))?;
            let baseline = run.baseline_y_px - glyph.y_offset_px;
            for ring in rings.iter() {
                contours.push(
                    ring.iter()
                        .map(|[x, y]| point((glyph.x_px + x * run.font_size_px) as f32, (baseline + y * run.font_size_px) as f32))
                        .collect(),
                );
            }
        }
    }
    Ok(Some(Laid { contours, lines: lines_of(&layout), box_width }))
}
/// Moda's degraded tier, the lane that shapes what the vector lane declines (right-to-left and bidi text,
/// Arabic joining, Indic and Southeast Asian clusters): the same paragraphs laid out and shaped by it, each
/// glyph's outline the face's at the size it is drawn. The tier draws through a raster, so it lays the text
/// out at a size whose raster fits and the result is scaled to font units; its lines start at the left and
/// are aligned in the widest line here, by their advances, as text-align aligns them.
fn degraded_laid(o: &Options, hosted: &Hosted, style: &str, upem: f64) -> Result<Laid> {
    let engine = hosted.degraded();
    let spec_at = |size: f64| moda_text::ops::TextNodeSpec {
        node_id: o.name.clone(),
        html: paragraphs(&o.body, &format!("text-align: left; {style}; font-size: {size}px")).into(),
        opacity: 1.,
        auto_width: true,
        ..Default::default()
    };
    // Raster budget: the walk draws at twice its logical size and refuses textures past 8192 px.
    let (w, h) = engine.canvas_logical_size(&spec_at(upem));
    let size = upem * (1500. / w.max(1.)).min(1500. / h.max(1.)).min(1.);
    let k = upem / size;
    let spec = spec_at(size);
    let layout = engine.layout_json(&spec).map_err(|error| anyhow::anyhow!("std/text.body: Moda's degraded text lane: {error}"))?;
    let mut lines = lines_of(&layout);
    let outlines = engine.glyph_outlines(&spec).map_err(|error| anyhow::anyhow!("std/text.body: Moda's degraded text lane: {error}"))?;
    let box_width = lines.iter().map(|line| line.width).fold(0., f64::max);
    // Each line's shift into the box, by text-align.
    let shift = |line: &Line| match o.align_x.as_str() {
        "center" => (box_width - line.width) * 0.5 - line.start,
        "right" => box_width - line.width - line.start,
        _ => -line.start,
    };
    let mut contours = Vec::new();
    for glyph in &outlines {
        let Some(line) = lines.iter().min_by(|a, b| {
            (a.baseline - f64::from(glyph.pen[1])).abs().total_cmp(&(b.baseline - f64::from(glyph.pen[1])).abs())
        }) else {
            continue;
        };
        let dx = shift(line);
        let at = |p: [f32; 2]| point(((f64::from(p[0]) + dx) * k) as f32, (f64::from(p[1]) * k) as f32);
        let mut builder = Path::builder();
        let mut open = false;
        for segment in &glyph.segments {
            use moda_text::degraded::OutlineSegment as S;
            match *segment {
                S::MoveTo(p) => {
                    if open {
                        builder.end(true);
                    }
                    builder.begin(at(p));
                    open = true;
                }
                S::LineTo(p) if open => _ = builder.line_to(at(p)),
                S::QuadTo(c, p) if open => _ = builder.quadratic_bezier_to(at(c), at(p)),
                S::CurveTo(a, b, p) if open => _ = builder.cubic_bezier_to(at(a), at(b), at(p)),
                S::Close if open => {
                    builder.end(true);
                    open = false;
                }
                _ => {}
            }
        }
        if open {
            builder.end(true);
        }
        use lyon_path::iterator::PathIterator;
        let mut ring = Vec::new();
        // A quarter of a font unit, five times the distance at which points merge (`geometry`).
        for event in builder.build().iter().flattened(0.25) {
            match event {
                lyon_path::PathEvent::Begin { at } => ring = vec![at],
                lyon_path::PathEvent::Line { to, .. } => ring.push(to),
                lyon_path::PathEvent::End { .. } => contours.push(std::mem::take(&mut ring)),
                _ => {}
            }
        }
    }
    for line in &mut lines {
        let dx = shift(line);
        *line = Line { top: line.top * k, height: line.height * k, baseline: line.baseline * k, start: (line.start + dx) * k, width: line.width * k };
    }
    Ok(Laid { contours, lines, box_width: box_width * k })
}
/// One paragraph per line of `body`, styled with `style`.
fn paragraphs(body: &str, style: &str) -> String {
    body.split('\n').map(|line| format!("<p style=\"{style}\">{}</p>", moda::escape_html(line))).collect()
}
fn geometry(o: Options, label: &str, bytes: &[u8]) -> Result<Value> {
    validate_options(&o)?;
    for ch in o.body.chars() {
        ensure!(ch == '\n' || !ch.is_control(), "unsupported text control character {ch:?}");
    }
    let face = ttf_parser::Face::parse(bytes, 0)
        .map_err(|error| anyhow::anyhow!("font {label}: not a TrueType or OpenType font ({error})"))?;
    for ch in o.body.chars().filter(|c| !c.is_whitespace()) {
        ensure!(face.glyph_index(ch).is_some_and(|g| g.0 != 0), "font has no glyph for {ch:?}");
    }
    ensure!(o.body.chars().any(|c| !c.is_whitespace()), "text has no outlined glyphs");
    // Moda lays out in px: at one px per font unit its numbers are the font's own.
    let upem = f64::from(face.units_per_em());
    let scale = o.size / upem as f32;
    // Lyon's absolute fill tolerance also joins vertices within half of it of an edge.
    // Keep that well below one font unit at any size, and remove the points it would
    // join from the contours too, so caps and walls share exactly one outline.
    let merge = scale * 0.05;
    let hosted = hosted(label, bytes)?;
    // One paragraph per line. CSS line-height 1 puts the lines one size apart (Blender Text's
    // space_line 1); the box is the widest line, so text-align places each line within it.
    let style = format!(
        "font-family: '{}'; font-weight: {}; font-style: {}; line-height: 1",
        moda::family(0),
        face.weight().to_number(),
        if face.is_italic() { "italic" } else { "normal" }
    );
    // Moda's vector lane draws what it can shape; what it declines its degraded tier shapes.
    let laid = match vector_laid(&o, label, &hosted, &style, upem)? {
        Some(laid) => laid,
        None => degraded_laid(&o, &hosted, &style, upem)?,
    };
    let (Some(first), Some(last)) = (laid.lines.first(), laid.lines.last()) else {
        anyhow::bail!("text has no outlined glyphs");
    };
    let (top, bottom) = (first.top, last.top + last.height);
    // The layout's point (px, y down) that becomes the geometry's origin.
    let origin_x = match o.align_x.as_str() {
        "center" => laid.box_width * 0.5,
        "right" => laid.box_width,
        _ => 0.,
    };
    let origin_y = match o.align_y.as_str() {
        "top" => top,
        "center" => (top + bottom) * 0.5,
        "bottom_baseline" => last.baseline,
        "bottom" => bottom,
        _ => first.baseline,
    };
    let mut contours = Vec::<Vec<Point>>::new();
    for ring in &laid.contours {
        let mut contour: Vec<Point> = ring
            .iter()
            .map(|p| point((f64::from(p.x) - origin_x) as f32 * scale, (origin_y - f64::from(p.y)) as f32 * scale))
            .collect();
        simplify(&mut contour, merge);
        if contour.len() >= 3 {
            contours.push(contour);
        }
    }
    ensure!(!contours.is_empty(), "text has no outlined glyphs");
    // Non-zero winding preserves counters and overlapping glyph outlines.
    let caps = fill(&contours, 2. * merge).context("tessellate font outlines")?;
    let mut mesh = Mesh::default();
    let depth = o.extrude + o.bevel_depth;
    for sign in if depth > 0. { vec![1., -1.] } else { vec![1.] } {
        let base = mesh.p.len() as u32;
        for p in &caps.vertices {
            mesh.vertex([p.x, p.y, sign * depth], [0., 0., sign]);
        }
        for tri in caps.indices.chunks_exact(3) {
            mesh.tri(base + tri[0], base + tri[1], base + tri[2]);
        }
    }
    if depth > 0. {
        for contour in &contours {
            let area: f32 = contour
                .iter()
                .zip(contour.iter().cycle().skip(1))
                .take(contour.len())
                .map(|(a, b)| a.x * b.y - b.x * a.y)
                .sum();
            let hole = contours
                .iter()
                .filter(|other| !std::ptr::eq(*other, contour) && contains(other, contour[0]))
                .count()
                % 2
                == 1;
            let sign = if (area > 0.) != hole { 1. } else { -1. };
            let normals: Vec<_> = (0..contour.len())
                .map(|i| {
                    let prev = contour[(i + contour.len() - 1) % contour.len()];
                    let at = contour[i];
                    let next = contour[(i + 1) % contour.len()];
                    let a = (at - prev).normalize();
                    let b = (next - at).normalize();
                    let n = lyon_path::math::vector((a.y + b.y) * sign, -(a.x + b.x) * sign)
                        .normalize();
                    let edge = lyon_path::math::vector(b.y * sign, -b.x * sign);
                    (n, 1. / n.dot(edge).max(0.1))
                })
                .collect();
            let steps = o.bevel_resolution.max(1);
            let mut rings = Vec::new();
            if o.bevel_depth > 0. {
                for k in 0..=steps {
                    let angle = k as f32 / steps as f32 * std::f32::consts::FRAC_PI_2;
                    rings.push((
                        o.bevel_depth * angle.sin(),
                        o.extrude + o.bevel_depth * angle.cos(),
                        angle.sin(),
                        angle.cos(),
                    ));
                }
                for k in (0..=steps).rev() {
                    let angle = k as f32 / steps as f32 * std::f32::consts::FRAC_PI_2;
                    rings.push((
                        o.bevel_depth * angle.sin(),
                        -o.extrude - o.bevel_depth * angle.cos(),
                        angle.sin(),
                        -angle.cos(),
                    ));
                }
            } else {
                rings.extend([(0., o.extrude, 1., 0.), (0., -o.extrude, 1., 0.)]);
            }
            let base = mesh.p.len() as u32;
            let count = contour.len() as u32;
            for &(offset, z, nxy, nz) in &rings {
                for (p, (n, miter)) in contour.iter().zip(&normals) {
                    mesh.vertex(
                        [p.x + n.x * offset * miter, p.y + n.y * offset * miter, z],
                        [n.x * nxy, n.y * nxy, nz],
                    );
                }
            }
            for ring in 0..rings.len() - 1 {
                for i in 0..count {
                    let a = base + ring as u32 * count + i;
                    let b = base + ring as u32 * count + (i + 1) % count;
                    mesh.tri(a, b, b + count);
                    mesh.tri(a, b + count, a + count);
                }
            }
        }
    }
    Ok(mesh.fragment(&o.name))
}
impl Mesh {
    /// The std geometry fragment: `{name}_position`, `_normal`, `_TEXCOORD_0` and `_indices`.
    pub(crate) fn fragment(&self, name: &str) -> Value {
        let p = format!("{name}_position");
        let n = format!("{name}_normal");
        let uv = format!("{name}_TEXCOORD_0");
        let i = format!("{name}_indices");
        json!({"buffers":{&p:{"type":"vec3","data":self.p},&n:{"type":"vec3","data":self.n},&uv:{"type":"vec2","data":self.uv},&i:{"type":"u32","data":self.i}},
        "geometry":{name:{"topology":"triangles","attributes":{"POSITION":p,"NORMAL":n,"TEXCOORD_0":uv},"indices":i}}})
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use lyon_path::{PathEvent, iterator::PathIterator};
    const FONT: &[u8] = include_bytes!("../tests/fonts/LiberationSans-Regular.ttf");
    fn text(o: Options) -> Result<Value> {
        geometry(o, "LiberationSans-Regular.ttf", FONT)
    }
    fn points(g: &Value) -> Vec<[f32; 3]> {
        serde_json::from_value(g["buffers"]["text_position"]["data"].clone()).unwrap()
    }
    #[test]
    fn retired_parsed_hosts_are_evicted() {
        HOSTS.with(|hosts| hosts.borrow_mut().clear());
        let original = include_bytes!("../tests/fonts/LiberationSans-Regular.ttf");
        let first = hosted("first", original).unwrap();
        let retired = std::rc::Rc::downgrade(&first);
        drop(first);
        for suffix in 0..16u8 {
            let mut bytes = original.to_vec();
            bytes.push(suffix);
            drop(hosted("varied", &bytes).unwrap());
        }
        assert!(retired.upgrade().is_none(), "retired parsed font host remains rooted");
    }

    #[test]
    fn complex_scripts_are_shaped_by_the_degraded_tier() {
        // Arabic joins: the shaped word is narrower than its letters set apart. (Joined letters overlap where they
        // connect, so, as with any overlapping outlines, walls stand inside the caps: the mesh is not closed.)
        let arabic = include_bytes!("../tests/fonts/NotoSansArabic-Regular.ttf");
        let width = |body: &str| {
            let g = geometry(Options { body: body.into(), extrude: 0.01, ..Options::default() }, "NotoSansArabic-Regular.ttf", arabic).unwrap();
            let x: Vec<f32> = points(&g).iter().map(|p| p[0]).collect();
            x.iter().copied().fold(f32::NEG_INFINITY, f32::max) - x.iter().copied().fold(f32::INFINITY, f32::min)
        };
        let joined = width("سلام");
        let apart = width("س ل ا م");
        assert!(joined > 0.3 && joined < apart, "{joined} {apart}");
        // align_x centres the line's advance on the origin.
        let g = geometry(Options { body: "سلام".into(), align_x: "center".into(), ..Options::default() }, "NotoSansArabic-Regular.ttf", arabic).unwrap();
        let x: Vec<f32> = points(&g).iter().map(|p| p[0]).collect();
        let (left, right) = (x.iter().copied().fold(f32::INFINITY, f32::min), x.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        assert!((left + right).abs() < 0.1 * (right - left), "{left} {right}");
    }
    #[test]
    fn what_moda_declines_is_refused() {
        for align_x in ["justify", "flush"] {
            let error = text(Options {
                body: "a b\nc".into(),
                align_x: align_x.into(),
                ..Options::default()
            })
            .unwrap_err()
            .to_string();
            assert!(error.contains(&format!("std/text.align_x {align_x} was removed")), "{error}");
            assert_eq!(error.lines().count(), 1, "{error}");
            assert!(error.contains("write left, center or right") && error.ends_with("see vork help upgrade"), "{error}");
        }
    }
    #[test]
    fn lines_are_one_size_apart_and_aligned() {
        let lowest = |body: &str, align_x: &str| {
            let p = points(
                &text(Options {
                    body: body.into(),
                    size: 2.,
                    align_x: align_x.into(),
                    ..Options::default()
                })
                .unwrap(),
            );
            let x = p.iter().map(|p| p[0]);
            (
                p.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min),
                x.clone().fold(f32::INFINITY, f32::min),
                x.fold(f32::NEG_INFINITY, f32::max),
            )
        };
        let (one, ..) = lowest("H", "left");
        let (two, ..) = lowest("H\nH", "left");
        let (three, ..) = lowest("H\n\nH", "left");
        assert!((one - two - 2.).abs() < 1e-3, "{one} {two}");
        assert!((one - three - 4.).abs() < 1e-3, "{one} {three}");
        let (_, left, _) = lowest("HH", "left");
        let (_, _, right) = lowest("HH", "right");
        let (_, centre_left, centre_right) = lowest("HH", "center");
        assert!(left >= 0. && left < 0.3, "{left}");
        assert!(right <= 0. && right > -0.3, "{right}");
        assert!((centre_left + centre_right).abs() < 1e-3, "{centre_left} {centre_right}");
    }
    #[test]
    fn counters_survive_caps_and_extrusion() {
        let o = Options {
            body: "O8".into(),
            extrude: 0.1,
            bevel_depth: 0.02,
            ..Options::default()
        };
        let g = text(o).unwrap();
        let p = g["buffers"]["text_position"]["data"].as_array().unwrap();
        let indices = g["buffers"]["text_indices"]["data"].as_array().unwrap();
        assert!(indices.len() > 100);
        assert!(p.iter().all(|v| {
            v.as_array()
                .unwrap()
                .iter()
                .all(|n| n.as_f64().unwrap().is_finite())
        }));
        assert!(
            p.iter()
                .any(|p| (p[2].as_f64().unwrap() - 0.12).abs() < 1e-6)
        );
        // No cap triangle may fill O's central counter.
        for t in indices.chunks_exact(3) {
            let pts: Vec<_> = t.iter().map(|i| &p[i.as_u64().unwrap() as usize]).collect();
            if pts
                .iter()
                .all(|v| (v[2].as_f64().unwrap() - 0.12).abs() < 1e-6)
            {
                let x = pts.iter().map(|p| p[0].as_f64().unwrap()).sum::<f64>() / 3.;
                let y = pts.iter().map(|p| p[1].as_f64().unwrap()).sum::<f64>() / 3.;
                assert!(!(x > 0.3 && x < 0.45 && y > 0.25 && y < 0.45));
            }
        }
    }
    #[test]
    fn invalid_options_and_missing_glyphs_fail() {
        assert!(
            text(Options {
                size: -1.,
                ..Options::default()
            })
            .is_err()
        );
        assert!(
            text(Options {
                body: "\u{10ffff}".into(),
                ..Options::default()
            })
            .is_err()
        );
        assert!(serde_json::from_str::<Options>(r#"{"bevel":1}"#).is_err());
    }
    fn closed(body: &str, bevel_depth: f32) -> Result<(), String> {
        let g = text(Options {
            body: body.into(),
            extrude: 0.01,
            bevel_depth,
            ..Options::default()
        })
        .map_err(|e| e.to_string())?;
        watertight(&g, "text")
    }
    /// Welded geometry uses every directed edge once and its reverse once, without
    /// zero-area triangles, and is wound outward. A cap triangle outside the outline
    /// leaves an open edge and reuses another.
    pub(crate) fn watertight(g: &Value, name: &str) -> Result<(), String> {
        let data = |key: &str| g["buffers"][key]["data"].clone();
        let p: Vec<[f64; 3]> = serde_json::from_value(data(&format!("{name}_position"))).unwrap();
        let indices: Vec<usize> = serde_json::from_value(data(&format!("{name}_indices"))).unwrap();
        let mut welded = std::collections::HashMap::new();
        let id: Vec<usize> = p
            .iter()
            .map(|v| {
                let n = welded.len();
                *welded.entry(v.map(f64::to_bits)).or_insert(n)
            })
            .collect();
        let mut edges = std::collections::HashMap::<_, u32>::new();
        let (mut volume, mut slivers) = (0., 0);
        for t in indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| p[i]);
            let cross = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            if u[1] * w[2] == u[2] * w[1]
                && u[2] * w[0] == u[0] * w[2]
                && u[0] * w[1] == u[1] * w[0]
            {
                slivers += 1;
            }
            volume += a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((id[x], id[y])).or_default() += 1;
            }
        }
        let open = edges
            .iter()
            .filter(|&(&(a, b), &n)| n != 1 || edges.get(&(b, a)) != Some(&1))
            .count();
        if open > 0 || slivers > 0 || volume <= 0. {
            return Err(format!(
                "{open} unpaired edges, {slivers} zero-area triangles, volume {volume}"
            ));
        }
        Ok(())
    }
    /// Whether a glyph's outline crosses itself: the nonzero cap then unions the overlap,
    /// and walls remain inside it.
    fn overlapping(face: &ttf_parser::Face<'_>, glyph: ttf_parser::GlyphId) -> bool {
        let mut outline = Outline(Path::builder());
        if face.outline_glyph(glyph, &mut outline).is_none() {
            return false;
        }
        let mut segments = Vec::new();
        for event in outline.0.build().iter().flattened(1.) {
            match event {
                PathEvent::Line { from, to } => segments.push(LineSegment { from, to }),
                PathEvent::End { last, first, .. } => segments.push(LineSegment {
                    from: last,
                    to: first,
                }),
                _ => {}
            }
        }
        segments.iter().enumerate().any(|(i, s)| {
            segments[i + 1..]
                .iter()
                .any(|t| s.to != t.from && s.from != t.to && s.intersects(t))
        })
    }
    #[test]
    fn extruded_glyphs_are_closed_and_outward() {
        let face = ttf_parser::Face::parse(FONT, 0).unwrap();
        let mut failures = Vec::new();
        let mut checked = 0;
        for c in (' '..='\u{ffff}').filter(|c| !c.is_control()) {
            let Some(glyph) = face.glyph_index(c) else {
                continue;
            };
            if glyph.0 == 0 || overlapping(&face, glyph) {
                continue;
            }
            match closed(&c.to_string(), 0.) {
                // Text Moda's vector lane declines (complex scripts, lone marks) is refused, not built.
                Err(e) if e.contains("declined") => {}
                Err(e) if !e.contains("no outlined glyphs") => failures.push(format!("{c:?}: {e}")),
                _ => checked += 1,
            }
        }
        assert!(checked > 2000, "{checked} glyphs");
        let ascii: String = (' '..='~').collect();
        for bevel in [0., 0.0004] {
            if let Err(e) = closed(&format!("{ascii}\n{ascii}"), bevel) {
                failures.push(format!("ASCII, bevel {bevel}: {e}"));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
