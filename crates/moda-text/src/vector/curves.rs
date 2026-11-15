//! Glyph outline → quadratic-segment extraction for the analytic vector text
//! fill — the Rust port of `renderers/webgpu/text/vector/glyphCurves.ts`
//! (`quadraticsFromCommands` / `getGlyphQuadratics`) over an opentype.js-exact
//! glyf outline reader.
//!
//! The browser gets outlines from `glyph.getPath(0, 0, 1)` (opentype.js 1.3.4
//! glyph.js:123-176): the glyf-table path commands scaled by
//! `1 / (unitsPerEm || 1000)` and y-FLIPPED (em-normalized, y-down, baseline
//! origin). Per the `otjs.rs` house rule we hand-port opentype.js's glyf
//! reader (tables/glyf.js + tables/loca.js) instead of using skrifa's outline
//! API, so command construction matches byte-for-byte:
//!
//!  - contour start selection (glyf.js getPath:226-241): last point if
//!    on-curve, else first point if on-curve, else their midpoint;
//!  - implied on-curve midpoints between consecutive off-curve points;
//!  - the duplicate `lineTo`/`quadTo` re-emission of the start point at the
//!    end of every contour walk (i wraps to `contour[0]`), which
//!    `quadraticsFromCommands` then drops as a zero-length degenerate;
//!  - composite glyphs flattened recursively via `transformPoints`
//!    (xScale/scale01/scale10/yScale F2Dot14 + dx/dy) — with the off-diagonal
//!    terms applied per the OT spec rather than opentype.js's transposed form
//!    (see Deviations) — including the matched-points positioning branch and its
//!    "Matched points out of range" throw (glyf.js buildPath:270-306).
//!
//! Deviations (recorded):
//!  - the composite 2×2 matrix is applied per the OpenType spec's ROW-vector
//!    convention (`x' = xScale·x + scale10·y`), NOT opentype.js's transposed
//!    column-vector form. opentype.js 1.3.4 `transformPoints` (glyf.js:175-191)
//!    swaps the off-diagonal terms; for a diagonal matrix the two agree, so the
//!    bug only shows on a rotated or skewed component. Fraunces `plus` (two
//!    `minus` refs, the second rotated 90°) rendered as two detached bars — the
//!    vertical one past the advance and below the baseline. fontTools and
//!    HarfBuzz both agree with the spec, so the TS lane corrects opentype.js at
//!    `renderers/webgpu/text/vector/compositeGlyphOutline.ts` and the two lanes
//!    stay bit-identical. See `transform_points` below;
//!  - composite recursion is depth-capped at 16 (opentype.js would blow the
//!    JS stack on a cyclic font; trusted fonts never recurse past 2);
//!  - CFF ('C' command) outlines are not parsed (no .otf in the catalog —
//!    `glyph_quadratics` errors → the node falls back like a vector-error).
//!    The cubic→quadratic subdivision itself IS ported (the segment contract
//!    pins `CUBIC_TOLERANCE_EM`/`MAX_CUBIC_SEGMENTS`), so a CFF source only
//!    needs a charstring reader.
//!
//! Numbers gate: `crates/moda-text/tests/vector_curve_parity.rs` compares
//! every extracted segment against the production dump
//! (`tools/oracle/dump-glyph-curves.mjs` → fixtures/oracle-vector/
//! glyph-curves.json) bit-exactly on the f32 values.

use crate::vector::otjs::ShapeError;

/// Manifest-pinned extraction constants (`moda_shaders` manifest
/// `vector_text` section mirrors these; glyphCurves.ts:33-38).
pub const CUBIC_TOLERANCE_EM: f64 = 0.001;
pub const MAX_CUBIC_SEGMENTS: i64 = 8;
pub const DEGENERATE_EPSILON_SQ: f64 = 1e-12;
/// 3 control points × 2 floats per quadratic segment (GlyphCurveStore.ts:21).
pub const FLOATS_PER_CURVE: usize = 6;

/// `GlyphQuadratics` (glyphCurves.ts:21-31). Values are f32 — the browser
/// stores a Float32Array and computes bounds over the CONVERTED values, so
/// bounds here are min/max of the f32-rounded control points.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphQuadratics {
    /// Flat control points [p0x, p0y, cx, cy, p1x, p1y, ...].
    pub points: Vec<f32>,
    /// `points.len() / 6`.
    pub curve_count: u32,
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

/// opentype.js `PathCommand` (the subset glyf/CFF emit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathCommand {
    M {
        x: f64,
        y: f64,
    },
    L {
        x: f64,
        y: f64,
    },
    Q {
        x1: f64,
        y1: f64,
        x: f64,
        y: f64,
    },
    C {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x: f64,
        y: f64,
    },
    Z,
}

/// `quadraticsFromCommands` (glyphCurves.ts:45-126): lines → degenerate
/// quadratics (control at midpoint), quadratics pass through, cubics
/// subdivide within `CUBIC_TOLERANCE_EM`, `Z`/unclosed contours emit the
/// closing edge; zero-length segments dropped.
pub fn quadratics_from_commands(commands: &[PathCommand]) -> GlyphQuadratics {
    let mut pts: Vec<f64> = Vec::new();
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut start_x = 0.0f64;
    let mut start_y = 0.0f64;
    let mut open = false;

    fn push_line(pts: &mut Vec<f64>, x0: f64, y0: f64, x1: f64, y1: f64) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        if dx * dx + dy * dy < DEGENERATE_EPSILON_SQ {
            return;
        }
        pts.extend_from_slice(&[x0, y0, (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1, y1]);
    }

    macro_rules! close_contour {
        () => {
            if open {
                push_line(&mut pts, cx, cy, start_x, start_y);
            }
            #[allow(unused_assignments)]
            {
                open = false;
            }
        };
    }

    for c in commands {
        match *c {
            PathCommand::M { x, y } => {
                close_contour!();
                start_x = x;
                start_y = y;
                cx = x;
                cy = y;
                open = true;
            }
            PathCommand::L { x, y } => {
                push_line(&mut pts, cx, cy, x, y);
                cx = x;
                cy = y;
            }
            PathCommand::Q { x1, y1, x, y } => {
                let end_dx = x - cx;
                let end_dy = y - cy;
                let ctrl_dx = x1 - cx;
                let ctrl_dy = y1 - cy;
                let degenerate = end_dx * end_dx + end_dy * end_dy < DEGENERATE_EPSILON_SQ
                    && ctrl_dx * ctrl_dx + ctrl_dy * ctrl_dy < DEGENERATE_EPSILON_SQ;
                if !degenerate {
                    pts.extend_from_slice(&[cx, cy, x1, y1, x, y]);
                }
                cx = x;
                cy = y;
            }
            PathCommand::C {
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => {
                append_cubic_as_quadratics(&mut pts, cx, cy, x1, y1, x2, y2, x, y);
                cx = x;
                cy = y;
            }
            PathCommand::Z => {
                close_contour!();
                cx = start_x;
                cy = start_y;
            }
        }
    }
    close_contour!();

    // `new Float32Array(pts)` — convert FIRST, then bounds over f32 values.
    let points: Vec<f32> = pts.iter().map(|v| *v as f32).collect();
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    if !points.is_empty() {
        min_x = f32::INFINITY;
        min_y = f32::INFINITY;
        max_x = f32::NEG_INFINITY;
        max_y = f32::NEG_INFINITY;
        let mut i = 0;
        while i < points.len() {
            let x = points[i];
            let y = points[i + 1];
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
            i += 2;
        }
    }
    let curve_count = (points.len() / FLOATS_PER_CURVE) as u32;
    GlyphQuadratics {
        points,
        curve_count,
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

/// `appendCubicAsQuadratics` (glyphCurves.ts:135-195): split the cubic into
/// n = ⌈∛(err/tol)⌉ pieces (err = √3/36·‖p3−3c2+3c1−p0‖, capped at 8) and
/// approximate each sub-cubic with control (3(c1+c2)−(p0+p3))/4.
#[allow(clippy::too_many_arguments)]
fn append_cubic_as_quadratics(
    out: &mut Vec<f64>,
    x0: f64,
    y0: f64,
    c1x: f64,
    c1y: f64,
    c2x: f64,
    c2y: f64,
    x1: f64,
    y1: f64,
) {
    let ex = x1 - 3.0 * c2x + 3.0 * c1x - x0;
    let ey = y1 - 3.0 * c2y + 3.0 * c1y - y0;
    let err = (3.0f64.sqrt() / 36.0) * ex.hypot(ey);
    // Math.ceil(Math.cbrt(err / tol)) — NaN-free for err >= 0; cbrt(0)=0 → max(1).
    let n_raw = (err / CUBIC_TOLERANCE_EM).cbrt().ceil();
    // Math.min(MAX, Math.max(1, n)) — clamp is equivalent (MAX >= 1).
    let n = (n_raw as i64).clamp(1, MAX_CUBIC_SEGMENTS);

    let point = |t: f64| -> (f64, f64) {
        let u = 1.0 - t;
        let a = u * u * u;
        let b = 3.0 * u * u * t;
        let c = 3.0 * u * t * t;
        let d = t * t * t;
        (
            a * x0 + b * c1x + c * c2x + d * x1,
            a * y0 + b * c1y + c * c2y + d * y1,
        )
    };
    let derivative = |t: f64| -> (f64, f64) {
        let u = 1.0 - t;
        (
            3.0 * u * u * (c1x - x0) + 6.0 * u * t * (c2x - c1x) + 3.0 * t * t * (x1 - c2x),
            3.0 * u * u * (c1y - y0) + 6.0 * u * t * (c2y - c1y) + 3.0 * t * t * (y1 - c2y),
        )
    };

    let (mut px, mut py) = point(0.0);
    let (mut dx, mut dy) = derivative(0.0);
    for i in 0..n {
        let t1 = (i + 1) as f64 / n as f64;
        let dt = 1.0 / n as f64;
        let (qx, qy) = point(t1);
        let (d1x, d1y) = derivative(t1);
        let s1x = px + (dt / 3.0) * dx;
        let s1y = py + (dt / 3.0) * dy;
        let s2x = qx - (dt / 3.0) * d1x;
        let s2y = qy - (dt / 3.0) * d1y;
        let ctrl_x = (3.0 * (s1x + s2x) - (px + qx)) / 4.0;
        let ctrl_y = (3.0 * (s1y + s2y) - (py + qy)) / 4.0;
        let seg_dx = qx - px;
        let seg_dy = qy - py;
        let ctrl_dx = ctrl_x - px;
        let ctrl_dy = ctrl_y - py;
        let degenerate = seg_dx * seg_dx + seg_dy * seg_dy < DEGENERATE_EPSILON_SQ
            && ctrl_dx * ctrl_dx + ctrl_dy * ctrl_dy < DEGENERATE_EPSILON_SQ;
        if !degenerate {
            out.extend_from_slice(&[px, py, ctrl_x, ctrl_y, qx, qy]);
        }
        px = qx;
        py = qy;
        dx = d1x;
        dy = d1y;
    }
}

// ─────────────────── glyf outline reader (tables/glyf.js) ───────────────────

/// One parsed glyf point (glyf.js parseGlyph). Coordinates are f64 because
/// composite transforms (F2Dot14 scales, midpoints) leave the integer grid.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GlyfPoint {
    x: f64,
    y: f64,
    on_curve: bool,
    last_point_of_contour: bool,
}

/// Vendored glyf+loca slices, kept on `OtjsFont` for on-demand outline reads.
#[derive(Debug, Clone)]
pub(crate) struct GlyfOutlines {
    /// The glyf table bytes (offsets in `loca` are relative to this slice).
    glyf: Vec<u8>,
    /// `numGlyphs + 1` absolute-into-`glyf` offsets (short version pre-×2,
    /// tables/loca.js:20-25).
    loca: Vec<u32>,
}

fn u16at(d: &[u8], o: usize) -> Result<u16, String> {
    match (d.get(o), d.get(o + 1)) {
        (Some(a), Some(b)) => Ok(u16::from_be_bytes([*a, *b])),
        _ => Err(format!("glyf u16 read out of bounds at {o}")),
    }
}
fn i16at(d: &[u8], o: usize) -> Result<i16, String> {
    Ok(u16at(d, o)? as i16)
}
fn u8at(d: &[u8], o: usize) -> Result<u8, String> {
    d.get(o)
        .copied()
        .ok_or_else(|| format!("glyf u8 read out of bounds at {o}"))
}
/// `parseF2Dot14` (parse.js): i16 / 16384.
fn f2dot14at(d: &[u8], o: usize) -> Result<f64, String> {
    Ok(i16at(d, o)? as f64 / 16384.0)
}

impl GlyfOutlines {
    /// Parse loca (tables/loca.js) + slice the glyf table.
    pub(crate) fn parse(
        data: &[u8],
        glyf_off: usize,
        glyf_len: usize,
        loca_off: usize,
        num_glyphs: u16,
        short_version: bool,
    ) -> Result<GlyfOutlines, String> {
        let mut loca = Vec::with_capacity(num_glyphs as usize + 1);
        for i in 0..=(num_glyphs as usize) {
            let v = if short_version {
                u16be_raw(data, loca_off + i * 2)? as u32 * 2
            } else {
                u32be_raw(data, loca_off + i * 4)?
            };
            loca.push(v);
        }
        let glyf = data
            .get(glyf_off..glyf_off + glyf_len)
            .ok_or("glyf table truncated")?
            .to_vec();
        Ok(GlyfOutlines { glyf, loca })
    }

    /// Flattened outline points of one glyph (composites resolved), exactly
    /// like opentype.js's `glyph.points` after `buildPath`.
    pub(crate) fn glyph_points(&self, gid: u32, depth: u32) -> Result<Vec<GlyfPoint>, String> {
        if depth > 16 {
            // Deviation note (module header): cycle guard, untriggerable on
            // well-formed fonts (opentype.js has none and would overflow).
            return Err("composite glyph recursion too deep".to_string());
        }
        let i = gid as usize;
        let (Some(&off), Some(&next)) = (self.loca.get(i), self.loca.get(i + 1)) else {
            // glyphs.get(out-of-range) → undefined.getPath() TypeError in JS.
            return Err(format!("glyph index {gid} out of loca range"));
        };
        if off == next {
            // Empty glyph (space): glyphLoader → no points, empty path.
            return Ok(Vec::new());
        }
        let d = &self.glyf[..];
        let p = off as usize;
        let number_of_contours = i16at(d, p)?;
        // skip xMin/yMin/xMax/yMax (4 × i16)
        let mut pos = p + 10;

        if number_of_contours > 0 {
            // Simple glyph (glyf.js parseGlyph:46-110).
            let mut end_point_indices: Vec<u16> = Vec::new();
            for _ in 0..number_of_contours {
                end_point_indices.push(u16at(d, pos)?);
                pos += 2;
            }
            let instruction_length = u16at(d, pos)? as usize;
            pos += 2 + instruction_length;

            let number_of_coordinates = *end_point_indices.last().unwrap() as usize + 1;
            let mut flags: Vec<u8> = Vec::with_capacity(number_of_coordinates);
            {
                let mut i = 0usize;
                while i < number_of_coordinates {
                    let flag = u8at(d, pos)?;
                    pos += 1;
                    flags.push(flag);
                    if (flag & 8) > 0 {
                        let repeat = u8at(d, pos)? as usize;
                        pos += 1;
                        for _ in 0..repeat {
                            flags.push(flag);
                            i += 1;
                        }
                    }
                    i += 1;
                }
            }
            if flags.len() != number_of_coordinates {
                return Err("Bad flags.".to_string());
            }

            let mut points: Vec<GlyfPoint> = flags
                .iter()
                .enumerate()
                .map(|(i, f)| GlyfPoint {
                    x: 0.0,
                    y: 0.0,
                    on_curve: (f & 1) != 0,
                    last_point_of_contour: end_point_indices.contains(&(i as u16)),
                })
                .collect();
            // x coordinates (parseGlyphCoordinate, shortVectorBitMask 2, sameBitMask 16)
            let mut prev = 0i64;
            for (i, f) in flags.iter().enumerate() {
                let v = parse_glyph_coordinate(d, &mut pos, *f, prev, 2, 16)?;
                points[i].x = v as f64;
                prev = v;
            }
            // y coordinates (masks 4 / 32)
            let mut prev = 0i64;
            for (i, f) in flags.iter().enumerate() {
                let v = parse_glyph_coordinate(d, &mut pos, *f, prev, 4, 32)?;
                points[i].y = v as f64;
                prev = v;
            }
            Ok(points)
        } else if number_of_contours == 0 {
            Ok(Vec::new())
        } else {
            // Composite glyph (glyf.js parseGlyph:112-171 + buildPath:270-306).
            let mut points: Vec<GlyfPoint> = Vec::new();
            let mut more_components = true;
            while more_components {
                let flags = u16at(d, pos)?;
                pos += 2;
                let glyph_index = u16at(d, pos)?;
                pos += 2;
                let mut x_scale = 1.0f64;
                let mut scale01 = 0.0f64;
                let mut scale10 = 0.0f64;
                let mut y_scale = 1.0f64;
                let mut dx = 0.0f64;
                let mut dy = 0.0f64;
                let mut matched_points: Option<[i64; 2]> = None;
                if (flags & 1) > 0 {
                    if (flags & 2) > 0 {
                        dx = i16at(d, pos)? as f64;
                        dy = i16at(d, pos + 2)? as f64;
                    } else {
                        matched_points = Some([u16at(d, pos)? as i64, u16at(d, pos + 2)? as i64]);
                    }
                    pos += 4;
                } else {
                    if (flags & 2) > 0 {
                        dx = u8at(d, pos)? as i8 as f64;
                        dy = u8at(d, pos + 1)? as i8 as f64;
                    } else {
                        matched_points = Some([u8at(d, pos)? as i64, u8at(d, pos + 1)? as i64]);
                    }
                    pos += 2;
                }
                if (flags & 8) > 0 {
                    x_scale = f2dot14at(d, pos)?;
                    y_scale = x_scale;
                    pos += 2;
                } else if (flags & 64) > 0 {
                    x_scale = f2dot14at(d, pos)?;
                    y_scale = f2dot14at(d, pos + 2)?;
                    pos += 4;
                } else if (flags & 128) > 0 {
                    x_scale = f2dot14at(d, pos)?;
                    scale01 = f2dot14at(d, pos + 2)?;
                    scale10 = f2dot14at(d, pos + 4)?;
                    y_scale = f2dot14at(d, pos + 6)?;
                    pos += 8;
                }
                more_components = (flags & 32) != 0;

                let component_points = self.glyph_points(glyph_index as u32, depth + 1)?;
                if component_points.is_empty() {
                    continue; // `if (componentGlyph.points)` — empty stays empty
                }
                let transformed: Vec<GlyfPoint> = match matched_points {
                    None => transform_points(
                        &component_points,
                        x_scale,
                        scale01,
                        scale10,
                        y_scale,
                        dx,
                        dy,
                    ),
                    Some([m0, m1]) => {
                        // buildPath matched-points branch (glyf.js:283-300).
                        if m0 > points.len() as i64 - 1 || m1 > component_points.len() as i64 - 1 {
                            return Err("Matched points out of range in ".to_string());
                        }
                        let first_pt = points[m0 as usize];
                        let second = transform_points(
                            &[component_points[m1 as usize]],
                            x_scale,
                            scale01,
                            scale10,
                            y_scale,
                            0.0,
                            0.0,
                        )[0];
                        transform_points(
                            &component_points,
                            x_scale,
                            scale01,
                            scale10,
                            y_scale,
                            first_pt.x - second.x,
                            first_pt.y - second.y,
                        )
                    }
                };
                points.extend(transformed);
            }
            Ok(points)
        }
    }

    /// Font-unit path commands for one glyph — `getPath(glyph.points)`
    /// (glyf.js:212-268), y-up.
    pub(crate) fn glyph_commands(&self, gid: u32) -> Result<Vec<PathCommand>, String> {
        let points = self.glyph_points(gid, 0)?;
        Ok(commands_from_points(&points))
    }
}

/// `parseGlyphCoordinate` (glyf.js:10-33).
fn parse_glyph_coordinate(
    d: &[u8],
    pos: &mut usize,
    flag: u8,
    previous: i64,
    short_vector_bit_mask: u8,
    same_bit_mask: u8,
) -> Result<i64, String> {
    let v;
    if (flag & short_vector_bit_mask) > 0 {
        let mut b = u8at(d, *pos)? as i64;
        *pos += 1;
        if (flag & same_bit_mask) == 0 {
            b = -b;
        }
        v = previous + b;
    } else if (flag & same_bit_mask) > 0 {
        v = previous;
    } else {
        let delta = i16at(d, *pos)? as i64;
        *pos += 2;
        v = previous + delta;
    }
    Ok(v)
}

/// `transformPoints` (glyf.js:175-191), corrected to the OpenType spec.
///
/// The spec treats the point as a ROW vector against the composite matrix, so
/// the off-diagonal terms cross over relative to the file order
/// (`xScale, scale01, scale10, yScale`):
///
/// ```text
/// x' = xScale·x + scale10·y + dx
/// y' = scale01·x + yScale·y + dy
/// ```
///
/// opentype.js has these two swapped, which is invisible for a diagonal matrix
/// and wrong for every rotated or skewed component (see the module Deviations).
fn transform_points(
    points: &[GlyfPoint],
    x_scale: f64,
    scale01: f64,
    scale10: f64,
    y_scale: f64,
    dx: f64,
    dy: f64,
) -> Vec<GlyfPoint> {
    points
        .iter()
        .map(|pt| GlyfPoint {
            x: x_scale * pt.x + scale10 * pt.y + dx,
            y: scale01 * pt.x + y_scale * pt.y + dy,
            on_curve: pt.on_curve,
            last_point_of_contour: pt.last_point_of_contour,
        })
        .collect()
}

/// `getContours` + `getPath` (glyf.js:193-268).
fn commands_from_points(points: &[GlyfPoint]) -> Vec<PathCommand> {
    let mut commands: Vec<PathCommand> = Vec::new();
    let mut contour: Vec<GlyfPoint> = Vec::new();
    let mut contours: Vec<Vec<GlyfPoint>> = Vec::new();
    for pt in points {
        contour.push(*pt);
        if pt.last_point_of_contour {
            contours.push(std::mem::take(&mut contour));
        }
    }
    // check.argument(currentContour.length === 0) — a trailing unterminated
    // contour throws in JS; surface as no commands for it is unreachable on
    // well-formed fonts (lastPointOfContour comes from endPointIndices).
    for contour in &contours {
        let mut curr = contour[contour.len() - 1];
        let mut next = contour[0];

        if curr.on_curve {
            commands.push(PathCommand::M {
                x: curr.x,
                y: curr.y,
            });
        } else if next.on_curve {
            commands.push(PathCommand::M {
                x: next.x,
                y: next.y,
            });
        } else {
            commands.push(PathCommand::M {
                x: (curr.x + next.x) * 0.5,
                y: (curr.y + next.y) * 0.5,
            });
        }

        for i in 0..contour.len() {
            curr = next;
            next = contour[(i + 1) % contour.len()];
            if curr.on_curve {
                commands.push(PathCommand::L {
                    x: curr.x,
                    y: curr.y,
                });
            } else {
                let next2 = if !next.on_curve {
                    ((curr.x + next.x) * 0.5, (curr.y + next.y) * 0.5)
                } else {
                    (next.x, next.y)
                };
                commands.push(PathCommand::Q {
                    x1: curr.x,
                    y1: curr.y,
                    x: next2.0,
                    y: next2.1,
                });
            }
        }
        commands.push(PathCommand::Z);
    }
    commands
}

/// Scale font-unit commands to em-normalized, y-down, baseline-origin space —
/// `glyph.getPath(0, 0, 1)` (glyph.js:123-176): scale = `1/(unitsPerEm||1000)`,
/// x' = x·scale, y' = −y·scale.
pub(crate) fn em_normalize(commands: &[PathCommand], units_per_em: u16) -> Vec<PathCommand> {
    let upm = if units_per_em != 0 {
        units_per_em as f64
    } else {
        1000.0
    };
    let s = 1.0 / upm;
    // JS computes `x + (cmd.x * xScale)` / `y + (-cmd.y * yScale)` with
    // x = y = 0 — the additions normalize `-0` to `+0` (IEEE x + 0.0), which
    // the f32 bit-exact parity gate observes. Mirror the adds.
    let sx = |v: f64| 0.0 + v * s;
    let sy = |v: f64| 0.0 + (-v * s);
    commands
        .iter()
        .map(|c| match *c {
            PathCommand::M { x, y } => PathCommand::M { x: sx(x), y: sy(y) },
            PathCommand::L { x, y } => PathCommand::L { x: sx(x), y: sy(y) },
            PathCommand::Q { x1, y1, x, y } => PathCommand::Q {
                x1: sx(x1),
                y1: sy(y1),
                x: sx(x),
                y: sy(y),
            },
            PathCommand::C {
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => PathCommand::C {
                x1: sx(x1),
                y1: sy(y1),
                x2: sx(x2),
                y2: sy(y2),
                x: sx(x),
                y: sy(y),
            },
            PathCommand::Z => PathCommand::Z,
        })
        .collect()
}

// raw big-endian readers over the WHOLE font file (loca offsets)
fn u16be_raw(d: &[u8], o: usize) -> Result<u16, String> {
    match (d.get(o), d.get(o + 1)) {
        (Some(a), Some(b)) => Ok(u16::from_be_bytes([*a, *b])),
        _ => Err(format!("loca u16 read out of bounds at {o}")),
    }
}
fn u32be_raw(d: &[u8], o: usize) -> Result<u32, String> {
    if d.len() >= o + 4 {
        Ok(u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]))
    } else {
        Err(format!("loca u32 read out of bounds at {o}"))
    }
}

/// Map the glyf reader's String errors into the vector lane's `ShapeError`
/// (an outline failure unwinds renderWithVector's catch like any other
/// vector-path throw → whole-node MSDF fallback).
pub(crate) fn outline_error(e: String) -> ShapeError {
    ShapeError::Other(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_become_degenerate_quadratics_and_z_closes() {
        // Unit square, unclosed (no Z): closeContour at end emits the edge.
        let cmds = [
            PathCommand::M { x: 0.0, y: 0.0 },
            PathCommand::L { x: 1.0, y: 0.0 },
            PathCommand::L { x: 1.0, y: 1.0 },
            PathCommand::L { x: 0.0, y: 1.0 },
        ];
        let q = quadratics_from_commands(&cmds);
        assert_eq!(q.curve_count, 4); // 3 lines + closing edge
                                      // control point of the first line = midpoint
        assert_eq!(&q.points[0..6], &[0.0, 0.0, 0.5, 0.0, 1.0, 0.0]);
        assert_eq!((q.min_x, q.min_y, q.max_x, q.max_y), (0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn zero_length_segments_dropped() {
        let cmds = [
            PathCommand::M { x: 0.0, y: 0.0 },
            PathCommand::L { x: 0.0, y: 0.0 }, // degenerate
            PathCommand::L { x: 1.0, y: 0.0 },
            PathCommand::Z,
        ];
        let q = quadratics_from_commands(&cmds);
        assert_eq!(q.curve_count, 2); // line + closing edge
    }

    #[test]
    fn cubic_subdivision_count_matches_formula() {
        // A big cubic: err = √3/36·‖p3−3c2+3c1−p0‖.
        let (x0, y0, c1x, c1y, c2x, c2y, x1, y1) = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0);
        let ex: f64 = x1 - 3.0 * c2x + 3.0 * c1x - x0;
        let ey: f64 = y1 - 3.0 * c2y + 3.0 * c1y - y0;
        let err = (3.0f64.sqrt() / 36.0) * ex.hypot(ey);
        let n = ((err / CUBIC_TOLERANCE_EM).cbrt().ceil() as i64)
            .max(1)
            .min(MAX_CUBIC_SEGMENTS);
        let cmds = [
            PathCommand::M { x: x0, y: y0 },
            PathCommand::C {
                x1: c1x,
                y1: c1y,
                x2: c2x,
                y2: c2y,
                x: x1,
                y: y1,
            },
        ];
        let q = quadratics_from_commands(&cmds);
        // n cubic pieces + the closing edge back to the start
        assert_eq!(q.curve_count as i64, n + 1);
    }
}
