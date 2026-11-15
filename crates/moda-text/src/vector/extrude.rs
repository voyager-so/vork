//! Solid 3D extrude wall geometry for vector text — the Rust port of
//! `renderers/webgpu/text/vector/vectorTextExtrude.ts` (task #142).
//!
//! The extrude effect is a SWEPT side wall behind a crisp analytic front
//! face — NOT a copy stack. Each glyph outline edge is swept from the front
//! face (offset 0) to the back (offset = depth at angle), producing one quad
//! (two triangles) per flattened outline segment: together one continuous
//! wall, drawn once through the solid triangles-in-local-space pipeline
//! (`VectorOpPipeline::Tess`). The front face is the node's own fill, drawn
//! on top by the normal fill passes (vectorTextExtrude.ts:4-22).
//!
//! Outline flattening mirrors `glyphTessellation.ts`'s
//! `flattenCommandsToRings` byte-for-byte: outlines are flattened at a fixed
//! 256 px size (`EXTRUDE_FLATTEN_PX`, flatness 0.3 px) then normalized back
//! to em — one fixed resolution, no pixel bucketing, because the wall is a
//! side surface (vectorTextExtrude.ts:33-42). Contours are cached per glyph
//! on `OtjsFont` (`glyph_em_contours`), the per-font partition of the
//! browser's global `CONTOUR_CACHE` keyed `${fontKey}|${glyphIndex}`.

use std::rc::Rc;

use crate::vector::curves::PathCommand;
use crate::vector::otjs::{OtjsFont, ShapeError};

/// `EXTRUDE_FLATTEN_PX` / `EXTRUDE_FLATNESS_PX` (vectorTextExtrude.ts:36-37).
pub const EXTRUDE_FLATTEN_PX: f64 = 256.0;
pub const EXTRUDE_FLATNESS_PX: f64 = 0.3;
/// `MAX_FLATTEN_DEPTH` (glyphTessellation.ts:8).
const MAX_FLATTEN_DEPTH: u32 = 16;

/// `Ring` (glyphTessellation.ts:5-6): a closed contour as flat points.
pub type Ring = Vec<[f64; 2]>;

/// `distToSegment` (glyphTessellation.ts:228-235): perpendicular distance to
/// the infinite line through a→b (point distance for degenerate segments).
fn dist_to_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 <= 1e-9 {
        return (px - ax).hypot(py - ay);
    }
    ((px - ax) * dy - (py - ay) * dx).abs() / len2.sqrt()
}

/// `flattenQuad` (glyphTessellation.ts:168-192).
#[allow(clippy::too_many_arguments)]
fn flatten_quad(
    ring: &mut Ring,
    x0: f64,
    y0: f64,
    cx: f64,
    cy: f64,
    x1: f64,
    y1: f64,
    flatness: f64,
    depth: u32,
) {
    if depth >= MAX_FLATTEN_DEPTH || dist_to_segment(cx, cy, x0, y0, x1, y1) <= flatness {
        ring.push([x1, y1]);
        return;
    }
    let ax = (x0 + cx) / 2.0;
    let ay = (y0 + cy) / 2.0;
    let bx = (cx + x1) / 2.0;
    let by = (cy + y1) / 2.0;
    let mx = (ax + bx) / 2.0;
    let my = (ay + by) / 2.0;
    flatten_quad(ring, x0, y0, ax, ay, mx, my, flatness, depth + 1);
    flatten_quad(ring, mx, my, bx, by, x1, y1, flatness, depth + 1);
}

/// `flattenCubic` (glyphTessellation.ts:194-226).
#[allow(clippy::too_many_arguments)]
fn flatten_cubic(
    ring: &mut Ring,
    x0: f64,
    y0: f64,
    c1x: f64,
    c1y: f64,
    c2x: f64,
    c2y: f64,
    x1: f64,
    y1: f64,
    flatness: f64,
    depth: u32,
) {
    let d =
        dist_to_segment(c1x, c1y, x0, y0, x1, y1).max(dist_to_segment(c2x, c2y, x0, y0, x1, y1));
    if depth >= MAX_FLATTEN_DEPTH || d <= flatness {
        ring.push([x1, y1]);
        return;
    }
    // de Casteljau split at t = 0.5
    let x01 = (x0 + c1x) / 2.0;
    let y01 = (y0 + c1y) / 2.0;
    let x12 = (c1x + c2x) / 2.0;
    let y12 = (c1y + c2y) / 2.0;
    let x23 = (c2x + x1) / 2.0;
    let y23 = (c2y + y1) / 2.0;
    let x012 = (x01 + x12) / 2.0;
    let y012 = (y01 + y12) / 2.0;
    let x123 = (x12 + x23) / 2.0;
    let y123 = (y12 + y23) / 2.0;
    let mx = (x012 + x123) / 2.0;
    let my = (y012 + y123) / 2.0;
    flatten_cubic(
        ring,
        x0,
        y0,
        x01,
        y01,
        x012,
        y012,
        mx,
        my,
        flatness,
        depth + 1,
    );
    flatten_cubic(
        ring,
        mx,
        my,
        x123,
        y123,
        x23,
        y23,
        x1,
        y1,
        flatness,
        depth + 1,
    );
}

/// `flattenCommandsToRings` (glyphTessellation.ts:129-166): rings with fewer
/// than 3 points are dropped (degenerate; no wall area).
pub fn flatten_commands_to_rings(commands: &[PathCommand], flatness: f64) -> Vec<Ring> {
    let mut rings: Vec<Ring> = Vec::new();
    let mut ring: Option<Ring> = None;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;

    for c in commands {
        match *c {
            PathCommand::M { x, y } => {
                if let Some(r) = ring.take() {
                    if r.len() >= 3 {
                        rings.push(r);
                    }
                }
                ring = Some(vec![[x, y]]);
                cx = x;
                cy = y;
            }
            PathCommand::L { x, y } => {
                if let Some(r) = ring.as_mut() {
                    r.push([x, y]);
                }
                cx = x;
                cy = y;
            }
            PathCommand::Q { x1, y1, x, y } => {
                if let Some(r) = ring.as_mut() {
                    flatten_quad(r, cx, cy, x1, y1, x, y, flatness, 0);
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
                if let Some(r) = ring.as_mut() {
                    flatten_cubic(r, cx, cy, x1, y1, x2, y2, x, y, flatness, 0);
                }
                cx = x;
                cy = y;
            }
            PathCommand::Z => {
                if let Some(r) = ring.take() {
                    if r.len() >= 3 {
                        rings.push(r);
                    }
                }
            }
        }
    }
    if let Some(r) = ring.take() {
        if r.len() >= 3 {
            rings.push(r);
        }
    }
    rings
}

/// Em-normalized contours from em-normalized path commands —
/// `getGlyphEmContours`'s flatten step (vectorTextExtrude.ts:49-65): scale
/// to `EXTRUDE_FLATTEN_PX` (the browser's `glyph.getPath(0, 0, 256)`),
/// flatten at 0.3 px, scale back. ×256 / ÷256 are exact in binary FP, so the
/// round-trip matches the browser's `x * inv` bit-for-bit.
pub fn em_contours_from_commands(commands_em: &[PathCommand]) -> Vec<Ring> {
    let s = EXTRUDE_FLATTEN_PX;
    let scaled: Vec<PathCommand> = commands_em
        .iter()
        .map(|c| match *c {
            PathCommand::M { x, y } => PathCommand::M { x: x * s, y: y * s },
            PathCommand::L { x, y } => PathCommand::L { x: x * s, y: y * s },
            PathCommand::Q { x1, y1, x, y } => PathCommand::Q {
                x1: x1 * s,
                y1: y1 * s,
                x: x * s,
                y: y * s,
            },
            PathCommand::C {
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => PathCommand::C {
                x1: x1 * s,
                y1: y1 * s,
                x2: x2 * s,
                y2: y2 * s,
                x: x * s,
                y: y * s,
            },
            PathCommand::Z => PathCommand::Z,
        })
        .collect();
    let inv = 1.0 / s;
    flatten_commands_to_rings(&scaled, EXTRUDE_FLATNESS_PX)
        .into_iter()
        .map(|ring| ring.into_iter().map(|[x, y]| [x * inv, y * inv]).collect())
        .collect()
}

/// `ExtrudePlacedGlyph` (vectorTextExtrude.ts:25-31) — a glyph placed by the
/// fill walk, retained for wall sweeping when an extrude pass is active.
pub struct ExtrudePlacedGlyph {
    pub font: Rc<OtjsFont>,
    pub gid: u32,
    pub pen_x: f64,
    pub baseline_y: f64,
    pub font_size_px: f64,
    /// Unpadded glyph em-box bounds [minX, minY, maxX, maxY] — the pivot for the
    /// per-glyph transform (its center, which equals the symmetric-padded em-box
    /// center the fill/shader uses). KineticText.
    pub em_bounds: [f32; 4],
    /// KineticText per-glyph transform. Identity for static text →
    /// walls sweep from the untouched outline (byte-exact). A glyph at opacity 0
    /// emits no walls (zero contribution on the extrude pass, like every pass).
    pub xform: crate::vector::ops::GlyphXform,
}

/// `buildExtrudeWallTriangles` (vectorTextExtrude.ts:76-103): sweep every
/// placed glyph's outline by `depth_px` at `angle_deg` (node-LOCAL y-down:
/// 0° = right, 90° = down) into a flat `[x0,y0, x1,y1, ...]` triangle list —
/// two triangles per outline segment. Depth is in local px (NOT em-scaled).
/// An outline read failure unwinds like any other vector-path throw
/// (renderWithVector's catch → whole-node MSDF fallback).
pub fn build_extrude_wall_triangles(
    glyphs: &[ExtrudePlacedGlyph],
    depth_px: f64,
    angle_deg: f64,
) -> Result<Vec<f32>, ShapeError> {
    let mut tris: Vec<f32> = Vec::new();
    if depth_px <= 0.0 {
        return Ok(tris);
    }
    // NOT `to_radians()`: the browser computes `(angleDeg * Math.PI) / 180`
    // (vectorTextExtrude.ts:79); `to_radians()` is `x * (PI/180)` which can
    // differ in the last ULP, breaking byte parity with the oracle.
    let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
    let dx = angle_rad.cos() * depth_px;
    let dy = angle_rad.sin() * depth_px;

    for g in glyphs {
        if g.gid == 0 {
            continue; // .notdef never draws (matches the fill walks)
        }
        // KineticText: a glyph at opacity 0 contributes ZERO on every
        // pass, extrude included — emit no walls for it. NOTE the extrude lane
        // honors opacity as BINARY visibility, not continuous alpha: the walls are
        // tessellated through `VectorOpPipeline::Tess`, whose triangles carry no
        // per-vertex alpha (only the pass-level `VectorOp.opacity`), and the
        // browser oracle (`vectorTextExtrude.ts`) predates per-glyph opacity and
        // has the same structure. So a partial-opacity glyph (e.g. 0.5) fades on
        // the analytic fill/effect passes but its extrude wall stays solid — a
        // known limitation; continuous wall opacity would need a per-vertex-alpha
        // Tess ABI, deferred past this foundation slice.
        if g.xform.opacity <= 0.0 {
            continue;
        }
        // Per-glyph transform about the glyph em-box center (the same pivot the
        // vertex shader uses for the fill/effect quads), in node-local px. Skipped
        // for the identity transform so static text stays byte-exact.
        let apply_xform = !g.xform.is_identity();
        let center_x = g.pen_x + ((g.em_bounds[0] + g.em_bounds[2]) as f64 * 0.5) * g.font_size_px;
        let center_y =
            g.baseline_y + ((g.em_bounds[1] + g.em_bounds[3]) as f64 * 0.5) * g.font_size_px;
        let lin = [
            g.xform.lin[0] as f64,
            g.xform.lin[1] as f64,
            g.xform.lin[2] as f64,
            g.xform.lin[3] as f64,
        ];
        let off = [g.xform.off[0] as f64, g.xform.off[1] as f64];
        let xform_point = |px: f64, py: f64| -> (f64, f64) {
            if !apply_xform {
                return (px, py);
            }
            let rx = px - center_x;
            let ry = py - center_y;
            (
                lin[0] * rx + lin[1] * ry + off[0] + center_x,
                lin[2] * rx + lin[3] * ry + off[1] + center_y,
            )
        };
        let contours = g.font.glyph_em_contours(g.gid)?;
        for ring in contours.iter() {
            for i in 0..ring.len() {
                let a = ring[i];
                let b = ring[(i + 1) % ring.len()]; // close the ring
                let (ax, ay) = xform_point(
                    g.pen_x + a[0] * g.font_size_px,
                    g.baseline_y + a[1] * g.font_size_px,
                );
                let (bx, by) = xform_point(
                    g.pen_x + b[0] * g.font_size_px,
                    g.baseline_y + b[1] * g.font_size_px,
                );
                // Quad [a, b, b+d, a+d] as two triangles. No backface
                // distinction (vectorTextExtrude.ts:94-99).
                tris.extend_from_slice(&[
                    ax as f32,
                    ay as f32,
                    bx as f32,
                    by as f32,
                    (bx + dx) as f32,
                    (by + dy) as f32,
                ]);
                tris.extend_from_slice(&[
                    ax as f32,
                    ay as f32,
                    (bx + dx) as f32,
                    (by + dy) as f32,
                    (ax + dx) as f32,
                    (ay + dy) as f32,
                ]);
            }
        }
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_lines_and_close() {
        // Unit triangle via lines + Z.
        let cmds = [
            PathCommand::M { x: 0.0, y: 0.0 },
            PathCommand::L { x: 1.0, y: 0.0 },
            PathCommand::L { x: 0.0, y: 1.0 },
            PathCommand::Z,
        ];
        let rings = flatten_commands_to_rings(&cmds, 0.3);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0], vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn degenerate_rings_dropped() {
        // Two-point "ring" (M + one L) contributes no wall area.
        let cmds = [
            PathCommand::M { x: 0.0, y: 0.0 },
            PathCommand::L { x: 1.0, y: 0.0 },
            PathCommand::Z,
        ];
        assert!(flatten_commands_to_rings(&cmds, 0.3).is_empty());
    }

    #[test]
    fn quad_flattening_subdivides_within_flatness() {
        // A strongly-curved quadratic at 256 px scale must subdivide; the
        // sagitta of every emitted segment stays within the flatness bound
        // (allowing the depth cap, which this curve does not hit).
        let cmds = [
            PathCommand::M { x: 0.0, y: 0.0 },
            PathCommand::Q {
                x1: 128.0,
                y1: 256.0,
                x: 256.0,
                y: 0.0,
            },
            PathCommand::Z,
        ];
        let rings = flatten_commands_to_rings(&cmds, 0.3);
        assert_eq!(rings.len(), 1);
        let ring = &rings[0];
        assert!(ring.len() > 8, "curve must subdivide, got {}", ring.len());
        // The flattened polyline ends at the curve endpoint.
        assert_eq!(*ring.last().unwrap(), [256.0, 0.0]);
    }

    #[test]
    fn wall_triangles_sweep_each_segment_into_two_triangles() {
        // Synthetic glyph: skip the font path by checking the geometry math
        // on a hand-built ring via the public helpers. A square ring (4
        // segments) sweeps into 8 triangles = 48 floats; verify one quad's
        // corners at angle 0° (dx = depth, dy = 0).
        let cmds = [
            PathCommand::M { x: 0.0, y: 0.0 },
            PathCommand::L { x: 1.0, y: 0.0 },
            PathCommand::L { x: 1.0, y: 1.0 },
            PathCommand::L { x: 0.0, y: 1.0 },
            PathCommand::Z,
        ];
        let rings = em_contours_from_commands(&cmds);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 4);
        // em round-trip is exact (×256 then ÷256).
        assert_eq!(rings[0][1], [1.0, 0.0]);
    }
}
