//! Rectangle corner styles (Illustrator audit §4, live corners) — a line-for-line port of
//! `packages/editor/src/core/geometry/rectCorners.ts`; both are pinned to `fixtures/rect-corners/cases.json`.
//!
//! A rectangle's `cornerStyle` (`inverted`, `chamfer`) cuts its corners the way a path's live corners do
//! (`path_geom::round_corners`), at the radii a round rectangle has (`normalize_rounded_rect_radii`): a 90° corner of
//! radius r is trimmed r along each edge and joined by a concave quarter circle centred on the corner (`inverted`) or
//! a straight cut (`chamfer`). `round` (the default) keeps the rectangle's own lanes (SDF, KAPPA corners,
//! smoothing); a styled rectangle has no SDF and draws this outline everywhere, like a smoothed one.

use crate::corner_smoothing::Seg;
use crate::path_geom::round_corners::CornerStyle;
use crate::shape_outline::{expand_corner_radii, normalize_rounded_rect_radii, KAPPA};
use crate::value::Value;

/// The corner style a rectangle draws with (`Round` for absent or unknown).
pub fn rect_corner_style(style: Option<&Value>) -> CornerStyle {
    style
        .and_then(Value::as_str)
        .and_then(CornerStyle::parse)
        .unwrap_or_default()
}

/// Whether a rectangle has styled (inverted / chamfer) corners: a non-round style and a rounded corner.
pub fn is_styled_rect(corner_radius: Option<&Value>, style: Option<&Value>) -> bool {
    rect_corner_style(style) != CornerStyle::Round
        && expand_corner_radii(corner_radius).iter().any(|r| *r > 0.0)
}

/// The styled outline in node-local space (origin top-left), clockwise from the top edge, closed by returning to
/// the start point. `corner_radius` is the raw stored value.
pub fn styled_rect_path(
    width: f64,
    height: f64,
    corner_radius: Option<&Value>,
    style: CornerStyle,
) -> Vec<Seg> {
    let (w, h) = (width, height);
    let radii = normalize_rounded_rect_radii(w, h, corner_radius);
    type V = (f64, f64);
    // Clockwise from top-right: (radius index, corner, incoming direction, outgoing direction).
    let corners: [(usize, V, V, V); 4] = [
        (1, (w, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (2, (w, h), (0.0, 1.0), (-1.0, 0.0)),
        (3, (0.0, h), (-1.0, 0.0), (0.0, -1.0)),
        (0, (0.0, 0.0), (0.0, -1.0), (1.0, 0.0)),
    ];
    let at = |k: V, d: V, s: f64| -> V { (k.0 + d.0 * s, k.1 + d.1 * s) };
    let mut out = Vec::with_capacity(9);
    let (i0, k0, u0, _) = corners[0];
    let start = at(k0, u0, -radii[i0]);
    out.push(Seg::Move(start.0, start.1));
    let mut cur = start;
    for (n, &(i, k, u, v)) in corners.iter().enumerate() {
        let r = radii[i];
        let to = at(k, v, r);
        if r > 0.0 {
            if style == CornerStyle::Chamfer {
                out.push(Seg::Line(to.0, to.1));
            } else {
                // Round: handles along the edges; inverted: across them (the quarter circle centred on the corner).
                let from = cur;
                let kr = KAPPA * r;
                let (d1, d2) = if style == CornerStyle::Inverted {
                    (v, u)
                } else {
                    (u, v)
                };
                out.push(Seg::Cubic {
                    cp1: (from.0 + d1.0 * kr, from.1 + d1.1 * kr),
                    cp2: (to.0 - d2.0 * kr, to.1 - d2.1 * kr),
                    to,
                });
            }
            cur = to;
        }
        let (ni, nk, nu, _) = corners[(n + 1) % 4];
        let next = if n == 3 { start } else { at(nk, nu, -radii[ni]) };
        if (next.0 - cur.0).hypot(next.1 - cur.1) > 1e-9 {
            out.push(Seg::Line(next.0, next.1));
        }
        cur = next;
    }
    out
}
