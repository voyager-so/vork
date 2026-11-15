// Shared vector text coverage core. build.rs splices it in place of the include line
// (`#include` plus this file's name, behind a line comment) in the analytic, -additive and
// -gradient system shaders and every <shader>.vector-text-fill module: one copy, so a fix
// here reaches all 40. The module must declare `curvePoints: array<vec2<f32>>` (3 control
// points per quadratic, em units).

// Signed winding contribution of one quadratic segment to the pixel at the origin,
// along the +x ray, with a linear AA filter over the pixel footprint (1/invDiameter
// wide): each y(t)=0 crossing counts by how far across the pixel it sits.
//
// WHICH crossings count is decided from the signs of the three control points' y
// (Lengyel's Slug classification, the 0x2E74 table), never from the computed roots:
// a vertex shared by two segments has one y, so it is classified the same way for
// both and the ray crosses it exactly once. Testing the roots against [0, 1) instead
// let each segment round its own root at a shared vertex — for a pixel centre level
// with an outline point (at 64 px, 1/32 of Inter's 2048-unit grid lands on a pixel
// centre, so stem corners and extrema often are) one segment counted the crossing and
// the other did too, or neither did, and the whole pixel column (or row) above the
// vertex read as half covered: 1 px hairlines above B, d, i, l, m.
fn vectorWindingAlongRay(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, invDiameter: f32) -> f32 {
  // Bit 0: the first root crosses (+), bit 1: the second root crosses (−). A point
  // exactly on the ray counts as below it (y ≤ 0), for every segment alike.
  let shift = select(0u, 2u, p0.y > 0.0) + select(0u, 4u, p1.y > 0.0) + select(0u, 8u, p2.y > 0.0);
  let code = (0x2E74u >> shift) & 3u;
  if (code == 0u) { return 0.0; }

  // y(t) = a.y·t² − 2·b.y·t + c.y (a factor of −2 extracted from the middle term).
  let a = p0 - 2.0 * p1 + p2;
  let b = p0 - p1;
  let c = p0;

  var t0: f32;
  var t1: f32;
  if (abs(a.y) >= 1e-5) {
    // A radicand rounded below zero is a grazing ray: both roots meet and cancel.
    let s = sqrt(max(b.y * b.y - a.y * c.y, 0.0));
    t0 = (b.y - s) / a.y;
    t1 = (b.y + s) / a.y;
  } else {
    // Degenerate-quadratic (line) branch. The classification only selects a crossing
    // when the endpoints straddle the ray, so b.y (half of p0.y − p2.y) is not zero.
    t0 = c.y / (2.0 * b.y);
    t1 = t0;
  }

  var alpha = 0.0;
  if ((code & 1u) != 0u) {
    let x = (a.x * t0 - 2.0 * b.x) * t0 + c.x;
    alpha = alpha + clamp(x * invDiameter + 0.5, 0.0, 1.0);
  }
  if ((code & 2u) != 0u) {
    let x = (a.x * t1 - 2.0 * b.x) * t1 + c.x;
    alpha = alpha - clamp(x * invDiameter + 0.5, 0.0, 1.0);
  }
  return alpha;
}

// Fractional winding coverage of one glyph at the pixel centered on `em` (em units).
// Two perpendicular rays are averaged so both vertical-ish and horizontal-ish edges
// get filtered; abs() makes coverage winding-direction agnostic (some fonts wind
// contours backwards) and clamping resolves overlapping contours (|winding| ≥ 1
// fills, with no conflation seam where contours overlap).
fn vectorGlyphCoverage(first: u32, count: u32, em: vec2<f32>, invDiameter: vec2<f32>) -> f32 {
  var windingX = 0.0; // +x ray: anti-aliases vertical-ish edges
  var windingY = 0.0; // +y ray: anti-aliases horizontal-ish edges
  for (var i = 0u; i < count; i = i + 1u) {
    let base = (first + i) * 3u;
    let p0 = curvePoints[base] - em;
    let p1 = curvePoints[base + 1u] - em;
    let p2 = curvePoints[base + 2u] - em;
    windingX = windingX + vectorWindingAlongRay(p0, p1, p2, invDiameter.x);
    windingY = windingY + vectorWindingAlongRay(
      vec2<f32>(p0.y, -p0.x), vec2<f32>(p1.y, -p1.x), vec2<f32>(p2.y, -p2.x), invDiameter.y);
  }
  return clamp(0.5 * (abs(windingX) + abs(windingY)), 0.0, 1.0);
}
