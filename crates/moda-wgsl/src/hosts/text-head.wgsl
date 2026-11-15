// Common constants

// Mathematical constants
const INVERSE_SQRT_2: f32 = 0.7071067811865476;

// Shader constants
const AA_PIXEL_EXPANSION: f32 = 0.5;
const MIN_BLUR_RADIUS: f32 = 0.01;
const MIN_GRADIENT_STOP_DIFF: f32 = 0.0001;


// Common color utilities

// Premultiply RGB by alpha for correct blending with premultiplied alpha mode
fn premultiplyAlpha(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

// Convert from sRGB to linear color space
fn srgbToLinear(srgb: vec3<f32>) -> vec3<f32> {
    return pow(srgb, vec3<f32>(2.2));
}

// Convert from linear to sRGB color space
fn linearToSrgb(linear: vec3<f32>) -> vec3<f32> {
    return pow(linear, vec3<f32>(1.0 / 2.2));
}

// Luminance calculation using standard weights
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// Desaturate a color by a given amount (0 = full color, 1 = grayscale)
fn desaturate(color: vec3<f32>, amount: f32) -> vec3<f32> {
    let gray = vec3<f32>(luminance(color));
    return mix(color, gray, amount);
}

// Apply opacity to a color (non-premultiplied)
fn applyOpacity(color: vec4<f32>, opacity: f32) -> vec4<f32> {
    return vec4<f32>(color.rgb, color.a * opacity);
}

// Blend two colors using normal blend mode
fn blendNormal(base: vec4<f32>, blend: vec4<f32>) -> vec4<f32> {
    let alpha = blend.a + base.a * (1.0 - blend.a);
    let rgb = (blend.rgb * blend.a + base.rgb * base.a * (1.0 - blend.a)) / alpha;
    return vec4<f32>(rgb, alpha);
}


// Main shader code

// Vector shader fill: view-projection (vertex) + shared shader ABI + glyph curves.
@group(0) @binding(0) var<uniform> frame: mat4x4<f32>;

struct CustomShaderData {
  model: mat4x4<f32>,
  fillColor: vec4<f32>,
  strokeColor: vec4<f32>,
  pathInfo: vec4<f32>,
  bounds: vec4<f32>,
  pageClipBounds: vec4<f32>,
  shaderParams: array<vec4<f32>, 16>,
  colorData: array<vec4<f32>, 16>,
}

@group(1) @binding(0) var<storage, read> paths: array<CustomShaderData>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) instanceId: u32,
  @location(1) @interpolate(flat) isStroke: f32,
  @location(2) localPos: vec2<f32>,
  @location(3) worldPos: vec2<f32>,
}

// Quadratic outline segments: 3 control points per curve, em units (y-down, pen origin).
@group(2) @binding(0) var<storage, read> curvePoints: array<vec2<f32>>;

struct VectorShadeVSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec2<f32>,
  @location(1) em: vec2<f32>,
  @location(2) @interpolate(flat) curveRange: vec2<u32>, // firstCurve, curveCount
  @location(3) local: vec2<f32>, // text-box-local glyph position: continuous across glyphs
  @location(4) @interpolate(flat) glyphOpacity: f32, // KineticText per-glyph opacity
};


fn kineticGlyphLocal(
  pen: vec2<f32>, em: vec2<f32>, emBox: vec4<f32>, fontSizePx: f32,
  glyphLin: vec4<f32>, glyphOff: vec2<f32>,
) -> vec2<f32> {
  let baseLocal = pen + em * fontSizePx;
  if (glyphLin.x == 1.0 && glyphLin.y == 0.0 && glyphLin.z == 0.0 && glyphLin.w == 1.0 &&
      glyphOff.x == 0.0 && glyphOff.y == 0.0) {
    return baseLocal;
  }
  let center = pen + (emBox.xy + emBox.zw) * 0.5 * fontSizePx;
  let rel = baseLocal - center;
  return vec2<f32>(
    glyphLin.x * rel.x + glyphLin.y * rel.y + glyphOff.x,
    glyphLin.z * rel.x + glyphLin.w * rel.y + glyphOff.y,
  ) + center;
}

// NOTE: shader-fill text does NOT apply the node distort warp: its
// vertex stage reads the shared CustomShaderData ABI (paths[0]), which has no
// homography fields. Distort-warped shader-fill nodes render unwarped — the same
// accepted gap as MSDF text.
@vertex fn vs(
  @location(0) corner: vec2<f32>,      // unit quad corner (0..1)
  @location(1) pen: vec2<f32>,         // local-space pen position (x, baselineY)
  @location(2) fontSizePx: f32,
  @location(3) emBox: vec4<f32>,       // padded glyph bounds in em units
  @location(4) curveRange: vec2<u32>,
  @location(5) glyphLin: vec4<f32>,    // KineticText per-glyph 2×2 [a,c,b,d]
  @location(6) glyphOff: vec2<f32>,    // KineticText per-glyph px offset
  @location(7) glyphOpacity: f32,      // KineticText per-glyph opacity
) -> VectorShadeVSOut {
  let path = paths[0];
  let em = mix(emBox.xy, emBox.zw, corner);
  let local = kineticGlyphLocal(pen, em, emBox, fontSizePx, glyphLin, glyphOff);
  let world = path.model * vec4<f32>(local, 0.0, 1.0);
  var o: VectorShadeVSOut;
  o.pos = frame * world;
  o.pos.z = path.pathInfo.w; // enforce scene depth (text z-ordering), matches solid/gradient text
  o.world = world.xy;
  o.em = em;
  o.curveRange = curveRange;
  o.local = local;
  o.glyphOpacity = glyphOpacity;
  return o;
}

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


