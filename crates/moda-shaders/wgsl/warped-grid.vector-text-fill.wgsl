// Common constants

// Mathematical constants
const INVERSE_SQRT_2: f32 = 0.7071067811865476;

// Shader constants
const AA_PIXEL_EXPANSION: f32 = 0.5;
const MIN_BLUR_RADIUS: f32 = 0.01;
const MIN_GRADIENT_STOP_DIFF: f32 = 0.0001;


// Common geometry utilities

// Generate a unit quad (0,0) to (1,1) - used for fill rendering
fn getUnitQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(0.0, 0.0); }
    case 1u: { pos = vec2<f32>(1.0, 0.0); }
    case 2u: { pos = vec2<f32>(0.0, 1.0); }
    case 3u: { pos = vec2<f32>(0.0, 1.0); }
    case 4u: { pos = vec2<f32>(1.0, 0.0); }
    default: { pos = vec2<f32>(1.0, 1.0); }
  }
  return pos;
}

// Generate a normalized quad (-1,-1) to (1,1) - used for stroke rendering
fn getNormalizedQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(-1.0, -1.0); }
    case 1u: { pos = vec2<f32>( 1.0, -1.0); }
    case 2u: { pos = vec2<f32>(-1.0,  1.0); }
    case 3u: { pos = vec2<f32>(-1.0,  1.0); }
    case 4u: { pos = vec2<f32>( 1.0, -1.0); }
    default: { pos = vec2<f32>( 1.0,  1.0); }
  }
  return pos;
}

// Simple 2x2 rotation helper
fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let s = sin(angle);
  let c = cos(angle);
  return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Generate UV coordinates for a quad (0,0) to (1,1)
fn getQuadUV(vertexIndex: u32) -> vec2<f32> {
  var uv: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { uv = vec2<f32>(0.0, 0.0); }
    case 1u: { uv = vec2<f32>(1.0, 0.0); }
    case 2u: { uv = vec2<f32>(0.0, 1.0); }
    case 3u: { uv = vec2<f32>(0.0, 1.0); }
    case 4u: { uv = vec2<f32>(1.0, 0.0); }
    default: { uv = vec2<f32>(1.0, 1.0); }
  }
  return uv;
}

// Generate a centered unit quad (-0.5,-0.5) to (0.5,0.5)
fn getCenteredQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(-0.5, -0.5); }
    case 1u: { pos = vec2<f32>( 0.5, -0.5); }
    case 2u: { pos = vec2<f32>(-0.5,  0.5); }
    case 3u: { pos = vec2<f32>(-0.5,  0.5); }
    case 4u: { pos = vec2<f32>( 0.5, -0.5); }
    default: { pos = vec2<f32>( 0.5,  0.5); }
  }
  return pos;
}

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


// Common noise utilities

// Integer-exact hash core (lowbias32). u32 ops are bit-exact across WGSL
// compilers (Tint/Naga) and devices (GPU and CPU/lavapipe), unlike the old
// fract(sin(x) * K) hash whose sin() ULP differences amplified into a
// per-pixel grain divergence between backends.
fn hashU32(x: u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

// Top 24 bits -> [0, 1): exactly representable in f32, no rounding.
fn hashToUnit(h: u32) -> f32 {
  return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Hash-based 2D value noise with smooth interpolation
fn hash21(p: vec2<f32>) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(p.x) ^ hashU32(bitcast<u32>(p.y))));
}

fn valueNoise(p: vec2<f32>) -> f32 {
  let ip = floor(p);
  let fp = fract(p);
  let a = hash21(ip);
  let b = hash21(ip + vec2<f32>(1.0, 0.0));
  let c = hash21(ip + vec2<f32>(0.0, 1.0));
  let d = hash21(ip + vec2<f32>(1.0, 1.0));
  let u = fp * fp * (3.0 - 2.0 * fp);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
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

// #include "vector-text-coverage.wgsl"


// Rotate helper
fn rot2(a: f32) -> mat2x2<f32> { let s = sin(a); let c = cos(a); return mat2x2<f32>(c,-s,s,c); }

// Distance to nearest grid line using reciprocals (fewer divisions)
fn grid_distance(p: vec2<f32>, spacing: vec2<f32>) -> f32 {
  let invS = 1.0 / max(spacing, vec2<f32>(1e-4));
  let fx = fract(p.x * invS.x);
  let fy = fract(p.y * invS.y);
  let dx = abs(fx * spacing.x - 0.5 * spacing.x);
  let dy = abs(fy * spacing.y - 0.5 * spacing.y);
  return min(dx, dy);
}

// Value noise from common include
fn value2(p: vec2<f32>) -> f32 { return valueNoise(p); }

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  var uv = (input.localPos - vec2<f32>(b.x, b.y)) / vec2<f32>(b.z, b.w);
  uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));

  // Params
  let spacing = vec2<f32>(max(path.shaderParams[0].x, 1e-4), max(path.shaderParams[0].y, 1e-4));
  let thickness = max(path.shaderParams[0].z, 1e-5);
  let time = path.shaderParams[0].w;

  let aaWidth = max(path.shaderParams[1].x, 1e-5);
  let noiseAmp = path.shaderParams[1].y;
  let noiseFreq = path.shaderParams[1].z;
  let warpCount = u32(clamp(path.shaderParams[1].w, 0.0, 3.0) + 0.5);

  let seed = path.shaderParams[2].x;

  // Build displacement field — simplified sine-warp with lightweight swirl for richer motion
  var p = uv;

  // Derive simple parameters. If a warp is provided, reuse its amp/speed as controls.
  var ampUV = min(spacing.x, spacing.y) * 2.2;
  var freq = 2.0;
  var spd = 1.2;
  var phase = seed * 3.1;
  if (warpCount > 0u) {
    let row0 = path.shaderParams[8u];
    let row1 = path.shaderParams[9u];
    let radius = max(row0.z, 1e-4);
    let amp = row0.w;
    ampUV = max(amp, 1e-4) * min(spacing.x, spacing.y) * 5.0;
    freq = max(1.2, 1.0 / radius * 1.1);
    spd = max(0.35, row1.x * 1.2);
    phase = row1.w + seed * 1.7;
  }

  let TAU = 6.28318530718;
  // Seed-driven randomizers for variety without heavy noise. Integer-exact streams
  // (hashU32/hashToUnit from the noise include): bit-exact across WGSL compilers and
  // devices, unlike the old fract(sin(x) * K) hash.
  let seedBits = hashU32(bitcast<u32>(seed));
  let r1 = hashToUnit(hashU32(seedBits ^ 1u));
  let r2 = hashToUnit(hashU32(seedBits ^ 2u));
  let r3 = hashToUnit(hashU32(seedBits ^ 3u));
  let r4 = hashToUnit(hashU32(seedBits ^ 4u));

  // Precompute per-orb random values (loop-invariant)
  let tight = fma(5.0, r3, 5.5);
  var orbCoreR: array<f32, 6>;
  var orbSign: array<f32, 6>;
  for (var j: u32 = 0u; j < 6u; j = j + 1u) {
    let jBits = hashU32((j + 1u) * 0x9e3779b9u) ^ seedBits;
    let coreRand = hashToUnit(hashU32(jBits ^ 5u));
    orbCoreR[j] = fma(0.05, coreRand, 0.05);
    let signRand = hashToUnit(hashU32(jBits ^ 6u));
    orbSign[j] = fma(signRand, 2.0, -1.0);
  }

  // Localized moving orbs (radial bulge/sink) for wandering deformation
  var orbDisp = vec2<f32>(0.0, 0.0);
  for (var i: u32 = 0u; i < 6u; i = i + 1u) {
    let k = f32(i);
    let aTime = fma(0.25, r1, fma(0.17, k, 0.60));
    let aPhase = fma(0.20, k, 1.00);
    let a = fma(time, aTime, phase * aPhase);
    let bTime = fma(0.20, r2, fma(0.13, k, 0.50));
    let bPhase = fma(0.15, k, 0.80);
    let bVal = fma(time, bTime, phase * bPhase);
    // Smooth Lissajous-like paths inside the domain
    let cx = fma(0.38, sin(fma(r3, 2.0, fma(k, 1.7, a))), 0.5);
    let cy = fma(0.38, cos(fma(r4, 2.1, fma(k, 1.3, bVal))), 0.5);
    let c = vec2<f32>(cx, cy);
    let d2 = p - c;
    let r2v = dot(d2, d2);
    let dir = d2 * inverseSqrt(r2v + 1e-6);

    // Core-softened radial kernel: simplified to single division
    let coreR = orbCoreR[i];
    let core2 = coreR * coreR;
    let env = r2v / ((r2v + core2) * fma(r2v, tight, 1.0));

    let sgn = orbSign[i];
    let breatheArg = fma(time, fma(0.25, k, 0.70), phase * fma(0.35, k, 1.10));
    let breathe = fma(0.30, sin(breatheArg), 0.70);
    let strength = ampUV * 4.5 * breathe;
    orbDisp = fma(dir, vec2<f32>(sgn * strength * env), orbDisp);
  }
  p += orbDisp;

  // Noise jitter disabled while previewing swirls only

  // Distance to nearest grid line after warp
  // Scale spacing to pixel space for consistent thickness
  let pxScale = vec2<f32>(b.z, b.w);
  let pp = p * pxScale;
  let pSpacing = spacing * pxScale;
  let dist = grid_distance(pp, pSpacing);
  // thickness and aaWidth are in pixels
  // Interpret thickness/aaWidth as pixel units
  let pxThickness = max(thickness, 1.2);
  let pxAA = max(aaWidth, 0.7);
  // Both returns below are the bespoke premultiplied composite from the hand-written shader: the RGB is
  // already folded over the background (e.g. bg.rgb * bg.a), NOT straight-alpha — do not normalize by outA.
  // The epilogue applies fillColor.a + page-clip + clip-mask and premultiplies on top, reproducing the
  // original output (including the bg.a^2 background quirk).
  // Early background path: outside line AA band → return background color.
  if (dist > pxThickness + pxAA) {
    let bg = path.colorData[14];
    return vec4<f32>(bg.rgb * bg.a, bg.a);
  }
  let alpha = 1.0 - smoothstep(pxThickness, pxThickness + pxAA, dist);

  // Colors
  let line = path.colorData[0];
  let bg = path.colorData[14];
  let srcA = clamp(alpha * line.a, 0.0, 1.0);
  let outColor = line.rgb * srcA + bg.rgb * bg.a * (1.0 - srcA);
  let outA = srcA + bg.a * (1.0 - srcA);

  return vec4<f32>(clamp(outColor, vec3<f32>(0.0), vec3<f32>(1.0)), outA);
}


@fragment fn fs(in: VectorShadeVSOut) -> @location(0) vec4<f32> {
  // Pixel footprint in em units (derivatives must be computed before any discard).
  let footprint = max(fwidth(in.em), vec2<f32>(1e-6));
  let invDiameter = 1.0 / footprint;

  let path = paths[0];
  // World-space page clip. discard (not return) keeps the derivative builtins below valid.
  let clip = path.pageClipBounds;
  if (in.world.x < clip.x || in.world.x > clip.x + clip.z ||
      in.world.y < clip.y || in.world.y > clip.y + clip.w) {
    discard;
  }

  // Analytic glyph coverage (same per-pixel winding as the solid/gradient pipelines).
  let coverage = vectorGlyphCoverage(in.curveRange.x, in.curveRange.y, in.em, invDiameter);
  // Early-out on empty regions of the padded glyph quad (mirrors the solid/gradient
  // fills) so an expensive shadeMain isn't evaluated just to be multiplied by ~0.
  // Safe before shadeMain: discard only demotes the invocation to a helper, so
  // derivative builtins inside shadeMain stay valid for live neighbors.
  if (coverage < 0.004) {
    discard;
  }

  // Resolve shader color in uniform control flow (discard demotes, it does not
  // diverge). Build the shared VertexOutput shadeMain expects via the struct
  // constructor — instanceId = 0 (single packed instance), isStroke = 0 (fill pass;
  // vector text has no stroke pass yet).
  let input = VertexOutput(in.pos, 0u, 0.0, in.local, in.world);
  let c = shadeMain(input, path);

  // Straight alpha: the blend applies srcColor * srcAlpha. Node opacity rides in
  // path.fillColor.a (commonData fillColor = [1, 1, 1, nodeOpacity]). KineticText
  // per-glyph opacity (1.0 for static text) multiplies in like the solid/gradient.
  let finalAlpha = c.a * coverage * path.fillColor.a * in.glyphOpacity;
  return vec4<f32>(c.rgb, finalAlpha);
}
