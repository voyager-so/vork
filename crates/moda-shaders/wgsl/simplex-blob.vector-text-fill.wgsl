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


// Common dither utilities

// 8x8 Bayer threshold matrix. Shared by ordered-dither shaders so the
// 64-entry table is declared once at module scope instead of inside the
// per-fragment helper.
const BAYER8_MATRIX: array<f32, 64> = array<f32, 64>(
  0.0, 48.0, 12.0, 60.0, 3.0, 51.0, 15.0, 63.0,
  32.0, 16.0, 44.0, 28.0, 35.0, 19.0, 47.0, 31.0,
  8.0, 56.0, 4.0, 52.0, 11.0, 59.0, 7.0, 55.0,
  40.0, 24.0, 36.0, 20.0, 43.0, 27.0, 39.0, 23.0,
  2.0, 50.0, 14.0, 62.0, 1.0, 49.0, 13.0, 61.0,
  34.0, 18.0, 46.0, 30.0, 33.0, 17.0, 45.0, 29.0,
  10.0, 58.0, 6.0, 54.0, 9.0, 57.0, 5.0, 53.0,
  42.0, 26.0, 38.0, 22.0, 41.0, 25.0, 37.0, 21.0
);

fn bayer8(cell: vec2<f32>) -> f32 {
  // Clamp to non-negative before the f32->u32 conversion: WGSL leaves
  // u32(<negative f32>) implementation-defined, which can produce arbitrary
  // indices on some drivers and visibly flicker dither at shape edges where
  // floor(pixel) may dip slightly below zero.
  let safeCell = max(cell, vec2<f32>(0.0));
  let x = u32(safeCell.x) % 8u;
  let y = u32(safeCell.y) % 8u;
  return (BAYER8_MATRIX[y * 8u + x] + 0.5) / 64.0;
}

fn orderedDitherColor(color: vec3<f32>, threshold: f32, strength: f32) -> vec3<f32> {
  let levels = 7.0;
  let dithered = floor(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)) * levels + vec3<f32>(threshold)) / levels;

  return mix(color, clamp(dithered, vec3<f32>(0.0), vec3<f32>(1.0)), strength);
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


// Shader ID: simplex-blob

const MAX_STEPS: i32 = 96;
const HIT_EPSILON: f32 = 0.0045;
const PI: f32 = 3.141592653589793;
fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn mod289_1(x: f32) -> f32 {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute1(x: f32) -> f32 {
  return mod289_1(((x * 34.0) + 1.0) * x);
}

fn permute4(x: vec4<f32>) -> vec4<f32> {
  return mod289_4(((x * 34.0) + vec4<f32>(1.0)) * x);
}

fn taylorInvSqrt1(r: f32) -> f32 {
  return 1.79284291400159 - 0.85373472095314 * r;
}

fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(1.79284291400159) - 0.85373472095314 * r;
}

fn grad4(j: f32, ip: vec4<f32>) -> vec4<f32> {
  let pxyz = floor(fract(vec3<f32>(j) * ip.xyz) * 7.0) * ip.z - vec3<f32>(1.0);
  let pw = 1.5 - dot(abs(pxyz), vec3<f32>(1.0));
  let p = vec4<f32>(pxyz, pw);
  let s = select(vec4<f32>(0.0), vec4<f32>(1.0), p < vec4<f32>(0.0));
  let adjusted = p.xyz + (s.xyz * 2.0 - vec3<f32>(1.0)) * s.www;

  return vec4<f32>(adjusted, p.w);
}

fn simplex4(v: vec4<f32>) -> f32 {
  let c = vec2<f32>(0.1381966011250105, 0.30901699437494745);
  var i = floor(v + dot(v, vec4<f32>(c.y)));
  let x0 = v - i + dot(i, vec4<f32>(c.x));

  let isX = step(x0.yzw, x0.xxx);
  let isYZ = step(x0.zww, x0.yyz);
  let i0x = isX.x + isX.y + isX.z;
  let i0y = 1.0 - isX.x + isYZ.x + isYZ.y;
  let i0z = 1.0 - isX.y + 1.0 - isYZ.x + isYZ.z;
  let i0w = 1.0 - isX.z + 1.0 - isYZ.y + 1.0 - isYZ.z;
  let i0 = vec4<f32>(i0x, i0y, i0z, i0w);
  let i3 = clamp(i0, vec4<f32>(0.0), vec4<f32>(1.0));
  let i2 = clamp(i0 - vec4<f32>(1.0), vec4<f32>(0.0), vec4<f32>(1.0));
  let i1 = clamp(i0 - vec4<f32>(2.0), vec4<f32>(0.0), vec4<f32>(1.0));

  let x1 = x0 - i1 + vec4<f32>(c.x);
  let x2 = x0 - i2 + vec4<f32>(2.0 * c.x);
  let x3 = x0 - i3 + vec4<f32>(3.0 * c.x);
  let x4 = x0 - vec4<f32>(1.0) + vec4<f32>(4.0 * c.x);

  i = mod289_4(i);
  let j0 = permute1(permute1(permute1(permute1(i.w) + i.z) + i.y) + i.x);
  let j1 = permute4(
    permute4(
      permute4(
        permute4(i.w + vec4<f32>(i1.w, i2.w, i3.w, 1.0)) +
          i.z + vec4<f32>(i1.z, i2.z, i3.z, 1.0)
      ) + i.y + vec4<f32>(i1.y, i2.y, i3.y, 1.0)
    ) + i.x + vec4<f32>(i1.x, i2.x, i3.x, 1.0)
  );
  let ip = vec4<f32>(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);

  var p0 = grad4(j0, ip);
  var p1 = grad4(j1.x, ip);
  var p2 = grad4(j1.y, ip);
  var p3 = grad4(j1.z, ip);
  var p4 = grad4(j1.w, ip);
  let norm = taylorInvSqrt4(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt1(dot(p4, p4));

  var m0 = max(vec3<f32>(0.6) - vec3<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3<f32>(0.0));
  var m1 = max(vec2<f32>(0.6) - vec2<f32>(dot(x3, x3), dot(x4, x4)), vec2<f32>(0.0));
  m0 *= m0;
  m1 *= m1;

  return 49.0 * (
    dot(m0 * m0, vec3<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
    dot(m1 * m1, vec2<f32>(dot(p3, x3), dot(p4, x4)))
  );
}

fn blobNoise(direction: vec3<f32>, clock: f32, noiseBands: f32) -> f32 {
  var n = simplex4(vec4<f32>(direction * 0.875, clock));
  n = sin(n * PI * noiseBands);
  n = n * 0.5 + 0.5;

  return n * n;
}

fn blobSdf(p: vec3<f32>, clock: f32, noiseBands: f32) -> f32 {
  let radius = length(p);
  let direction = p / max(radius, 0.0001);
  let displacementAmount = blobNoise(direction, clock, noiseBands) * 0.065;

  return radius - (0.98 + displacementAmount);
}

fn colorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  if (index == 1u) {
    return clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  } else if (index == 2u) {
    return clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  } else if (index >= 3u) {
    return clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  return clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let pixel = input.localPos - b.xy;

  // Packing map:
  // path.colorData[0..1].rgb = background colors
  // path.colorData[2..3].rgb = blob colors
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = speed
  // path.shaderParams[1].x = offsetX
  // path.shaderParams[1].y = offsetY
  // path.shaderParams[1].z = ditherStrength
  // path.shaderParams[1].w = blobOffsetX
  // path.shaderParams[2].x = noiseBands
  // path.shaderParams[2].y = ditherSize
  // path.shaderParams[2].z = blobOffsetY
  let time = path.shaderParams[0].x;
  let scale = max(path.shaderParams[0].y, 0.001);
  let rotation = path.shaderParams[0].z;
  let speed = path.shaderParams[0].w;
  let offset = vec2<f32>(path.shaderParams[1].x, path.shaderParams[1].y);
  let ditherStrength = clamp(path.shaderParams[1].z, 0.0, 1.0);
  let blobOffset = vec2<f32>(path.shaderParams[1].w, path.shaderParams[2].z);
  let noiseBands = max(path.shaderParams[2].x, 0.001);
  let ditherSize = max(path.shaderParams[2].y, 1.0);
  let clock = time * speed * 0.1;

  let aspect = b.z / max(b.w, 1.0);
  let centered = (uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
  let screenOffset = vec2<f32>(-offset.x * 0.42 * aspect, offset.y * 0.42);
  let screen = rotate2(centered + screenOffset, -rotation) / scale;
  let blobScreen = screen + vec2<f32>(-blobOffset.x * 0.42 * aspect, blobOffset.y * 0.42);
  let rayOrigin = vec3<f32>(0.0, 0.0, 3.2);
  let rayDirection = normalize(vec3<f32>(blobScreen, -1.85));
  let backgroundMix = smoothstep(-0.72, 0.92, screen.y + valueNoise(screen * 1.3 + vec2<f32>(clock * 0.2, -clock * 0.1)) * 0.35);
  var color = mix(colorAt(0u, path), colorAt(1u, path), vec3<f32>(backgroundMix));
  var t = 1.2;
  var hit = false;
  var hitPoint = rayOrigin;

  // The blob lies inside the sphere of radius 0.98 + 0.065 (the largest displacement) around the
  // origin, and the march hits only where the SDF drops below HIT_EPSILON: a ray passing farther than
  // that from the origin can never hit, so it skips the (up to 96-step) march — most of the frame.
  let closest = dot(rayOrigin, rayDirection);
  let missDistanceSq = dot(rayOrigin, rayOrigin) - closest * closest;
  let reach = 0.98 + 0.065 + HIT_EPSILON + 0.001;
  if (missDistanceSq > reach * reach) {
    t = 6.0;
  }

  for (var i = 0; i < MAX_STEPS && t <= 5.8; i = i + 1) {
    hitPoint = rayOrigin + rayDirection * t;
    let distance = blobSdf(hitPoint, clock, noiseBands);

    if (distance < HIT_EPSILON) {
      hit = true;
      break;
    }

    t += max(distance * 0.36, 0.004);

    if (t > 5.8) {
      break;
    }
  }

  if (hit) {
    let surfaceNoise = blobNoise(normalize(hitPoint), clock, noiseBands);
    let surfaceMix = smoothstep(0.1, 0.92, surfaceNoise);
    let blobBase = mix(colorAt(2u, path), colorAt(3u, path), vec3<f32>(surfaceMix));

    color = mix(color, blobBase, 0.96);
  }

  let ditherCell = floor(pixel / ditherSize);
  let ditheredColor = orderedDitherColor(color, bayer8(ditherCell), ditherStrength);

  return vec4<f32>(clamp(ditheredColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
