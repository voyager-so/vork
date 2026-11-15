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


// Shader ID: dithered-wave-ribbon

const MAX_RIBBON_COLORS: u32 = 4u;
const RIBBON_COLOR_OFFSET: u32 = 2u;
const RIBBON_CUTOFF: f32 = 0.12;
const BACKGROUND_DITHER_STRENGTH: f32 = 0.85 / 255.0;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn fieldFade(value: vec2<f32>) -> vec2<f32> {
  return value * value * (vec2<f32>(3.0) - value * 2.0);
}

// Integer-exact hash core (lowbias32): bit-exact across WGSL compilers and devices,
// unlike the old fract(sin(x) * K) hash.
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

fn fieldSeed(cell: vec2<f32>) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(cell.x) ^ hashU32(bitcast<u32>(cell.y))));
}

fn fieldNoise(point: vec2<f32>) -> f32 {
  let cell = floor(point);
  let local = fract(point);
  let blend = fieldFade(local);
  let a = fieldSeed(cell);
  let b = fieldSeed(cell + vec2<f32>(1.0, 0.0));
  let c = fieldSeed(cell + vec2<f32>(0.0, 1.0));
  let d = fieldSeed(cell + vec2<f32>(1.0, 1.0));
  let bottom = mix(a, b, blend.x);
  let top = mix(c, d, blend.x);

  return mix(bottom, top, blend.y);
}

fn layeredField(point: vec2<f32>, clock: f32) -> f32 {
  var p = point;
  var amp = 0.56;
  var sum = 0.0;
  var norm = 0.0;

  for (var layer = 0; layer < 4; layer = layer + 1) {
    let layerF = f32(layer);
    let drift = vec2<f32>(
      sin(clock * 0.24 + layerF * 1.41),
      cos(clock * 0.19 - layerF * 1.67)
    ) * 0.28;

    sum += fieldNoise(p + drift) * amp;
    norm += amp;
    p = mat2x2<f32>(
      vec2<f32>(1.58, 0.28),
      vec2<f32>(-0.36, 1.72)
    ) * p + vec2<f32>(0.17, 0.31);
    amp *= 0.53;
  }

  return sum / max(norm, 0.001);
}

fn packedColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  var color = path.colorData[0].rgb;

  if (index == 1u) {
    color = path.colorData[1].rgb;
  } else if (index == 2u) {
    color = path.colorData[2].rgb;
  } else if (index == 3u) {
    color = path.colorData[3].rgb;
  } else if (index == 4u) {
    color = path.colorData[4].rgb;
  } else if (index >= 5u) {
    color = path.colorData[5].rgb;
  }

  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn backgroundColor(haze: f32, path: CustomShaderData) -> vec3<f32> {
  return mix(packedColorAt(0u, path), packedColorAt(1u, path), clamp(haze, 0.0, 1.0));
}

fn ribbonColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  return packedColorAt(RIBBON_COLOR_OFFSET + min(index, MAX_RIBBON_COLORS - 1u), path);
}

fn ribbonGradient(centerStrength: f32, colorCount: u32, path: CustomShaderData) -> vec3<f32> {
  let count = max(colorCount, 2u);
  let last = count - 1u;
  let scaled = (1.0 - clamp(centerStrength, 0.0, 1.0)) * f32(last);
  let left = min(u32(floor(scaled)), last - 1u);
  let right = min(left + 1u, last);
  let local = fract(scaled);

  return mix(ribbonColorAt(left, path), ribbonColorAt(right, path), local);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let pixel = input.localPos - b.xy;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = ribbon color count
  // path.shaderParams[0].z = scale
  // path.shaderParams[0].w = rotation
  // path.shaderParams[1].x = offsetX
  // path.shaderParams[1].y = offsetY
  // path.shaderParams[1].z = speed
  // path.shaderParams[1].w = grain
  // path.shaderParams[2].x = ribbonWidth
  // path.shaderParams[2].y = pixelSize
  // path.shaderParams[2].z = ditherStrength
  // path.colorData[0..1].rgb = background colors
  // path.colorData[2..5].rgb = ribbon colors, center to outer edge
  let time = path.shaderParams[0].x;
  let ribbonColorCount = min(max(u32(path.shaderParams[0].y + 0.5), 2u), MAX_RIBBON_COLORS);
  let scale = max(path.shaderParams[0].z, 0.001);
  let rotation = path.shaderParams[0].w;
  let offset = vec2<f32>(path.shaderParams[1].x, path.shaderParams[1].y);
  let speed = path.shaderParams[1].z;
  let grain = clamp(path.shaderParams[1].w, 0.0, 1.0);
  let ribbonWidth = clamp(path.shaderParams[2].x, 0.35, 1.5);
  let pixelSize = max(path.shaderParams[2].y, 1.0);
  let ditherStrength = clamp(path.shaderParams[2].z, 0.0, 1.0);

  let aspect = b.z / max(b.w, 1.0);
  let screenOffset = vec2<f32>(-offset.x * 0.42 * aspect, offset.y * 0.42);
  let clock = time * (0.04 + speed * 0.1);
  let centered = (uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
  let backgroundPoint = rotate2(centered + screenOffset, -rotation) / scale;
  let baseHaze = layeredField(backgroundPoint * 0.82 + vec2<f32>(clock * 0.18, -clock * 0.08), clock);
  let backgroundDither = bayer8(floor(pixel)) - 0.5;
  let baseColor = clamp(
    backgroundColor(baseHaze, path) + vec3<f32>(backgroundDither * BACKGROUND_DITHER_STRENGTH),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );

  let blockPixel = floor(pixel / pixelSize) * pixelSize + vec2<f32>(pixelSize * 0.5);
  let samplePixel = clamp(blockPixel, vec2<f32>(0.0), max(b.zw - vec2<f32>(1.0), vec2<f32>(0.0)));
  let sampleUv = samplePixel / b.zw;
  let sampleCentered = (sampleUv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
  var point = rotate2(sampleCentered + screenOffset, -rotation) / scale;

  point.x += sin(point.y * 1.35 + clock * 0.42) * 0.18;
  point.y += cos(point.x * 0.72 - clock * 0.31) * 0.12;

  let haze = layeredField(point * 0.82 + vec2<f32>(clock * 0.18, -clock * 0.08), clock);
  let wavePattern =
    cos(0.5 * point.x - 4.0 * clock) *
    sin(1.5 * point.x + 2.0 * clock) *
    (0.75 + 0.25 * cos(6.0 * clock));
  let waveEdge = -point.y + wavePattern * 0.44 + (haze - 0.5) * 0.34;
  let waveA = 1.0 - smoothstep(-0.62 * ribbonWidth, 0.78 * ribbonWidth, waveEdge);
  let wavePatternB =
    cos(0.42 * point.x + 2.7 - 2.8 * clock) *
    sin(1.15 * point.x - 1.4 + 1.3 * clock) *
    (0.72 + 0.18 * cos(4.2 * clock));
  let waveEdgeB = -point.y + wavePatternB * 0.32 + (haze - 0.5) * 0.22 + 0.18;
  let waveB = 1.0 - smoothstep(-0.46 * ribbonWidth, 0.7 * ribbonWidth, waveEdgeB);
  let wave = max(waveA, waveB * 0.55);
  let ribbonStrength = pow(wave, 1.32);
  let dust = fieldSeed(floor(samplePixel * 0.62)) - 0.5;
  let ribbonPresence = step(RIBBON_CUTOFF, ribbonStrength);
  let cleanRibbonMask = smoothstep(RIBBON_CUTOFF, 0.92, ribbonStrength) * ribbonPresence;
  let ribbonMask = clamp(cleanRibbonMask + dust * grain * 0.045 * cleanRibbonMask, 0.0, 1.0);

  let ribbonColor = ribbonGradient(ribbonStrength, ribbonColorCount, path);
  let ditherCellSize = max(floor(pixelSize / 8.0), 1.0);
  let ditherCell = floor(pixel / ditherCellSize);
  let ditheredRibbonColor = orderedDitherColor(ribbonColor, bayer8(ditherCell), ditherStrength);
  let color = mix(baseColor, ditheredRibbonColor, ribbonMask);

  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
