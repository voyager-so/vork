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

// #include "vector-text-coverage.wgsl"


// Shader ID: clouds

fn rotate2(point: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);

  return vec2<f32>(point.x * c - point.y * s, point.x * s + point.y * c);
}

fn fade2(value: vec2<f32>) -> vec2<f32> {
  return value * value * (vec2<f32>(3.0, 3.0) - value * 2.0);
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

fn speck(cell: vec2<f32>) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(cell.x) ^ hashU32(bitcast<u32>(cell.y))));
}

fn softGrid(point: vec2<f32>) -> f32 {
  let cell = floor(point);
  let local = fract(point);
  let blend = fade2(local);
  let a = speck(cell);
  let b = speck(cell + vec2<f32>(1.0, 0.0));
  let c = speck(cell + vec2<f32>(0.0, 1.0));
  let d = speck(cell + vec2<f32>(1.0, 1.0));
  let low = mix(a, b, blend.x);
  let high = mix(c, d, blend.x);

  return mix(low, high, blend.y);
}

fn driftStack(point: vec2<f32>, clock: f32) -> f32 {
  var p = point;
  var strength = 0.55;
  var total = 0.0;
  var normalizer = 0.0;

  for (var octave = 0; octave < 5; octave = octave + 1) {
    let curl = vec2<f32>(
      sin(p.y * 0.72 + clock * 0.31),
      cos(p.x * 0.66 - clock * 0.24)
    ) * 0.15;

    total += softGrid(p + curl) * strength;
    normalizer += strength;
    p = mat2x2<f32>(
      vec2<f32>(1.68, 0.42),
      vec2<f32>(-0.36, 1.74)
    ) * p + vec2<f32>(0.13, 0.29);
    strength *= 0.52;
  }

  return total / normalizer;
}

fn skyTone(uv: vec2<f32>, sun: vec2<f32>, sunlight: f32) -> vec3<f32> {
  let horizon = smoothstep(-0.1, 0.9, uv.y);
  let lower = vec3<f32>(0.87, 0.93, 0.98);
  let upper = vec3<f32>(0.36, 0.55, 0.88);
  let air = mix(lower, upper, vec3<f32>(horizon));
  let sunDistance = distance(uv, sun);
  let wash = smoothstep(0.78, 0.0, sunDistance) * sunlight;
  let core = smoothstep(0.18, 0.0, sunDistance) * sunlight;
  let warm = vec3<f32>(1.0, 0.92, 0.68);

  return mix(air, warm, vec3<f32>(wash * 0.32)) + warm * core * 0.12;
}

fn cloudMask(noise: f32, height: f32, coverage: f32, softness: f32) -> f32 {
  let lift = mix(-0.12, 0.14, height);
  let low = mix(0.62, 0.42, coverage) + lift * 0.12;
  let high = low + mix(0.06, 0.24, softness);

  return smoothstep(low, high, noise);
}

fn cloudTone(noise: f32, uv: vec2<f32>, sun: vec2<f32>, sunlight: f32) -> vec3<f32> {
  let lightSide = smoothstep(0.75, 0.0, distance(uv, sun));
  let shadow = vec3<f32>(0.70, 0.80, 0.92);
  let body = vec3<f32>(0.92, 0.96, 1.0);
  let crest = vec3<f32>(1.0, 0.985, 0.92);
  let thickness = smoothstep(0.45, 0.88, noise);

  return mix(shadow, body, vec3<f32>(thickness)) + crest * lightSide * sunlight * 0.18;
}

fn transformUv(uv: vec2<f32>, aspect: f32, scale: f32, rotation: f32, offset: vec2<f32>) -> vec2<f32> {
  let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let transformed = rotate2(centered / max(scale, 0.001) + vec2<f32>(-offset.x * 0.38 * aspect, offset.y * 0.38), -rotation);

  return vec2<f32>(transformed.x / aspect + 0.5, transformed.y + 0.5);
}

fn shadeClouds(
  uv: vec2<f32>,
  pixel: vec2<f32>,
  time: f32,
  resolution: vec2<f32>,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>,
  speed: f32,
  coverage: f32,
  softness: f32,
  sunlight: f32,
  exposure: f32,
) -> vec4<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let transformedUv = transformUv(uv, aspect, scale, rotation, offset);
  let clock = time * speed * 0.14;
  let sun = vec2<f32>(0.62, 0.43);
  let wind = vec2<f32>(clock * 0.42, clock * 0.08);
  var cloudPoint = transformedUv * vec2<f32>(aspect, 1.0);

  cloudPoint.x += transformedUv.y * 0.16;
  cloudPoint = cloudPoint * 2.8 + wind;

  let broad = driftStack(cloudPoint, clock);
  let detail = driftStack(cloudPoint * 2.15 + vec2<f32>(4.7, 1.9), clock * 1.23);
  let vapor = mix(broad, detail, 0.28);
  let mask = cloudMask(vapor, transformedUv.y, coverage, softness);
  let edge = smoothstep(0.0, 0.55, mask);
  let sky = skyTone(transformedUv, sun, sunlight);
  let clouds = cloudTone(vapor, transformedUv, sun, sunlight);
  let dither = speck(floor(pixel * 0.5)) - 0.5;

  var color = mix(sky, clouds, vec3<f32>(edge));
  color += vec3<f32>(dither * 0.018 * softness);
  color = pow(max(color * exposure, vec3<f32>(0.0)), vec3<f32>(0.9));

  return vec4<f32>(color, 1.0);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let pixel = input.localPos - b.xy;
  let resolution = b.zw;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = speed
  // path.shaderParams[1].z = coverage
  // path.shaderParams[2].x = softness
  // path.shaderParams[2].y = sunlight
  // path.shaderParams[2].z = exposure
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let speed = path.shaderParams[1].y;
  let coverage = path.shaderParams[1].z;
  let softness = path.shaderParams[2].x;
  let sunlight = path.shaderParams[2].y;
  let exposure = path.shaderParams[2].z;

  let color = shadeClouds(uv, pixel, time, resolution, scale, rotation, offset, speed, coverage, softness, sunlight, exposure);

  return color;
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
