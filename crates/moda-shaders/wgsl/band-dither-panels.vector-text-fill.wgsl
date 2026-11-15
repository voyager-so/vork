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


// Shader ID: band-dither-panels

// Sandbox named this helper `modf`, which collides with the WGSL builtin
// modf(); renamed to modBand to keep Studio's WGSL pipeline happy.
fn modBand(x: f32, y: f32) -> f32 {
  return x - floor(x / y) * y;
}

fn bayerQuadrant(x: f32, y: f32) -> f32 {
  return y * 3.0 + x * 2.0 - x * y * 4.0;
}

fn bayer8(pixelCoord: vec2<f32>) -> f32 {
  let x = modBand(pixelCoord.x, 8.0);
  let y = modBand(pixelCoord.y, 8.0);
  let b2 = bayerQuadrant(modBand(x, 2.0), modBand(y, 2.0));
  let b4 = bayerQuadrant(floor(modBand(x, 4.0) / 2.0), floor(modBand(y, 4.0) / 2.0));
  let b8 = bayerQuadrant(floor(x / 4.0), floor(y / 4.0));

  return (b2 * 16.0 + b4 * 4.0 + b8) / 64.0;
}

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn transformUv(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  rotation: f32
) -> vec2<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  var p = uv - vec2<f32>(0.5);
  p.x *= aspect;
  p = rotate2(p, -rotation);

  return vec2<f32>(p.x / aspect, p.y) + vec2<f32>(0.5);
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

fn rand1(v: f32) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(v)));
}

fn panelSource(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  panelWidth: f32
) -> f32 {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let direction = normalize(vec2<f32>(0.72, 1.0));
  let tangent = vec2<f32>(-direction.y, direction.x);
  let alongPanel = dot(p, tangent);
  let panelFrequency = 5.5 / max(panelWidth, 0.001);
  let axis = (dot(p, direction) + time * 0.035) * panelFrequency + 14.0;
  let lane = floor(axis);
  let laneRandom = rand1(lane * 17.31);
  let laneRandomB = rand1(lane * 31.73 + 4.8);
  let laneRandomC = rand1(lane * 51.17 + 8.2);
  let laneRandomD = rand1(lane * 73.91 + 2.6);
  let laneSpeed = mix(0.035, 0.42, laneRandom);
  let laneLength = mix(0.42, 0.84, laneRandomB);
  let laneOffset = laneRandom * 1.37 + laneRandomC * 0.41;
  let segmentPhase = fract(alongPanel * laneLength - time * laneSpeed + laneOffset);
  let head = smoothstep(0.035, 0.08, segmentPhase);
  let tail = 1.0 - smoothstep(0.62, 0.78, segmentPhase);
  let segment = head * tail;
  let laneIntensity = mix(0.42, 1.0, pow(laneRandomD, 0.55));
  let screenAlong = dot(vec2<f32>(uv.x * aspect, uv.y), vec2<f32>(1.0, 1.0));
  let screenGate = smoothstep(-0.22, 0.18, screenAlong - 0.32) * (1.0 - smoothstep(2.72, 3.28, screenAlong));

  return mix(0.06, 0.74, segment * screenGate * laneIntensity);
}

fn distortedSource(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  panelWidth: f32
) -> vec4<f32> {
  return vec4<f32>(vec3<f32>(panelSource(uv, resolution, time, panelWidth)), 1.0);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = pixel size
  // path.shaderParams[0].z = dither strength
  // path.shaderParams[0].w = threshold
  // path.shaderParams[1].x = panel width
  // path.shaderParams[1].y = rotation
  // path.shaderParams[1].z = speed
  // path.colorData[0].rgb = background color
  // path.colorData[1].rgb = panel color
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];

  let time = p0.x;
  let pixelSize = max(p0.y, 1.0);
  let ditherStrength = p0.z;
  let thresholdCenter = p0.w;
  let panelWidth = max(p1.x, 0.001);
  let rotation = p1.y;
  let speed = max(p1.z, 0.0);
  let t = time * speed;

  let pixelCoord = floor(uv * resolution / pixelSize);
  let pixelUv = (pixelCoord + vec2<f32>(0.5)) * pixelSize / resolution;
  let sourceUv = transformUv(pixelUv, resolution, rotation);
  let sourceColor = distortedSource(sourceUv, resolution, t, panelWidth);
  // Renamed from `luminance` to avoid shadowing the common color include's
  // luminance() helper that `color: true` pulls in.
  let lum = dot(sourceColor.rgb, vec3<f32>(0.299, 0.587, 0.114)) * sourceColor.a;
  let ditherValue = bayer8(pixelCoord);
  let threshold = 0.5 + (ditherValue - 0.5) * ditherStrength;
  let result = step(threshold, lum + (thresholdCenter - 0.5));
  let colorA = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorB = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color = mix(colorA, colorB, result);

  return vec4<f32>(color, 1.0);
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
