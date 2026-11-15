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

// Frame uniforms
@group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>;

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

const STROKE_DEPTH_OFFSET: f32 = -0.000001490116119384765625;

struct VertexInput {
  @location(0) position: vec2<f32>,
  @location(1) vertexType: f32,
}
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) instanceId: u32,
  @location(1) @interpolate(flat) isStroke: f32,
  @location(2) localPos: vec2<f32>,
  @location(3) worldPos: vec2<f32>,
}

@vertex
fn vertexMain(vertex: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var o: VertexOutput;
  let path = paths[instanceIndex];
  let lp = vertex.position;
  let wp = path.model * vec4<f32>(lp.x, lp.y, 0.0, 1.0);
  o.worldPos = wp.xy;
  o.position = frameUniforms * wp;
  let depthOffset = select(0.0, STROKE_DEPTH_OFFSET, vertex.vertexType > 0.5);
  o.position.z = path.pathInfo.w + depthOffset;
  o.instanceId = instanceIndex;
  o.isStroke = vertex.vertexType;
  o.localPos = lp;
  return o;
}


struct ClipMaskUniforms {
  maskActive: u32,
  maskOrigin: vec2<f32>,
  maskInvSize: vec2<f32>,
}
@group(2) @binding(0) var clipMaskSampler: sampler;
@group(2) @binding(1) var clipMaskTexture: texture_2d<f32>;
@group(2) @binding(2) var<uniform> clipMask: ClipMaskUniforms;

fn clipMaskCoverage(worldPos: vec2<f32>) -> f32 {
  if (clipMask.maskActive == 0u) {
    return 1.0;
  }
  let uv = (worldPos - clipMask.maskOrigin) * clipMask.maskInvSize;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return 0.0;
  }
  return textureSampleLevel(clipMaskTexture, clipMaskSampler, uv, 0.0).r;
}


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


fn shadeStroke(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  return vec4<f32>(path.strokeColor.rgb, 1.0);
}


fn pageClip(worldPos: vec2<f32>, path: CustomShaderData) -> f32 {
  let clip = path.pageClipBounds;
  return step(clip.x, worldPos.x) * step(worldPos.x, clip.x + clip.z) *
         step(clip.y, worldPos.y) * step(worldPos.y, clip.y + clip.w);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let path = paths[input.instanceId];
  let clipIn = pageClip(input.worldPos, path) * clipMaskCoverage(input.worldPos);
  // Page-clip / clip-mask early-out before the (possibly expensive) shade. A discard (not a return)
  // keeps shadeMain below in uniform control flow, so its derivative builtins (fwidth/dpdx/dpdy) stay valid.
  if (clipIn <= 0.0) { discard; }
  let c = shadeMain(input, path);
  // Stroke fragments paint the stroke shader (default: node strokeColor) instead of letting the shader
  // fill bleed across the border. Resolved after shadeMain so the uniformity above holds. Mirrors the fill
  // path: shadeStroke returns intrinsic straight-alpha and the epilogue applies the node strokeColor.a.
  if (input.isStroke > 0.5) {
    let sc = shadeStroke(input, path);
    let strokeAlpha = sc.a * path.strokeColor.a * clipIn;
    if (strokeAlpha <= 0.0) { discard; }
    return premultiplyAlpha(vec4<f32>(sc.rgb, strokeAlpha));
  }
  let baseAlpha = path.fillColor.a * clipIn;
  let outA = c.a * baseAlpha;
  if (outA <= 0.0) { discard; }
  return premultiplyAlpha(vec4<f32>(c.rgb, outA));
}
