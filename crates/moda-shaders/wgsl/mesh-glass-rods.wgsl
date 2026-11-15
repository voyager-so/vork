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


// Shader ID: mesh-glass-rods

const PI: f32 = 3.141592653589793;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn transformUv(
  uv: vec2<f32>,
  aspect: f32,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>
) -> vec2<f32> {
  let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let transformed = rotate2(
    centered / max(scale, 0.001) + vec2<f32>(-offset.x * aspect, offset.y),
    -rotation
  );

  return vec2<f32>(transformed.x / aspect + 0.5, transformed.y + 0.5);
}

fn signedValueNoise(p: vec2<f32>) -> f32 {
  return valueNoise(p) * 2.0 - 1.0;
}

fn fbm1(p: vec2<f32>) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.0;

  for (var i = 0; i < 5; i = i + 1) {
    sum += signedValueNoise(q) * amp;
    q = rotate2(q * 2.03 + vec2<f32>(19.17, -7.31), 0.62);
    amp *= 0.52;
  }

  return sum;
}

fn mirrored(value: vec2<f32>) -> vec2<f32> {
  let m = value - floor(value / 2.0) * 2.0;

  return mix(m, vec2<f32>(2.0) - m, step(vec2<f32>(1.0), m));
}

fn meshBackground(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let drift = vec2<f32>(time * 0.028, -time * 0.021);
  let warpA = fbm1(p * 1.35 + drift);
  let warpB = fbm1(rotate2(p * 1.8 - drift.yx, 0.86));
  let wave = vec2<f32>(
    sin(p.y * 4.8 + time * 0.95),
    cos(p.x * 5.7 - time * 0.82)
  );
  p += vec2<f32>(warpA, warpB) * 0.16 + wave * vec2<f32>(0.045, 0.06);

  let color1 = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color2 = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color3 = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color4 = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let anchor1 = vec2<f32>(-0.46 * aspect, -0.34) + vec2<f32>(sin(time * 0.19), cos(time * 0.13)) * vec2<f32>(0.14 * aspect, 0.1);
  let anchor2 = vec2<f32>(0.42 * aspect, -0.29) + vec2<f32>(cos(time * 0.17), sin(time * 0.11)) * vec2<f32>(0.16 * aspect, 0.11);
  let anchor3 = vec2<f32>(-0.38 * aspect, 0.33) + vec2<f32>(sin(time * 0.14 + 1.7), cos(time * 0.18)) * vec2<f32>(0.13 * aspect, 0.12);
  let anchor4 = vec2<f32>(0.44 * aspect, 0.31) + vec2<f32>(cos(time * 0.12 + 0.8), sin(time * 0.16)) * vec2<f32>(0.15 * aspect, 0.1);
  let w1 = 1.0 / max(dot(p - anchor1, p - anchor1), 0.035);
  let w2 = 1.0 / max(dot(p - anchor2, p - anchor2), 0.035);
  let w3 = 1.0 / max(dot(p - anchor3, p - anchor3), 0.035);
  let w4 = 1.0 / max(dot(p - anchor4, p - anchor4), 0.035);
  let weightSum = w1 + w2 + w3 + w4;
  let pooled = (color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4) / max(weightSum, 0.001);

  return pooled;
}

fn filmGrainNoise(fragCoord: vec2<f32>) -> f32 {
  return hash21(fragCoord) * 2.0 - 1.0;
}

fn rodsLayer(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  distortion: f32,
  chromaticAberration: f32,
  waviness: f32,
  frequency: f32,
  rods: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  var p = uv * 2.0 - vec2<f32>(1.0);
  p.x *= aspect;

  let angle = 45.0 * PI / 180.0;
  let pRot = rotate2(p, angle);
  let rodTravel = time * 0.08;
  let travelingY = pRot.y + rodTravel;
  let wave = waviness * sin(travelingY * frequency);
  let rodX = fract((pRot.x + wave) * rods) * 2.0 - 1.0;
  let rodZSq = 1.0 - rodX * rodX;
  let rodZ = sqrt(max(rodZSq, 0.0));
  let n = vec3<f32>(rodX, 0.0, -rodZ);
  let rd = vec3<f32>(0.0, 0.0, -1.0);
  let refractiveIndex = 0.6;
  let refractedRay = mix(n, rd, refractiveIndex);
  let zDist = 0.5 / (abs(refractedRay.z) + 0.001);
  let hitPos = vec3<f32>(vec2<f32>(pRot.x, travelingY), 0.0) + (zDist * distortion) * refractedRay;
  var uvHit = rotate2(hitPos.xy, -angle);

  uvHit.x /= aspect;
  uvHit = uvHit * 0.5 + vec2<f32>(0.5);

  let flowClock = time * 0.1;
  let flow = vec2<f32>(sin(flowClock), cos(flowClock * 0.8)) * 0.05;
  let warpedUv = mirrored(uvHit + flow);
  let r = meshBackground(mirrored(warpedUv + vec2<f32>(chromaticAberration, 0.0)), resolution, time, path).r;
  let g = meshBackground(warpedUv, resolution, time, path).g;
  let b = meshBackground(mirrored(warpedUv - vec2<f32>(chromaticAberration, 0.0)), resolution, time, path).b;
  let glassFactor = smoothstep(0.0, 1.0, 1.0 - abs(n.z));
  let glass = glassFactor * 0.0025;

  return vec3<f32>(r, g, b) + vec3<f32>(glass);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = distortion
  // path.shaderParams[0].z = scale
  // path.shaderParams[0].w = rotation
  // path.shaderParams[1].x = offsetX
  // path.shaderParams[1].y = offsetY
  // path.shaderParams[1].z = chromatic aberration
  // path.shaderParams[1].w = waviness
  // path.shaderParams[2].x = frequency
  // path.shaderParams[2].y = rods
  // path.shaderParams[2].z = grain
  // path.shaderParams[2].w = speed
  // path.colorData[0..3].rgb = mesh palette
  let time = path.shaderParams[0].x;
  let distortion = path.shaderParams[0].y;
  let scale = path.shaderParams[0].z;
  let rotation = path.shaderParams[0].w;
  let offset = vec2<f32>(path.shaderParams[1].x, path.shaderParams[1].y);
  let chromaticAberration = path.shaderParams[1].z;
  let waviness = path.shaderParams[1].w;
  let frequency = path.shaderParams[2].x;
  let rods = path.shaderParams[2].y;
  let grain = path.shaderParams[2].z;
  let speed = path.shaderParams[2].w;
  let t = time * speed;
  let aspect = resolution.x / resolution.y;
  let effectUv = transformUv(uv, aspect, scale, rotation, offset);
  var color = rodsLayer(
    effectUv,
    resolution,
    t,
    distortion,
    chromaticAberration,
    waviness,
    frequency,
    rods,
    path
  );

  // Framebuffer coords for film grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  color += vec3<f32>(filmGrainNoise(input.position.xy) * grain);

  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
