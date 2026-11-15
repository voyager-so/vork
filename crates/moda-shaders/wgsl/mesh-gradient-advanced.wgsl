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


// Constants
const TWO_PI: f32 = 6.283185307179586;

// Procedural position for i-th spot
fn spotPosition(i: u32, t: f32) -> vec2<f32> {
  let fi = f32(i);
  let a = fi * 0.37;
  let b = 0.6 + f32(i % 3u) * 0.3;
  let c = 0.8 + f32((i + 1u) % 4u) * 0.25;
  let x = sin(t * b + a);
  let y = cos(t * c + a * 1.5);
  return 0.5 + 0.5 * vec2<f32>(x, y);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  var uv = (input.localPos - vec2<f32>(b.x, b.y)) / vec2<f32>(b.z, b.w);
  uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));

  // Params - batch reads (compacted layout)
  let params0 = path.shaderParams[0];
  let params1 = path.shaderParams[1];
  let params2 = path.shaderParams[2];
  let count = u32(params0.x + 0.5);
  let distortion = params0.y;
  let swirl = params0.z;
  let t = params0.w;

  let positionsSeed = params1.x;
  let waveX = params1.y;
  let waveXShift = params1.z;
  let waveY = params1.w;
  let waveYShift = params2.x;
  let mixing = clamp(params2.y, 0.0, 1.0);
  let grainMixer = clamp(params2.z, 0.0, 1.0);
  let grainOverlay = clamp(params2.w, 0.0, 1.0);

  // Single hash for grain - derive RGB cheaply instead of 3 separate hash calls.
  // Framebuffer coords: @builtin(position) is bit-exact across rasterizers
  // (interpolated varyings like localPos are not).
  let grainUV = input.position.xy;
  let grain = hash21(grainUV);
  let mixerGrain = 0.1 * grainMixer * (grain - 0.5);

  // Precompute wave phase offsets
  let waveXPhase = TWO_PI * waveXShift;
  let waveYPhase = TWO_PI * waveYShift;

  // Distortion with unrolled loop (k=1 and k=2)
  var shape_uv = uv;
  let centerVec = shape_uv - vec2<f32>(0.5);
  let radius = length(centerVec);
  let center = 1.0 - radius;

  // k=1 iteration
  let sy1 = smoothstep(0.0, 1.0, shape_uv.y);
  let sx1 = smoothstep(0.0, 1.0, shape_uv.x);
  shape_uv.x += waveX * center * cos(t + waveXPhase + 2.0 * sy1);
  shape_uv.y += waveY * center * cos(0.2 * t + waveYPhase + 2.0 * sx1);
  shape_uv += distortion * center * vec2<f32>(sin(t + 0.4 * shape_uv.y), cos(t + 0.6 * shape_uv.x)) * 0.5;

  // k=2 iteration
  let sy2 = smoothstep(0.0, 1.0, shape_uv.y);
  let sx2 = smoothstep(0.0, 1.0, shape_uv.x);
  shape_uv.x += waveX * center * 0.5 * cos(t + waveXPhase + 4.0 * sy2);
  shape_uv.y += waveY * center * 0.5 * cos(0.2 * t + waveYPhase + 4.0 * sx2);
  shape_uv += distortion * center * 0.25 * vec2<f32>(sin(t + 0.8 * shape_uv.y), cos(t + 1.2 * shape_uv.x));

  // Swirl around center
  var uvRot = shape_uv - vec2<f32>(0.5);
  let angle = 3.0 * swirl * radius;
  uvRot = rotate2(uvRot, -angle);
  uvRot += vec2<f32>(0.5);

  var color = vec3<f32>(0.0);
  var wsum = 0.0;

  // Precompute mixing-dependent values OUTSIDE the loop (was computed 10x per fragment!)
  let mixingHigh = max(0.0, (mixing - 0.5) * 2.0);
  let power = mix(4.0, 0.75, mixingHigh);
  let halfPower = power * 0.5; // For use with dot(d,d) instead of length(d)
  let useSmoothBlend = mixing < 0.5;
  let smoothBlendFactor = mixing * 2.0;

  // Loop over colors as spots
  let maxN = min(count, 10u);
  let tSeed = t + positionsSeed;
  for (var i: u32 = 0u; i < maxN; i = i + 1u) {
    let pos = spotPosition(i, tSeed) + mixerGrain;
    let col = path.colorData[i].rgb;
    let d = uvRot - pos;
    // Use dot(d,d) instead of length(d) - avoids sqrt, use half power
    let r2 = dot(d, d);
    let rd = pow(max(r2, 1e-8), halfPower);
    var w = 1.0 / (rd + 1e-3);

    // Branchless smooth blend: always compute but conditionally apply
    let bias = mix(0.01, 5.0, clamp(w, 0.0, 1.0));
    let smoothW = pow(w, mix(bias, 1.0, smoothBlendFactor));
    w = select(w, smoothW, useSmoothBlend);

    color += col * w;
    wsum += w;
  }

  // Fallback if no weight - simplified
  color = select(color / wsum, path.colorData[0].rgb, wsum <= 1e-6);

  // Single hash-derived RGB grain (avoids 2 extra sin calls from hash21)
  let grainR = grain;
  let grainG = fract(grain * 127.1);
  let grainB = fract(grain * 311.7);
  let grainColor = vec3<f32>(grainR, grainG, grainB) - 0.5;
  color = color + grainColor * grainOverlay * 0.24;

  // Straight-alpha; the epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
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
