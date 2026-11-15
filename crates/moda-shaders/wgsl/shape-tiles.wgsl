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


fn encode_int(v: f32) -> u32 { return u32(v + 0.5); }

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

// Hash utility for palette index selection
fn hash12(p: vec2<f32>) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(p.x) ^ hashU32(bitcast<u32>(p.y))));
}

// Palette sampling: linear blend across provided colors (RGBA)
fn sample_palette(path: CustomShaderData, t: f32, count: u32) -> vec4<f32> {
  let c = clamp(count, 1u, 8u);
  if (c == 1u) { return path.colorData[0]; }
  let x = clamp(t, 0.0, 1.0) * (f32(c) - 1.0);
  let i0 = u32(floor(x));
  let i1 = min(i0 + 1u, c - 1u);
  let f = fract(x);
  let a = path.colorData[i0];
  let b = path.colorData[i1];
  return mix(a, b, f);
}

// Triangle constants
const SQRT3: f32 = 1.73205080757;
const SQRT3_HALF: f32 = 0.86602540378;

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  var uv = (input.localPos - vec2<f32>(b.x, b.y)) / vec2<f32>(b.z, b.w);
  uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));

  // Aspect-correct UV: keep shapes isotropic regardless of node aspect ratio
  let aspect = b.z / max(b.w, 1e-4);
  var uvA = uv;
  if (aspect > 1.0) {
    uvA = vec2<f32>((uv.x - 0.5) * aspect + 0.5, uv.y);
  } else {
    uvA = vec2<f32>(uv.x, (uv.y - 0.5) / aspect + 0.5);
  }

  // Params layout mirrors ShapeGridShader where possible
  let time = path.shaderParams[0].w;

  let cellSize = max(path.shaderParams[1].x, 1e-4);
  let strokeWidth = max(path.shaderParams[1].w, 0.0);

  let rippleAmp = path.shaderParams[2].w;

  let rippleWave = max(path.shaderParams[3].x, 1e-4);
  let rippleSpeed = path.shaderParams[3].y;
  let rippleAng = path.shaderParams[3].z;
  let seed = path.shaderParams[3].w;

  let colorMode = encode_int(path.shaderParams[4].x);  // 0=uniform,1=gradient,2=hashed
  let gradAngle = path.shaderParams[4].y;
  let gradRepeat = max(path.shaderParams[4].z, 1e-4);
  let colorCount = encode_int(path.shaderParams[4].w);

  // Triangle tiling field in UV space
  let cs = max(cellSize, 1e-4);
  let u = uvA.x / cs - uvA.y / (cs * SQRT3);
  let v = uvA.y / (cs * SQRT3_HALF);
  let fu = fract(u);
  let fv = fract(v);
  let sum = fu + fv;
  var du = 0.0;
  var dv = 0.0;
  var dw = 0.0;
  if (sum <= 1.0) {
    du = fu; dv = fv; dw = 1.0 - sum;
  } else {
    du = 1.0 - fu; dv = 1.0 - fv; dw = sum - 1.0;
  }
  let edgeUV = min(du, min(dv, dw)) * cs * SQRT3_HALF;

  // Stroke/fill alpha
  let aa = 0.002;
  var outlineAlpha = 1.0 - smoothstep(max(strokeWidth, 0.0005) - aa, max(strokeWidth, 0.0005) + aa, edgeUV);

  // Interior fill mask hugging outline
  let interior = smoothstep(max(strokeWidth, 0.0005) - aa * 0.5, max(strokeWidth, 0.0005), edgeUV);

  // Per-triangle blinking using hashed cell id
  let ui = floor(u);
  let vi = floor(v);
  let h = hash12(vec2<f32>(ui, vi));
  let h2 = hash12(vec2<f32>(ui + 17.0 * seed, vi + 43.0 * seed));
  let waveDir = vec2<f32>(cos(rippleAng), sin(rippleAng));
  let wave = rippleAmp * sin((dot(uvA, waveDir) * (6.28318530718 / rippleWave)) - time * rippleSpeed);
  let phase = time * rippleSpeed * 2.5 + (h * 6.28318530718) + wave;
  let base = 0.5 + 0.5 * sin(phase);
  let blink = select(0.0, 1.0, base > (0.75 + 0.2 * (h2 - 0.5)));
  let fillBlink = interior * blink;

  // Colors
  var col: vec4<f32>;
  switch(i32(colorMode)) {
    case 0: { col = path.colorData[0]; break; }
    case 1: {
      let gdir = vec2<f32>(cos(gradAngle), sin(gradAngle));
      let s = dot(uvA - vec2<f32>(0.5), gdir) * gradRepeat * 6.28318530718;
      let t = 0.5 + 0.5 * sin(s);
      col = sample_palette(path, t, colorCount);
      break;
    }
    default: {
      let hh = hashToUnit(hashU32(bitcast<u32>(h) ^ hashU32(bitcast<u32>(seed))));
      col = sample_palette(path, hh, colorCount);
      break;
    }
  }

  // Background compositing
  let bg = path.colorData[14];
  let srcA = clamp((outlineAlpha + clamp(fillBlink, 0.0, 1.0)) * col.a, 0.0, 1.0);
  let outColor = col.rgb * srcA + bg.rgb * bg.a * (1.0 - srcA);
  let outA = srcA + bg.a * (1.0 - srcA);

  // Premultiplied composite preserved from the hand-written shader: outColor is already folded over the
  // background (RGB premultiplied), NOT straight-alpha — do not normalize by outA. The epilogue applies
  // fillColor.a + page-clip + clip-mask and premultiplies on top, reproducing the original output.
  return vec4<f32>(clamp(outColor, vec3<f32>(0.0), vec3<f32>(1.0)), outA);
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
