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
