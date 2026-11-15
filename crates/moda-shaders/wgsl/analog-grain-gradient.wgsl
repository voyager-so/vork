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



const MAX_COLORS: u32 = 6u;
const PI: f32 = 3.141592653589793;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
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

fn fbm4(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, d: vec2<f32>) -> vec4<f32> {
  return vec4<f32>(fbm1(a), fbm1(b), fbm1(c), fbm1(d));
}

fn colorAt(index: u32, path: CustomShaderData) -> vec4<f32> {
  return path.colorData[index];
}

fn premulStop(index: u32, path: CustomShaderData) -> vec4<f32> {
  let color = clamp(colorAt(index, path), vec4<f32>(0.0), vec4<f32>(1.0));
  return vec4<f32>(color.rgb * color.a, color.a);
}

fn gradientPremul(t: f32, count: u32, softness: f32, aa: f32, path: CustomShaderData) -> vec4<f32> {
  let n = max(count, 2u);
  let last = n - 1u;
  let scaled = clamp(t, 0.0, 1.0) * f32(last);
  let left = min(u32(floor(scaled)), last - 1u);
  let right = min(left + 1u, last);
  let local = fract(scaled);
  let width = max(aa * 1.5, mix(0.002, 0.5, clamp(softness, 0.0, 1.0)));
  let blend = smoothstep(0.5 - width, 0.5 + width, local);

  return mix(premulStop(left, path), premulStop(right, path), blend);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let pixel = input.position.xy;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = color count
  // path.shaderParams[0].z = softness
  // path.shaderParams[0].w = intensity
  // path.shaderParams[1].x = noise
  // path.shaderParams[1].y = scale
  // path.shaderParams[1].z = rotation
  // path.shaderParams[1].w = offsetX
  // path.shaderParams[2].x = offsetY
  // path.shaderParams[2].y = speed
  let time = path.shaderParams[0].x;
  let count = min(max(u32(path.shaderParams[0].y + 0.5), 2u), MAX_COLORS);
  let softness = clamp(path.shaderParams[0].z, 0.0, 1.0);
  let intensity = clamp(path.shaderParams[0].w, 0.0, 1.0);
  let noise = clamp(path.shaderParams[1].x, 0.0, 1.0);
  let scale = max(path.shaderParams[1].y, 0.001);
  let rotation = path.shaderParams[1].z;
  let offset = vec2<f32>(path.shaderParams[1].w, path.shaderParams[2].x);
  let speed = path.shaderParams[2].y;
  let t = time * speed;

  let aspect = b.z / max(b.w, 1.0);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let zoom = 1.0 / scale;
  p = rotate2(p * zoom + vec2<f32>(-offset.x * 0.35 * aspect, -offset.y * 0.35), -rotation);

  let slowDrift = vec2<f32>(t * 0.026, -t * 0.018);
  let organicA = fbm1(p * 1.15 + slowDrift);
  let organicB = fbm1(rotate2(p * 1.85 - slowDrift.yx, 0.9));
  let wave = sin((p.x * 1.35 + p.y * 0.52) * PI + organicA * 1.7 + t * 0.16);
  var shape = 0.5 + 0.28 * wave + 0.18 * organicA - 0.12 * organicB + p.y * 0.18;

  let grainUV = pixel + vec2<f32>(17.0, 43.0);
  let baseNoise = signedValueNoise(grainUV * 0.5 + vec2<f32>(t * 0.7, 0.0));
  let fbmVals = fbm4(
    grainUV * 0.002 + vec2<f32>(10.0 + t * 0.018, -4.0),
    grainUV * 0.003 + vec2<f32>(-31.0, t * 0.014),
    grainUV * 0.001 + vec2<f32>(55.0 - t * 0.012, 20.0),
    rotate2(grainUV * 0.4, 2.0) + vec2<f32>(0.0, t * 0.025)
  );

  let grainDist = baseNoise * signedValueNoise(grainUV * 0.2 + vec2<f32>(3.0, t * 0.05)) - fbmVals.x * 0.75 - fbmVals.y * 0.55;
  let rawNoise = 0.75 * baseNoise - fbmVals.z * 0.65 - fbmVals.w * 0.25;
  let grain = clamp(rawNoise * 0.5 + 0.5, 0.0, 1.0);
  let analogSpeckle = (grain - 0.5) * (0.55 + 0.45 * valueNoise(grainUV * 0.045 + fbmVals.xy));

  shape += intensity * 0.16 * grainDist;
  shape += noise * 0.105 * analogSpeckle;
  shape = clamp(shape, 0.0, 1.0);

  let aa = fwidth(shape) * f32(count - 1u);
  var premul = gradientPremul(shape, count, softness, aa, path);

  let grainOverlay = analogSpeckle * noise * 0.16 + grainDist * intensity * noise * 0.025;
  let grainOverlayColor = vec3<f32>(grainOverlay * max(premul.a, 0.18));
  premul = vec4<f32>(clamp(premul.rgb + grainOverlayColor, vec3<f32>(0.0), vec3<f32>(1.0)), premul.a);

  // premul.a is always 1.0 (packed gradient-stop alpha is always 1.0), so premul.rgb is already
  // valid straight-alpha and the framework epilogue's premultiply reproduces the original output.
  return vec4<f32>(premul.rgb, premul.a);
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
