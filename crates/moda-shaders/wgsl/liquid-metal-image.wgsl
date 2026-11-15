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


@group(2) @binding(0) var shaderImage: texture_2d<f32>;
@group(2) @binding(1) var shaderImageSampler: sampler;


struct ClipMaskUniforms {
  maskActive: u32,
  maskOrigin: vec2<f32>,
  maskInvSize: vec2<f32>,
}
@group(3) @binding(0) var clipMaskSampler: sampler;
@group(3) @binding(1) var clipMaskTexture: texture_2d<f32>;
@group(3) @binding(2) var<uniform> clipMask: ClipMaskUniforms;

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


const MAX_COLORS: u32 = 4u;
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

fn rotateUV(p: vec2<f32>, a: f32) -> vec2<f32> {
  let ca = cos(a);
  let sa = sin(a);
  return vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
}

fn colorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  var color = path.colorData[1].rgb;

  if (index == 1u) {
    color = path.colorData[2].rgb;
  } else if (index == 2u) {
    color = path.colorData[3].rgb;
  } else if (index >= 3u) {
    color = path.colorData[4].rgb;
  }

  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn smoother(t: f32) -> f32 {
  let x = clamp(t, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

fn paletteMetal(t: f32, count: u32, path: CustomShaderData) -> vec3<f32> {
  let n = max(count, 2u);
  let last = n - 1u;
  let scaled = clamp(t, 0.0, 1.0) * f32(last);
  let left = min(u32(floor(scaled)), last - 1u);
  let right = min(left + 1u, last);
  let blend = smoother(fract(scaled));

  return mix(colorAt(left, path), colorAt(right, path), blend);
}

fn softShade(phase: f32, ridge: f32) -> f32 {
  let primary = 0.5 + 0.5 * sin(phase * TAU);
  let secondary = 0.5 + 0.5 * sin((phase * 0.43 + ridge * 0.16 + 0.21) * TAU);
  let glow = smoother(primary);
  let valley = smoother(secondary);

  return clamp(0.12 + glow * 0.78 - valley * 0.16 + ridge * 0.2, 0.0, 1.0);
}

fn frameMask(u: vec2<f32>) -> f32 {
  let leftBottom = vec2<f32>(smoothstep(0.0, 0.001, u.x), smoothstep(0.0, 0.001, u.y));
  let rightTop = vec2<f32>(1.0) - vec2<f32>(smoothstep(0.999, 1.0, u.x), smoothstep(0.999, 1.0, u.y));
  return leftBottom.x * leftBottom.y * rightTop.x * rightTop.y;
}

fn sampleEdgeSmooth(u: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, rr: f32, c: f32) -> f32 {
  let e = rr / max(vec2<f32>(textureDimensions(shaderImage)), vec2<f32>(1.0));
  let ed = e * 0.707106781;

  var s = 0.159576912 * c;
  s += 0.098320332 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>( e.x, 0.0), dx, dy).r;
  s += 0.098320332 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>(-e.x, 0.0), dx, dy).r;
  s += 0.098320332 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>( 0.0, e.y), dx, dy).r;
  s += 0.098320332 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>( 0.0,-e.y), dx, dy).r;
  s += 0.059634295 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>( ed.x, ed.y), dx, dy).r;
  s += 0.059634295 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>(-ed.x, ed.y), dx, dy).r;
  s += 0.059634295 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>( ed.x,-ed.y), dx, dy).r;
  s += 0.059634295 * textureSampleGrad(shaderImage, shaderImageSampler, u + vec2<f32>(-ed.x,-ed.y), dx, dy).r;

  return s;
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = repetition
  // path.shaderParams[0].z = speed
  // path.shaderParams[0].w = color count
  // path.shaderParams[1].x = stripe angle degrees
  // path.shaderParams[1].y = distortion
  // path.shaderParams[1].z = image scale
  // path.shaderParams[1].w = image rotation radians
  // path.shaderParams[2].x = image offset X
  // path.shaderParams[2].y = image offset Y
  // path.shaderParams[2].z = image aspect
  // path.shaderParams[2].w = image ready
  // path.colorData[0].rgb = background color
  // path.colorData[1..4].rgb = metal gradient colors
  // poisson-mask image channels: r = edge field, g = opacity mask

  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];

  let time = p0.x;
  let rep = clamp(p0.y, 1.0, 4.0);
  let speed = max(p0.z, 0.0);
  let count = min(max(u32(p0.w + 0.5), 2u), MAX_COLORS);

  let stripeAngle = p1.x * PI / 180.0;
  let distortion = clamp(p1.y, 0.0, 2.0);
  let imageScale = vec2<f32>(max(p1.z, 0.001), max(path.shaderParams[3].x, 0.001));
  let imageRotation = p1.w;
  let imageOffset = vec2<f32>(p2.x, p2.y);

  let sourceDims = max(path.shaderParams[3].yz, vec2<f32>(1.0));
  let imageReady = p2.w;

  let u0 = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let sourceLocal = (u0 - vec2<f32>(0.5)) * b.zw + b.zw * 0.5;
  var u = (sourceLocal - imageOffset) / sourceDims;
  u = rotateUV(u - vec2<f32>(0.5), imageRotation) + vec2<f32>(0.5);
  u = u / imageScale;

  let dx = dpdx(u);
  let dy = dpdy(u);
  let px = textureSampleGrad(shaderImage, shaderImageSampler, u, dx, dy);

  let tm = time * speed * 0.5;
  let edgeField = pow(sampleEdgeSmooth(u, dx, dy, 6.0, px.r), 1.6);
  let imageMask = px.g * frameMask(u) * step(0.5, imageReady);

  let g0 = u - vec2<f32>(0.5);
  let dir = vec2<f32>(cos(stripeAngle), sin(stripeAngle));
  let linear = dot(g0, dir);

  let edgeWrap = edgeField * distortion;
  let z = clamp(1.0 - pow(1.7 * length(g0 + dir * edgeWrap * 0.18), 1.25), 0.0, 1.0);

  let repNorm = clamp((rep - 1.0) / 3.0, 0.0, 1.0);
  let visualRep = mix(0.85, 4.25, pow(repNorm, 0.65));

  let phase = linear * visualRep - tm - edgeWrap * 1.35 + z * 0.08;
  let paletteT = 0.5 + 0.5 * sin(phase * TAU);
  let shade = softShade(phase, z);

  let metalBase = paletteMetal(paletteT, count, path);
  let sheen = vec3<f32>(0.98, 0.99, 1.0) * smoothstep(0.7, 1.0, shade);

  let metal = clamp(
    metalBase * (0.32 + 0.8 * shade) +
    sheen * 0.14,
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );

  let bg = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  // Framebuffer coords for grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  let grain = (hash21(input.position.xy + vec2<f32>(time * 17.13, -time * 9.47)) - 0.5) / 255.0;
  let rgb = clamp(mix(bg, metal, imageMask) + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(1.0));

  return vec4<f32>(rgb, 1.0);
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
