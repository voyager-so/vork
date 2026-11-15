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


const MAX_COLORS: u32 = 7u;
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const DISTORTION: f32 = 0.2;

fn rotateUV(p: vec2<f32>, a: f32) -> vec2<f32> {
  let ca = cos(a);
  let sa = sin(a);
  return vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
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
    q = rotateUV(q * 2.03 + vec2<f32>(19.17, -7.31), 0.62);
    amp *= 0.52;
  }

  return sum;
}

fn fbm4(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, d: vec2<f32>) -> vec4<f32> {
  return vec4<f32>(fbm1(a), fbm1(b), fbm1(c), fbm1(d));
}

fn colorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  var color = path.colorData[1].rgb;

  if (index == 1u) {
    color = path.colorData[2].rgb;
  } else if (index == 2u) {
    color = path.colorData[3].rgb;
  } else if (index == 3u) {
    color = path.colorData[4].rgb;
  } else if (index == 4u) {
    color = path.colorData[5].rgb;
  } else if (index == 5u) {
    color = path.colorData[6].rgb;
  } else if (index >= 6u) {
    color = path.colorData[7].rgb;
  }

  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn smoother(t: f32) -> f32 {
  let x = clamp(t, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

fn heatmapPalette(t: f32, count: u32, path: CustomShaderData) -> vec3<f32> {
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

fn getImgFrame(uv: vec2<f32>, th: f32) -> f32 {
  var frame = 1.0;
  frame *= smoothstep(0.0, th, uv.y);
  frame *= 1.0 - smoothstep(1.0 - th, 1.0, uv.y);
  frame *= smoothstep(0.0, th, uv.x);
  frame *= 1.0 - smoothstep(1.0 - th, 1.0, uv.x);
  return frame;
}

fn sst(edge0: f32, edge1: f32, x: f32) -> f32 {
  return smoothstep(edge0, edge1, x);
}

fn blurEdge3x3(tex: texture_2d<f32>, samplerTex: sampler, uv: vec2<f32>, dudx: vec2<f32>, dudy: vec2<f32>, radius: f32, centerSample: f32) -> f32 {
  let texel = 1.0 / vec2<f32>(textureDimensions(tex));
  let r = radius * texel;

  var sum = 4.0 * centerSample;
  sum += 2.0 * textureSampleGrad(tex, samplerTex, uv + vec2<f32>(0.0, -r.y), dudx, dudy).g;
  sum += 2.0 * textureSampleGrad(tex, samplerTex, uv + vec2<f32>(0.0, r.y), dudx, dudy).g;
  sum += 2.0 * textureSampleGrad(tex, samplerTex, uv + vec2<f32>(-r.x, 0.0), dudx, dudy).g;
  sum += 2.0 * textureSampleGrad(tex, samplerTex, uv + vec2<f32>(r.x, 0.0), dudx, dudy).g;
  sum += textureSampleGrad(tex, samplerTex, uv + vec2<f32>(-r.x, -r.y), dudx, dudy).g;
  sum += textureSampleGrad(tex, samplerTex, uv + vec2<f32>(r.x, -r.y), dudx, dudy).g;
  sum += textureSampleGrad(tex, samplerTex, uv + vec2<f32>(-r.x, r.y), dudx, dudy).g;
  sum += textureSampleGrad(tex, samplerTex, uv + vec2<f32>(r.x, r.y), dudx, dudy).g;

  return sum / 16.0;
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = repetition
  // path.shaderParams[0].z = speed
  // path.shaderParams[0].w = color count
  // path.shaderParams[1].x = stripe angle degrees
  // path.shaderParams[1].y = inner glow
  // path.shaderParams[1].z = outer glow
  // path.shaderParams[1].w = image scale
  // path.shaderParams[2].x = image rotation radians
  // path.shaderParams[2].y = image offset X
  // path.shaderParams[2].z = image offset Y
  // path.shaderParams[2].w = image aspect
  // path.shaderParams[3].x = image ready
  // path.shaderParams[3].y = analog grain
  // path.colorData[0].rgb = background color
  // path.colorData[1..7].rgb = heatmap gradient colors
  // heatmap image channels: r = contour, g = outer blur, b = inner blur

  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];

  let time = p0.x;
  let rep = clamp(p0.y, 1.0, 4.0);
  let speed = max(p0.z, 0.0);
  let count = min(max(u32(p0.w + 0.5), 2u), MAX_COLORS);

  let stripeAngle = p1.x * PI / 180.0;
  let innerGlow = clamp(p1.y, 0.0, 1.0);
  let outerGlow = clamp(p1.z, 0.0, 1.0);
  let imageScale = vec2<f32>(max(p1.w, 0.001), max(path.shaderParams[3].z, 0.001));
  let imageRotation = p2.x;
  let imageOffset = vec2<f32>(p2.y, p2.z);

  let sourceDims = max(vec2<f32>(path.shaderParams[3].w, path.shaderParams[4].x), vec2<f32>(1.0));
  let imageReady = path.shaderParams[3].x;
  let analogGrainAmount = clamp(path.shaderParams[3].y, 0.0, 1.0);

  let u0 = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let sourceLocal = (u0 - vec2<f32>(0.5)) * b.zw + b.zw * 0.5;
  var u = (sourceLocal - imageOffset) / sourceDims;
  u = rotateUV(u - vec2<f32>(0.5), imageRotation) + vec2<f32>(0.5);
  u = u / imageScale;
  let heatmapContentOffset = path.shaderParams[4].yz;
  let heatmapContentScale = vec2<f32>(path.shaderParams[4].w, path.shaderParams[5].x);
  u = heatmapContentOffset + u * heatmapContentScale;

  let dx = dpdx(u);
  let dy = dpdy(u);
  let px = textureSampleGrad(shaderImage, shaderImageSampler, u, dx, dy);
  let imgSoftFrame = getImgFrame(u, 0.03);

  let tm = time * speed * 0.5;
  // Framebuffer coords for grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  let pixel = input.position.xy;
  let grainUV = pixel + vec2<f32>(17.0, 43.0);
  let baseNoise = signedValueNoise(grainUV * 2.0 + vec2<f32>(tm * 0.7, 0.0));
  let fbmVals = fbm4(
    grainUV * 0.002 + vec2<f32>(10.0 + tm * 0.018, -4.0),
    grainUV * 0.003 + vec2<f32>(-31.0, tm * 0.014),
    grainUV * 0.001 + vec2<f32>(55.0 - tm * 0.012, 20.0),
    rotateUV(grainUV * 0.4, 2.0) + vec2<f32>(0.0, tm * 0.025)
  );

  let grainDist = baseNoise * signedValueNoise(grainUV * 0.2 + vec2<f32>(3.0, tm * 0.05)) - fbmVals.x * 0.75 - fbmVals.y * 0.55;
  let rawNoise = 0.75 * baseNoise - fbmVals.z * 0.65 - fbmVals.w * 0.25;
  let analogGrain = clamp(rawNoise * 0.5 + 0.5, 0.0, 1.0);
  let analogSpeckle = sign(analogGrain - 0.5) * pow(abs(analogGrain - 0.5) * 50.0, 0.55) * 0.25;

  let shape = px.r;
  let outerBlurSample = blurEdge3x3(shaderImage, shaderImageSampler, u, dx, dy, 8.0, px.g);
  let outerBlur = (1.0 - mix(1.0, outerBlurSample, shape)) * imgSoftFrame;
  let innerBlur = mix(outerBlurSample, 0.0, shape);
  let contour = clamp(mix(px.b, 0.0, shape) + analogGrainAmount * (0.06 * grainDist + 0.0125 * analogSpeckle), 0.0, 1.0);
  let imageReadyMask = step(0.5, imageReady);
  let logoMask = clamp(1.0 - shape, 0.0, 1.0) * frameMask(u) * imageReadyMask;

  let g0 = u - vec2<f32>(0.5);
  let dir = vec2<f32>(cos(stripeAngle), sin(stripeAngle));
  let linear = dot(g0, dir);

  let edgeWrap = contour * DISTORTION;
  let z = clamp(1.0 - pow(1.7 * length(g0 + dir * edgeWrap * 0.18), 1.25), 0.0, 1.0);

  let repNorm = clamp((rep - 1.0) / 3.0, 0.0, 1.0);
  let visualRep = mix(0.85, 4.25, pow(repNorm, 0.65));

  let phase = linear * visualRep - tm - edgeWrap * 1.35 + z * 0.08;
  let paletteT = clamp(0.5 + 0.5 * sin(phase * TAU) + analogGrainAmount * (0.06 * grainDist + 0.05 * analogSpeckle), 0.0, 1.0);
  let shade = softShade(phase, z);

  let heatmapBase = heatmapPalette(paletteT, count, path);
  let sheen = vec3<f32>(0.98, 0.99, 1.0) * smoothstep(0.7, 1.0, shade);
  let heatmap = clamp(
    heatmapBase * (0.38 + 0.86 * shade) +
    sheen * 0.1,
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );

  let bg = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  var outerT = fract(tm - 0.25);

  var outer = 0.9 * pow(outerBlur, 0.8);
  let y = fract(linear * visualRep - outerT - edgeWrap * 1.35 + z * 0.08);
  var animatedMask = sst(0.3, 0.65, y) * (1.0 - sst(0.65, 1.0, y));
  animatedMask += 0.5;
  outer *= animatedMask;
  outer *= mix(0.0, 5.0, pow(outerGlow, 2.0));
  outer *= imgSoftFrame * imageReadyMask;

  var inner = (0.8 + 0.8 * innerBlur) * innerGlow;

  let innerY = fract(linear * visualRep - tm - edgeWrap * 1.35 + z * 0.08);
  let innerWave = sst(0.3, 0.65, innerY) * (1.0 - sst(0.65, 1.0, innerY));

  inner *= 0.55 + 0.45 * innerWave;
  inner += contour * 2.0;
  inner = pow(min(1.0, inner) * logoMask, 1.2);

  let outerHeat = clamp(outer, 0.0, 1.0);
  let outerMixer = outerHeat * f32(count);
  var outerShape = 0.0;
  var outerColor = colorAt(0u, path);
  for (var i = 1u; i <= MAX_COLORS; i += 1u) {
    if (i > count) {
      break;
    }
    let m = clamp(outerMixer - f32(i - 1u), 0.0, 1.0);
    if (i == 1u) {
      outerShape = m;
    }
    outerColor = mix(outerColor, colorAt(i - 1u, path), m);
  }

  let grainOverlay = analogSpeckle * analogGrainAmount * 0.16 + grainDist * analogGrainAmount * analogGrainAmount * 0.025;
  // Framebuffer coords for grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  let grain = (hash21(input.position.xy + vec2<f32>(time * 17.13, -time * 9.47)) - 0.5) / 255.0;
  let innerRgb = mix(bg, heatmap, inner);
  let outerRgb = mix(innerRgb, outerColor, outerShape * (1.0 - logoMask));
  let rgb = clamp(outerRgb + vec3<f32>(grain + grainOverlay), vec3<f32>(0.0), vec3<f32>(1.0));

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
