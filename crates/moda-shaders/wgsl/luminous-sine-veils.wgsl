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


// Shader ID: luminous-sine-veils

const MAX_VEILS: u32 = 20u;
const MAX_VEIL_COLORS: u32 = 4u;
const VEIL_COLOR_OFFSET: u32 = 2u;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn packedColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  // colorData is a storage-buffer array: dynamic index (clamped to the last of
  // the 6 packed slots) instead of a branch chain.
  let safeIndex = min(index, 5u);
  return clamp(path.colorData[safeIndex].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn backgroundColor(uv: vec2<f32>, path: CustomShaderData) -> vec3<f32> {
  let topLight = smoothstep(-0.25, 0.85, uv.y + 0.18 * sin(uv.x * 2.0));
  return mix(packedColorAt(0u, path), packedColorAt(1u, path), topLight);
}

fn veilColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  return packedColorAt(VEIL_COLOR_OFFSET + min(index, MAX_VEIL_COLORS - 1u), path);
}

fn veilGradient(t: f32, colorCount: u32, path: CustomShaderData) -> vec3<f32> {
  let count = max(colorCount, 2u);
  let last = count - 1u;
  let scaled = clamp(t, 0.0, 1.0) * f32(last);
  let left = min(u32(floor(scaled)), last - 1u);
  let right = min(left + 1u, last);
  let local = smoothstep(0.0, 1.0, fract(scaled));
  return mix(veilColorAt(left, path), veilColorAt(right, path), local);
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  return color + vec3<f32>(grain * grainStrength);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  // Cache the raw shape-local UV once: the clamped copy drives the veil field
  // (which mutates uv below), while the raw copy feeds the background sample.
  let localUv = (input.localPos - b.xy) / b.zw;
  var uv = clamp(localUv, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = veil color count
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = exposure
  // path.shaderParams[1].x = glow
  // path.shaderParams[1].y = grain
  // path.shaderParams[1].z = speed
  // path.colorData[0..1].rgb = background colors
  // path.colorData[2..5].rgb = veil colors
  let time = path.shaderParams[0].x;
  let colorCount = min(max(u32(path.shaderParams[0].y + 0.5), 2u), MAX_VEIL_COLORS);
  let rotation = path.shaderParams[0].z;
  let exposure = path.shaderParams[0].w;
  let glow = path.shaderParams[1].x;
  let grain = path.shaderParams[1].y;
  let speed = path.shaderParams[1].z;

  let clock = time * speed;
  uv = rotate2(uv - vec2<f32>(0.5), rotation) + vec2<f32>(0.5);
  uv.y -= 0.58;
  uv = uv * 2.0;

  // Perf (docs/qa/shader-perf.md): the veil loop is the whole cost of this shader, so it carries only
  // per-pixel work. uv.x is fixed through the loop, so sin(uv.x + t + clock) advances by a constant
  // angle (1/20) each step: one sin/cos pair and a rotation recurrence replace 20 sins. The veil
  // gradient depends only on the step, so each step's two stop weights are accumulated per color stop
  // (4 scalars) and resolved against the colors once after the loop, instead of indexing the packed
  // colors 40 times per pixel. shaderColor only feeds a length: its sums are kept as two scalars.
  let lastStop = f32(colorCount - 1u);
  let stepSin = 0.04997916927067833; // sin(1 / 20)
  let stepCos = 0.9987502603949663; // cos(1 / 20)
  var waveSin = sin(uv.x + clock);
  var waveCos = cos(uv.x + clock);
  let xx = uv.x * uv.x + 0.16;
  var widthSum = 0.0;
  var widthYSum = 0.0;
  var veilStops = vec4<f32>(0.0);
  var glowStops = vec4<f32>(0.0);
  var colorWeight = 0.0;

  for (var i = 0u; i < MAX_VEILS; i = i + 1u) {
    let t = f32(i) / 20.0;
    uv.y += 0.04 * waveSin;
    let nextSin = waveSin * stepCos + waveCos * stepSin;
    waveCos = waveCos * stepCos - waveSin * stepSin;
    waveSin = nextSin;

    let safeY = max(abs(uv.y), 0.0001);
    let width = 1.0 / (250.0 * safeY);
    let glowWidth = width * (250.0 / 72.0);
    let layerWeight = clamp(width * sqrt(xx + 0.64 * uv.y * uv.y), 0.0, 1.0);
    // veilGradient(t) as weights on the 4 stops: (1 - local) on the left stop, local on the right.
    let scaled = t * lastStop;
    let left = min(floor(scaled), lastStop - 1.0);
    let local = smoothstep(0.0, 1.0, scaled - left);
    let stops = vec4<f32>(0.0, 1.0, 2.0, 3.0);
    let stopWeights = select(vec4<f32>(0.0), vec4<f32>(1.0 - local), stops == vec4<f32>(left)) +
      select(vec4<f32>(0.0), vec4<f32>(local), stops == vec4<f32>(left + 1.0));
    widthSum += width;
    widthYSum += width * uv.y;
    veilStops += stopWeights * layerWeight;
    glowStops += stopWeights * clamp(glowWidth, 0.0, 0.2);
    colorWeight += layerWeight;
  }

  let v0 = veilColorAt(0u, path);
  let v1 = veilColorAt(1u, path);
  let v2 = veilColorAt(2u, path);
  let v3 = veilColorAt(3u, path);
  let shaderColor = vec3<f32>(widthSum * uv.x, widthSum * 0.4, widthYSum * 0.8);
  let colorizedVeils = v0 * veilStops.x + v1 * veilStops.y + v2 * veilStops.z + v3 * veilStops.w;
  let glowColor = v0 * glowStops.x + v1 * glowStops.y + v2 * glowStops.z + v3 * glowStops.w;

  let mask = clamp(length(shaderColor) * exposure, 0.0, 1.0);
  let veilColor = colorizedVeils / max(colorWeight, 0.0001);
  let baseColor = backgroundColor(localUv, path);
  let glowContribution = glowColor * glow * 0.18 * exposure;
  var color = clamp(mix(baseColor, veilColor, mask) + glowContribution, vec3<f32>(0.0), vec3<f32>(1.0));
  color = filmGrain(color, pixel, grain);

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
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
