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


// Shader ID: swirl-edge-bands-gradient

const TWO_PI: f32 = 6.283185307179586;
const NOISE_DISTORTION_GAIN: f32 = 0.58;
const PATTERN_SCALE: f32 = 0.58;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn remapSceneScale(controlScale: f32) -> f32 {
  let s = clamp(controlScale, 0.0, 1.0);
  let closestZoom = 0.38;
  let formerZeroZoom = 1.15;
  return mix(closestZoom, formerZeroZoom, s);
}

fn colorStop(index: u32, path: CustomShaderData) -> vec4<f32> {
  if (index == 1u) {
    return clamp(path.colorData[1], vec4<f32>(0.0), vec4<f32>(1.0));
  }

  if (index >= 2u) {
    return clamp(path.colorData[2], vec4<f32>(0.0), vec4<f32>(1.0));
  }

  return clamp(path.colorData[0], vec4<f32>(0.0), vec4<f32>(1.0));
}

fn blendThreeColors(mixer: f32, edgeWidth: f32, edgeBlur: f32, path: CustomShaderData) -> vec4<f32> {
  let c1 = colorStop(0u, path);
  let c2 = colorStop(1u, path);
  let c3 = colorStop(2u, path);
  let r1 = smoothstep(0.35 * edgeWidth, 0.7 - 0.35 * edgeWidth + 0.5 * edgeBlur, mixer);
  let r2 = smoothstep(0.3 + 0.35 * edgeWidth, 1.0 - 0.35 * edgeWidth + edgeBlur, mixer);
  let color12 = mix(c1.rgb, c2.rgb, r1);
  let alpha12 = mix(c1.a, c2.a, r1);

  return vec4<f32>(mix(color12, c3.rgb, r2), mix(alpha12, c3.a, r2));
}

fn warpedCoord(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>,
  distortion: f32,
  swirl: f32,
  iterations: u32
) -> vec2<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let zoom = max(scale, 0.001);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  p = rotate2(p * zoom + vec2<f32>(-offset.x * aspect, -offset.y), -rotation);

  let n1 = valueNoise(p * 1.15 + vec2<f32>(time * 0.16, time * 0.13));
  let n2 = valueNoise(p * 2.35 + vec2<f32>(-time * 0.21, time * 0.18));
  let angle = n1 * TWO_PI;
  p += vec2<f32>(cos(angle), sin(angle)) * n2 * distortion * NOISE_DISTORTION_GAIN;

  let swirlAmount = clamp(swirl, 0.0, 2.0);
  for (var i = 1u; i <= 30u; i = i + 1u) {
    if (i <= iterations) {
      let fi = f32(i);
      p.x += swirlAmount / fi * cos(time + fi * 1.5 * p.y) * 0.12;
      p.y += swirlAmount / fi * cos(time + fi * p.x) * 0.12;
    }
  }

  return p;
}

fn patternMixer(p: vec2<f32>, proportion: f32, shapeScale: f32) -> f32 {
  let prop = clamp(proportion, 0.0, 1.0);
  let scale = clamp(shapeScale, 0.0, 1.0);
  let edgeWidth = 0.34 * (1.0 - scale);
  return smoothstep(
    -edgeWidth,
    edgeWidth,
    p.y + 0.34 * (prop - 0.5) + sin(p.x * 1.7) * 0.04
  );
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  let luma = clamp(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  return color + vec3<f32>(grain * grainStrength * (0.35 + 0.65 * pow(1.0 - luma, 2.0)));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = proportion
  // path.shaderParams[1].z = softness
  // path.shaderParams[1].w = distortion
  // path.shaderParams[2].x = swirl
  // path.shaderParams[2].y = swirl iterations
  // path.shaderParams[2].z = grain
  // path.shaderParams[2].w = speed
  // path.colorData[0..2] = colors
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let proportion = path.shaderParams[1].y;
  let softness = clamp(path.shaderParams[1].z, 0.0, 1.0);
  let distortion = path.shaderParams[1].w;
  let swirl = path.shaderParams[2].x;
  let iterations = min(max(u32(path.shaderParams[2].y + 0.5), 1u), 30u);
  let grain = path.shaderParams[2].z;
  let speed = path.shaderParams[2].w;
  let t = time * speed * 0.5;
  let sceneScale = remapSceneScale(scale);

  let p = warpedCoord(uv, resolution, t, sceneScale, rotation, offset, distortion, swirl, iterations);
  let mixer = patternMixer(p, proportion, PATTERN_SCALE);
  let edgesWidth = 1.0 - softness;
  let edgeBlur = mix(0.012, 0.034, scale);
  let mixed = blendThreeColors(mixer, edgesWidth, edgeBlur, path);
  let vignetteP = vec2<f32>((uv.x - 0.5) * resolution.x / resolution.y, uv.y - 0.5);
  let vignette = smoothstep(1.05, 0.18, length(vignetteP));
  let color = filmGrain(clamp(mixed.rgb * mix(0.86, 1.08, vignette), vec3<f32>(0.0), vec3<f32>(1.0)), pixel, grain);

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), mixed.a);
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
