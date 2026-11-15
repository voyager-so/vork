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


// Shader ID: center-wave-bars

const WAVE_BARS_PI: f32 = 3.14159265359;

fn waveBarsGradient(t: f32, path: CustomShaderData) -> vec3<f32> {
  let c0 = path.colorData[1].rgb;
  let c1 = path.colorData[2].rgb;
  let c2 = path.colorData[3].rgb;
  let c3 = path.colorData[4].rgb;

  if (t < 0.3333333333) {
    return mix(c0, c1, vec3<f32>(smoothstep(0.0, 0.3333333333, t)));
  } else if (t < 0.6666666667) {
    return mix(c1, c2, vec3<f32>(smoothstep(0.3333333333, 0.6666666667, t)));
  }

  return mix(c2, c3, vec3<f32>(smoothstep(0.6666666667, 1.0, t)));
}

fn waveBarHeight(
  x: f32,
  time: f32,
  center: f32,
  baseHeight: f32,
  heightSpread: f32,
  waveDelay: f32
) -> f32 {
  let distanceFromCenter = abs(x - center);
  let height = baseHeight + heightSpread * (distanceFromCenter * 3.0);
  let animationPhase = time / 4.75 * 2.0 * WAVE_BARS_PI;
  let delay = distanceFromCenter * waveDelay;
  let delayedWave = sin(animationPhase - delay);
  let heightModifier = clamp(1.0 + delayedWave * 0.12, 0.8, 1.2);

  return height * heightModifier;
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let hostUv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let uv = vec2<f32>(hostUv.x, 1.0 - hostUv.y);
  // Grain per framebuffer pixel: an integer coordinate hashes the same in the direct draw and the pre-pass.
  let pixel = floor(input.position.xy);

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = bar width
  // path.shaderParams[0].z = center
  // path.shaderParams[0].w = base height
  // path.shaderParams[1].x = height spread
  // path.shaderParams[1].y = speed
  // path.shaderParams[1].z = wave delay
  // path.shaderParams[1].w = grain
  // path.colorData[0].rgb = background color
  // path.colorData[1..4].rgb = bar gradient colors, bottom to top
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];

  let time = p0.x;
  let barWidth = max(p0.y, 0.0001);
  let center = p0.z;
  let baseHeight = p0.w;
  let heightSpread = p1.x;
  let speed = max(p1.y, 0.0);
  let waveDelay = p1.z;
  let grain = p1.w;
  let t = time * speed;

  let barIndex = floor(uv.x / barWidth);
  let barCenterX = (barIndex + 0.5) * barWidth;
  let height = waveBarHeight(barCenterX, t, center, baseHeight, heightSpread, waveDelay);
  let barTopY = height * 0.85;
  let backgroundColor = path.colorData[0].rgb;
  var finalColor = backgroundColor;

  if (uv.y < barTopY && uv.y > 0.0) {
    let heightFactor = uv.y / barTopY;
    let barColor = waveBarsGradient(heightFactor, path);
    let fade = 1.0 - smoothstep(barTopY - 0.25, barTopY, uv.y);
    finalColor = mix(backgroundColor, barColor, vec3<f32>(fade));
  }

  finalColor += vec3<f32>((hash21(pixel) - 0.5) * grain);

  return vec4<f32>(clamp(finalColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
