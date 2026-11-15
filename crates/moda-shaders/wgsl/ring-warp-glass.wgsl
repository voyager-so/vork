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


// Shader ID: ring-warp-glass

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

fn swirl(uv: vec2<f32>, time: f32, path: CustomShaderData) -> vec3<f32> {
  let f = swirlField(uv, time);
  let colorA = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorB = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  return colorA * f.x + (colorB - colorA) * f.y;
}

// The swirl colour is mix(colorA, colorB, blend) * shimmer = colorA * shimmer + (colorB - colorA) *
// (blend * shimmer): linear in (shimmer, blend * shimmer), so the 27-tap chromatic blur below averages
// these two scalars per channel and resolves the colours once.
fn swirlField(uv: vec2<f32>, time: f32) -> vec2<f32> {
  let t = time * 0.5;
  let detail = 4.2;
  let d1 = vec2<f32>(
    uv.x + sin(uv.y * detail * 1.7 + t * 0.8) * 0.12 + cos(uv.x * detail * 0.9 - t * 0.5) * 0.05,
    uv.y + cos(uv.x * detail * 1.3 - t * 0.6) * 0.12 + sin(uv.y * detail * 1.1 + t * 0.7) * 0.05
  );
  let p1 = sin(d1.x * detail * 2.1 + d1.y * detail * 1.8 + t * 0.4);
  let freq2 = detail * 2.1;
  let d2 = vec2<f32>(
    d1.x + cos(d1.y * freq2 * 2.7 - t * 0.45) * 0.07 + sin(d1.x * freq2 * 1.9 + t * 0.6) * 0.04,
    d1.y + sin(d1.x * freq2 * 2.3 + t * 0.65) * 0.07 + cos(d1.y * freq2 * 1.6 - t * 0.4) * 0.04
  );
  let p2 = cos(d2.x * freq2 * 1.4 - d2.y * freq2 * 1.9 + t * 0.35);
  let freq3 = detail * 3.7;
  let d3 = vec2<f32>(
    d2.x + sin(d2.y * freq3 * 1.8 + t * 0.85) * 0.04 + cos(d2.x * freq3 * 1.3 - t * 0.55) * 0.025 + sin((d2.x + d2.y) * freq3 * 0.7 + t * 0.9) * 0.02,
    d2.y + cos(d2.x * freq3 * 1.6 - t * 0.75) * 0.04 + sin(d2.y * freq3 * 1.1 + t * 0.5) * 0.025 + cos((d2.x + d2.y) * freq3 * 0.8 - t * 0.95) * 0.02
  );
  let p3 = sin(d3.x * freq3 * 1.1 + d3.y * freq3 * 1.5 - t * 0.55);
  let combined = p1 * 0.45 + p2 * 0.35 + p3 * 0.2;
  let blendFactor = smoothstep(0.3, 0.7, combined * 0.5 + 0.5 - 0.192);
  let shimmer = sin(t * 2.5 + combined * 8.0) * 0.015 + 1.0;

  return vec2<f32>(shimmer, blendFactor * shimmer);
}

fn ringSdfAt(uv: vec2<f32>, resolution: vec2<f32>, ringWidth: f32) -> f32 {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let center = vec2<f32>(0.5, 1.0);
  let d = vec2<f32>((uv.x - center.x) * aspect, uv.y - center.y);
  let ringFrequency = 5.5 / max(ringWidth, 0.001);
  let radius = length(d) * ringFrequency;
  let cell = fract(radius);
  let distanceToBoundary = min(cell, 1.0 - cell);

  return -distanceToBoundary / ringFrequency;
}

fn blurredSwirlField(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  blur: f32
) -> vec2<f32> {
  let pixel = 1.0 / max(resolution, vec2<f32>(1.0));
  let radius = 40.0 * blur;
  var acc = vec2<f32>(0.0);

  acc += swirlField(uv + pixel * vec2<f32>(0.0, 0.0) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(-0.737, 0.675) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(0.087, -0.996) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(0.608, 0.794) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(-0.985, -0.174) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(0.844, -0.537) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(-0.259, 0.966) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(-0.460, -0.888) * radius, time);
  acc += swirlField(uv + pixel * vec2<f32>(0.940, 0.342) * radius, time);

  return acc / 9.0;
}

fn glass(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  refraction: f32,
  chromaticAberration: f32,
  blur: f32,
  ringWidth: f32,
  path: CustomShaderData
) -> vec4<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let center = vec2<f32>(0.5);
  let rawSdf = ringSdfAt(uv, resolution, ringWidth);
  let pxH = 1.0 / max(resolution.y, 1.0);

  let eps = 0.01;
  let gradX = (ringSdfAt(uv + vec2<f32>(eps, 0.0), resolution, ringWidth) - rawSdf) / eps;
  let gradY = (ringSdfAt(uv + vec2<f32>(0.0, eps), resolution, ringWidth) - rawSdf) / eps;
  let sharp = 0.05;
  let rb1 = clamp(-rawSdf / sharp * 32.0, 0.0, 1.0);
  let rb2Base = clamp(-(rawSdf - pxH) / sharp * 16.0, 0.0, 1.0) - clamp(-rawSdf / sharp * 16.0, 0.0, 1.0);
  let lightDir = vec2<f32>(cos(300.0 * PI / 180.0), sin(300.0 * PI / 180.0));
  let lightFacing = dot(vec2<f32>(gradX, gradY), lightDir) * 0.5 + 0.5;
  let rb2 = rb2Base * lightFacing * 0.3;
  let depthNorm = clamp(-rawSdf / 0.3, 0.0, 1.0);
  let refrStrength = (1.0 - depthNorm) * (1.0 - depthNorm);
  let offset = vec2<f32>(-gradX / aspect, -gradY) * 1.57 * 0.15 * refraction * refrStrength;
  let lensUv = center + (uv - center) + offset;
  let chroma = offset * 0.06 * chromaticAberration;
  let fieldR = blurredSwirlField(lensUv + chroma, resolution, time, blur);
  let fieldG = blurredSwirlField(lensUv, resolution, time, blur);
  let fieldB = blurredSwirlField(lensUv - chroma, resolution, time, blur);
  let colorA = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorB = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let blurred = colorA * vec3<f32>(fieldR.x, fieldG.x, fieldB.x) +
    (colorB - colorA) * vec3<f32>(fieldR.y, fieldG.y, fieldB.y);
  let n = normalize(vec3<f32>(gradX, gradY, 2.0));
  let h = normalize(vec3<f32>(lightDir, 2.0));
  let shininess = exp2(8.0 - 0.5 * 7.0);
  let spec = pow(clamp(dot(n, h), 0.0, 1.0), shininess) * 0.3 * refrStrength;
  let fresnelDepth = clamp(-rawSdf / max(0.31 * 0.06, 0.001), 0.0, 1.0);
  let fresnel = pow(1.0 - fresnelDepth, 2.0) * 0.02 * rb1;
  let lighting = blurred + vec3<f32>(rb2 + spec + fresnel);
  let transition = smoothstep(0.0, 1.0, rb1);
  // Away from a ring boundary the lens covers the swirl entirely (transition = 1).
  var base = lighting;
  if (transition < 1.0) {
    base = swirl(uv, time, path);
  }

  return vec4<f32>(mix(base, lighting, transition), 1.0);
}

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

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hashToUnit(hashU32(bitcast<u32>(fragCoord.x) ^ hashU32(bitcast<u32>(fragCoord.y)))) * 2.0 - 1.0;
  let brightness = clamp(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  let darkFactor = pow(1.0 - brightness + 0.000001, 2.0);

  return color + vec3<f32>(grain * darkFactor * 0.035 * grainStrength);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = refraction
  // path.shaderParams[1].z = chromatic aberration
  // path.shaderParams[1].w = blur
  // path.shaderParams[2].x = ring width
  // path.shaderParams[2].y = grain
  // path.shaderParams[2].z = speed
  // path.colorData[0..1].rgb = swirl dark, light
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let refraction = path.shaderParams[1].y;
  let chromaticAberration = path.shaderParams[1].z;
  let blur = path.shaderParams[1].w;
  let ringWidth = path.shaderParams[2].x;
  let grain = path.shaderParams[2].y;
  let speed = path.shaderParams[2].z;
  let t = time * speed;
  let aspect = resolution.x / resolution.y;
  let effectUv = transformUv(uv, aspect, scale, rotation, offset);

  let lens = glass(effectUv, resolution, t, refraction, chromaticAberration, blur, ringWidth, path);
  let composed = lens.rgb;
  // Framebuffer coords for film grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not), so the grain is
  // identical on every backend.
  let color = filmGrain(composed, input.position.xy, grain);

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
