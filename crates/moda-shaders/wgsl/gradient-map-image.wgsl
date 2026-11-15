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


const MAX_COLORS: u32 = 8u;
const IMAGE_LOADING_COLOR = vec3<f32>(0.9, 0.9, 0.9);
const IMAGE_FAILED_COLOR = vec3<f32>(0.85, 0.74, 0.74);

fn rotateSourceUv(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn sourceImageUv(layerPos: vec2<f32>, path: CustomShaderData) -> vec2<f32> {
  let rawScale = path.shaderParams[0].zw;
  let scaleSign = select(vec2<f32>(-1.0), vec2<f32>(1.0), rawScale >= vec2<f32>(0.0));
  let imageScale = scaleSign * max(abs(rawScale), vec2<f32>(0.001));
  let imageRotation = path.shaderParams[1].x;
  let imageOffset = path.shaderParams[1].yz;
  let sourceDims = max(vec2<f32>(path.shaderParams[1].w, path.shaderParams[2].x), vec2<f32>(1.0));

  var uv = (layerPos - imageOffset) / sourceDims;
  uv = rotateSourceUv(uv, imageRotation);
  return uv / imageScale;
}

fn sourceImageFrame(uv: vec2<f32>) -> f32 {
  let lower = step(vec2<f32>(0.0), uv);
  let upper = step(uv, vec2<f32>(1.0));
  return lower.x * lower.y * upper.x * upper.y;
}

fn gradientMapColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  var color = path.colorData[0].rgb;
  if (index == 1u) { color = path.colorData[1].rgb; }
  else if (index == 2u) { color = path.colorData[2].rgb; }
  else if (index == 3u) { color = path.colorData[3].rgb; }
  else if (index == 4u) { color = path.colorData[4].rgb; }
  else if (index == 5u) { color = path.colorData[5].rgb; }
  else if (index == 6u) { color = path.colorData[6].rgb; }
  else if (index >= 7u) { color = path.colorData[7].rgb; }
  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gradientMapColor(value: f32, colorCount: u32, path: CustomShaderData) -> vec3<f32> {
  let count = min(max(colorCount, 2u), MAX_COLORS);
  let last = count - 1u;
  let scaled = clamp(value, 0.0, 1.0) * f32(last);
  let left = min(u32(floor(scaled)), last);
  let right = min(left + 1u, last);
  let amount = smoothstep(0.0, 1.0, fract(scaled));
  return mix(gradientMapColorAt(left, path), gradientMapColorAt(right, path), vec3<f32>(amount));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  // Shared source placement ABI: [0..8] = aspect, ready, scale X/Y,
  // rotation, offset X/Y, source width/height. [9..11] are Gradient Map.
  // Normal pattern fills map from raw fill-local coordinates; tessellation
  // bounds include stroke expansion and must not shift the image field.
  let layerPos = input.localPos;
  let uv = sourceImageUv(layerPos, path);
  let source = textureSampleGrad(shaderImage, shaderImageSampler, uv, dpdx(uv), dpdy(uv));
  let imageState = path.shaderParams[2].w;
  if (imageState < 0.5) {
    return vec4<f32>(IMAGE_LOADING_COLOR, 1.0);
  }
  if (imageState > 1.5) {
    return vec4<f32>(IMAGE_FAILED_COLOR, 1.0);
  }
  var luma = dot(source.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  luma = 1.0 - abs(fract((luma + path.shaderParams[2].y) * 0.5) * 2.0 - 1.0);
  let count = min(max(u32(path.shaderParams[2].z + 0.5), 2u), MAX_COLORS);
  let color = gradientMapColor(luma, count, path);
  let coverage = source.a * sourceImageFrame(uv) * step(0.5, path.shaderParams[0].y);
  return vec4<f32>(color, coverage);
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
