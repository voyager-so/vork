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


// Shader ID: smooth-sine-lines

const MAX_LINES: u32 = 12u;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn sstep(edgeA: f32, edgeB: f32, x: f32) -> f32 {
  if (edgeA < edgeB) {
    return smoothstep(edgeA, edgeB, x);
  }

  return 1.0 - smoothstep(edgeB, edgeA, x);
}

fn lineMasks(
  uvIn: vec2<f32>,
  speed: f32,
  height: f32,
  time: f32,
  waveAmount: f32,
  lineWidth: f32
) -> vec2<f32> {
  var uv = uvIn;
  let xEnvelope = sstep(1.0, 0.0, abs(uv.x));
  uv.y += xEnvelope * sin(time * speed + uv.x * height) * waveAmount;

  let width = max(0.0001, 0.06 * sstep(0.2, 0.9, abs(uv.x)) * lineWidth);
  let core = sstep(width, 0.0, abs(uv.y) - 0.004 * lineWidth);
  let edgeFade = sstep(1.0, 0.3, abs(uv.x));
  let halo = exp(-abs(uv.y) / max(width * 5.4, 0.004)) * edgeFade;

  return vec2<f32>(core * edgeFade, max(halo - core * 0.35, 0.0));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv01 = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let aspect = b.z / max(b.w, 1.0);
  let centered = vec2<f32>((uv01.x - 0.5) * aspect, uv01.y - 0.5);

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = line count
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = wave amount
  // path.shaderParams[1].x = frequency multiplier
  // path.shaderParams[1].y = speed multiplier
  // path.shaderParams[1].z = glow
  // path.colorData[0].rgb = background color
  // path.colorData[1].rgb = line color
  let time = path.shaderParams[0].x;
  let lineCount = min(max(u32(path.shaderParams[0].y + 0.5), 2u), MAX_LINES);
  let rotation = path.shaderParams[0].z;
  let lineWidth = 0.5;
  let waveAmount = path.shaderParams[0].w;
  let frequency = path.shaderParams[1].x;
  let brightness = 1.0;
  let speedMultiplier = path.shaderParams[1].y;
  let glow = path.shaderParams[1].z;
  let uv = rotate2(centered, -rotation);

  let background = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let lineColor = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  var coreCoverage = 0.0;
  var glowCoverage = 0.0;
  let denominator = max(f32(lineCount - 1u), 1.0);

  for (var i = 0u; i < MAX_LINES; i = i + 1u) {
    if (i < lineCount) {
      let t = f32(i) / denominator;
      let masks = lineMasks(
        uv,
        1.0 + t,
        (4.0 + t) * frequency,
        time * speedMultiplier,
        waveAmount,
        lineWidth
      );
      let coreAlpha = clamp(masks.x * brightness, 0.0, 1.0);
      let glowAlpha = clamp(masks.y * glow * 0.65, 0.0, 1.0);
      coreCoverage = 1.0 - (1.0 - coreCoverage) * (1.0 - coreAlpha);
      glowCoverage = 1.0 - (1.0 - glowCoverage) * (1.0 - glowAlpha);
    }
  }

  let glowColor = mix(background, lineColor, clamp(glowCoverage, 0.0, 1.0));
  let color = mix(glowColor, lineColor, clamp(coreCoverage, 0.0, 1.0));

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(color, 1.0);
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
