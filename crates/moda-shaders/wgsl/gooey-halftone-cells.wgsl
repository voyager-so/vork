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


// Shader ID: gooey-halftone-cells

const PI: f32 = 3.141592653589793;

// Renamed from the sandbox's `modf` to avoid shadowing the WGSL builtin modf().
fn wrapMod(x: f32, y: f32) -> f32 {
  return x - floor(x / y) * y;
}

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn transformUv(
  uv: vec2<f32>,
  aspect: f32,
  rotation: f32,
  offset: vec2<f32>
) -> vec2<f32> {
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  p = rotate2(p, -rotation);
  p += vec2<f32>(-offset.x * aspect, -offset.y);

  return vec2<f32>(p.x / aspect + 0.5, p.y + 0.5);
}

fn smoothMin(a: f32, b: f32, k: f32) -> f32 {
  let safeK = max(k, 0.0001);
  let h = max(safeK - abs(a - b), 0.0) / safeK;
  return min(a, b) - h * h * safeK * 0.25;
}

fn sourceLuma(
  uv: vec2<f32>,
  aspect: f32,
  time: f32,
  bloomCenterA: vec2<f32>,
  bloomCenterB: vec2<f32>,
  noiseDrift: vec2<f32>
) -> f32 {
  let p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let sweep = 0.5 + 0.5 * sin((p.x * 2.4 + p.y * 1.8) * PI + time * 0.38);
  let bloomA = smoothstep(0.86, 0.12, length(p - bloomCenterA));
  let bloomB = smoothstep(0.72, 0.05, length(p - bloomCenterB));
  let noise = valueNoise(p * 3.0 + noiseDrift);

  return clamp(sweep * 0.42 + bloomA * 0.34 + bloomB * 0.28 + noise * 0.18, 0.0, 1.0);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));
  // Framebuffer-space coords for film/fiber grain (avoids chunky Retina grain); the cell grid stays
  // in shape-local space via effectPixel below.
  let grainPixel = input.position.xy;

  // Packing map:
  // path.shaderParams[0].x = time
  // scale = 1
  // path.shaderParams[0].y = pixel size
  // path.shaderParams[0].z = dot size
  // path.shaderParams[0].w = gooeyness
  // path.shaderParams[1].x = contrast
  // path.shaderParams[1].y = rotation
  // path.shaderParams[1].z = offsetX
  // path.shaderParams[1].w = offsetY
  // path.shaderParams[2].x = grain
  // path.shaderParams[2].y = speed
  // path.colorData[0].rgb = paper color
  // path.colorData[1].rgb = ink color
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];

  let time = p0.x;
  let cellSize = max(p0.y, 3.0);
  let dotSize = p0.z;
  let gooeyness = p0.w;
  let contrast = p1.x;
  let rotation = p1.y;
  let offset = vec2<f32>(p1.z, p1.w);
  let grain = p2.x;
  let t = time * p2.y;

  let aspect = resolution.x / max(resolution.y, 1.0);
  let invResolution = 1.0 / resolution;
  let effectUv = transformUv(uv, aspect, rotation, offset);
  let effectPixel = effectUv * resolution;
  let baseCell = floor(effectPixel / cellSize);
  let jitterScale = cellSize * 0.18;
  let radiusScale = cellSize * 0.74;
  let smoothK = clamp(gooeyness, 0.0, 1.2) * cellSize * 1.35;
  let bloomCenterA = vec2<f32>(sin(t * 0.16) * 0.28, cos(t * 0.13) * 0.18);
  let bloomCenterB = vec2<f32>(cos(t * 0.11) * -0.22, sin(t * 0.19) * 0.16);
  let noiseDrift = vec2<f32>(t * 0.08, -t * 0.05);
  var minDist = 100000.0;

  for (var dx: i32 = -3; dx <= 3; dx = dx + 1) {
    for (var dy: i32 = -3; dy <= 3; dy = dy + 1) {
      let neighbor = vec2<f32>(f32(dx), f32(dy));
      let cellIndex = baseCell + neighbor;
      let isActiveCell = wrapMod(cellIndex.x + cellIndex.y, 2.0) < 0.5;

      if (isActiveCell) {
        let jitter = vec2<f32>(
          hash21(cellIndex + vec2<f32>(13.7, 4.1)) - 0.5,
          hash21(cellIndex + vec2<f32>(2.9, 19.3)) - 0.5
        ) * jitterScale;
        let cellCenter = (cellIndex + vec2<f32>(0.5)) * cellSize + jitter;
        let sampleUv = clamp(cellCenter * invResolution, vec2<f32>(0.0), vec2<f32>(1.0));
        let luma = sourceLuma(sampleUv, aspect, t, bloomCenterA, bloomCenterB, noiseDrift);
        let darkness = pow(clamp(1.0 - luma, 0.0, 1.0), contrast);
        let radius = (0.08 + darkness * dotSize) * radiusScale;
        let dist = length(effectPixel - cellCenter);
        minDist = smoothMin(minDist, dist - radius, smoothK);
      }
    }
  }

  let aaShape = max(cellSize * 0.018, 0.75);
  let shape = 1.0 - smoothstep(-aaShape, aaShape, minDist);
  let paper = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let ink = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorDelta = length(paper - ink);
  let fiberStrength = max(grain, 0.0);
  let fiber =
    (valueNoise(uv * 54.0) * 0.035 + hash21(grainPixel) - 0.5) * fiberStrength;
  var color = mix(paper, ink, vec3<f32>(shape));
  color += vec3<f32>(fiber * smoothstep(0.0, 0.25, colorDelta));

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
