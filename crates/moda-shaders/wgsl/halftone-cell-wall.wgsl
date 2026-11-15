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


// Shader ID: halftone-cell-wall

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
    centered / max(scale, 0.001) + vec2<f32>(-offset.x * aspect, -offset.y),
    -rotation
  );

  return vec2<f32>(transformed.x / aspect + 0.5, transformed.y + 0.5);
}

fn halftoneSwirlField(
  uv: vec2<f32>,
  aspect: f32,
  time: f32,
  swirlDetail: f32,
  swirlStrength: f32,
  colorA: vec3<f32>,
  colorB: vec3<f32>,
  colorC: vec3<f32>
) -> vec3<f32> {
  let detail = max(swirlDetail, 0.001);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let radius = length(p);
  let spin = atan2(p.y, p.x) + sin(radius * detail * 3.4 - time * 0.55) * 0.18 * swirlStrength;
  p = vec2<f32>(cos(spin), sin(spin)) * radius;

  var q = rotate2(p, 0.35 + sin(time * 0.13) * 0.16);
  q += vec2<f32>(
    sin(q.y * detail * 2.25 + time * 0.82) * 0.115,
    cos(q.x * detail * 2.05 - time * 0.74) * 0.105
  ) * swirlStrength;

  let layerA = sin((q.x * 2.55 + q.y * 1.35) * detail + time * 0.44);
  let layerB = cos((q.x * -1.7 + q.y * 2.85) * detail - time * 0.36);
  let layerC = sin((q.x + q.y) * detail * 2.2 + sin(time * 0.21) * 1.7);
  let combined = layerA * 0.42 + layerB * 0.36 + layerC * 0.22;

  let mixAB = smoothstep(0.18, 0.82, combined * 0.5 + 0.5);
  let accent = smoothstep(0.66, 0.98, sin(layerA * 2.1 + layerC * 1.4 + time * 0.7) * 0.5 + 0.5);
  var color = mix(colorA, colorB, mixAB);
  color = mix(color, colorC, accent * 0.28);
  color *= 1.0 + sin(time * 2.0 + combined * 7.0) * 0.018;

  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  let brightness = clamp(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  let shadow = 1.0 - brightness + 0.000001;
  return color + vec3<f32>(grain * shadow * shadow * grainStrength);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain); the cell grid stays in
  // shape-local space via effectPixel below.
  let grainPixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = cell size
  // path.shaderParams[0].w = radius
  // path.shaderParams[1].x = rotation
  // path.shaderParams[1].y = offsetX
  // path.shaderParams[1].z = offsetY
  // path.shaderParams[1].w = swirl detail
  // path.shaderParams[2].x = swirl strength
  // path.shaderParams[2].y = jitter
  // path.shaderParams[2].z = grain
  // path.shaderParams[2].w = speed
  // path.colorData[0..2].rgb = swirl colors
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];

  let time = p0.x;
  let scale = p0.y;
  let cellSize = max(p0.z, 2.0);
  let radius = p0.w;
  let rotation = p1.x;
  let offset = vec2<f32>(p1.y, p1.z);
  let swirlDetail = p1.w;
  let swirlStrength = p2.x;
  let jitter = p2.y;
  let grain = p2.z;
  let speed = p2.w;
  let t = time * speed;
  let aspect = resolution.x / resolution.y;
  let invResolution = 1.0 / resolution;
  let colorA = path.colorData[0].rgb;
  let colorB = path.colorData[1].rgb;
  let colorC = path.colorData[2].rgb;
  let effectUv = transformUv(uv, aspect, scale, rotation, offset);
  let effectPixel = effectUv * resolution;
  let baseCell = floor(effectPixel / cellSize);
  let radiusBase = cellSize * radius;
  let aa = max(cellSize * 0.025, 0.65);

  var finalColor = vec3<f32>(0.0);
  var maxCircle = 0.0;
  // A cell's dot is at most radiusBase * 1.14 wide plus the aa band, and its centre moves at most
  // cellSize * jitter / 2 per axis: a cell whose unjittered centre is farther than that from this pixel
  // draws circle = 0 here, which never changes the result (a zero circle only wins while maxCircle is
  // still 0, and then the background shows). Skipping those cells skips their swirl-field sample —
  // most of the 25.
  let reach = radiusBase * 1.14 + aa + cellSize * max(jitter, 0.0) * 0.7072 + 0.01;

  for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
      let neighbor = vec2<f32>(f32(dx), f32(dy));
      let cellIndex = baseCell + neighbor;
      let plainDelta = effectPixel - (cellIndex + vec2<f32>(0.5)) * cellSize;
      if (dot(plainDelta, plainDelta) > reach * reach) {
        continue;
      }
      var jitterVec = vec2<f32>(0.0);
      if (jitter > 0.0001) {
        jitterVec = vec2<f32>(
          hash21(cellIndex + vec2<f32>(17.31, 4.97)) - 0.5,
          hash21(cellIndex + vec2<f32>(2.43, 29.11)) - 0.5
        ) * cellSize * jitter;
      }
      let cellCenter = (cellIndex + vec2<f32>(0.5)) * cellSize + jitterVec;
      let sampleUv = clamp(cellCenter * invResolution, vec2<f32>(0.0), vec2<f32>(1.0));
      let texColor = halftoneSwirlField(sampleUv, aspect, t, swirlDetail, swirlStrength, colorA, colorB, colorC);
      let luma = dot(texColor, vec3<f32>(0.2126, 0.7152, 0.0722));
      let localRadius = radiusBase * mix(0.72, 1.14, smoothstep(0.05, 0.95, luma));
      let delta = effectPixel - cellCenter;
      let distSq = dot(delta, delta);
      let radiusSq = localRadius * localRadius;
      let aaSq = max(2.0 * localRadius * aa, aa * aa);
      let circle = 1.0 - smoothstep(radiusSq - aaSq, radiusSq + aaSq, distSq);
      let winner = step(maxCircle, circle);
      finalColor = mix(finalColor, texColor, vec3<f32>(winner));
      maxCircle = max(maxCircle, circle);
    }
  }

  let background = halftoneSwirlField(effectUv, aspect, t, swirlDetail, swirlStrength, colorA, colorB, colorC) * 0.16;
  let vignetteP = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let vignette = smoothstep(1.06, 0.16, length(vignetteP));
  let color = mix(background, finalColor, vec3<f32>(maxCircle)) * mix(0.86, 1.08, vignette);
  var finalColorGrain = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  if (grain > 0.0001) {
    finalColorGrain = filmGrain(finalColorGrain, grainPixel, grain);
  }

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(clamp(finalColorGrain, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
