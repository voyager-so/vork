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


// Shader ID: metal-plates

fn turn(angle: f32) -> mat2x2<f32> {
  let s = sin(angle);
  let c = cos(angle);

  return mat2x2<f32>(
    vec2<f32>(c, s),
    vec2<f32>(-s, c)
  );
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

fn random(st: vec2<f32>) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(st.x) ^ hashU32(bitcast<u32>(st.y))));
}

fn noise(st: vec2<f32>) -> f32 {
  let i = floor(st);
  let f = fract(st);

  let a = random(i);
  let b = random(i + vec2<f32>(1.0, 0.0));
  let c = random(i + vec2<f32>(0.0, 1.0));
  let d = random(i + vec2<f32>(1.0, 1.0));

  let u = f * f * (vec2<f32>(3.0) - 2.0 * f);

  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

fn fbm(stIn: vec2<f32>, time: f32, detail: f32) -> f32 {
  var st = stIn;
  var v = sin(time * 0.5) * 0.2;
  var a = 0.8;
  let shift = vec2<f32>(100.0);
  let angle = 0.5;
  let rot = mat2x2<f32>(
    vec2<f32>(cos(angle), sin(angle)),
    vec2<f32>(-sin(angle), cos(angle))
  );

  for (var i = 0; i < 5; i = i + 1) {
    v += a * noise(st);
    st = rot * st * 2.0 + shift;
    a *= detail;
  }

  return v;
}

fn transformSt(st: vec2<f32>, scale: f32, rotation: f32, offset: vec2<f32>) -> vec2<f32> {
  return turn(-rotation) * (st / max(scale, 0.001) + vec2<f32>(-offset.x, offset.y));
}

fn shadeMetalPlates(
  fragCoord: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>,
  speed: f32,
  plateLineStrength: f32,
  detail: f32,
  path: CustomShaderData,
) -> vec4<f32> {
  let t = time * speed;
  var st = (fragCoord.xy - resolution.xy) / min(resolution.x, resolution.y);
  st = transformSt(st, scale, rotation, offset);

  var co = st;
  var len = 0.0;

  for (var i = 0; i < 3; i = i + 1) {
    len = length(co);
    co.x += sin(co.y + t * 0.620) * 0.3;
    co.y += cos(co.x + t * 0.164 + cos(len)) * 0.3;
  }

  len -= 3.0;

  var q = vec2<f32>(0.0);
  q.x = fbm(st + vec2<f32>(1.0), t, detail);
  q.y = fbm(st + vec2<f32>(-0.450, 0.650), t, detail);

  var r = vec2<f32>(0.0);
  r.x = fbm(st + q + vec2<f32>(0.570, 0.520) + vec2<f32>(0.5 * t), t, detail);
  r.y = fbm(st + q + vec2<f32>(0.340, -0.570) + vec2<f32>(0.4 * t), t, detail);

  for (var i = 1.0; i < 1.1; i = i + 1.0) {
    let p1 = 0.2 * i;
    var a = st.x + st.y;
    a = a - floor(a / p1) * p1;
    a = min(a, p1 - a);
    r += vec2<f32>((0.00125 * plateLineStrength) / (a + 0.0001));
  }

  let f = fbm(st + r, t, detail);
  let light = 0.5 + 0.5 * cos(len);

  let metalDark = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let metalMid = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let metalLight = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  var col = mix(metalDark, metalMid, light);
  col = mix(col, metalLight, pow(light, 6.0));
  col *= vec3<f32>(0.92, 0.95, 0.96);

  let intensity = 1.8 * (f * f * f + 0.6 * f * f + 0.5 * f);

  return vec4<f32>(col * intensity, 1.0);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let pixel = input.localPos - b.xy;
  let resolution = b.zw;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = speed
  // path.shaderParams[1].z = plate line strength
  // path.shaderParams[1].w = detail (fbm amplitude falloff)
  // path.colorData[0..2].rgb = metal dark, mid, light
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let speed = path.shaderParams[1].y;
  let plateLineStrength = path.shaderParams[1].z;
  let detail = path.shaderParams[1].w;

  let color = shadeMetalPlates(pixel, resolution, time, scale, rotation, offset, speed, plateLineStrength, detail, path);

  return color;
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
