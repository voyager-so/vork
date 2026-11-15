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


// Shader ID: soft-glass-panels

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

fn panelOptics(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  panelWidth: f32,
  waveFrequency: f32,
  waveAmplitude: f32
) -> vec4<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let angle = 138.0 * PI / 180.0;
  let cosA = cos(angle);
  let sinA = sin(angle);
  let centered = uv - vec2<f32>(0.5);
  let asp = vec2<f32>(centered.x * aspect, centered.y);
  let u = asp.x * cosA + asp.y * sinA;
  let v = -asp.x * sinA + asp.y * cosA;
  let frequency = 9.02 / max(panelWidth, 0.001);
  let cell = fract((u + sin(v * waveFrequency * PI * 2.0) * waveAmplitude) * frequency) - 0.5;
  let cellPos = cell * 2.0;
  let slope = sign(cellPos) * pow(max(abs(cellPos), 0.0001), 3.0);
  let refrU = -(slope * 3.37) * (0.5 / frequency);

  return vec4<f32>(cellPos, slope, refrU, frequency);
}

fn backgroundField(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  blobScale: f32,
  blobStrength: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let baseColor = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let blobColor = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  var flow = p * max(blobScale, 0.001);
  flow += vec2<f32>(
    sin(p.y * 2.9 + time * 0.48) * 0.14 + cos(p.x * 1.7 - time * 0.32) * 0.08,
    cos(p.x * 2.4 - time * 0.41) * 0.13 + sin(p.y * 1.9 + time * 0.27) * 0.07
  );
  let folded = rotate2(flow, 0.38 + sin(time * 0.11) * 0.08);
  let sweepA = sin((folded.x * 1.45 + folded.y * 0.92) * PI + time * 0.34);
  let sweepB = cos((folded.x * -0.72 + folded.y * 1.85) * PI - time * 0.29);
  let sweepC = sin((flow.x + flow.y) * 1.35 * PI + sin(time * 0.19) * 1.4);
  let mask = smoothstep(0.02, 0.72, sweepA * 0.5 + sweepB * 0.32 + sweepC * 0.18 + 0.22);
  let feather = smoothstep(0.0, 0.9, mask) * blobStrength;
  let milkyBase = mix(baseColor, vec3<f32>(1.0), 0.08);

  return mix(milkyBase, blobColor, clamp(feather, 0.0, 1.0));
}

fn refractedBackground(
  refractedUv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  chroma: vec2<f32>,
  blobScale: f32,
  blobStrength: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let r = backgroundField(refractedUv + chroma, resolution, time, blobScale, blobStrength, path).r;
  let g = backgroundField(refractedUv, resolution, time, blobScale, blobStrength, path).g;
  let b = backgroundField(refractedUv - chroma, resolution, time, blobScale, blobStrength, path).b;

  return vec3<f32>(r, g, b);
}

fn glassPanels(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  refraction: f32,
  chromaticAberration: f32,
  panelWidth: f32,
  waveFrequency: f32,
  waveAmplitude: f32,
  blobScale: f32,
  blobStrength: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let angle = 138.0 * PI / 180.0;
  let cosA = cos(angle);
  let sinA = sin(angle);
  let optics = panelOptics(uv, resolution, panelWidth, waveFrequency, waveAmplitude);
  let slope = optics.y;
  let refrU = optics.z * refraction;
  let refractedUv = vec2<f32>(
    uv.x + (refrU * cosA) / aspect,
    uv.y + refrU * sinA
  );
  let chromaU = refrU * 0.15 * chromaticAberration;
  let chroma = vec2<f32>((chromaU * cosA) / aspect, chromaU * sinA);
  let refracted = refractedBackground(refractedUv, resolution, time, chroma, blobScale, blobStrength, path);
  let nz = sqrt(1.0 - min(slope * slope, 1.0));
  let halfLight = (-90.0 * PI / 180.0) * 0.5;
  let hx = sin(halfLight);
  let hy = cos(halfLight);
  let nDotH = max(slope * hx + nz * hy, 0.0);
  let shininess = exp2(8.0 - 0.11 * 7.0);
  let fresnel = pow(1.0 - nz, 5.0);
  let spec = pow(nDotH, shininess) * (0.04 + 0.96 * fresnel) * 2.0 * max(refraction, 0.0);

  return refracted + vec3<f32>(spec);
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  return color + vec3<f32>(grain * grainStrength);
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
  // path.shaderParams[1].w = panel width
  // path.shaderParams[2].x = wave frequency
  // path.shaderParams[2].y = wave amplitude
  // path.shaderParams[2].z = blob scale
  // path.shaderParams[2].w = blob strength
  // path.shaderParams[3].x = grain
  // path.shaderParams[3].y = speed
  // path.colorData[0].rgb = background color
  // path.colorData[1].rgb = blob color
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let refraction = path.shaderParams[1].y;
  let chromaticAberration = path.shaderParams[1].z;
  let panelWidth = path.shaderParams[1].w;
  let waveFrequency = path.shaderParams[2].x;
  let waveAmplitude = path.shaderParams[2].y;
  let blobScale = path.shaderParams[2].z;
  let blobStrength = path.shaderParams[2].w;
  let grain = path.shaderParams[3].x;
  let speed = path.shaderParams[3].y;
  let t = time * speed;
  let aspect = resolution.x / resolution.y;
  let effectUv = transformUv(uv, aspect, scale, rotation, offset);
  let color = glassPanels(
    effectUv,
    resolution,
    t,
    refraction,
    chromaticAberration,
    panelWidth,
    waveFrequency,
    waveAmplitude,
    blobScale,
    blobStrength,
    path
  );
  // Framebuffer coords for film grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  let finalColor = filmGrain(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), input.position.xy, grain);

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
