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


// Shader ID: opening-gates

const PI: f32 = 3.141592653589793;

fn saturate(v: f32) -> f32 {
  return clamp(v, 0.0, 1.0);
}

fn safeAspect(resolution: vec2<f32>) -> f32 {
  return resolution.x / max(resolution.y, 1.0);
}

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn paletteGradient(uv: vec2<f32>, local: vec2<f32>, phase: f32, side: f32, layer: f32, path: CustomShaderData) -> vec3<f32> {
  let c0 = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c1 = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c2 = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c3 = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let diagonal = dot(local, normalize(vec2<f32>(0.72 + side * 0.12, 1.0)));
  let broad = 0.5 + 0.5 * sin(diagonal * 3.2 + phase * 2.7 + layer * 0.71);
  let plume = 0.5 + 0.5 * sin((uv.x * 2.0 - uv.y * 1.35) * PI + phase * 4.2 + layer);
  let wash = smoothstep(0.08, 0.95, uv.y + sin(uv.x * 2.4 + phase) * 0.08);
  let base = mix(c0, c1, saturate(broad * 0.8 + wash * 0.18));
  let cool = mix(base, c2, smoothstep(0.42, 0.92, plume) * (0.35 + 0.18 * side));
  let warm = mix(cool, c3, smoothstep(0.55, 0.98, broad) * 0.38);
  let milk = smoothstep(0.48, 1.0, broad + plume * 0.22) * 0.34;

  return mix(warm, vec3<f32>(1.0), milk);
}

fn softBackground(uv: vec2<f32>, p: vec2<f32>, time: f32, path: CustomShaderData) -> vec3<f32> {
  let c0 = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c1 = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c2 = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c3 = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let a = 0.5 + 0.5 * sin((p.x * 0.7 + p.y * 1.4) * PI + time * 0.55);
  let b = 0.5 + 0.5 * cos((p.x * -1.2 + p.y * 0.55) * PI - time * 0.41);
  let base = mix(c0, c1, smoothstep(0.15, 0.95, a));
  let cool = mix(base, c2, smoothstep(0.48, 1.0, b) * 0.62);
  let glow = mix(cool, c3, smoothstep(0.72, 1.0, a + b * 0.36) * 0.35);

  return mix(glow, vec3<f32>(1.0), 0.08 + (1.0 - uv.y) * 0.1);
}

fn gateLayer(
  uv: vec2<f32>,
  p: vec2<f32>,
  resolution: vec2<f32>,
  spawnId: f32,
  phase: f32,
  opening: f32,
  edgeStrength: f32,
  path: CustomShaderData
) -> vec4<f32> {
  let angleSlot = fract(spawnId * 0.38196601125);
  let angleJitter = (hash21(vec2<f32>(spawnId, 3.17)) - 0.5) * 0.18;
  let angle = -1.05 + angleSlot * 2.1 + angleJitter;
  let n = vec2<f32>(cos(angle), sin(angle));
  let tangent = vec2<f32>(-n.y, n.x);
  let seamReveal = smoothstep(0.09, 0.22, phase);
  let openCurve = smoothstep(0.12, 1.0, phase);
  let openDistance = openCurve * openCurve * (2.75 * opening);
  let birth = smoothstep(0.0, 0.045, phase);
  let layerAlpha = birth;
  let lineOffset = 0.0;
  let sideNow = select(-1.0, 1.0, dot(p, n) - lineOffset >= 0.0);
  let localP = p - n * sideNow * openDistance;
  let splitDistance = dot(localP, n) - lineOffset;
  let sideMask = select(0.0, 1.0, splitDistance * sideNow >= 0.0);
  let px = 1.0 / max(resolution.y, 1.0);
  let cutFeather = smoothstep(0.0, px * 1.5, abs(splitDistance));
  let screenExtent =
    abs(n.x) * (safeAspect(resolution) * 0.5 + 0.25) +
    abs(n.y) * 0.75 +
    abs(lineOffset);
  let onScreen = 1.0 - smoothstep(screenExtent - 0.08, screenExtent + 0.08, openDistance);
  let openPanelMask = sideMask * cutFeather;
  let closedPanelMask = 1.0;
  let panelMask = mix(closedPanelMask, openPanelMask, seamReveal) * layerAlpha * onScreen;
  let local = vec2<f32>(dot(localP, tangent), splitDistance);
  let sourceUv = uv + vec2<f32>(n.x / max(safeAspect(resolution), 0.001), -n.y) * sideNow * openDistance * 0.09;
  let grad = paletteGradient(sourceUv, local, phase, sideNow, spawnId, path);
  let distanceToCut = abs(splitDistance);
  let fineEdge = exp(-distanceToCut * 220.0) * edgeStrength * seamReveal;
  let broadEdge = exp(-distanceToCut * 55.0) * 0.12 * edgeStrength * seamReveal;
  let shadow = exp(-distanceToCut * 90.0) * 0.14 * edgeStrength * seamReveal;
  let lightFacing = dot(n * sideNow, normalize(vec2<f32>(-0.32, 0.95))) * 0.5 + 0.5;
  let edgeColor = vec3<f32>(fineEdge * (0.28 + lightFacing * 0.62) + broadEdge);
  let seamDark = vec3<f32>(shadow * (1.0 - lightFacing) * 0.55);
  let depthWash = mix(0.94, 1.08, 1.0 - phase);
  let color = grad * depthWash + edgeColor - seamDark;

  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), panelMask);
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  // Integer hash (noise include) instead of fract(sin(...)) — the sine hash diverges across GPU
  // backends, breaking WebGPU/native parity.
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  return color + vec3<f32>(grain * grainStrength * (0.45 + (1.0 - luma) * 0.55));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));
  let aspect = safeAspect(resolution);
  let p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = speed
  // path.shaderParams[0].z = opening
  // path.shaderParams[0].w = edge strength
  // path.shaderParams[1].x = grain
  // path.colorData[0..3].rgb = palette
  let time = path.shaderParams[0].x;
  let speed = path.shaderParams[0].y;
  let opening = path.shaderParams[0].z;
  let edgeStrength = path.shaderParams[0].w;
  let grain = path.shaderParams[1].x;
  let t = time * speed;

  var color = softBackground(uv, p, t, path);

  let gateClock = t * 1.08;
  let gateBase = floor(gateClock);
  let gateFrac = fract(gateClock);

  for (var i = 0; i < 7; i = i + 1) {
    let slot = f32(i);
    let phase = (gateFrac + slot) / 7.0;
    let spawnId = gateBase - slot;
    let layer = gateLayer(uv, p, resolution, spawnId, phase, opening, edgeStrength, path);
    color = mix(color, layer.rgb, layer.a);
  }

  color = filmGrain(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), pixel, grain);

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
