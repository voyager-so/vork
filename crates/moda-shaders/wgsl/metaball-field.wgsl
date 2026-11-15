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



const MAX_STEPS: i32 = 64;
const METABALL_COUNT: u32 = 9u;
const HIT_EPSILON: f32 = 0.0045;
const NORMAL_EPSILON: f32 = 0.012;
const LIGHT_DIRECTION: vec3<f32> = vec3<f32>(-0.4683, 0.6452, 0.6036);

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn noirSeed(cell: vec2<f32>) -> f32 {
  // Integer-exact lattice hash (hashU32/hashToUnit from the noise include):
  // bit-exact across WGSL compilers and devices, unlike the old fract(sin(x) * K) hash.
  return hashToUnit(hashU32(bitcast<u32>(cell.x) ^ hashU32(bitcast<u32>(cell.y))));
}

fn noirField(point: vec2<f32>, clock: f32) -> f32 {
  var p = point;
  var amp = 0.56;
  var sum = 0.0;
  var norm = 0.0;

  for (var layer = 0; layer < 4; layer = layer + 1) {
    let layerF = f32(layer);
    let drift = vec2<f32>(
      sin(clock * 0.24 + layerF * 1.41),
      cos(clock * 0.19 - layerF * 1.67)
    ) * 0.28;

    sum += valueNoise(p + drift) * amp;
    norm += amp;
    p = mat2x2<f32>(
      vec2<f32>(1.58, 0.28),
      vec2<f32>(-0.36, 1.72)
    ) * p + vec2<f32>(0.17, 0.31);
    amp *= 0.53;
  }

  return sum / max(norm, 0.001);
}

fn smoothMin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

fn metaballCenter(index: i32, clock: f32, blobSpread: f32) -> vec3<f32> {
  let i = f32(index);
  let baseAngle = i * 2.094395 + 0.45;
  let slow = clock * (0.42 + i * 0.07);
  let radius = 0.58 + 0.18 * sin(clock * 0.17 + i * 1.9);
  let center = vec3<f32>(
    cos(baseAngle + slow * 0.42) * radius + sin(clock * 0.11 + i) * 0.18,
    sin(baseAngle - slow * 0.35) * 0.42 + cos(clock * 0.13 + i * 0.7) * 0.14,
    sin(clock * 0.23 + i * 1.31) * 0.34
  );

  return center * blobSpread;
}

fn buildMetaballs(clock: f32, blobSize: f32, blobSpread: f32, blobCount: u32) -> array<vec4<f32>, 9> {
  var balls: array<vec4<f32>, 9>;

  for (var i = 0u; i < blobCount; i = i + 1u) {
    let center = metaballCenter(i32(i), clock, blobSpread);
    let radius = (0.46 + 0.09 * sin(clock * 0.2 + f32(i) * 1.7)) * blobSize;
    balls[i] = vec4<f32>(center, radius);
  }

  return balls;
}

fn sceneSdf(p: vec3<f32>, balls: array<vec4<f32>, 9>, smoothK: f32, blobCount: u32) -> f32 {
  var d = 100.0;

  for (var i = 0u; i < blobCount; i = i + 1u) {
    let ball = balls[i];
    let sphere = length(p - ball.xyz) - ball.w;
    d = smoothMin(d, sphere, smoothK);
  }

  return d;
}

fn intersectAabb(rayOrigin: vec3<f32>, rayDirection: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
  let safeDirection = select(vec3<f32>(0.0001), rayDirection, abs(rayDirection) > vec3<f32>(0.0001));
  let invDirection = vec3<f32>(1.0) / safeDirection;
  let t0 = (boxMin - rayOrigin) * invDirection;
  let t1 = (boxMax - rayOrigin) * invDirection;
  let nearT = min(t0, t1);
  let farT = max(t0, t1);
  let enter = max(max(nearT.x, nearT.y), nearT.z);
  let exit = min(min(farT.x, farT.y), farT.z);

  return vec2<f32>(enter, exit);
}

fn estimateNormal(p: vec3<f32>, balls: array<vec4<f32>, 9>, smoothK: f32, blobCount: u32) -> vec3<f32> {
  let e = NORMAL_EPSILON;
  let k1 = vec3<f32>(1.0, -1.0, -1.0);
  let k2 = vec3<f32>(-1.0, -1.0, 1.0);
  let k3 = vec3<f32>(-1.0, 1.0, -1.0);
  let k4 = vec3<f32>(1.0, 1.0, 1.0);

  return normalize(
    k1 * sceneSdf(p + k1 * e, balls, smoothK, blobCount) +
    k2 * sceneSdf(p + k2 * e, balls, smoothK, blobCount) +
    k3 * sceneSdf(p + k3 * e, balls, smoothK, blobCount) +
    k4 * sceneSdf(p + k4 * e, balls, smoothK, blobCount)
  );
}

fn palette3(wave: f32, colorA: vec3<f32>, colorB: vec3<f32>, colorC: vec3<f32>) -> vec3<f32> {
  let w = clamp(wave, 0.0, 1.0);
  let low = mix(colorA, colorB, vec3<f32>(smoothstep(0.0, 0.5, w)));
  let high = mix(colorB, colorC, vec3<f32>(smoothstep(0.5, 1.0, w)));

  return mix(low, high, vec3<f32>(step(0.5, w)));
}

fn backgroundPalette(wave: f32, path: CustomShaderData) -> vec3<f32> {
  let colorA = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorB = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorC = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  return palette3(wave, colorA, colorB, colorC);
}

fn metaballPalette(wave: f32, path: CustomShaderData) -> vec3<f32> {
  let colorA = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let colorB = clamp(path.colorData[4].rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  return mix(colorA, colorB, vec3<f32>(smoothstep(0.0, 1.0, clamp(wave, 0.0, 1.0))));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let pixel = input.localPos - b.xy;

  // Packing map:
  // path.colorData[0..2].rgb = background colors
  // path.colorData[3..4].rgb = metaball colors
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = speed
  // path.shaderParams[1].x = offsetX
  // path.shaderParams[1].y = offsetY
  // path.shaderParams[1].z = grain
  // path.shaderParams[1].w = exposure
  // path.shaderParams[2].x = contrast
  // path.shaderParams[2].y = blobSize
  // path.shaderParams[2].z = blobCount
  // path.shaderParams[2].w = blobSpread
  let time = path.shaderParams[0].x;
  let scale = max(path.shaderParams[0].y, 0.001);
  let rotation = path.shaderParams[0].z;
  let speed = path.shaderParams[0].w;
  let offset = vec2<f32>(path.shaderParams[1].x, path.shaderParams[1].y);
  let grain = clamp(path.shaderParams[1].z, 0.0, 1.0);
  let exposure = max(path.shaderParams[1].w, 0.01);
  let contrast = max(path.shaderParams[2].x, 0.01);
  let blobSize = max(path.shaderParams[2].y, 0.1);
  let blobCount = u32(clamp(round(path.shaderParams[2].z), 1.0, f32(METABALL_COUNT)));
  let blobSpread = max(path.shaderParams[2].w, 0.1);
  let clock = time * (0.04 + speed * 0.12);

  let aspect = b.z / max(b.w, 1.0);
  var screen = (uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
  screen = rotate2(screen / scale + vec2<f32>(-offset.x * 0.38 * aspect, -offset.y * 0.38), -rotation);

  let haze = noirField(screen * 0.82 + vec2<f32>(clock * 0.18, -clock * 0.08), clock);
  let backgroundWave = smoothstep(0.18, 0.95, haze);
  let vignette = smoothstep(1.36, 0.18, length(screen));
  let backgroundColor = backgroundPalette(backgroundWave, path);
  let backgroundShadow = backgroundPalette(clamp(backgroundWave - 0.22, 0.0, 1.0), path);
  var color = mix(backgroundShadow, backgroundColor, vec3<f32>(0.72 + 0.28 * vignette));

  let rayOrigin = vec3<f32>(0.0, 0.0, 3.15);
  let rayDirection = normalize(vec3<f32>(screen, -1.86));
  let metaballs = buildMetaballs(clock, blobSize, blobSpread, blobCount);
  let smoothK = 0.36 * blobSize;
  // AABB sized to enclose the maximum possible metaball extent at param maxima
  // (blobSpread <= 1.85, blobSize <= 1.45). Tighter bounds clipped blobs at
  // extreme settings because rays exited before reaching them.
  let boundsHit = intersectAabb(rayOrigin, rayDirection, vec3<f32>(-2.8, -2.4, -2.0), vec3<f32>(2.8, 2.4, 1.8));
  var t = max(boundsHit.x, 0.0);
  let tExit = boundsHit.y;
  var hit = false;
  var steps = 0.0;
  var hitPoint = rayOrigin;

  if (tExit > t) {
    for (var step = 0; step < MAX_STEPS; step = step + 1) {
      hitPoint = rayOrigin + rayDirection * t;
      let distance = sceneSdf(hitPoint, metaballs, smoothK, blobCount);
      steps = f32(step);

      if (distance < HIT_EPSILON) {
        hit = true;
        break;
      }

      t += max(distance * 0.72, 0.012);

      if (t > tExit) {
        break;
      }
    }
  }

  if (hit) {
    let normal = estimateNormal(hitPoint, metaballs, smoothK, blobCount);
    let rim = clamp(1.0 - abs(dot(normal, -rayDirection)), 0.0, 1.0);
    let viewAmount = rim * sqrt(rim);
    let diffuse = clamp(dot(normal, LIGHT_DIRECTION) * 0.5 + 0.5, 0.0, 1.0);
    let thickness = smoothstep(0.0, 1.0, 1.0 - steps / f32(MAX_STEPS));
    let blobWave = clamp(diffuse * 0.38 + viewAmount * 0.48 + thickness * 0.22 + haze * 0.12, 0.0, 1.0);
    let blobColor = metaballPalette(blobWave, path);
    let lightAmount = 0.72 + 0.28 * diffuse;
    let colorPresence = smoothstep(0.46, 0.92, max(max(viewAmount, thickness), blobWave));
    let shadedBlob = mix(blobColor * lightAmount, blobColor, vec3<f32>(colorPresence * 0.9));

    color = mix(color, shadedBlob, vec3<f32>(0.9));
  }

  color += vec3<f32>((noirSeed(floor(pixel * 0.62)) - 0.5) * grain * 0.022);
  // Power-curve contrast: contrast > 1 darkens midtones (more visual contrast),
  // contrast < 1 lifts them. Default 1.0 is a no-op.
  color = pow(max(color * exposure, vec3<f32>(0.0)), vec3<f32>(contrast));

  let outRgb = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(outRgb, 1.0);
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
