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


// Shader ID: stars

const STARS_TAU: f32 = 6.283185307179586;

fn starsRotate2(point: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * point.x - s * point.y, s * point.x + c * point.y);
}

fn starsTransformPoint(
  uv: vec2<f32>,
  aspect: f32,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>
) -> vec2<f32> {
  let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  return starsRotate2(
    centered / max(scale, 0.001) + vec2<f32>(-offset.x * aspect, offset.y),
    -rotation
  );
}

fn starsHash22(point: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    hash21(point + vec2<f32>(17.13, 91.71)),
    hash21(point + vec2<f32>(63.29, 11.47))
  );
}

fn starsNebulaFbm(pointIn: vec2<f32>) -> f32 {
  var point = pointIn;
  var amplitude = 0.54;
  var total = 0.0;
  var normalizer = 0.0;

  for (var octave = 0; octave < 5; octave = octave + 1) {
    total += valueNoise(point) * amplitude;
    normalizer += amplitude;
    point = mat2x2<f32>(
      vec2<f32>(1.72, 0.51),
      vec2<f32>(-0.43, 1.81)
    ) * point + vec2<f32>(0.37, 0.19);
    amplitude *= 0.5;
  }

  return total / max(normalizer, 0.001);
}

fn starsDustFbm(pointIn: vec2<f32>) -> f32 {
  var point = pointIn;
  var amplitude = 0.52;
  var total = 0.0;
  var normalizer = 0.0;

  for (var octave = 0; octave < 7; octave = octave + 1) {
    total += valueNoise(point) * amplitude;
    normalizer += amplitude;
    point = mat2x2<f32>(
      vec2<f32>(1.58, 1.12),
      vec2<f32>(-1.12, 1.58)
    ) * point + vec2<f32>(0.73, -0.41);
    amplitude *= 0.55;
  }

  return total / max(normalizer, 0.001);
}

fn starsLayer(
  point: vec2<f32>,
  resolution: vec2<f32>,
  gridScale: f32,
  layerSeed: f32,
  clock: f32,
  density: f32,
  size: f32,
  coolColor: vec3<f32>,
  warmColor: vec3<f32>
) -> vec3<f32> {
  let drift = vec2<f32>(clock * (0.002 + layerSeed * 0.0007), -clock * 0.0013);
  let gridPoint = (point + drift) * gridScale;
  let cell = floor(gridPoint);
  let local = fract(gridPoint) - vec2<f32>(0.5);
  let seedCell = cell + vec2<f32>(layerSeed * 31.7, layerSeed * 67.3);
  // Most cells hold no star: decide that first and skip the rest of the star's shading.
  let birth = hash21(seedCell + vec2<f32>(41.3, 13.1));
  if (birth < 1.0 - clamp(density * 0.055, 0.0, 0.18)) {
    return vec3<f32>(0.0);
  }
  let jitter = (starsHash22(seedCell) - vec2<f32>(0.5)) * 0.72;
  let delta = local - jitter;
  let distanceToStar = length(delta);
  let character = hash21(seedCell + vec2<f32>(7.9, 103.1));
  let brightness = pow(character, 3.4);
  let radius = mix(0.012, 0.045, brightness) * max(size, 0.01);
  let aa = max(0.5 * gridScale / max(min(resolution.x, resolution.y), 1.0), 0.0015);
  let core = 1.0 - smoothstep(radius - aa, radius + aa, distanceToStar);
  let halo = exp(-distanceToStar * mix(19.0, 10.0, brightness)) * brightness * 0.16;
  let flareGate = smoothstep(0.86, 1.0, brightness);
  let horizontal = exp(-abs(delta.x) * 85.0) * exp(-abs(delta.y) * 12.0);
  let vertical = exp(-abs(delta.y) * 85.0) * exp(-abs(delta.x) * 12.0);
  let flare = (horizontal + vertical) * flareGate * 0.2;
  let pulseRate = mix(0.7, 2.6, hash21(seedCell + vec2<f32>(53.7, 5.3)));
  let pulsePhase = hash21(seedCell + vec2<f32>(19.4, 71.2)) * STARS_TAU;
  let pulse = mix(1.0, 0.72 + 0.28 * sin(clock * pulseRate + pulsePhase), 0.42);
  let temperature = hash21(seedCell + vec2<f32>(83.2, 29.6));
  let starColor = mix(coolColor, warmColor, vec3<f32>(temperature));

  return starColor * pulse * (core * mix(0.7, 2.4, brightness) + halo + flare);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0] = time, scale, rotation, offsetX
  // path.shaderParams[1] = offsetY, speed, star density, star size
  // path.shaderParams[2] = nebula strength, dust, exposure, unused
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];
  let aspect = b.z / max(b.w, 1.0);
  let point = starsTransformPoint(uv, aspect, p0.y, p0.z, vec2<f32>(p0.w, p1.x));
  let clock = p0.x * max(p1.y, 0.0);
  let slowDrift = vec2<f32>(clock * 0.004, -clock * 0.003);
  let laneWarp = starsNebulaFbm(
    starsRotate2(point * vec2<f32>(1.35, 0.82), 1.73) + slowDrift * 0.47
  );
  let laneBend = (laneWarp - 0.5) * 0.075 + sin(point.y * 1.85 - 0.4) * 0.018;
  let crossBand = point.x - laneBend;
  let bandPoint = vec2<f32>(crossBand, point.y);

  let planeCoordinate = crossBand * 5.65;
  let galaxyPlane = 1.0 / (planeCoordinate * planeCoordinate + 1.0);
  let centerDirection = clamp(0.76 + point.y * 0.2 - crossBand * 0.045, 0.42, 1.0);
  let galaxyCenter = centerDirection * centerDirection;

  let broadNoise = starsNebulaFbm(
    bandPoint * vec2<f32>(2.1, 1.25) + vec2<f32>(7.4, 2.1) + slowDrift * 0.35
  );
  let cloudNoise = starsNebulaFbm(
    bandPoint * vec2<f32>(5.4, 3.15) + vec2<f32>(19.7, 8.3) - slowDrift * 0.8
  );
  let fineGas = starsNebulaFbm(
    bandPoint * vec2<f32>(11.5, 7.1) + vec2<f32>(37.1, 15.9) + slowDrift * 1.4
  );
  let gasTexture =
    0.7 +
    broadNoise * 0.28 +
    cloudNoise * 0.16 +
    pow(smoothstep(0.54, 0.82, fineGas), 1.4) * 0.13;

  let dustDrift = slowDrift * 0.18;
  let dustBase =
    starsRotate2(bandPoint * vec2<f32>(3.1, 1.7), 0.47) +
    vec2<f32>(51.2, 23.7) +
    dustDrift;
  let dustWarpX = starsDustFbm(dustBase) - 0.5;
  let dustWarpY =
    starsDustFbm(starsRotate2(dustBase * 1.37 + vec2<f32>(17.8, 43.1), 1.11)) - 0.5;
  let warpedDustPoint = dustBase + vec2<f32>(dustWarpX, dustWarpY) * 0.9;
  let broadDust = starsDustFbm(warpedDustPoint * 1.45 + vec2<f32>(73.4, 31.8));
  let fineDust = starsDustFbm(
    starsRotate2(warpedDustPoint * 3.35 + vec2<f32>(11.8, 89.1), 0.88)
  );

  let dustCenter = crossBand + dustWarpX * 0.12 + (broadDust - 0.5) * 0.035;
  let dustWidth = mix(0.055, 0.15, smoothstep(0.2, 0.8, dustWarpY + 0.5));
  let dustDistance = abs(dustCenter);
  let veilProfile = 1.0 - smoothstep(dustWidth * 0.4, dustWidth * 10.65, dustDistance);
  let filamentProfile =
    1.0 - smoothstep(dustWidth * 0.08, dustWidth * 0.78, dustDistance);
  let veilDensity = smoothstep(0.38, 0.72, broadDust);
  let filamentDensity = smoothstep(0.48, 0.78, fineDust);
  let dustLane = clamp(
    (veilProfile * veilDensity * 0.88 + filamentProfile * filamentDensity * 0.72) *
      galaxyCenter,
    0.0,
    1.0
  );
  let nebulaStrength = max(p2.x, 0.0);
  let dustAmount = clamp(p2.y, 0.0, 1.5);

  let galaxyIntensity = galaxyPlane * galaxyCenter;
  var nebula = vec3<f32>(0.255, 0.245, 0.43) * galaxyIntensity * gasTexture;
  nebula +=
    vec3<f32>(0.085, 0.078, 0.145) *
    galaxyIntensity *
    pow(smoothstep(0.48, 0.78, cloudNoise), 1.3);
  let dustAbsorption = clamp(dustLane * dustAmount * 1.12, 0.0, 0.92);
  nebula *= mix(
    vec3<f32>(1.0),
    vec3<f32>(0.11, 0.085, 0.16),
    vec3<f32>(dustAbsorption)
  );

  let coolStar = vec3<f32>(0.152941, 0.419608, 0.560784);
  let warmStar = vec3<f32>(1.0, 0.839216, 0.639216);
  var stars = vec3<f32>(0.0);
  stars += starsLayer(point, resolution, 31.0, 1.0, clock, p1.z, p1.w * 1.2, coolStar, warmStar);
  stars += starsLayer(
    point,
    resolution,
    57.0,
    2.0,
    clock,
    p1.z,
    p1.w,
    mix(coolStar, vec3<f32>(1.0), vec3<f32>(0.35)),
    warmStar
  );
  stars += starsLayer(point, resolution, 93.0, 3.0, clock, p1.z * 0.9, p1.w * 0.75, vec3<f32>(0.72, 0.82, 1.0), warmStar);
  stars *= 1.0 - clamp(dustAbsorption * 0.9, 0.0, 0.88);

  let skyVariation = valueNoise(point * 2.3 + vec2<f32>(clock * 0.003, 0.0));
  var color =
    vec3<f32>(0.0028, 0.0032, 0.0105) * (0.82 + skyVariation * 0.25) +
    nebula * nebulaStrength +
    stars;
  color = vec3<f32>(1.0) - exp(-max(color, vec3<f32>(0.0)) * max(p2.z, 0.0));

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
