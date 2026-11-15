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



const PI: f32 = 3.141592653589793;
const CELL_SIZE: vec2<f32> = vec2<f32>(12.0, 18.0);

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn signedNoise(p: vec2<f32>) -> f32 {
  return valueNoise(p) * 2.0 - 1.0;
}

fn fbm(p: vec2<f32>) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.0;

  for (var i = 0; i < 5; i = i + 1) {
    sum += signedNoise(q) * amp;
    q = rotate2(q * 2.02 + vec2<f32>(13.7, -9.2), 0.74);
    amp *= 0.52;
  }

  return sum;
}

fn sdRoundRect(p: vec2<f32>, center: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
  let q = abs(p - center) - halfSize;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn digitSegmentOn(digit: u32, segment: u32) -> bool {
  if (digit == 0u) {
    return segment == 1u || segment == 2u || segment == 3u || segment == 5u || segment == 6u || segment == 7u;
  } else if (digit == 1u) {
    return segment == 3u || segment == 6u;
  } else if (digit == 2u) {
    return segment == 1u || segment == 3u || segment == 4u || segment == 5u || segment == 7u;
  } else if (digit == 3u) {
    return segment == 1u || segment == 3u || segment == 4u || segment == 6u || segment == 7u;
  } else if (digit == 4u) {
    return segment == 2u || segment == 3u || segment == 4u || segment == 6u;
  } else if (digit == 5u) {
    return segment == 1u || segment == 2u || segment == 4u || segment == 6u || segment == 7u;
  } else if (digit == 6u) {
    return segment == 1u || segment == 2u || segment == 4u || segment == 5u || segment == 6u || segment == 7u;
  } else if (digit == 7u) {
    return segment == 1u || segment == 3u || segment == 6u;
  } else if (digit == 8u) {
    return true;
  } else if (digit == 9u) {
    return segment == 1u || segment == 2u || segment == 3u || segment == 4u || segment == 6u || segment == 7u;
  }

  return false;
}

fn segmentDistance(p: vec2<f32>, segment: u32) -> f32 {
  let horizontalSize = vec2<f32>(0.155, 0.012);
  let verticalSize = vec2<f32>(0.012, 0.115);
  let radius = 0.022;

  if (segment == 1u) {
    return sdRoundRect(p, vec2<f32>(0.0, 0.405), horizontalSize, radius);
  } else if (segment == 2u) {
    return sdRoundRect(p, vec2<f32>(-0.24, 0.215), verticalSize, radius);
  } else if (segment == 3u) {
    return sdRoundRect(p, vec2<f32>(0.24, 0.215), verticalSize, radius);
  } else if (segment == 4u) {
    return sdRoundRect(p, vec2<f32>(0.0, 0.0), horizontalSize, radius);
  } else if (segment == 5u) {
    return sdRoundRect(p, vec2<f32>(-0.24, -0.215), verticalSize, radius);
  } else if (segment == 6u) {
    return sdRoundRect(p, vec2<f32>(0.24, -0.215), verticalSize, radius);
  }

  return sdRoundRect(p, vec2<f32>(0.0, -0.405), horizontalSize, radius);
}

fn sevenSegmentDigit(p: vec2<f32>, digit: u32) -> vec2<f32> {
  var activeDist = 10.0;
  var inactiveDist = 10.0;

  for (var segment = 1u; segment <= 7u; segment = segment + 1u) {
    let dist = segmentDistance(p, segment);

    if (digitSegmentOn(digit, segment)) {
      activeDist = min(activeDist, dist);
    } else {
      inactiveDist = min(inactiveDist, dist);
    }
  }

  return vec2<f32>(activeDist, inactiveDist);
}

fn glyphMask(digit: u32, localPos: vec2<f32>, gridScale: f32, backlight: f32) -> f32 {
  let cellSize = CELL_SIZE * gridScale;
  let glyphSize = cellSize * vec2<f32>(0.74, 0.86);
  let glyphOrigin = (cellSize - glyphSize) * 0.5;
  let glyphPos = localPos - glyphOrigin;
  let glyphUv = glyphPos / glyphSize;
  let insideGlyph = step(vec2<f32>(0.0), glyphUv) * step(glyphUv, vec2<f32>(0.999));

  if (insideGlyph.x * insideGlyph.y <= 0.0) {
    return 0.0;
  }

  let p = (vec2<f32>(glyphUv.x, 1.0 - glyphUv.y) - 0.5) * vec2<f32>(0.82, 1.0);
  let distances = sevenSegmentDigit(p, digit);
  let aa = max(0.014, 0.9 / max(min(cellSize.x, cellSize.y), 1.0));
  let activeMask = 1.0 - smoothstep(0.0, aa, distances.x);
  let inactiveMask = 1.0 - smoothstep(0.0, aa, distances.y);

  return max(activeMask, inactiveMask * backlight);
}

fn abstractMatrixMask(level: u32, cellId: vec2<f32>, localPos: vec2<f32>, gridScale: f32, backlight: f32) -> f32 {
  let cellSize = CELL_SIZE * gridScale;
  let tileSize = cellSize * vec2<f32>(0.78, 0.82);
  let tileOrigin = (cellSize - tileSize) * 0.5;
  let tilePos = localPos - tileOrigin;
  let tileUv = tilePos / tileSize;
  let insideTile = step(vec2<f32>(0.0), tileUv) * step(tileUv, vec2<f32>(0.999));

  if (insideTile.x * insideTile.y <= 0.0) {
    return 0.0;
  }

  let matrixSize = vec2<f32>(3.0, 4.0);
  let blockCell = floor(tileUv * matrixSize);
  let blockUv = fract(tileUv * matrixSize);
  let blockLocal = abs(blockUv - 0.5);
  let density = f32(level) / 10.0;
  let cellHash = hash21(cellId + vec2<f32>(31.0, 71.0));
  let centerFalloff = length((blockCell + 0.5) / matrixSize - vec2<f32>(0.5, 0.5));
  let vertical = 1.0 - smoothstep(0.25, 0.95, abs(blockCell.x - 1.0));
  let horizontal = 1.0 - smoothstep(0.25, 0.95, abs(blockCell.y - 1.5));
  let mirroredX = min(blockCell.x, 2.0 - blockCell.x);
  let mirroredY = min(blockCell.y, 3.0 - blockCell.y);
  let corners = step(mirroredX + mirroredY, 0.1);
  let sidePair = step(mirroredX, 0.1) * step(abs(blockCell.y - 1.5), 0.75);
  let capPair = step(mirroredY, 0.1) * step(abs(blockCell.x - 1.0), 0.75);
  let frame = max(max(corners, sidePair), capPair);
  let centerCluster = 1.0 - smoothstep(0.72, 1.65, centerFalloff * 3.2);
  let ring = smoothstep(0.28, 0.7, centerFalloff * 3.2) * (1.0 - smoothstep(1.45, 1.95, centerFalloff * 3.2));
  let cross = max(vertical, horizontal);
  let shapeA = max(centerCluster, corners);
  let shapeB = max(cross, centerCluster * 0.6);
  let shapeC = max(frame, ring * 0.75);
  let shapeAB = mix(shapeA, shapeB, step(0.34, cellHash));
  let shape = mix(shapeAB, shapeC, step(0.67, cellHash));
  let activeGate = smoothstep(0.12, 0.36, density);
  let blockEnergy = density * 1.16 + shape * 0.42 - centerFalloff * 0.22;
  let litAmount = smoothstep(0.18, 0.68, blockEnergy) * activeGate;
  let block = 1.0 - smoothstep(0.36, 0.46, max(blockLocal.x * 0.78, blockLocal.y));
  let blockGradient = 0.68 + blockUv.y * 0.24 + blockUv.x * 0.08;
  let ghost = block * backlight;
  let litBlock = block * blockGradient;

  return mix(ghost, litBlock, litAmount);
}

fn fieldValue(p: vec2<f32>, t: f32) -> f32 {
  let slow = vec2<f32>(sin(t * 0.09) * 0.24, cos(t * 0.07) * 0.2);
  let q = p * 1.35 + slow;
  let directionalA = sin(q.x * 2.2 + t * 0.24);
  let directionalB = sin(2.35 * (q.x * sin(t * 0.11) + q.y * cos(t * 0.08)) + t * 0.18);
  let radialAOrigin = q + vec2<f32>(0.42 * sin(t * 0.13), 0.36 * cos(t * 0.1));
  let radialBOrigin = q + vec2<f32>(-0.48 * cos(t * 0.08), 0.3 * sin(t * 0.12));
  let radialA = sin(sqrt(dot(radialAOrigin, radialAOrigin) * 22.0 + 1.0) - t * 0.3);
  let radialB = sin(sqrt(dot(radialBOrigin, radialBOrigin) * 16.0 + 1.0) + t * 0.22);
  let organicWarp = fbm(q * 0.58 + vec2<f32>(t * 0.025, -t * 0.018)) * 0.22;
  let plasma = directionalA * 0.24 + directionalB * 0.22 + radialA * 0.28 + radialB * 0.2 + organicWarp;
  let grouped = smoothstep(-0.72, 0.78, plasma);
  let softBands = sin((plasma + organicWarp * 0.65) * PI * 0.72) * 0.12;
  let value = 0.2 + grouped * 0.68 + softBands;

  return clamp(value, 0.0, 1.0);
}

// Returns CRT-warped UVs without clamping so callers can detect curved-out pixels.
// Sample sites must clamp before reading from textures or grids.
fn crtCurve(uv: vec2<f32>, warp: f32) -> vec2<f32> {
  let strength = clamp(warp * 0.5, 0.0, 1.0);
  var p = (uv - 0.5) * 2.0;
  p *= mix(1.0, 1.1, strength);
  p.x *= 1.0 + pow(abs(p.y) / 5.0, 2.0) * strength;
  p.y *= 1.0 + pow(abs(p.x) / 4.0, 2.0) * strength;
  p = p * 0.5 + 0.5;
  return mix(uv, p * 0.92 + 0.04, strength);
}

fn segmentSample(pixel: vec2<f32>, boundsSize: vec2<f32>, aspect: f32, scale: f32, gridScale: f32, rotation: f32, offset: vec2<f32>, t: f32, contrast: f32, glyphType: u32, backlight: f32) -> f32 {
  let cellSize = CELL_SIZE * gridScale;
  let cellId = floor(pixel / cellSize);
  let cellOrigin = cellId * cellSize;
  let cellCenter = cellOrigin + cellSize * 0.5;
  let localPos = pixel - cellOrigin;
  var p = vec2<f32>((cellCenter.x / max(boundsSize.x, 1.0) - 0.5) * aspect, cellCenter.y / max(boundsSize.y, 1.0) - 0.5);
  p = rotate2(p / scale + vec2<f32>(-offset.x * 0.35 * aspect, -offset.y * 0.35), -rotation);

  let brightness = fieldValue(p, t);
  let blackPoint = mix(0.02, 0.5, (contrast + 1.0) * 0.5);
  let whitePoint = mix(0.55, 1.0, (contrast + 1.0) * 0.5);
  let mappedBrightness = smoothstep(blackPoint, whitePoint, brightness);
  let dither = hash21(cellId + vec2<f32>(17.0, 43.0)) - 0.5;
  let shapedBrightness = clamp(mappedBrightness * 0.9 + dither * 0.1, 0.0, 0.999);
  let level = min(u32(floor(shapedBrightness * 11.0)), 10u);
  var digit = 10u;

  if (level > 0u) {
    digit = level - 1u;
  }

  let mask = select(
    glyphMask(digit, localPos, gridScale, backlight),
    abstractMatrixMask(level, cellId, localPos, gridScale, backlight),
    glyphType == 1u
  );

  return mask;
}

fn segmentSampleUv(uv: vec2<f32>, boundsSize: vec2<f32>, aspect: f32, scale: f32, gridScale: f32, rotation: f32, offset: vec2<f32>, t: f32, contrast: f32, glyphType: u32, backlight: f32) -> f32 {
  return segmentSample(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * boundsSize, boundsSize, aspect, scale, gridScale, rotation, offset, t, contrast, glyphType, backlight);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let boundsSize = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = gridScale
  // path.shaderParams[0].w = rotation
  // path.shaderParams[1].x = offsetX
  // path.shaderParams[1].y = offsetY
  // path.shaderParams[1].z = speed
  // path.shaderParams[1].w = contrast
  // path.shaderParams[2].x = crtWarp
  // path.shaderParams[2].y = vignette
  // path.shaderParams[2].z = glyphType
  // path.shaderParams[2].w = backlight
  // path.colorData[0].rgb = glyph color
  // path.colorData[1].rgb = background color
  let time = path.shaderParams[0].x;
  let scale = max(path.shaderParams[0].y * 0.35, 0.001);
  let gridScale = max(path.shaderParams[0].z * 2.75, 0.1);
  let rotation = path.shaderParams[0].w;
  let offset = vec2<f32>(path.shaderParams[1].x, path.shaderParams[1].y);
  let speed = path.shaderParams[1].z;
  let contrast = clamp(path.shaderParams[1].w, -1.0, 1.0);
  let crtWarp = clamp(path.shaderParams[2].x, 0.0, 2.0);
  let vignetteAmount = clamp(path.shaderParams[2].y, 0.0, 2.0);
  // Glyph type: 0 = readable seven-segment digits, 1 = abstract matrix marks.
  let glyphType = u32(clamp(round(path.shaderParams[2].z), 0.0, 1.0));
  let backlight = clamp(path.shaderParams[2].w, 0.0, 0.16);
  let t = time * speed;

  let curvedUv = crtCurve(uv, crtWarp);
  let safeUv = clamp(curvedUv, vec2<f32>(0.0), vec2<f32>(1.0));
  let aspect = b.z / max(b.w, 1.0);
  let glyphColor = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let backgroundColor = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let coverage = segmentSampleUv(curvedUv, boundsSize, aspect, scale, gridScale, rotation, offset, t, contrast, glyphType, backlight);
  var coverageRgb = vec3<f32>(coverage);

  let vignette = max(16.0 * safeUv.x * safeUv.y * (1.0 - safeUv.x) * (1.0 - safeUv.y), 0.0);

  // insideCurve uses the unclamped curvedUv so corners that warped past the edge can be blanked.
  let insideCurve = step(vec2<f32>(0.0), curvedUv) * step(curvedUv, vec2<f32>(1.0));
  coverageRgb *= insideCurve.x * insideCurve.y;

  // Smooth aperture grille (0.35..1.0 across each pixel pair) so columns are actually attenuated.
  let aperture = 1.0 - 0.65 * clamp((fract((input.localPos.x - b.x) * 0.5) * 2.0 - 1.0) * 2.0, 0.0, 1.0);
  coverageRgb *= aperture;
  var rgb = mix(backgroundColor, glyphColor, coverageRgb);
  // Guard against WGSL's indeterminate pow(0, 0) when both vignette and vignetteAmount are 0.
  let vignettePower = 0.3 * vignetteAmount;
  let vignetteFactor = select(1.0, pow(max(vignette, 1e-6), vignettePower), vignettePower > 0.0);
  rgb *= vignetteFactor;

  return vec4<f32>(rgb, 1.0);
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
