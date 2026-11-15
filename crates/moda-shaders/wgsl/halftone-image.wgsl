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


// Shader ID: halftone-image

// Shared source-image ABI (Gradient Map's): shaderParams [0..8] = image aspect, ready, scale X/Y, rotation,
// offset X/Y, source width/height (the node's image fill placement); [11] = image state (0 loading, 1 ready,
// 2 failed); the shader's own params from [12] (shaderParams[3]).
const IMAGE_LOADING_COLOR = vec3<f32>(0.9, 0.9, 0.9);
const IMAGE_FAILED_COLOR = vec3<f32>(0.85, 0.74, 0.74);

fn rotateSourceUv(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

// The image fill's own mapping from fill-local coordinates to image UV (Fill / Fit / Crop / Tile resolved).
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

// The loading / failed placeholder, else the shaded colour. Called last, after every derivative.
fn sourceImageResult(shaded: vec4<f32>, path: CustomShaderData) -> vec4<f32> {
  let imageState = path.shaderParams[2].w;
  if (imageState < 0.5) {
    return vec4<f32>(IMAGE_LOADING_COLOR, 1.0);
  }
  if (imageState > 1.5) {
    return vec4<f32>(IMAGE_FAILED_COLOR, 1.0);
  }
  return shaded;
}

// Integer-exact hash (lowbias32), as in the procedural fills.
fn imageHashU32(x: u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

fn imageHash21(p: vec2<f32>) -> f32 {
  return f32(imageHashU32(bitcast<u32>(p.x) ^ imageHashU32(bitcast<u32>(p.y))) >> 8u) * (1.0 / 16777216.0);
}

fn imageLuma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn halftoneRowParity(row: f32) -> f32 {
  return row - floor(row * 0.5) * 2.0;
}

fn halftoneCellCenter(
  cell: vec2<f32>,
  cellSize: f32,
  stagger: f32,
  gridRotation: f32,
  imageCenter: vec2<f32>
) -> vec2<f32> {
  let rowOffset = halftoneRowParity(cell.y) * stagger * 0.5;
  let gridCenter = (cell + vec2<f32>(0.5 + rowOffset, 0.5)) * cellSize;
  return rotateSourceUv(gridCenter - imageCenter, -gridRotation) + imageCenter;
}

fn halftoneDotMask(delta: vec2<f32>, radius: f32, antialias: f32) -> f32 {
  let distance = length(delta);
  let aa = max(antialias, 0.65);
  let presence = smoothstep(0.0, aa * 1.5, radius);
  let outer = 1.0 - smoothstep(radius - aa, radius + aa, distance);
  return outer * presence;
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  // Packing map (after the source-image ABI):
  // path.shaderParams[3] = cell size px, dot size, luma contrast, row stagger
  // path.shaderParams[4] = grid rotation, source colour, invert, paper grain
  // path.colorData[0].rgb = paper, path.colorData[1].rgb = ink
  let p3 = path.shaderParams[3];
  let p4 = path.shaderParams[4];
  let sourceUv = sourceImageUv(input.localPos, path);
  let sourceDx = dpdx(sourceUv);
  let sourceDy = dpdy(sourceUv);
  let resolution = max(vec2<f32>(textureDimensions(shaderImage)), vec2<f32>(1.0));
  let imageCenter = resolution * 0.5;
  let sourcePixel = sourceUv * resolution;
  let antialias = max(0.65, 0.5 * (length(sourceDx * resolution) + length(sourceDy * resolution)));
  let cellSize = max(p3.x, 2.0);
  let stagger = clamp(p3.w, 0.0, 1.0);
  let gridRotation = p4.x;
  let gridPixel = rotateSourceUv(sourcePixel - imageCenter, gridRotation) + imageCenter;
  let gridRow = floor(gridPixel.y / cellSize);
  let rowOffset = halftoneRowParity(gridRow) * stagger * 0.5;
  let baseCell = vec2<f32>(floor(gridPixel.x / cellSize - rowOffset), gridRow);
  let maxRadius = cellSize * 0.72 * max(p3.y, 0.0);
  let lumaContrast = max(p3.z, 0.001);
  let sourceColorMix = clamp(p4.y, 0.0, 1.0);
  let inverted = p4.z >= 0.5;
  let paper = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let ink = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  var strongestMask = 0.0;
  var strongestColor = ink;

  // Neighbouring dots can reach into this cell: the largest one wins.
  for (var cellX: i32 = -1; cellX <= 1; cellX = cellX + 1) {
    for (var cellY: i32 = -1; cellY <= 1; cellY = cellY + 1) {
      let cell = baseCell + vec2<f32>(f32(cellX), f32(cellY));
      let centerPixel = halftoneCellCenter(cell, cellSize, stagger, gridRotation, imageCenter);
      let sampleUv = clamp(centerPixel / resolution, vec2<f32>(0.0), vec2<f32>(1.0));
      let cellSample = textureSampleGrad(shaderImage, shaderImageSampler, sampleUv, sourceDx, sourceDy);
      let contrastedLuma = clamp((imageLuma(cellSample.rgb) - 0.5) * lumaContrast + 0.5, 0.0, 1.0);
      let coverage = select(1.0 - contrastedLuma, contrastedLuma, inverted);
      let radius = maxRadius * sqrt(clamp(coverage, 0.0, 1.0));
      let mask = halftoneDotMask(sourcePixel - centerPixel, radius, antialias);
      let wins = step(strongestMask, mask);
      strongestColor = mix(
        strongestColor,
        mix(ink, clamp(cellSample.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), sourceColorMix),
        vec3<f32>(wins)
      );
      strongestMask = max(strongestMask, mask);
    }
  }

  let original = textureSampleGrad(shaderImage, shaderImageSampler, sourceUv, sourceDx, sourceDy);
  let frame = sourceImageFrame(sourceUv) * step(0.5, path.shaderParams[0].y);
  let grain = (imageHash21(floor(sourcePixel)) - 0.5) * max(p4.w, 0.0) * frame * original.a;
  let color = mix(paper, strongestColor, vec3<f32>(strongestMask)) + vec3<f32>(grain);

  return sourceImageResult(vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), original.a * frame), path);
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
