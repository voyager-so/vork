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

// Vector shader fill: view-projection (vertex) + shared shader ABI + glyph curves.
@group(0) @binding(0) var<uniform> frame: mat4x4<f32>;

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

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) instanceId: u32,
  @location(1) @interpolate(flat) isStroke: f32,
  @location(2) localPos: vec2<f32>,
  @location(3) worldPos: vec2<f32>,
}

// Quadratic outline segments: 3 control points per curve, em units (y-down, pen origin).
@group(2) @binding(0) var<storage, read> curvePoints: array<vec2<f32>>;

struct VectorShadeVSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec2<f32>,
  @location(1) em: vec2<f32>,
  @location(2) @interpolate(flat) curveRange: vec2<u32>, // firstCurve, curveCount
  @location(3) local: vec2<f32>, // text-box-local glyph position: continuous across glyphs
  @location(4) @interpolate(flat) glyphOpacity: f32, // KineticText per-glyph opacity
};


fn kineticGlyphLocal(
  pen: vec2<f32>, em: vec2<f32>, emBox: vec4<f32>, fontSizePx: f32,
  glyphLin: vec4<f32>, glyphOff: vec2<f32>,
) -> vec2<f32> {
  let baseLocal = pen + em * fontSizePx;
  if (glyphLin.x == 1.0 && glyphLin.y == 0.0 && glyphLin.z == 0.0 && glyphLin.w == 1.0 &&
      glyphOff.x == 0.0 && glyphOff.y == 0.0) {
    return baseLocal;
  }
  let center = pen + (emBox.xy + emBox.zw) * 0.5 * fontSizePx;
  let rel = baseLocal - center;
  return vec2<f32>(
    glyphLin.x * rel.x + glyphLin.y * rel.y + glyphOff.x,
    glyphLin.z * rel.x + glyphLin.w * rel.y + glyphOff.y,
  ) + center;
}

// NOTE: shader-fill text does NOT apply the node distort warp: its
// vertex stage reads the shared CustomShaderData ABI (paths[0]), which has no
// homography fields. Distort-warped shader-fill nodes render unwarped — the same
// accepted gap as MSDF text.
@vertex fn vs(
  @location(0) corner: vec2<f32>,      // unit quad corner (0..1)
  @location(1) pen: vec2<f32>,         // local-space pen position (x, baselineY)
  @location(2) fontSizePx: f32,
  @location(3) emBox: vec4<f32>,       // padded glyph bounds in em units
  @location(4) curveRange: vec2<u32>,
  @location(5) glyphLin: vec4<f32>,    // KineticText per-glyph 2×2 [a,c,b,d]
  @location(6) glyphOff: vec2<f32>,    // KineticText per-glyph px offset
  @location(7) glyphOpacity: f32,      // KineticText per-glyph opacity
) -> VectorShadeVSOut {
  let path = paths[0];
  let em = mix(emBox.xy, emBox.zw, corner);
  let local = kineticGlyphLocal(pen, em, emBox, fontSizePx, glyphLin, glyphOff);
  let world = path.model * vec4<f32>(local, 0.0, 1.0);
  var o: VectorShadeVSOut;
  o.pos = frame * world;
  o.pos.z = path.pathInfo.w; // enforce scene depth (text z-ordering), matches solid/gradient text
  o.world = world.xy;
  o.em = em;
  o.curveRange = curveRange;
  o.local = local;
  o.glyphOpacity = glyphOpacity;
  return o;
}

// #include "vector-text-coverage.wgsl"


// Shader ID: cmyk-halftone-press

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn transformUv(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>
) -> vec2<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  p = rotate2(p / max(scale, 0.001), -rotation);
  p += vec2<f32>(-offset.x * aspect, -offset.y);

  return vec2<f32>(p.x / aspect + 0.5, p.y + 0.5);
}

fn sourceField(uv: vec2<f32>, resolution: vec2<f32>, time: f32, path: CustomShaderData) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let a = vec3<f32>(0.0);
  let b = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  p += vec2<f32>(
    sin(p.y * 2.6 + time * 0.18) * 0.055,
    cos(p.x * 2.1 - time * 0.16) * 0.05
  );
  let noiseA = valueNoise(p * 2.1 + vec2<f32>(time * 0.045, -time * 0.03));
  let noiseB = valueNoise(rotate2(p * 4.3, 0.64) + vec2<f32>(-time * 0.025, time * 0.04));
  let cA = vec2<f32>(sin(time * 0.13) * 0.32, cos(time * 0.11) * 0.22);
  let cB = vec2<f32>(cos(time * 0.09 + 1.4) * -0.28, sin(time * 0.15 - 0.7) * 0.26);
  let cC = vec2<f32>(sin(time * 0.07 + 2.2) * 0.18, cos(time * 0.1 + 0.9) * -0.2);
  let blobA = smoothstep(0.72, 0.04, length(p - cA));
  let blobB = smoothstep(0.64, 0.05, length(p - cB));
  let blobC = smoothstep(0.58, 0.04, length(p - cC));
  let mixAB = smoothstep(0.15, 1.0, blobB + noiseA * 0.36);
  let mixC = smoothstep(0.1, 0.92, blobC + noiseB * 0.32);
  let softBase = mix(a, b, vec3<f32>(mixAB));
  let pooled = mix(softBase, c, vec3<f32>(mixC));
  let light = blobA * 0.24 + smoothstep(0.0, 1.0, noiseA * noiseB) * 0.08;

  return clamp(pooled + vec3<f32>(light), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rgbToCmyk(rgb: vec3<f32>) -> vec4<f32> {
  let r = rgb.r;
  let g = rgb.g;
  let b = rgb.b;
  let k = min(1.0 - r, min(1.0 - g, 1.0 - b));
  let invK = 1.0 - k;
  var cmy = vec3<f32>(0.0);

  if (invK > 0.0001) {
    cmy = vec3<f32>(
      (1.0 - r - k) / invK,
      (1.0 - g - k) / invK,
      (1.0 - b - k) / invK
    );
  }

  return clamp(vec4<f32>(cmy, k), vec4<f32>(0.0), vec4<f32>(1.0));
}

fn toGridUv(pixel: vec2<f32>, angle: f32, pixelSize: f32) -> vec2<f32> {
  return rotate2(pixel, angle) / max(pixelSize, 1.0);
}

fn getCellCenterUv(
  pixel: vec2<f32>,
  resolution: vec2<f32>,
  angle: f32,
  pixelSize: f32
) -> vec2<f32> {
  let gridUv = toGridUv(pixel, angle, pixelSize);
  let cellCenter = floor(gridUv) + vec2<f32>(0.5);
  let centerScreen = rotate2(cellCenter * pixelSize, -angle);

  return clamp(centerScreen / resolution, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn halftoneDot(pixel: vec2<f32>, angle: f32, pixelSize: f32, dotSize: f32, coverage: f32) -> f32 {
  let gridUv = toGridUv(pixel, angle, pixelSize);
  let distToCenter = fract(gridUv) - vec2<f32>(0.5);
  let radius = dotSize * sqrt(clamp(coverage, 0.0, 1.0));
  let d = length(distToCenter);
  let aa = max(fwidth(d), 0.002);

  return 1.0 - smoothstep(radius - aa, radius + aa, d);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain); the dot grid stays in
  // shape-local space via effectPixel below.
  let grainPixel = input.position.xy;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = pixel size
  // path.shaderParams[0].w = dot size
  // path.shaderParams[1].x = ink strength
  // path.shaderParams[1].y = rotation
  // path.shaderParams[1].z = offsetX
  // path.shaderParams[1].w = offsetY
  // path.shaderParams[2].x = grain
  // path.shaderParams[2].y = speed
  // paper color = black; first source color is effectively black in sourceField
  // path.colorData[0..1].rgb = source colors
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];

  let time = p0.x;
  let scale = p0.y;
  let pixelSize = max(p0.z, 2.0);
  let dotSize = p0.w;
  let inkStrength = p1.x;
  let rotation = p1.y;
  let offset = vec2<f32>(p1.z, p1.w);
  let grain = p2.x;
  let t = time * p2.y * 2.0;

  let effectUv = transformUv(uv, resolution, scale, rotation, offset);
  let effectPixel = effectUv * resolution;
  // All CMYK layers share one screen (angle 0), so they sample the same cell center: compute the
  // source sample + CMYK split once and reuse for every channel + the highlight ink.
  let cellUv = getCellCenterUv(effectPixel, resolution, 0.0, pixelSize);
  let cellSample = sourceField(cellUv, resolution, t, path);
  let cmyk = rgbToCmyk(cellSample);
  let highlightCoverage = smoothstep(
    0.72,
    0.96,
    dot(cellSample, vec3<f32>(0.2126, 0.7152, 0.0722))
  );

  let dotC = halftoneDot(effectPixel, 0.0, pixelSize, dotSize, cmyk.x);
  let dotM = halftoneDot(effectPixel, 0.0, pixelSize, dotSize, cmyk.y);
  let dotY = halftoneDot(effectPixel, 0.0, pixelSize, dotSize, cmyk.z);
  let dotK = halftoneDot(effectPixel, 0.0, pixelSize, dotSize * 0.94, cmyk.w);
  let dotHighlight = halftoneDot(
    effectPixel,
    0.0,
    pixelSize,
    min(dotSize * 0.58, 0.46),
    highlightCoverage
  );

  let paper = vec3<f32>(0.0);
  var printColor = vec3<f32>(1.0);
  printColor.r *= 1.0 - 0.95 * inkStrength * dotC;
  printColor.g *= 1.0 - 0.95 * inkStrength * dotM;
  printColor.b *= 1.0 - 0.9 * inkStrength * dotY;
  printColor *= 1.0 - 1.08 * inkStrength * dotK;

  let inkCoverage = clamp(max(max(dotC, dotM), max(dotY, max(dotK, dotHighlight))) * inkStrength, 0.0, 1.0);
  let paperLuma = dot(paper, vec3<f32>(0.2126, 0.7152, 0.0722));
  let darkPaperMix = 1.0 - smoothstep(0.08, 0.58, paperLuma);
  let sourceColor = sourceField(effectUv, resolution, t, path);
  let paperPrint = printColor * mix(vec3<f32>(1.0), paper, vec3<f32>(0.35));
  let luminousPrint = max(printColor, sourceColor * 0.86);
  let visiblePrint = mix(paperPrint, luminousPrint, vec3<f32>(darkPaperMix));
  var outColor = mix(paper, visiblePrint, vec3<f32>(inkCoverage));
  outColor += vec3<f32>((hash21(grainPixel) - 0.5) * grain);

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(clamp(outColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}


@fragment fn fs(in: VectorShadeVSOut) -> @location(0) vec4<f32> {
  // Pixel footprint in em units (derivatives must be computed before any discard).
  let footprint = max(fwidth(in.em), vec2<f32>(1e-6));
  let invDiameter = 1.0 / footprint;

  let path = paths[0];
  // World-space page clip. discard (not return) keeps the derivative builtins below valid.
  let clip = path.pageClipBounds;
  if (in.world.x < clip.x || in.world.x > clip.x + clip.z ||
      in.world.y < clip.y || in.world.y > clip.y + clip.w) {
    discard;
  }

  // Analytic glyph coverage (same per-pixel winding as the solid/gradient pipelines).
  let coverage = vectorGlyphCoverage(in.curveRange.x, in.curveRange.y, in.em, invDiameter);
  // Early-out on empty regions of the padded glyph quad (mirrors the solid/gradient
  // fills) so an expensive shadeMain isn't evaluated just to be multiplied by ~0.
  // Safe before shadeMain: discard only demotes the invocation to a helper, so
  // derivative builtins inside shadeMain stay valid for live neighbors.
  if (coverage < 0.004) {
    discard;
  }

  // Resolve shader color in uniform control flow (discard demotes, it does not
  // diverge). Build the shared VertexOutput shadeMain expects via the struct
  // constructor — instanceId = 0 (single packed instance), isStroke = 0 (fill pass;
  // vector text has no stroke pass yet).
  let input = VertexOutput(in.pos, 0u, 0.0, in.local, in.world);
  let c = shadeMain(input, path);

  // Straight alpha: the blend applies srcColor * srcAlpha. Node opacity rides in
  // path.fillColor.a (commonData fillColor = [1, 1, 1, nodeOpacity]). KineticText
  // per-glyph opacity (1.0 for static text) multiplies in like the solid/gradient.
  let finalAlpha = c.a * coverage * path.fillColor.a * in.glyphOpacity;
  return vec4<f32>(c.rgb, finalAlpha);
}
