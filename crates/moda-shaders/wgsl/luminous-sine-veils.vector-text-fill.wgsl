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


// Shader ID: luminous-sine-veils

const MAX_VEILS: u32 = 20u;
const MAX_VEIL_COLORS: u32 = 4u;
const VEIL_COLOR_OFFSET: u32 = 2u;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn packedColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  // colorData is a storage-buffer array: dynamic index (clamped to the last of
  // the 6 packed slots) instead of a branch chain.
  let safeIndex = min(index, 5u);
  return clamp(path.colorData[safeIndex].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn backgroundColor(uv: vec2<f32>, path: CustomShaderData) -> vec3<f32> {
  let topLight = smoothstep(-0.25, 0.85, uv.y + 0.18 * sin(uv.x * 2.0));
  return mix(packedColorAt(0u, path), packedColorAt(1u, path), topLight);
}

fn veilColorAt(index: u32, path: CustomShaderData) -> vec3<f32> {
  return packedColorAt(VEIL_COLOR_OFFSET + min(index, MAX_VEIL_COLORS - 1u), path);
}

fn veilGradient(t: f32, colorCount: u32, path: CustomShaderData) -> vec3<f32> {
  let count = max(colorCount, 2u);
  let last = count - 1u;
  let scaled = clamp(t, 0.0, 1.0) * f32(last);
  let left = min(u32(floor(scaled)), last - 1u);
  let right = min(left + 1u, last);
  let local = smoothstep(0.0, 1.0, fract(scaled));
  return mix(veilColorAt(left, path), veilColorAt(right, path), local);
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  return color + vec3<f32>(grain * grainStrength);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  // Cache the raw shape-local UV once: the clamped copy drives the veil field
  // (which mutates uv below), while the raw copy feeds the background sample.
  let localUv = (input.localPos - b.xy) / b.zw;
  var uv = clamp(localUv, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = veil color count
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = exposure
  // path.shaderParams[1].x = glow
  // path.shaderParams[1].y = grain
  // path.shaderParams[1].z = speed
  // path.colorData[0..1].rgb = background colors
  // path.colorData[2..5].rgb = veil colors
  let time = path.shaderParams[0].x;
  let colorCount = min(max(u32(path.shaderParams[0].y + 0.5), 2u), MAX_VEIL_COLORS);
  let rotation = path.shaderParams[0].z;
  let exposure = path.shaderParams[0].w;
  let glow = path.shaderParams[1].x;
  let grain = path.shaderParams[1].y;
  let speed = path.shaderParams[1].z;

  let clock = time * speed;
  uv = rotate2(uv - vec2<f32>(0.5), rotation) + vec2<f32>(0.5);
  uv.y -= 0.58;
  uv = uv * 2.0;

  // Perf (docs/qa/shader-perf.md): the veil loop is the whole cost of this shader, so it carries only
  // per-pixel work. uv.x is fixed through the loop, so sin(uv.x + t + clock) advances by a constant
  // angle (1/20) each step: one sin/cos pair and a rotation recurrence replace 20 sins. The veil
  // gradient depends only on the step, so each step's two stop weights are accumulated per color stop
  // (4 scalars) and resolved against the colors once after the loop, instead of indexing the packed
  // colors 40 times per pixel. shaderColor only feeds a length: its sums are kept as two scalars.
  let lastStop = f32(colorCount - 1u);
  let stepSin = 0.04997916927067833; // sin(1 / 20)
  let stepCos = 0.9987502603949663; // cos(1 / 20)
  var waveSin = sin(uv.x + clock);
  var waveCos = cos(uv.x + clock);
  let xx = uv.x * uv.x + 0.16;
  var widthSum = 0.0;
  var widthYSum = 0.0;
  var veilStops = vec4<f32>(0.0);
  var glowStops = vec4<f32>(0.0);
  var colorWeight = 0.0;

  for (var i = 0u; i < MAX_VEILS; i = i + 1u) {
    let t = f32(i) / 20.0;
    uv.y += 0.04 * waveSin;
    let nextSin = waveSin * stepCos + waveCos * stepSin;
    waveCos = waveCos * stepCos - waveSin * stepSin;
    waveSin = nextSin;

    let safeY = max(abs(uv.y), 0.0001);
    let width = 1.0 / (250.0 * safeY);
    let glowWidth = width * (250.0 / 72.0);
    let layerWeight = clamp(width * sqrt(xx + 0.64 * uv.y * uv.y), 0.0, 1.0);
    // veilGradient(t) as weights on the 4 stops: (1 - local) on the left stop, local on the right.
    let scaled = t * lastStop;
    let left = min(floor(scaled), lastStop - 1.0);
    let local = smoothstep(0.0, 1.0, scaled - left);
    let stops = vec4<f32>(0.0, 1.0, 2.0, 3.0);
    let stopWeights = select(vec4<f32>(0.0), vec4<f32>(1.0 - local), stops == vec4<f32>(left)) +
      select(vec4<f32>(0.0), vec4<f32>(local), stops == vec4<f32>(left + 1.0));
    widthSum += width;
    widthYSum += width * uv.y;
    veilStops += stopWeights * layerWeight;
    glowStops += stopWeights * clamp(glowWidth, 0.0, 0.2);
    colorWeight += layerWeight;
  }

  let v0 = veilColorAt(0u, path);
  let v1 = veilColorAt(1u, path);
  let v2 = veilColorAt(2u, path);
  let v3 = veilColorAt(3u, path);
  let shaderColor = vec3<f32>(widthSum * uv.x, widthSum * 0.4, widthYSum * 0.8);
  let colorizedVeils = v0 * veilStops.x + v1 * veilStops.y + v2 * veilStops.z + v3 * veilStops.w;
  let glowColor = v0 * glowStops.x + v1 * glowStops.y + v2 * glowStops.z + v3 * glowStops.w;

  let mask = clamp(length(shaderColor) * exposure, 0.0, 1.0);
  let veilColor = colorizedVeils / max(colorWeight, 0.0001);
  let baseColor = backgroundColor(localUv, path);
  let glowContribution = glowColor * glow * 0.18 * exposure;
  var color = clamp(mix(baseColor, veilColor, mask) + glowContribution, vec3<f32>(0.0), vec3<f32>(1.0));
  color = filmGrain(color, pixel, grain);

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
