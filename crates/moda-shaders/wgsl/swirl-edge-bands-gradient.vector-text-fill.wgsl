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


// Shader ID: swirl-edge-bands-gradient

const TWO_PI: f32 = 6.283185307179586;
const NOISE_DISTORTION_GAIN: f32 = 0.58;
const PATTERN_SCALE: f32 = 0.58;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn remapSceneScale(controlScale: f32) -> f32 {
  let s = clamp(controlScale, 0.0, 1.0);
  let closestZoom = 0.38;
  let formerZeroZoom = 1.15;
  return mix(closestZoom, formerZeroZoom, s);
}

fn colorStop(index: u32, path: CustomShaderData) -> vec4<f32> {
  if (index == 1u) {
    return clamp(path.colorData[1], vec4<f32>(0.0), vec4<f32>(1.0));
  }

  if (index >= 2u) {
    return clamp(path.colorData[2], vec4<f32>(0.0), vec4<f32>(1.0));
  }

  return clamp(path.colorData[0], vec4<f32>(0.0), vec4<f32>(1.0));
}

fn blendThreeColors(mixer: f32, edgeWidth: f32, edgeBlur: f32, path: CustomShaderData) -> vec4<f32> {
  let c1 = colorStop(0u, path);
  let c2 = colorStop(1u, path);
  let c3 = colorStop(2u, path);
  let r1 = smoothstep(0.35 * edgeWidth, 0.7 - 0.35 * edgeWidth + 0.5 * edgeBlur, mixer);
  let r2 = smoothstep(0.3 + 0.35 * edgeWidth, 1.0 - 0.35 * edgeWidth + edgeBlur, mixer);
  let color12 = mix(c1.rgb, c2.rgb, r1);
  let alpha12 = mix(c1.a, c2.a, r1);

  return vec4<f32>(mix(color12, c3.rgb, r2), mix(alpha12, c3.a, r2));
}

fn warpedCoord(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>,
  distortion: f32,
  swirl: f32,
  iterations: u32
) -> vec2<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let zoom = max(scale, 0.001);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  p = rotate2(p * zoom + vec2<f32>(-offset.x * aspect, -offset.y), -rotation);

  let n1 = valueNoise(p * 1.15 + vec2<f32>(time * 0.16, time * 0.13));
  let n2 = valueNoise(p * 2.35 + vec2<f32>(-time * 0.21, time * 0.18));
  let angle = n1 * TWO_PI;
  p += vec2<f32>(cos(angle), sin(angle)) * n2 * distortion * NOISE_DISTORTION_GAIN;

  let swirlAmount = clamp(swirl, 0.0, 2.0);
  for (var i = 1u; i <= 30u; i = i + 1u) {
    if (i <= iterations) {
      let fi = f32(i);
      p.x += swirlAmount / fi * cos(time + fi * 1.5 * p.y) * 0.12;
      p.y += swirlAmount / fi * cos(time + fi * p.x) * 0.12;
    }
  }

  return p;
}

fn patternMixer(p: vec2<f32>, proportion: f32, shapeScale: f32) -> f32 {
  let prop = clamp(proportion, 0.0, 1.0);
  let scale = clamp(shapeScale, 0.0, 1.0);
  let edgeWidth = 0.34 * (1.0 - scale);
  return smoothstep(
    -edgeWidth,
    edgeWidth,
    p.y + 0.34 * (prop - 0.5) + sin(p.x * 1.7) * 0.04
  );
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  let luma = clamp(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  return color + vec3<f32>(grain * grainStrength * (0.35 + 0.65 * pow(1.0 - luma, 2.0)));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = proportion
  // path.shaderParams[1].z = softness
  // path.shaderParams[1].w = distortion
  // path.shaderParams[2].x = swirl
  // path.shaderParams[2].y = swirl iterations
  // path.shaderParams[2].z = grain
  // path.shaderParams[2].w = speed
  // path.colorData[0..2] = colors
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let proportion = path.shaderParams[1].y;
  let softness = clamp(path.shaderParams[1].z, 0.0, 1.0);
  let distortion = path.shaderParams[1].w;
  let swirl = path.shaderParams[2].x;
  let iterations = min(max(u32(path.shaderParams[2].y + 0.5), 1u), 30u);
  let grain = path.shaderParams[2].z;
  let speed = path.shaderParams[2].w;
  let t = time * speed * 0.5;
  let sceneScale = remapSceneScale(scale);

  let p = warpedCoord(uv, resolution, t, sceneScale, rotation, offset, distortion, swirl, iterations);
  let mixer = patternMixer(p, proportion, PATTERN_SCALE);
  let edgesWidth = 1.0 - softness;
  let edgeBlur = mix(0.012, 0.034, scale);
  let mixed = blendThreeColors(mixer, edgesWidth, edgeBlur, path);
  let vignetteP = vec2<f32>((uv.x - 0.5) * resolution.x / resolution.y, uv.y - 0.5);
  let vignette = smoothstep(1.05, 0.18, length(vignetteP));
  let color = filmGrain(clamp(mixed.rgb * mix(0.86, 1.08, vignette), vec3<f32>(0.0), vec3<f32>(1.0)), pixel, grain);

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), mixed.a);
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
