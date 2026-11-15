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


// Shader ID: soft-voronoi

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

fn scalarSeed(value: f32) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(value)));
}

fn tileSeed(tile: vec2<f32>) -> vec2<f32> {
  let hx = hashU32(bitcast<u32>(tile.x) ^ hashU32(bitcast<u32>(tile.y)));
  let hy = hashU32(hx ^ 0x9e3779b9u);

  return vec2<f32>(hashToUnit(hx), hashToUnit(hy));
}

fn cellInk(tile: vec2<f32>, tint: f32) -> vec3<f32> {
  let n = scalarSeed(dot(tile, vec2<f32>(19.0, 131.0)));
  let phase = n * 4.2 + vec3<f32>(0.35, 2.15, 4.45) + tint;
  let soft = 0.52 + 0.48 * sin(phase);

  return soft * soft;
}

fn softVoronoi(point: vec2<f32>, clock: f32, width: f32, tint: f32) -> vec4<f32> {
  let tile = floor(point);
  let local = fract(point);

  var best = 8.0;
  var color = vec3<f32>(0.0);

  for (var y = -2; y <= 2; y += 1) {
    for (var x = -2; x <= 2; x += 1) {
      let neighbor = vec2<f32>(f32(x), f32(y));
      let id = tile + neighbor;
      let seed = tileSeed(id);
      let orbit = 0.5 + 0.5 * sin(clock + seed * 6.2831853);
      let delta = neighbor - local + orbit;
      let distance = length(delta);
      let blend = smoothstep(-1.0, 1.0, (best - distance) / width);
      let easing = blend * (1.0 - blend);
      let settle = easing * width / (1.0 + 3.0 * width);

      best = mix(best, distance, blend) - settle;
      color = mix(color, cellInk(id, tint), vec3<f32>(blend)) - vec3<f32>(settle);
    }
  }

  return vec4<f32>(best, color);
}

fn spectral(value: f32, tint: f32) -> vec3<f32> {
  let base = vec3<f32>(0.48, 0.50, 0.52);
  let span = vec3<f32>(0.47, 0.45, 0.43);
  let rate = vec3<f32>(0.72, 1.00, 1.28);
  let bias = vec3<f32>(0.66, 0.31, 0.27) + tint * vec3<f32>(0.05, -0.04, 0.08);

  return base + span * cos(6.2831853 * (rate * value + bias));
}

fn transformUv(uv: vec2<f32>, aspect: f32, scale: f32, rotation: f32, offset: vec2<f32>) -> vec2<f32> {
  let pivot = vec2<f32>(0.5 * aspect, 0.5);
  var point = uv * vec2<f32>(aspect, 1.0);
  point = turn(-rotation) * ((point - pivot) / max(scale, 0.001) + vec2<f32>(-offset.x * 0.38 * aspect, offset.y * 0.38)) + pivot;

  return point / vec2<f32>(aspect, 1.0);
}

fn shadeSoftVoronoi(
  uv: vec2<f32>,
  pixel: vec2<f32>,
  time: f32,
  resolution: vec2<f32>,
  tint: f32,
  speed: f32,
  density: f32,
  smoothness: f32,
  grain: f32,
  exposure: f32,
) -> vec4<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let clock = time * speed * 0.14;
  var point = uv * vec2<f32>(aspect, 1.0);
  // Per-pixel jitter hashed from exact framebuffer coords (rasterizer-independent),
  // not from the interpolated uv-derived point.
  point += tileSeed(pixel) * (0.0015 + grain * 0.004);
  point = turn(sin(clock * 0.27) * 0.035) * (point - vec2<f32>(0.5 * aspect, 0.5))
    + vec2<f32>(0.5 * aspect, 0.5);

  let field = softVoronoi(point * density, clock, smoothness, tint);
  let slope = (point.y - 0.5) * 0.34;
  let chroma = dot(field.yzw, vec3<f32>(0.32, 0.45, 0.23));
  let bands = field.x * 0.58 + chroma * 0.36 - slope + tint * 0.08;
  let paper = 0.08 + 0.92 * spectral(bands, tint);
  let luminous = 0.88 + 0.12 * field.yzw;
  let dust = scalarSeed(dot(floor(pixel * 0.6), vec2<f32>(17.0, 59.0)));

  var color = paper * luminous;
  color *= exposure;
  color += (dust - 0.5) * grain * 0.035;
  color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(0.55));

  return vec4<f32>(color, 1.0);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer coords for grain/jitter: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  let pixel = input.position.xy;
  let resolution = b.zw;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = tint
  // path.shaderParams[1].z = speed
  // path.shaderParams[2].x = smoothness
  // path.shaderParams[2].y = grain
  // path.shaderParams[2].z = exposure
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let tint = path.shaderParams[1].y;
  let speed = path.shaderParams[1].z;
  let density = 1.55;
  let smoothness = path.shaderParams[2].x;
  let grain = path.shaderParams[2].y;
  let exposure = path.shaderParams[2].z;

  let transformedUv = transformUv(uv, resolution.x / max(resolution.y, 1.0), scale, rotation, offset);
  let color = shadeSoftVoronoi(transformedUv, pixel, time, resolution, tint, speed, density, smoothness, grain, exposure);

  return color;
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
