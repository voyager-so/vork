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


// Shader ID: halftone-cell-wall

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
    centered / max(scale, 0.001) + vec2<f32>(-offset.x * aspect, -offset.y),
    -rotation
  );

  return vec2<f32>(transformed.x / aspect + 0.5, transformed.y + 0.5);
}

fn halftoneSwirlField(
  uv: vec2<f32>,
  aspect: f32,
  time: f32,
  swirlDetail: f32,
  swirlStrength: f32,
  colorA: vec3<f32>,
  colorB: vec3<f32>,
  colorC: vec3<f32>
) -> vec3<f32> {
  let detail = max(swirlDetail, 0.001);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let radius = length(p);
  let spin = atan2(p.y, p.x) + sin(radius * detail * 3.4 - time * 0.55) * 0.18 * swirlStrength;
  p = vec2<f32>(cos(spin), sin(spin)) * radius;

  var q = rotate2(p, 0.35 + sin(time * 0.13) * 0.16);
  q += vec2<f32>(
    sin(q.y * detail * 2.25 + time * 0.82) * 0.115,
    cos(q.x * detail * 2.05 - time * 0.74) * 0.105
  ) * swirlStrength;

  let layerA = sin((q.x * 2.55 + q.y * 1.35) * detail + time * 0.44);
  let layerB = cos((q.x * -1.7 + q.y * 2.85) * detail - time * 0.36);
  let layerC = sin((q.x + q.y) * detail * 2.2 + sin(time * 0.21) * 1.7);
  let combined = layerA * 0.42 + layerB * 0.36 + layerC * 0.22;

  let mixAB = smoothstep(0.18, 0.82, combined * 0.5 + 0.5);
  let accent = smoothstep(0.66, 0.98, sin(layerA * 2.1 + layerC * 1.4 + time * 0.7) * 0.5 + 0.5);
  var color = mix(colorA, colorB, mixAB);
  color = mix(color, colorC, accent * 0.28);
  color *= 1.0 + sin(time * 2.0 + combined * 7.0) * 0.018;

  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  let brightness = clamp(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  let shadow = 1.0 - brightness + 0.000001;
  return color + vec3<f32>(grain * shadow * shadow * grainStrength);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain); the cell grid stays in
  // shape-local space via effectPixel below.
  let grainPixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = cell size
  // path.shaderParams[0].w = radius
  // path.shaderParams[1].x = rotation
  // path.shaderParams[1].y = offsetX
  // path.shaderParams[1].z = offsetY
  // path.shaderParams[1].w = swirl detail
  // path.shaderParams[2].x = swirl strength
  // path.shaderParams[2].y = jitter
  // path.shaderParams[2].z = grain
  // path.shaderParams[2].w = speed
  // path.colorData[0..2].rgb = swirl colors
  let p0 = path.shaderParams[0];
  let p1 = path.shaderParams[1];
  let p2 = path.shaderParams[2];

  let time = p0.x;
  let scale = p0.y;
  let cellSize = max(p0.z, 2.0);
  let radius = p0.w;
  let rotation = p1.x;
  let offset = vec2<f32>(p1.y, p1.z);
  let swirlDetail = p1.w;
  let swirlStrength = p2.x;
  let jitter = p2.y;
  let grain = p2.z;
  let speed = p2.w;
  let t = time * speed;
  let aspect = resolution.x / resolution.y;
  let invResolution = 1.0 / resolution;
  let colorA = path.colorData[0].rgb;
  let colorB = path.colorData[1].rgb;
  let colorC = path.colorData[2].rgb;
  let effectUv = transformUv(uv, aspect, scale, rotation, offset);
  let effectPixel = effectUv * resolution;
  let baseCell = floor(effectPixel / cellSize);
  let radiusBase = cellSize * radius;
  let aa = max(cellSize * 0.025, 0.65);

  var finalColor = vec3<f32>(0.0);
  var maxCircle = 0.0;
  // A cell's dot is at most radiusBase * 1.14 wide plus the aa band, and its centre moves at most
  // cellSize * jitter / 2 per axis: a cell whose unjittered centre is farther than that from this pixel
  // draws circle = 0 here, which never changes the result (a zero circle only wins while maxCircle is
  // still 0, and then the background shows). Skipping those cells skips their swirl-field sample —
  // most of the 25.
  let reach = radiusBase * 1.14 + aa + cellSize * max(jitter, 0.0) * 0.7072 + 0.01;

  for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
      let neighbor = vec2<f32>(f32(dx), f32(dy));
      let cellIndex = baseCell + neighbor;
      let plainDelta = effectPixel - (cellIndex + vec2<f32>(0.5)) * cellSize;
      if (dot(plainDelta, plainDelta) > reach * reach) {
        continue;
      }
      var jitterVec = vec2<f32>(0.0);
      if (jitter > 0.0001) {
        jitterVec = vec2<f32>(
          hash21(cellIndex + vec2<f32>(17.31, 4.97)) - 0.5,
          hash21(cellIndex + vec2<f32>(2.43, 29.11)) - 0.5
        ) * cellSize * jitter;
      }
      let cellCenter = (cellIndex + vec2<f32>(0.5)) * cellSize + jitterVec;
      let sampleUv = clamp(cellCenter * invResolution, vec2<f32>(0.0), vec2<f32>(1.0));
      let texColor = halftoneSwirlField(sampleUv, aspect, t, swirlDetail, swirlStrength, colorA, colorB, colorC);
      let luma = dot(texColor, vec3<f32>(0.2126, 0.7152, 0.0722));
      let localRadius = radiusBase * mix(0.72, 1.14, smoothstep(0.05, 0.95, luma));
      let delta = effectPixel - cellCenter;
      let distSq = dot(delta, delta);
      let radiusSq = localRadius * localRadius;
      let aaSq = max(2.0 * localRadius * aa, aa * aa);
      let circle = 1.0 - smoothstep(radiusSq - aaSq, radiusSq + aaSq, distSq);
      let winner = step(maxCircle, circle);
      finalColor = mix(finalColor, texColor, vec3<f32>(winner));
      maxCircle = max(maxCircle, circle);
    }
  }

  let background = halftoneSwirlField(effectUv, aspect, t, swirlDetail, swirlStrength, colorA, colorB, colorC) * 0.16;
  let vignetteP = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let vignette = smoothstep(1.06, 0.16, length(vignetteP));
  let color = mix(background, finalColor, vec3<f32>(maxCircle)) * mix(0.86, 1.08, vignette);
  var finalColorGrain = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  if (grain > 0.0001) {
    finalColorGrain = filmGrain(finalColorGrain, grainPixel, grain);
  }

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(clamp(finalColorGrain, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
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
