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


// Shader ID: mesh-glass-rods

const PI: f32 = 3.141592653589793;

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
    centered / max(scale, 0.001) + vec2<f32>(-offset.x * aspect, offset.y),
    -rotation
  );

  return vec2<f32>(transformed.x / aspect + 0.5, transformed.y + 0.5);
}

fn signedValueNoise(p: vec2<f32>) -> f32 {
  return valueNoise(p) * 2.0 - 1.0;
}

fn fbm1(p: vec2<f32>) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.0;

  for (var i = 0; i < 5; i = i + 1) {
    sum += signedValueNoise(q) * amp;
    q = rotate2(q * 2.03 + vec2<f32>(19.17, -7.31), 0.62);
    amp *= 0.52;
  }

  return sum;
}

fn mirrored(value: vec2<f32>) -> vec2<f32> {
  let m = value - floor(value / 2.0) * 2.0;

  return mix(m, vec2<f32>(2.0) - m, step(vec2<f32>(1.0), m));
}

fn meshBackground(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  var p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let drift = vec2<f32>(time * 0.028, -time * 0.021);
  let warpA = fbm1(p * 1.35 + drift);
  let warpB = fbm1(rotate2(p * 1.8 - drift.yx, 0.86));
  let wave = vec2<f32>(
    sin(p.y * 4.8 + time * 0.95),
    cos(p.x * 5.7 - time * 0.82)
  );
  p += vec2<f32>(warpA, warpB) * 0.16 + wave * vec2<f32>(0.045, 0.06);

  let color1 = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color2 = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color3 = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let color4 = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let anchor1 = vec2<f32>(-0.46 * aspect, -0.34) + vec2<f32>(sin(time * 0.19), cos(time * 0.13)) * vec2<f32>(0.14 * aspect, 0.1);
  let anchor2 = vec2<f32>(0.42 * aspect, -0.29) + vec2<f32>(cos(time * 0.17), sin(time * 0.11)) * vec2<f32>(0.16 * aspect, 0.11);
  let anchor3 = vec2<f32>(-0.38 * aspect, 0.33) + vec2<f32>(sin(time * 0.14 + 1.7), cos(time * 0.18)) * vec2<f32>(0.13 * aspect, 0.12);
  let anchor4 = vec2<f32>(0.44 * aspect, 0.31) + vec2<f32>(cos(time * 0.12 + 0.8), sin(time * 0.16)) * vec2<f32>(0.15 * aspect, 0.1);
  let w1 = 1.0 / max(dot(p - anchor1, p - anchor1), 0.035);
  let w2 = 1.0 / max(dot(p - anchor2, p - anchor2), 0.035);
  let w3 = 1.0 / max(dot(p - anchor3, p - anchor3), 0.035);
  let w4 = 1.0 / max(dot(p - anchor4, p - anchor4), 0.035);
  let weightSum = w1 + w2 + w3 + w4;
  let pooled = (color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4) / max(weightSum, 0.001);

  return pooled;
}

fn filmGrainNoise(fragCoord: vec2<f32>) -> f32 {
  return hash21(fragCoord) * 2.0 - 1.0;
}

fn rodsLayer(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  distortion: f32,
  chromaticAberration: f32,
  waviness: f32,
  frequency: f32,
  rods: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  var p = uv * 2.0 - vec2<f32>(1.0);
  p.x *= aspect;

  let angle = 45.0 * PI / 180.0;
  let pRot = rotate2(p, angle);
  let rodTravel = time * 0.08;
  let travelingY = pRot.y + rodTravel;
  let wave = waviness * sin(travelingY * frequency);
  let rodX = fract((pRot.x + wave) * rods) * 2.0 - 1.0;
  let rodZSq = 1.0 - rodX * rodX;
  let rodZ = sqrt(max(rodZSq, 0.0));
  let n = vec3<f32>(rodX, 0.0, -rodZ);
  let rd = vec3<f32>(0.0, 0.0, -1.0);
  let refractiveIndex = 0.6;
  let refractedRay = mix(n, rd, refractiveIndex);
  let zDist = 0.5 / (abs(refractedRay.z) + 0.001);
  let hitPos = vec3<f32>(vec2<f32>(pRot.x, travelingY), 0.0) + (zDist * distortion) * refractedRay;
  var uvHit = rotate2(hitPos.xy, -angle);

  uvHit.x /= aspect;
  uvHit = uvHit * 0.5 + vec2<f32>(0.5);

  let flowClock = time * 0.1;
  let flow = vec2<f32>(sin(flowClock), cos(flowClock * 0.8)) * 0.05;
  let warpedUv = mirrored(uvHit + flow);
  let r = meshBackground(mirrored(warpedUv + vec2<f32>(chromaticAberration, 0.0)), resolution, time, path).r;
  let g = meshBackground(warpedUv, resolution, time, path).g;
  let b = meshBackground(mirrored(warpedUv - vec2<f32>(chromaticAberration, 0.0)), resolution, time, path).b;
  let glassFactor = smoothstep(0.0, 1.0, 1.0 - abs(n.z));
  let glass = glassFactor * 0.0025;

  return vec3<f32>(r, g, b) + vec3<f32>(glass);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = distortion
  // path.shaderParams[0].z = scale
  // path.shaderParams[0].w = rotation
  // path.shaderParams[1].x = offsetX
  // path.shaderParams[1].y = offsetY
  // path.shaderParams[1].z = chromatic aberration
  // path.shaderParams[1].w = waviness
  // path.shaderParams[2].x = frequency
  // path.shaderParams[2].y = rods
  // path.shaderParams[2].z = grain
  // path.shaderParams[2].w = speed
  // path.colorData[0..3].rgb = mesh palette
  let time = path.shaderParams[0].x;
  let distortion = path.shaderParams[0].y;
  let scale = path.shaderParams[0].z;
  let rotation = path.shaderParams[0].w;
  let offset = vec2<f32>(path.shaderParams[1].x, path.shaderParams[1].y);
  let chromaticAberration = path.shaderParams[1].z;
  let waviness = path.shaderParams[1].w;
  let frequency = path.shaderParams[2].x;
  let rods = path.shaderParams[2].y;
  let grain = path.shaderParams[2].z;
  let speed = path.shaderParams[2].w;
  let t = time * speed;
  let aspect = resolution.x / resolution.y;
  let effectUv = transformUv(uv, aspect, scale, rotation, offset);
  var color = rodsLayer(
    effectUv,
    resolution,
    t,
    distortion,
    chromaticAberration,
    waviness,
    frequency,
    rods,
    path
  );

  // Framebuffer coords for film grain: @builtin(position) is bit-exact across
  // rasterizers (interpolated varyings like localPos are not).
  color += vec3<f32>(filmGrainNoise(input.position.xy) * grain);

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
