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


// Shader ID: plasma-field

const ALPHA: f32 = 0.5;
const BETA: f32 = 0.5;
const GAMMA: f32 = 2.5;
const FIELD_SCALE: f32 = 6.0;
const FIELD_TWIST: f32 = 0.50;
const FIELD_DETAIL: f32 = 200.0;
const FIELD_DETAIL_INV: f32 = 1.0 / FIELD_DETAIL;

struct PlasmaContext {
  time: f32,
  aspect: f32,
  baseColor: vec3<f32>,
  warpDrift: vec2<f32>,
  fieldMove: mat2x2<f32>,
  fieldScale: f32,
  fieldOffset: vec3<f32>,
  fieldBias: vec3<f32>,
};

fn rotate2(point: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);

  return vec2<f32>(point.x * c - point.y * s, point.x * s + point.y * c);
}

fn fieldMove(a: f32) -> mat2x2<f32> {
  let c = cos(a);
  let s = sin(a);

  return mat2x2<f32>(
    vec2<f32>(c, s),
    vec2<f32>(-s, c)
  );
}

fn vfF(p: vec2<f32>, time: f32) -> f32 {
  return sin(p.x + sin(p.y + time * 0.1)) *
    sin(p.y * p.x * 0.1 + time * 0.2);
}

fn vectorWarp(pIn: vec2<f32>, ctx: PlasmaContext) -> vec2<f32> {
  var p = pIn;
  let ep = vec2<f32>(0.05, 0.0);
  var g = vec2<f32>(0.0);

  for (var i = 0; i < 20; i = i + 1) {
    let t0 = vfF(p, ctx.time);
    let t1 = vfF(p + ep.xy, ctx.time);
    let t2 = vfF(p + ep.yx, ctx.time);

    g = vec2<f32>(t1 - t0, t2 - t0) * 20.0;
    let tangent = vec2<f32>(-g.y, g.x);

    p += FIELD_TWIST * tangent + g * FIELD_DETAIL_INV + ctx.warpDrift;
  }

  return g;
}

fn mapField(stIn: vec3<f32>, ctx: PlasmaContext) -> f32 {
  var st = stIn;
  let movedXY = ctx.fieldMove * st.xy;
  st = vec3<f32>(movedXY.x, movedXY.y, st.z);
  let movedXZ = ctx.fieldMove * vec2<f32>(st.x, st.z);
  st = vec3<f32>(movedXZ.x, st.y, movedXZ.y);

  let p = st * ctx.fieldScale + ctx.fieldOffset;

  return length(st + ctx.fieldBias) +
    sin(p.x + sin(cos(p.y) + cos(p.z) * GAMMA)) * 0.5 - 2.0;
}

fn vignette(uvIn: vec2<f32>) -> f32 {
  var uv = clamp(uvIn, vec2<f32>(0.0), vec2<f32>(1.0));
  uv *= vec2<f32>(1.0) - uv.yx;
  var vig = uv.x * uv.y * 15.0;
  vig = pow(vig, 0.5);

  return vig;
}

fn pixel(p: vec2<f32>, resolution: vec2<f32>, ctx: PlasmaContext) -> vec3<f32> {
  let uv = p / resolution;
  var st = uv - vec2<f32>(1.0, 0.5);

  var fp = st;
  fp.x *= ctx.aspect;
  fp *= FIELD_SCALE;

  let warp = vectorWarp(fp, ctx);
  st += warp * 0.08;

  var col = vec3<f32>(BETA, GAMMA, ALPHA);
  let dist = 0.5;
  let ray = vec3<f32>(0.0, 0.0, BETA) + normalize(vec3<f32>(st, -1.0)) * dist;
  let rz = mapField(ray, ctx);
  let f = clamp((rz - mapField(ray + vec3<f32>(0.1), ctx)) * 0.5, -0.5, 1.0);

  let l = vec3<f32>(
    0.1588235294 * ALPHA,
    0.1254901961 * BETA,
    0.3490196080 * GAMMA
  ) + vec3<f32>(4.0, 2.5, 2.5) * f;

  let glow = smoothstep(5.0, 2.0, rz) * 0.4;

  // Intentional double application: stacking col*l + glow*l twice produces
  // the high-contrast metallic plasma look. Removing the second line makes
  // the shader visibly flatter / less interesting. Not a copy-paste bug.
  col = col * l + glow * l;
  col = col * l + glow * l;
  let brightness = dot(col, vec3<f32>(0.299, 0.587, 0.114));

  return ctx.baseColor * brightness;
}

fn blur3(p: vec2<f32>, resolution: vec2<f32>, direction: vec2<f32>, ctx: PlasmaContext) -> vec3<f32> {
  var color = vec3<f32>(0.0);

  let off1 = vec2<f32>(1.3846153846) * direction;

  color += pixel(p, resolution, ctx) * 0.2641839061;
  color += pixel(p + off1, resolution, ctx) * 0.3679080469;
  color += pixel(p - off1, resolution, ctx) * 0.3679080469;

  return color;
}

fn transformUv(uv: vec2<f32>, aspect: f32, scale: f32, rotation: f32, offset: vec2<f32>) -> vec2<f32> {
  let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let transformed = rotate2(centered / max(scale, 0.001) + vec2<f32>(-offset.x * 0.38 * aspect, offset.y * 0.38), -rotation);

  return vec2<f32>(transformed.x / aspect + 0.5, transformed.y + 0.5);
}

fn applyIridescence(color: vec3<f32>, uv: vec2<f32>, transformedUv: vec2<f32>, time: f32, amount: f32) -> vec3<f32> {
  let strength = clamp(amount, 0.0, 1.0);
  let centered = transformedUv - vec2<f32>(0.5);
  let brightness = dot(color, vec3<f32>(0.299, 0.587, 0.114));
  let phase = dot(centered, vec2<f32>(16.0, -11.0)) +
    length(centered) * 18.0 +
    brightness * 12.0 +
    sin((uv.x + uv.y) * 9.0 + time * 0.35) * 0.75;
  let film = 0.5 + 0.5 * cos(vec3<f32>(phase, phase + 2.0943951024, phase + 4.1887902048));
  let metallic = color * (0.72 + film * 1.45) + film * brightness * 0.32;

  return mix(color, metallic, strength);
}

fn shadePlasmaField(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  scale: f32,
  rotation: f32,
  offset: vec2<f32>,
  iridescence: f32,
  lightIntensity: f32,
  path: CustomShaderData
) -> vec4<f32> {
  let aspect = resolution.x / max(resolution.y, 1.0);
  let fieldTime = time / 5.0 + 1.0;
  let ctx = PlasmaContext(
    time,
    aspect,
    clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
    vec2<f32>(sin(time * 0.25) * 0.01, cos(time * 0.25) * 0.5),
    fieldMove(fieldTime * (0.2 + 0.2 * ALPHA)),
    (0.2 * sin(fieldTime / 10.0 * ALPHA) + 8.0 + 3.0 * BETA) * 2.0,
    vec3<f32>(fieldTime),
    vec3<f32>(sin(cos(fieldTime * 0.5 * GAMMA)))
  );
  let transformedUv = transformUv(uv, aspect, scale, rotation, offset);
  let fragCoord = transformedUv * resolution;
  let col = blur3(fragCoord, resolution, vec2<f32>(2.0, 2.0), ctx) * clamp(lightIntensity, 0.0, 3.0);
  let vig = vignette(uv);
  let mixed = applyIridescence(col, uv, transformedUv, time, iridescence);

  return vec4<f32>(mixed * vig, 1.0);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let resolution = b.zw;

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = scale
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = offsetX
  // path.shaderParams[1].x = offsetY
  // path.shaderParams[1].y = speed
  // path.shaderParams[1].z = iridescence
  // path.shaderParams[1].w = lightIntensity
  // path.colorData[0].rgb = base color
  let time = path.shaderParams[0].x;
  let scale = path.shaderParams[0].y;
  let rotation = path.shaderParams[0].z;
  let offset = vec2<f32>(path.shaderParams[0].w, path.shaderParams[1].x);
  let speed = path.shaderParams[1].y;
  let iridescence = path.shaderParams[1].z;
  let lightIntensity = path.shaderParams[1].w;
  let t = time * speed;

  let color = shadePlasmaField(uv, resolution, t, scale, rotation, offset, iridescence, lightIntensity, path);

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
