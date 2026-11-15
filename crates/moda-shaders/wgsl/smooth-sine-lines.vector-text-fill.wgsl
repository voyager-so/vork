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


// Shader ID: smooth-sine-lines

const MAX_LINES: u32 = 12u;

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn sstep(edgeA: f32, edgeB: f32, x: f32) -> f32 {
  if (edgeA < edgeB) {
    return smoothstep(edgeA, edgeB, x);
  }

  return 1.0 - smoothstep(edgeB, edgeA, x);
}

fn lineMasks(
  uvIn: vec2<f32>,
  speed: f32,
  height: f32,
  time: f32,
  waveAmount: f32,
  lineWidth: f32
) -> vec2<f32> {
  var uv = uvIn;
  let xEnvelope = sstep(1.0, 0.0, abs(uv.x));
  uv.y += xEnvelope * sin(time * speed + uv.x * height) * waveAmount;

  let width = max(0.0001, 0.06 * sstep(0.2, 0.9, abs(uv.x)) * lineWidth);
  let core = sstep(width, 0.0, abs(uv.y) - 0.004 * lineWidth);
  let edgeFade = sstep(1.0, 0.3, abs(uv.x));
  let halo = exp(-abs(uv.y) / max(width * 5.4, 0.004)) * edgeFade;

  return vec2<f32>(core * edgeFade, max(halo - core * 0.35, 0.0));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv01 = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  let aspect = b.z / max(b.w, 1.0);
  let centered = vec2<f32>((uv01.x - 0.5) * aspect, uv01.y - 0.5);

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = line count
  // path.shaderParams[0].z = rotation
  // path.shaderParams[0].w = wave amount
  // path.shaderParams[1].x = frequency multiplier
  // path.shaderParams[1].y = speed multiplier
  // path.shaderParams[1].z = glow
  // path.colorData[0].rgb = background color
  // path.colorData[1].rgb = line color
  let time = path.shaderParams[0].x;
  let lineCount = min(max(u32(path.shaderParams[0].y + 0.5), 2u), MAX_LINES);
  let rotation = path.shaderParams[0].z;
  let lineWidth = 0.5;
  let waveAmount = path.shaderParams[0].w;
  let frequency = path.shaderParams[1].x;
  let brightness = 1.0;
  let speedMultiplier = path.shaderParams[1].y;
  let glow = path.shaderParams[1].z;
  let uv = rotate2(centered, -rotation);

  let background = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let lineColor = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));

  var coreCoverage = 0.0;
  var glowCoverage = 0.0;
  let denominator = max(f32(lineCount - 1u), 1.0);

  for (var i = 0u; i < MAX_LINES; i = i + 1u) {
    if (i < lineCount) {
      let t = f32(i) / denominator;
      let masks = lineMasks(
        uv,
        1.0 + t,
        (4.0 + t) * frequency,
        time * speedMultiplier,
        waveAmount,
        lineWidth
      );
      let coreAlpha = clamp(masks.x * brightness, 0.0, 1.0);
      let glowAlpha = clamp(masks.y * glow * 0.65, 0.0, 1.0);
      coreCoverage = 1.0 - (1.0 - coreCoverage) * (1.0 - coreAlpha);
      glowCoverage = 1.0 - (1.0 - glowCoverage) * (1.0 - glowAlpha);
    }
  }

  let glowColor = mix(background, lineColor, clamp(glowCoverage, 0.0, 1.0));
  let color = mix(glowColor, lineColor, clamp(coreCoverage, 0.0, 1.0));

  // Straight-alpha; the framework epilogue applies fillColor.a + page-clip + clip-mask and premultiplies.
  return vec4<f32>(color, 1.0);
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
