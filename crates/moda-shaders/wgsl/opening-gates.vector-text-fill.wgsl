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


// Shader ID: opening-gates

const PI: f32 = 3.141592653589793;

fn saturate(v: f32) -> f32 {
  return clamp(v, 0.0, 1.0);
}

fn safeAspect(resolution: vec2<f32>) -> f32 {
  return resolution.x / max(resolution.y, 1.0);
}

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn paletteGradient(uv: vec2<f32>, local: vec2<f32>, phase: f32, side: f32, layer: f32, path: CustomShaderData) -> vec3<f32> {
  let c0 = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c1 = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c2 = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c3 = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let diagonal = dot(local, normalize(vec2<f32>(0.72 + side * 0.12, 1.0)));
  let broad = 0.5 + 0.5 * sin(diagonal * 3.2 + phase * 2.7 + layer * 0.71);
  let plume = 0.5 + 0.5 * sin((uv.x * 2.0 - uv.y * 1.35) * PI + phase * 4.2 + layer);
  let wash = smoothstep(0.08, 0.95, uv.y + sin(uv.x * 2.4 + phase) * 0.08);
  let base = mix(c0, c1, saturate(broad * 0.8 + wash * 0.18));
  let cool = mix(base, c2, smoothstep(0.42, 0.92, plume) * (0.35 + 0.18 * side));
  let warm = mix(cool, c3, smoothstep(0.55, 0.98, broad) * 0.38);
  let milk = smoothstep(0.48, 1.0, broad + plume * 0.22) * 0.34;

  return mix(warm, vec3<f32>(1.0), milk);
}

fn softBackground(uv: vec2<f32>, p: vec2<f32>, time: f32, path: CustomShaderData) -> vec3<f32> {
  let c0 = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c1 = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c2 = clamp(path.colorData[2].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let c3 = clamp(path.colorData[3].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let a = 0.5 + 0.5 * sin((p.x * 0.7 + p.y * 1.4) * PI + time * 0.55);
  let b = 0.5 + 0.5 * cos((p.x * -1.2 + p.y * 0.55) * PI - time * 0.41);
  let base = mix(c0, c1, smoothstep(0.15, 0.95, a));
  let cool = mix(base, c2, smoothstep(0.48, 1.0, b) * 0.62);
  let glow = mix(cool, c3, smoothstep(0.72, 1.0, a + b * 0.36) * 0.35);

  return mix(glow, vec3<f32>(1.0), 0.08 + (1.0 - uv.y) * 0.1);
}

fn gateLayer(
  uv: vec2<f32>,
  p: vec2<f32>,
  resolution: vec2<f32>,
  spawnId: f32,
  phase: f32,
  opening: f32,
  edgeStrength: f32,
  path: CustomShaderData
) -> vec4<f32> {
  let angleSlot = fract(spawnId * 0.38196601125);
  let angleJitter = (hash21(vec2<f32>(spawnId, 3.17)) - 0.5) * 0.18;
  let angle = -1.05 + angleSlot * 2.1 + angleJitter;
  let n = vec2<f32>(cos(angle), sin(angle));
  let tangent = vec2<f32>(-n.y, n.x);
  let seamReveal = smoothstep(0.09, 0.22, phase);
  let openCurve = smoothstep(0.12, 1.0, phase);
  let openDistance = openCurve * openCurve * (2.75 * opening);
  let birth = smoothstep(0.0, 0.045, phase);
  let layerAlpha = birth;
  let lineOffset = 0.0;
  let sideNow = select(-1.0, 1.0, dot(p, n) - lineOffset >= 0.0);
  let localP = p - n * sideNow * openDistance;
  let splitDistance = dot(localP, n) - lineOffset;
  let sideMask = select(0.0, 1.0, splitDistance * sideNow >= 0.0);
  let px = 1.0 / max(resolution.y, 1.0);
  let cutFeather = smoothstep(0.0, px * 1.5, abs(splitDistance));
  let screenExtent =
    abs(n.x) * (safeAspect(resolution) * 0.5 + 0.25) +
    abs(n.y) * 0.75 +
    abs(lineOffset);
  let onScreen = 1.0 - smoothstep(screenExtent - 0.08, screenExtent + 0.08, openDistance);
  let openPanelMask = sideMask * cutFeather;
  let closedPanelMask = 1.0;
  let panelMask = mix(closedPanelMask, openPanelMask, seamReveal) * layerAlpha * onScreen;
  let local = vec2<f32>(dot(localP, tangent), splitDistance);
  let sourceUv = uv + vec2<f32>(n.x / max(safeAspect(resolution), 0.001), -n.y) * sideNow * openDistance * 0.09;
  let grad = paletteGradient(sourceUv, local, phase, sideNow, spawnId, path);
  let distanceToCut = abs(splitDistance);
  let fineEdge = exp(-distanceToCut * 220.0) * edgeStrength * seamReveal;
  let broadEdge = exp(-distanceToCut * 55.0) * 0.12 * edgeStrength * seamReveal;
  let shadow = exp(-distanceToCut * 90.0) * 0.14 * edgeStrength * seamReveal;
  let lightFacing = dot(n * sideNow, normalize(vec2<f32>(-0.32, 0.95))) * 0.5 + 0.5;
  let edgeColor = vec3<f32>(fineEdge * (0.28 + lightFacing * 0.62) + broadEdge);
  let seamDark = vec3<f32>(shadow * (1.0 - lightFacing) * 0.55);
  let depthWash = mix(0.94, 1.08, 1.0 - phase);
  let color = grad * depthWash + edgeColor - seamDark;

  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), panelMask);
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  // Integer hash (noise include) instead of fract(sin(...)) — the sine hash diverges across GPU
  // backends, breaking WebGPU/native parity.
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  return color + vec3<f32>(grain * grainStrength * (0.45 + (1.0 - luma) * 0.55));
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));
  let aspect = safeAspect(resolution);
  let p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = speed
  // path.shaderParams[0].z = opening
  // path.shaderParams[0].w = edge strength
  // path.shaderParams[1].x = grain
  // path.colorData[0..3].rgb = palette
  let time = path.shaderParams[0].x;
  let speed = path.shaderParams[0].y;
  let opening = path.shaderParams[0].z;
  let edgeStrength = path.shaderParams[0].w;
  let grain = path.shaderParams[1].x;
  let t = time * speed;

  var color = softBackground(uv, p, t, path);

  let gateClock = t * 1.08;
  let gateBase = floor(gateClock);
  let gateFrac = fract(gateClock);

  for (var i = 0; i < 7; i = i + 1) {
    let slot = f32(i);
    let phase = (gateFrac + slot) / 7.0;
    let spawnId = gateBase - slot;
    let layer = gateLayer(uv, p, resolution, spawnId, phase, opening, edgeStrength, path);
    color = mix(color, layer.rgb, layer.a);
  }

  color = filmGrain(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), pixel, grain);

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
