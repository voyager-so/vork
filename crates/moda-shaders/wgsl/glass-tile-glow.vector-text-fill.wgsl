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


// Shader ID: glass-tile-glow

const PI: f32 = 3.141592653589793;

fn safeAspect(resolution: vec2<f32>) -> f32 {
  return resolution.x / max(resolution.y, 1.0);
}

fn roundedBoxSdf(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
  let q = abs(p) - halfSize + vec2<f32>(radius);
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn softNoise(p: vec2<f32>) -> f32 {
  let n = valueNoise(p);
  let m = valueNoise(p * 2.07 + vec2<f32>(17.3, 9.1));
  return n * 0.68 + m * 0.32;
}

// The background is a mix of two colours by one scalar glow, so the blurred (and chromatically split)
// background below averages the scalar and mixes once: 27 scalar evaluations instead of 27 colour ones.
fn backgroundField(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  glowStrength: f32,
  path: CustomShaderData
) -> vec3<f32> {
  let baseColor = clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let glowColor = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  return mix(baseColor, glowColor, backgroundGlow(uv, resolution, time, glowStrength));
}

fn backgroundGlow(uv: vec2<f32>, resolution: vec2<f32>, time: f32, glowStrength: f32) -> f32 {
  let aspect = safeAspect(resolution);
  let p = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
  let drift = vec2<f32>(sin(time * 0.21) * 0.16 * aspect, cos(time * 0.17) * 0.09);
  let verticalWave =
    sin((p.x * 1.15 + p.y * 0.55) * PI + time * 0.34) * 0.04 +
    cos((p.x * -0.45 + p.y * 1.35) * PI - time * 0.27) * 0.03;
  let vertical = pow(clamp(uv.y + verticalWave - 0.1, 0.0, 1.0), 1.38);
  let leftCenter = vec2<f32>(-0.42 * aspect, 0.28) + drift;
  let rightCenter = vec2<f32>(0.38 * aspect, 0.2) + vec2<f32>(cos(time * 0.19) * 0.2 * aspect, sin(time * 0.15) * 0.1);
  let leftDelta = p - leftCenter;
  let rightDelta = p - rightCenter;
  let leftBloom = exp(-dot(leftDelta, leftDelta) * 2.75);
  let rightBloom = exp(-dot(rightDelta, rightDelta) * 2.05);
  let lowBand = smoothstep(0.08, 0.82, vertical);
  let cloud = softNoise(vec2<f32>(p.x * 1.05 + time * 0.18, p.y * 1.35 - time * 0.14));
  let cloud2 = softNoise(vec2<f32>(p.x * 2.25 - time * 0.12, p.y * 1.75 + time * 0.16));
  let glow = clamp((lowBand * 0.78 + leftBloom * 0.34 + rightBloom * 0.48 + cloud * 0.14 + cloud2 * 0.06) * glowStrength, 0.0, 1.35);
  return clamp(glow, 0.0, 1.0);
}

fn tileHeight(column: f32, aspect: f32, coverage: f32, randomness: f32) -> f32 {
  let n = hash21(vec2<f32>(column, 4.17));
  let n2 = hash21(vec2<f32>(column * 1.73 + 8.0, 11.31));
  let wave = 0.5 + 0.5 * sin(column * 0.77 + n2 * 2.4);
  let sideLift =
    exp(-pow((column - 1.1) * 0.42, 2.0)) * 0.16 +
    exp(-pow((column - aspect * 8.1) * 0.08, 2.0)) * 0.16;
  let shaped = mix(0.62, 0.42 + wave * 0.72 + sideLift, randomness);
  return clamp(coverage * shaped, 0.08, 0.82);
}

fn tileMaskAndData(
  p: vec2<f32>,
  resolution: vec2<f32>,
  tileSize: f32,
  tileGap: f32,
  coverage: f32,
  randomness: f32,
  borderRadius: f32,
  time: f32
) -> vec4<f32> {
  let aspect = safeAspect(resolution);
  let size = max(tileSize, 0.001);
  let pitch = max(tileSize + tileGap, 0.001);
  let cellId = floor(p / pitch);
  let local = fract(p / pitch) * pitch - vec2<f32>(pitch * 0.5);
  let height = tileHeight(cellId.x, aspect, coverage, randomness);
  let visible = step((cellId.y + 0.5) * pitch, height);
  let topFade = 1.0 - smoothstep(height - pitch * 1.35, height + pitch * 0.5, p.y);
  let dropout = step(0.1, hash21(cellId + vec2<f32>(3.7, 1.9)));
  let halfSize = vec2<f32>(size * 0.5);
  let radius = size * clamp(borderRadius, 0.02, 0.48);
  let sdf = roundedBoxSdf(local, halfSize, radius);
  let aa = fwidth(sdf) * 2.2 + 0.001;
  let opacityTime = time * 0.22;
  let opacityStep = floor(opacityTime);
  let opacityBlend = smoothstep(0.0, 1.0, fract(opacityTime));
  let opacityA = hash21(cellId + vec2<f32>(9.2 + opacityStep * 13.7, 5.4 - opacityStep * 4.1));
  let opacityB = hash21(cellId + vec2<f32>(9.2 + (opacityStep + 1.0) * 13.7, 5.4 - (opacityStep + 1.0) * 4.1));
  let animatedHighlight = mix(opacityA, opacityB, opacityBlend);
  let opacityJitter = mix(0.9, 0.52 + animatedHighlight * 0.48, randomness);
  let tileAlpha = visible * topFade * dropout * opacityJitter;
  let body = (1.0 - smoothstep(0.0, aa, sdf)) * tileAlpha;

  return vec4<f32>(sdf, body, tileAlpha, opacityJitter);
}

fn tilePanelSdfAt(
  p: vec2<f32>,
  resolution: vec2<f32>,
  tileSize: f32,
  tileGap: f32,
  coverage: f32,
  randomness: f32,
  borderRadius: f32,
  time: f32
) -> f32 {
  return tileMaskAndData(p, resolution, tileSize, tileGap, coverage, randomness, borderRadius, time).x;
}

fn blurredGlow(
  uv: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  glowStrength: f32,
  blur: f32
) -> f32 {
  let pixel = 1.0 / max(resolution, vec2<f32>(1.0));
  let radius = 40.0 * blur;
  var acc = 0.0;

  acc += backgroundGlow(uv + pixel * vec2<f32>(0.0, 0.0) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(-0.737, 0.675) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(0.087, -0.996) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(0.608, 0.794) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(-0.985, -0.174) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(0.844, -0.537) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(-0.259, 0.966) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(-0.460, -0.888) * radius, resolution, time, glowStrength);
  acc += backgroundGlow(uv + pixel * vec2<f32>(0.940, 0.342) * radius, resolution, time, glowStrength);

  return acc / 9.0;
}

fn squareTileGlass(
  uv: vec2<f32>,
  p: vec2<f32>,
  resolution: vec2<f32>,
  time: f32,
  tileSize: f32,
  tileGap: f32,
  coverage: f32,
  randomness: f32,
  borderRadius: f32,
  glowStrength: f32,
  path: CustomShaderData
) -> vec4<f32> {
  let aspect = safeAspect(resolution);
  let center = vec2<f32>(0.5);
  let tile = tileMaskAndData(p, resolution, tileSize, tileGap, coverage, randomness, borderRadius, time);
  let rawSdf = tile.x;
  let body = tile.y;
  let opacityJitter = tile.w;
  let pxH = 1.0 / max(resolution.y, 1.0);
  let eps = 0.01;
  let gradX = (tilePanelSdfAt(p + vec2<f32>(eps, 0.0), resolution, tileSize, tileGap, coverage, randomness, borderRadius, time) - rawSdf) / eps;
  let gradY = (tilePanelSdfAt(p + vec2<f32>(0.0, eps), resolution, tileSize, tileGap, coverage, randomness, borderRadius, time) - rawSdf) / eps;
  let sharp = 0.05;
  let rb1 = clamp(-rawSdf / sharp * 32.0, 0.0, 1.0);
  let rb2Base = clamp(-(rawSdf - pxH) / sharp * 16.0, 0.0, 1.0) - clamp(-rawSdf / sharp * 16.0, 0.0, 1.0);
  let lightDir = vec2<f32>(cos(300.0 * PI / 180.0), sin(300.0 * PI / 180.0));
  let lightFacing = dot(vec2<f32>(gradX, gradY), lightDir) * 0.5 + 0.5;
  let rb2 = rb2Base * lightFacing * 0.3;
  let depthNorm = clamp(-rawSdf / 0.3, 0.0, 1.0);
  let refrStrength = (1.0 - depthNorm) * (1.0 - depthNorm);
  let refraction = 1.0;
  let chromaticAberration = 0.7;
  let blur = 2.3;
  let refractOffset = vec2<f32>(-gradX / aspect, gradY) * 1.57 * 0.15 * refraction * refrStrength;
  let lensUv = center + (uv - center) + refractOffset;
  let chroma = refractOffset * 0.06 * chromaticAberration;
  // Only a pixel a tile covers shows the refracted background (the caller mixes by `transition`), so
  // the 27-tap blur runs there alone. No derivative is taken past this branch.
  let transition = smoothstep(0.0, 1.0, rb1) * body;
  var blurred = vec3<f32>(0.0);
  if (transition > 0.0) {
    let blurredGlowRgb = vec3<f32>(
      blurredGlow(lensUv + chroma, resolution, time, glowStrength, blur),
      blurredGlow(lensUv, resolution, time, glowStrength, blur),
      blurredGlow(lensUv - chroma, resolution, time, glowStrength, blur)
    );
    blurred = mix(
      clamp(path.colorData[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
      clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
      blurredGlowRgb
    );
  }
  let n = normalize(vec3<f32>(gradX, -gradY, 2.0));
  let h = normalize(vec3<f32>(lightDir, 2.0));
  let shininess = exp2(8.0 - 0.5 * 7.0);
  let spec = pow(clamp(dot(n, h), 0.0, 1.0), shininess) * 0.3 * refrStrength;
  let fresnelDepth = clamp(-rawSdf / max(0.31 * 0.06, 0.001), 0.0, 1.0);
  let fresnel = pow(1.0 - fresnelDepth, 2.0) * 0.02 * rb1;
  let glowTint = clamp(path.colorData[1].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let bottomTint = smoothstep(0.18, 0.95, uv.y);
  let highlight = smoothstep(0.54, 1.0, opacityJitter);
  let milk = mix(0.08, 0.42, highlight);
  let edgeLight = mix(0.45, 1.25, highlight);
  let lighting =
    mix(blurred, vec3<f32>(1.0), milk) +
    vec3<f32>((rb2 + spec + fresnel) * edgeLight) +
    glowTint * body * bottomTint * mix(0.004, 0.026, highlight) * glowStrength;
  return vec4<f32>(lighting, transition);
}

fn filmGrain(color: vec3<f32>, fragCoord: vec2<f32>, grainStrength: f32) -> vec3<f32> {
  let grain = hash21(fragCoord) * 2.0 - 1.0;
  return color + vec3<f32>(grain * grainStrength);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.0), vec2<f32>(1.0));
  // Framebuffer-space coords for film grain (avoids chunky Retina grain).
  let pixel = input.position.xy;
  let resolution = max(b.zw, vec2<f32>(1.0));

  // Packing map:
  // path.shaderParams[0].x = time
  // path.shaderParams[0].y = tile size
  // path.shaderParams[0].z = tile gap
  // path.shaderParams[0].w = tile coverage
  // path.shaderParams[1].x = randomness
  // path.shaderParams[1].y = border radius
  // path.shaderParams[1].z = glow strength
  // path.shaderParams[1].w = grain
  // path.shaderParams[2].x = speed
  // path.colorData[0].rgb = background color
  // path.colorData[1].rgb = glow color
  let time = path.shaderParams[0].x;
  let tileSize = path.shaderParams[0].y;
  let tileGap = path.shaderParams[0].z;
  let tileCoverage = path.shaderParams[0].w;
  let randomness = path.shaderParams[1].x;
  let borderRadius = path.shaderParams[1].y;
  let glowStrength = path.shaderParams[1].z;
  let grain = path.shaderParams[1].w;
  let speed = path.shaderParams[2].x;
  let t = time * speed * 2.0;
  let base = backgroundField(uv, resolution, t, glowStrength, path);
  let p = vec2<f32>(uv.x * safeAspect(resolution), 1.0 - uv.y);
  let tileGlass = squareTileGlass(uv, p, resolution, t, tileSize, tileGap, tileCoverage, randomness, borderRadius, glowStrength, path);
  var color = mix(base, tileGlass.rgb, tileGlass.a);
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
