// Common constants

// Mathematical constants
const INVERSE_SQRT_2: f32 = 0.7071067811865476;

// Shader constants
const AA_PIXEL_EXPANSION: f32 = 0.5;
const MIN_BLUR_RADIUS: f32 = 0.01;
const MIN_GRADIENT_STOP_DIFF: f32 = 0.0001;


// Common geometry utilities

// Generate a unit quad (0,0) to (1,1) - used for fill rendering
fn getUnitQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(0.0, 0.0); }
    case 1u: { pos = vec2<f32>(1.0, 0.0); }
    case 2u: { pos = vec2<f32>(0.0, 1.0); }
    case 3u: { pos = vec2<f32>(0.0, 1.0); }
    case 4u: { pos = vec2<f32>(1.0, 0.0); }
    default: { pos = vec2<f32>(1.0, 1.0); }
  }
  return pos;
}

// Generate a normalized quad (-1,-1) to (1,1) - used for stroke rendering
fn getNormalizedQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(-1.0, -1.0); }
    case 1u: { pos = vec2<f32>( 1.0, -1.0); }
    case 2u: { pos = vec2<f32>(-1.0,  1.0); }
    case 3u: { pos = vec2<f32>(-1.0,  1.0); }
    case 4u: { pos = vec2<f32>( 1.0, -1.0); }
    default: { pos = vec2<f32>( 1.0,  1.0); }
  }
  return pos;
}

// Simple 2x2 rotation helper
fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let s = sin(angle);
  let c = cos(angle);
  return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Generate UV coordinates for a quad (0,0) to (1,1)
fn getQuadUV(vertexIndex: u32) -> vec2<f32> {
  var uv: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { uv = vec2<f32>(0.0, 0.0); }
    case 1u: { uv = vec2<f32>(1.0, 0.0); }
    case 2u: { uv = vec2<f32>(0.0, 1.0); }
    case 3u: { uv = vec2<f32>(0.0, 1.0); }
    case 4u: { uv = vec2<f32>(1.0, 0.0); }
    default: { uv = vec2<f32>(1.0, 1.0); }
  }
  return uv;
}

// Generate a centered unit quad (-0.5,-0.5) to (0.5,0.5)
fn getCenteredQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(-0.5, -0.5); }
    case 1u: { pos = vec2<f32>( 0.5, -0.5); }
    case 2u: { pos = vec2<f32>(-0.5,  0.5); }
    case 3u: { pos = vec2<f32>(-0.5,  0.5); }
    case 4u: { pos = vec2<f32>( 0.5, -0.5); }
    default: { pos = vec2<f32>( 0.5,  0.5); }
  }
  return pos;
}

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


      // Baked constants for optimal visual effect
      const TWO_PI: f32 = 6.283185307179586;
      const DISTORTION: f32 = 0.08;
      const SWIRL: f32 = 0.8;
      const SWIRL_MULT: f32 = 3.0;
      const WAVE: f32 = 0.25;
      const WAVE_PHASE: f32 = 0.15 * TWO_PI;
      const WAVE_Y_TIME_SCALE: f32 = 0.2;
      const POSITION_SEED: f32 = 25.0;
      const GRAIN_JITTER: f32 = 0.1;
      const GRAIN_OVERLAY: f32 = 0.24;

      // Spot animation frequency coefficients (module-level for guaranteed single allocation)
      const SPOT_FREQ_X: array<f32, 10> = array<f32, 10>(0.6, 0.9, 1.2, 0.6, 0.9, 1.2, 0.6, 0.9, 1.2, 0.6);
      const SPOT_FREQ_Y: array<f32, 10> = array<f32, 10>(0.8, 1.05, 1.3, 0.8, 1.05, 1.3, 0.8, 1.05, 1.3, 0.8);

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

      // Procedural position for i-th spot
      fn spotPos(i: u32, t: f32) -> vec2<f32> {
        let fi = f32(i);
        let a = fi * 0.37;
        let phase = t + a;
        return 0.5 + 0.5 * vec2<f32>(sin(phase * SPOT_FREQ_X[i]), cos(phase * SPOT_FREQ_Y[i] + a * 0.5));
      }

      fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
        // NOTE: This shader does not support strokes - designed for full-page backgrounds.
        // Stroke vertices will render as the mesh gradient (which is fine for backgrounds).

        // UV computation
        let b = path.bounds;
        let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.001), vec2<f32>(0.999));

        // Params: count, time, grain (minimal params)
        let params0 = path.shaderParams[0];
        let count = u32(params0.x + 0.5);
        let t = params0.y;
        let grain = params0.z;

        // Distortion - single pass, baked params
        let centerVec = uv - 0.5;
        let radius = length(centerVec);
        let center = 1.0 - radius;
        
        var shape_uv = uv;
        let sy = uv.y * uv.y * (3.0 - 2.0 * uv.y); // inline smoothstep(0,1,y)
        let sx = uv.x * uv.x * (3.0 - 2.0 * uv.x); // inline smoothstep(0,1,x)
        shape_uv.x += WAVE * center * cos(t + WAVE_PHASE + 2.0 * sy);
        shape_uv.y += WAVE * center * cos(WAVE_Y_TIME_SCALE * t + WAVE_PHASE + 2.0 * sx);
        shape_uv += DISTORTION * center * vec2<f32>(sin(t + 0.4 * shape_uv.y), cos(t + 0.6 * shape_uv.x)) * 0.5;

        // Swirl - always applied with fixed strength
        var uvRot = shape_uv - 0.5;
        let swirlAngle = SWIRL_MULT * SWIRL * radius;
        let ca = cos(-swirlAngle);
        let sa = sin(-swirlAngle);
        uvRot = vec2<f32>(uvRot.x * ca - uvRot.y * sa, uvRot.x * sa + uvRot.y * ca);
        uvRot += 0.5;

        // Hash for grain (computed once). Framebuffer coords: @builtin(position)
        // is bit-exact across rasterizers (interpolated varyings like localPos are not).
        let grainUV = input.position.xy;
        let h = hashToUnit(hashU32(bitcast<u32>(grainUV.x) ^ hashU32(bitcast<u32>(grainUV.y))));
        let grainOffset = GRAIN_JITTER * grain * (h - 0.5);

        // Color accumulation - branchless, fixed power (inverse distance)
        var color = vec3<f32>(0.0);
        var wsum = 0.0;
        let maxN = min(count, 10u);
        
        for (var i: u32 = 0u; i < maxN; i = i + 1u) {
          let pos = spotPos(i, t + POSITION_SEED) + grainOffset;
          let col = path.colorData[i].rgb;
          let d = uvRot - pos;
          let r2 = dot(d, d);
          // Fixed power=2 (halfPower=1) - simple inverse square falloff
          let w = 1.0 / (r2 + 1e-3);
          color += col * w;
          wsum += w;
        }

        // Normalize color (wsum always > 0 when count > 0)
        color = color / max(wsum, 1e-6);

        // Apply grain overlay
        let grainColor = vec3<f32>(h, fract(h * 127.1), fract(h * 311.7)) - 0.5;
        color = color + grainColor * grain * GRAIN_OVERLAY;

        // Straight-alpha output; the framework epilogue applies fillColor.a + clip/mask coverage.
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
