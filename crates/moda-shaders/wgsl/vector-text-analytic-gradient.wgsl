// #include "working-color.wgsl"

struct DrawUniforms {
  model: mat4x4<f32>,
  color: vec4<f32>,
  params: vec4<f32>,   // x=zDepth, y=opacity, z=blur(px), w=stroke(px) — z/w solid analytic only
  pageClip: vec4<f32>, // x, y, w, h in world space
  // Node distort warp: rows of the local-px 3×3 homography. distort0.w is the
  // enabled flag (1 = warp); disabled draws carry identity rows.
  distort0: vec4<f32>,
  distort1: vec4<f32>,
  distort2: vec4<f32>,
};

// Warp a local-space position through the node's distort homography, returning the
// HOMOGENEOUS result (x·w, y·w, w). Callers keep w through the model/frame transforms
// (both are affine in xy with last row [0,0,0,1]) and let the hardware perspective
// divide + perspective-correct varying interpolation do the rest — this is what makes
// glyph-space (em) varyings exact per fragment across a warped quad. Validation only
// proves w > 0 inside the node's box; glyph geometry that overflows the box (bearings,
// descenders) could in principle cross the horizon (w ≤ 0) under an extreme warp and
// get hardware-clipped — accepted: plausible matrices keep the horizon far outside.
fn vectorWarpLocal(local: vec2<f32>) -> vec3<f32> {
  if (draw.distort0.w < 0.5) {
    return vec3<f32>(local, 1.0);
  }
  let p = vec3<f32>(local, 1.0);
  return vec3<f32>(dot(draw.distort0.xyz, p), dot(draw.distort1.xyz, p), dot(draw.distort2.xyz, p));
}
@group(0) @binding(0) var<uniform> frame: mat4x4<f32>;
@group(0) @binding(1) var<uniform> draw: DrawUniforms;
// Quadratic outline segments: 3 control points per curve, em units (y-down, pen origin).
@group(0) @binding(2) var<storage, read> curvePoints: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> gradient: GradientUniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  // True world position: affine in screen space, so linear (non-perspective)
  // interpolation is exact on warped quads; em/local are affine in LOCAL
  // space and need the default perspective-correct interpolation.
  @location(0) @interpolate(linear) world: vec2<f32>,
  @location(1) em: vec2<f32>,
  @location(2) @interpolate(flat) curveRange: vec2<u32>, // firstCurve, curveCount
  @location(3) local: vec2<f32>,
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

@vertex fn vs(
  @location(0) corner: vec2<f32>,      // unit quad corner (0..1)
  @location(1) pen: vec2<f32>,         // local-space pen position (x, baselineY)
  @location(2) fontSizePx: f32,
  @location(3) emBox: vec4<f32>,       // padded glyph bounds in em units
  @location(4) curveRange: vec2<u32>,
  @location(5) glyphLin: vec4<f32>,    // KineticText per-glyph 2×2 [a,c,b,d]
  @location(6) glyphOff: vec2<f32>,    // KineticText per-glyph px offset
  @location(7) glyphOpacity: f32,      // KineticText per-glyph opacity
) -> VSOut {
  let em = mix(emBox.xy, emBox.zw, corner);
  // KineticText: per-glyph transform about the glyph em-box
  // center; identity instances fast-path to the original math (byte-exact static).
  let local = kineticGlyphLocal(pen, em, emBox, fontSizePx, glyphLin, glyphOff);
  let h = vectorWarpLocal(local);
  let world = draw.model * vec4<f32>(h.xy, 0.0, h.z);
  var o: VSOut;
  o.pos = frame * world;
  o.pos.z = draw.params.x * h.z; // hardware divide restores params.x
  o.world = world.xy / h.z;
  o.em = em;
  o.curveRange = curveRange;
  o.local = local;
  o.glyphOpacity = glyphOpacity;
  return o;
}

// #include "vector-text-coverage.wgsl"

// #include "gradient-space.wgsl"

struct GradientUniforms {
  textBounds: vec4<f32>,     // x, y, width, height in local space
  gradientParams: vec4<f32>, // linear: startX, startY, endX, endY; radial: centerX, centerY, radiusX, radiusY (0-100 %)
  gradientConfig: vec4<f32>, // x=numStops, y=fillType (1=linear, 2=radial), z=rotation(rad), w=flags (include/gradient-space.wgsl)
  gradientStops: array<vec4<f32>, 8>, // [offset, r, g, blue_alpha_packed]
};

fn vectorGradientStopColor(stop: vec4<f32>) -> vec4<f32> {
  let blue = floor(stop.w) / 255.0;
  let alpha = fract(stop.w) / 0.99999;
  return vec4<f32>(stop.y, stop.z, blue, alpha);
}

fn vectorInterpolateGradient(position: f32, numStops: u32) -> vec4<f32> {
  let safeNumStops = min(numStops, 8u);
  if (safeNumStops == 0u) {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
  }
  let firstStop = gradient.gradientStops[0];
  let t = gsRepeat(position, firstStop.x, gradient.gradientStops[safeNumStops - 1u].x, gradient.gradientConfig.w);
  if (safeNumStops == 1u || t <= firstStop.x) {
    return vectorGradientStopColor(firstStop);
  }
  let lastStop = gradient.gradientStops[safeNumStops - 1u];
  if (t >= lastStop.x) {
    return vectorGradientStopColor(lastStop);
  }
  for (var i = 1u; i < safeNumStops; i = i + 1u) {
    let currStop = gradient.gradientStops[i];
    let prevStop = gradient.gradientStops[i - 1u];
    if (t <= currStop.x) {
      let stopDiff = currStop.x - prevStop.x;
      var localT = 0.0;
      if (stopDiff > select(0.001, 0.0, (u32(gradient.gradientConfig.w + 0.5) & 16u) != 0u)) {
        localT = (t - prevStop.x) / stopDiff;
      }
      return gsMix(vectorGradientStopColor(prevStop), vectorGradientStopColor(currStop), localT, gradient.gradientConfig.w);
    }
  }
  return vectorGradientStopColor(lastStop);
}

fn vectorLinearGradientT(localPos: vec2<f32>) -> f32 {
  let bounds = gradient.textBounds;
  let gradStart = gradient.gradientParams.xy * 0.01 * bounds.zw;
  let gradEnd = gradient.gradientParams.zw * 0.01 * bounds.zw;
  let gradDir = gradEnd - gradStart;
  let gradLengthSq = dot(gradDir, gradDir);
  if (gradLengthSq <= 0.01) { return 0.0; }
  let posFromOrigin = localPos - bounds.xy;
  return dot(posFromOrigin - gradStart, gradDir) / gradLengthSq;
}

fn vectorRadialGradientT(localPos: vec2<f32>) -> f32 {
  let bounds = gradient.textBounds;
  let rotation = gradient.gradientConfig.z;
  let gradCenter = gradient.gradientParams.xy * 0.01 * bounds.zw;
  let radiusX = max(1.0, gradient.gradientParams.z * 0.01 * bounds.z);
  let radiusY = max(1.0, gradient.gradientParams.w * 0.01 * bounds.w);
  let relPos = (localPos - bounds.xy) - gradCenter;
  let cosR = cos(-rotation);
  let sinR = sin(-rotation);
  let rotatedPos = vec2<f32>(relPos.x * cosR - relPos.y * sinR, relPos.x * sinR + relPos.y * cosR);
  return length(vec2<f32>(rotatedPos.x / radiusX, rotatedPos.y / radiusY));
}

fn vectorGradientFillColor(localPos: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
  let fillType = u32(gradient.gradientConfig.y);
  let numStops = u32(gradient.gradientConfig.x);
  if (fillType == 1u && numStops > 0u) {
    return vectorInterpolateGradient(vectorLinearGradientT(localPos), numStops);
  }
  if (fillType == 2u && numStops > 0u) {
    return vectorInterpolateGradient(vectorRadialGradientT(localPos), numStops);
  }
  return fallback;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  // Pixel footprint in em units (derivatives must be computed before any discard).
  let footprint = max(fwidth(in.em), vec2<f32>(1e-6));
  let invDiameter = 1.0 / footprint;

  let clip = draw.pageClip;
  if (in.world.x < clip.x || in.world.x > clip.x + clip.z ||
      in.world.y < clip.y || in.world.y > clip.y + clip.w) {
    discard;
  }

  let coverage = vectorGlyphCoverage(in.curveRange.x, in.curveRange.y, in.em, invDiameter);
  if (coverage < 0.004) {
    discard;
  }
  let fillColor = gsDither(vectorGradientFillColor(in.local, draw.color), in.pos.xy, gradient.gradientConfig.w);
  // KineticText: per-glyph opacity (1.0 for static text). The gradient
  // is node-space — the glyph slides across the fixed fill via its transformed
  // local position, sampling different colors — so only alpha gains the glyph term.
  let alpha = fillColor.a * draw.params.y * in.glyphOpacity * coverage;
  return vec4<f32>(workingRgb(fillColor.rgb), alpha);
}
