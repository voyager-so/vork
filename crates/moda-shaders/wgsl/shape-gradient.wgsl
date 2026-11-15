// #include "working-color.wgsl"
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


// #include "gradient-space.wgsl"

// Main shader code

      @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>;

      struct GradientShapeData {
        model: mat4x4<f32>,
        color: vec4<f32>,         // r, g, b, opacity
        dimensions: vec4<f32>,    // width, height, depth, shapeType
        cornerRadii: vec4<f32>,   // tl,tr,br,bl (rect) or radiusX,radiusY,0,0 (ellipse)
        gradientStart: vec2<f32>,
        gradientEnd: vec2<f32>,
        gradientMeta: vec4<f32>,  // fillType, numStops, gradientRotation, flags (space + 8 × dither: include/gradient-space.wgsl)
        gradientStops: array<vec4<f32>, 8>,
        pageClipBounds: vec4<f32>,
        innerShadowColor: vec4<f32>,  // rgb + combined alpha (0 => no inner shadow)
        innerShadowParams: vec4<f32>, // offsetX, offsetY, blur, spread (shape-local px)
        strokeColor: vec4<f32>,       // rgb (straight) + combined alpha (0 => no stroke)
        strokeParams: vec4<f32>,      // width, alignCode (0=center,1=inside,2=outside), 0, 0
      }

      @group(1) @binding(0) var<storage, read> shapes: array<GradientShapeData>;

      struct VSOut {
        @builtin(position) position: vec4<f32>,
        @location(0) worldPos: vec2<f32>,
        @location(1) localPos: vec2<f32>,
        @location(2) @interpolate(flat) clip: vec4<f32>,
        @location(3) @interpolate(flat) size: vec2<f32>,
        @location(4) @interpolate(flat) radii: vec4<f32>,
        @location(5) @interpolate(flat) shapeType: f32,
        @location(6) @interpolate(flat) instanceId: u32,
      }

      fn sdRoundedBox(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
        let r = select(
          select(radii.w, radii.z, p.x > 0.0),
          select(radii.x, radii.y, p.x > 0.0),
          p.y < 0.0
        );
        let q = abs(p) - size + r;
        return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
      }

      // Unit gradient of sdRoundedBox at p, in local units. The field is only piecewise smooth: inside the box it is
      // max(q.x, q.y), whose gradient flips between the two axes along each corner's diagonal. fwidth(dist) over a
      // 2x2 pixel quad that straddles that diagonal adds BOTH axes' differences, twice a pixel at a square corner, so
      // a corner pixel whose quad reached outside the shape came out 3/4 covered (alpha 191).
      fn sdRoundedBoxGradient(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> vec2<f32> {
        let r = select(
          select(radii.w, radii.z, p.x > 0.0),
          select(radii.x, radii.y, p.x > 0.0),
          p.y < 0.0
        );
        let q = abs(p) - size + r;
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
        if (q.x > 0.0 || q.y > 0.0) {
          return s * normalize(max(q, vec2<f32>(0.0)));
        }
        return s * select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), q.x > q.y);
      }

      // One device pixel in the SDF's units, as fwidth(dist) measures it where the field is smooth
      // (|d·dx| + |d·dy| for the gradient d), but from the analytic gradient for a rectangle, so the
      // flip at its corners does not double it. Ellipses keep fwidth(dist) (their field is smooth).
      fn sdfEdgeWidth(fw: f32, p: vec2<f32>, dpx: vec2<f32>, dpy: vec2<f32>, size: vec2<f32>, radii: vec4<f32>, shapeType: f32) -> f32 {
        let g = sdRoundedBoxGradient(p, size, radii);
        return select(fw, abs(dot(g, dpx)) + abs(dot(g, dpy)), shapeType < 0.5);
      }

      fn sdEllipse(p: vec2<f32>, r: vec2<f32>) -> f32 {
        if (r.x < 0.0001 || r.y < 0.0001) { return 1.0; }
        let k0 = length(p / r);
        let k1 = length(p / (r * r));
        if (k1 < 0.0001) { return -min(r.x, r.y); }
        return k0 * (k0 - 1.0) / k1;
      }

      fn sdShape(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>, shapeType: f32) -> f32 {
        if (shapeType < 0.5) {
          return sdRoundedBox(p, size, radii);
        } else {
          return sdEllipse(p, vec2<f32>(radii.x, radii.y));
        }
      }

      // How far the stroke band reaches OUTSIDE the shape geometry, so
      // the vertex shader can grow the quad to contain it. center => width/2,
      // outside => width, inside => 0; 0 when there is no stroke (strokeColor.a
      // == 0), leaving the fill quad unchanged.
      fn strokeOutsetExtent(strokeColor: vec4<f32>, strokeParams: vec4<f32>) -> f32 {
        if (strokeColor.a <= 0.0 || strokeParams.x <= 0.0) {
          return 0.0;
        }
        let alignCode = strokeParams.y;
        if (alignCode > 1.5) {
          return strokeParams.x;        // outside
        } else if (alignCode > 0.5) {
          return 0.0;                   // inside
        }
        return strokeParams.x * 0.5;    // center
      }

      // Signed distance to the stroke RING (negative inside the band).
      // d = signed distance to the shape; h = strokeWidth/2. center => abs(d) - h,
      // inside => abs(d + h) - h, outside => abs(d - h) - h. For ellipses the band
      // rides sdEllipse's near-boundary approximation (same field the fill/inner
      // shadow use), so a wide stroke on a very eccentric ellipse can vary slightly
      // in width — acceptable per the golden gate.
      fn strokeRingDistance(d: f32, strokeParams: vec4<f32>) -> f32 {
        let h = strokeParams.x * 0.5;
        let alignCode = strokeParams.y;
        var alignOffset = 0.0;          // center
        if (alignCode > 1.5) {
          alignOffset = -h;             // outside
        } else if (alignCode > 0.5) {
          alignOffset = h;              // inside
        }
        return abs(d + alignOffset) - h;
      }

// Box-filter edge coverage in [0,1]: 1 at half a pixel inside the silhouette,
// 0.5 on the edge, 0 at half a pixel outside. edgeWidth is fwidth(dist) —
// one device pixel expressed in the SDF's units.
fn sdfCoverage(dist: f32, edgeWidth: f32) -> f32 {
  return clamp(0.5 - dist / max(edgeWidth, 1e-6), 0.0, 1.0);
}


      
// Inner (inset) shadow, CSS / Figma semantics: the complement of the silhouette shrunk
// by the spread and moved by the offset, blurred with a Gaussian of sigma = blur / 2, over
// the fill. The blur is evaluated exactly per silhouette row (a difference of erfs) and
// summed over the rows within 4 sigma (Evan Wallace, "Fast rounded rectangle shadows") —
// the same evaluation as the path-shadow lane, so both lanes agree.
const INSET_SHADOW_ROWS: i32 = 12;

fn insetErf2(x: vec2<f32>) -> vec2<f32> {
  let s = sign(x);
  let a = abs(x);
  var y = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
  y = y * y;
  return s - s / (y * y);
}

// The shape (half size `size`, radii tl/tr/br/bl — or radii.xy = (rx, ry) for an ellipse)
// convolved with a Gaussian of `sigma`, at `p`.
fn insetGaussianShape(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>, isEllipse: bool, sigma: f32) -> f32 {
  let halfW = select(size.x, radii.x, isEllipse);
  let halfH = select(size.y, radii.y, isEllipse);
  if (halfW <= 0.0 || halfH <= 0.0) {
    return 0.0;
  }
  let lo = clamp(-4.0 * sigma, p.y - halfH, p.y + halfH);
  let hi = clamp(4.0 * sigma, p.y - halfH, p.y + halfH);
  let stepK = (hi - lo) / f32(INSET_SHADOW_ROWS);
  let invS = INVERSE_SQRT_2 / sigma;
  let norm = 1.0 / (2.5066282746 * sigma);
  var total = 0.0;
  var k = lo + 0.5 * stepK;
  for (var i = 0; i < INSET_SHADOW_ROWS; i++) {
    let rowY = p.y - k;
    var e: vec2<f32>;
    if (isEllipse) {
      let t = rowY / halfH;
      let w = halfW * sqrt(max(0.0, 1.0 - t * t));
      e = vec2<f32>(w, w);
    } else {
      let rl = select(radii.w, radii.x, rowY < 0.0);
      let rr = select(radii.z, radii.y, rowY < 0.0);
      let dl = max(abs(rowY) - (size.y - rl), 0.0);
      let dr = max(abs(rowY) - (size.y - rr), 0.0);
      e = vec2<f32>(size.x - rl + sqrt(max(0.0, rl * rl - dl * dl)), size.x - rr + sqrt(max(0.0, rr * rr - dr * dr)));
    }
    let span = 0.5 * insetErf2(vec2<f32>(p.x + e.x, p.x - e.y) * invS);
    total += (span.x - span.y) * exp(-0.5 * k * k / (sigma * sigma)) * norm * stepK;
    k += stepK;
  }
  return clamp(total, 0.0, 1.0);
}

// Composite the inner shadow over a straight-alpha fill rgb and return the new rgb.
// innerColor: rgb = shadow color, a = innerShadowOpacity (NOT multiplied by node
//             opacity — node opacity is applied once by the host via the body/final
//             alpha). When a <= 0 there is no inner shadow and the fill is returned unchanged.
// innerParams: (offsetX, offsetY, blur, spread) in shape-local pixels.
// pixelLocal: one device pixel in shape-local units (fwidth of the SDF), which widens sigma
//             so a hard (blur 0) inner shadow keeps an anti-aliased edge.
fn applyInnerShadow(
  fillRgb: vec3<f32>,
  innerColor: vec4<f32>,
  innerParams: vec4<f32>,
  localPos: vec2<f32>,
  size: vec2<f32>,
  radii: vec4<f32>,
  shapeType: f32,
  pixelLocal: f32
) -> vec3<f32> {
  if (innerColor.a <= 0.0) {
    return fillRgb;
  }
  let isEllipse = shapeType >= 0.5;
  let spread = innerParams.w;
  // The caster: the silhouette shrunk by the spread (a rounded corner r → max(0, r − spread),
  // a square corner stays square; ellipse radii − spread).
  var casterSize = size - vec2<f32>(spread);
  var casterRadii: vec4<f32>;
  if (isEllipse) {
    casterRadii = vec4<f32>(radii.xy - vec2<f32>(spread), 0.0, 0.0);
  } else {
    let shrunk = select(vec4<f32>(0.0), max(radii - vec4<f32>(spread), vec4<f32>(0.0)), radii > vec4<f32>(0.0));
    casterRadii = min(shrunk, vec4<f32>(max(min(casterSize.x, casterSize.y), 0.0)));
  }
  let sigma = sqrt(0.25 * innerParams.z * innerParams.z + 0.09 * pixelLocal * pixelLocal) + 1e-6;
  let caster = insetGaussianShape(localPos - innerParams.xy, casterSize, casterRadii, isEllipse, sigma);
  let strength = clamp(innerColor.a * (1.0 - caster), 0.0, 1.0);
  return mix(fillRgb, innerColor.rgb, strength);
}


      
    // Interpolate gradient color at position t
    fn interpolateGradient(t: f32, numStops: u32, gradientStops: array<vec4<f32>, 8>, flags: f32) -> vec4<f32> {
      let t_clamped = gsRepeat(t, gradientStops[0].x, gradientStops[max(1u, min(numStops, 8u)) - 1u].x, flags);
      
      // Safety: ensure numStops is valid
      let safeNumStops = min(numStops, 8u);
      
      // Handle edge cases
      if (safeNumStops == 0u) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
      }
      if (safeNumStops == 1u) {
        // Single stop: unpack blue and alpha from w component
        let stop = gradientStops[0];
        let blueAlphaPacked = stop.w;
        let blue = clamp(floor(blueAlphaPacked) / 255.0, 0.0, 1.0);
        let alpha = clamp(fract(blueAlphaPacked) / 0.99999, 0.0, 1.0);
        // Clamp color components to valid range
        return vec4<f32>(
          clamp(stop.y, 0.0, 1.0),
          clamp(stop.z, 0.0, 1.0),
          blue,
          alpha
        );
      }
      
      // Get first and last stop colors for extension
      let firstStop = gradientStops[0];
      let firstBlueAlpha = firstStop.w;
      let firstBlue = clamp(floor(firstBlueAlpha) / 255.0, 0.0, 1.0);
      let firstAlpha = clamp(fract(firstBlueAlpha) / 0.99999, 0.0, 1.0);
      let firstColor = vec4<f32>(
        clamp(firstStop.y, 0.0, 1.0),
        clamp(firstStop.z, 0.0, 1.0),
        firstBlue,
        firstAlpha
      );
      
      let lastStop = gradientStops[safeNumStops - 1u];
      let lastBlueAlpha = lastStop.w;
      let lastBlue = clamp(floor(lastBlueAlpha) / 255.0, 0.0, 1.0);
      let lastAlpha = clamp(fract(lastBlueAlpha) / 0.99999, 0.0, 1.0);
      let lastColor = vec4<f32>(
        clamp(lastStop.y, 0.0, 1.0),
        clamp(lastStop.z, 0.0, 1.0),
        lastBlue,
        lastAlpha
      );
      
      // If t is before the first stop, use the first stop's color
      if (t_clamped <= firstStop.x) {
        return firstColor;
      }
      
      // If t is after the last stop, use the last stop's color
      if (t_clamped >= lastStop.x) {
        return lastColor;
      }
      
      // Find the two stops to interpolate between
      for (var i = 1u; i < safeNumStops; i++) {
        let currStop = gradientStops[i];
        let prevStop = gradientStops[i - 1u];
        
        if (t_clamped <= currStop.x) {
          // Interpolate between prevStop and currStop
          let stopDiff = currStop.x - prevStop.x;
          var localT = 0.0;
          
          // Use a larger threshold to avoid extreme interpolation values
          if (stopDiff > select(0.001, 0.0, (u32(flags + 0.5) & 16u) != 0u)) {
            localT = (t_clamped - prevStop.x) / stopDiff;
          } else {
            // If stops are too close, just use the color at t position
            localT = select(0.0, 1.0, t_clamped > (prevStop.x + currStop.x) * 0.5);
          }
          
          // Unpack blue and alpha from both stops
          let prevBlueAlpha = prevStop.w;
          let prevBlue = clamp(floor(prevBlueAlpha) / 255.0, 0.0, 1.0);
          let prevAlpha = clamp(fract(prevBlueAlpha) / 0.99999, 0.0, 1.0);
          
          let currBlueAlpha = currStop.w;
          let currBlue = clamp(floor(currBlueAlpha) / 255.0, 0.0, 1.0);
          let currAlpha = clamp(fract(currBlueAlpha) / 0.99999, 0.0, 1.0);
          
          // Interpolate all components, in the paint's space
          return gsMix(
            vec4<f32>(
              clamp(prevStop.y, 0.0, 1.0),
              clamp(prevStop.z, 0.0, 1.0),
              prevBlue,
              prevAlpha
            ),
            vec4<f32>(
              clamp(currStop.y, 0.0, 1.0),
              clamp(currStop.z, 0.0, 1.0),
              currBlue,
              currAlpha
            ),
            localT,
            flags
          );
        }
      }
      
      // Should never reach here, but return last color as fallback
      return lastColor;
    }
    
    // Calculate linear gradient t value
    fn calculateLinearGradientT(
      localPos: vec2<f32>,
      halfSize: vec2<f32>,
      gradStart: vec2<f32>,
      gradEnd: vec2<f32>,
      width: f32,
      height: f32
    ) -> f32 {
      
      // Convert gradient points from percentage (0-100) to local coordinates
      // Use 0,0 as origin (top-left)
      let gradStartLocal = (gradStart * 0.01) * vec2<f32>(width, height);
      let gradEndLocal = (gradEnd * 0.01) * vec2<f32>(width, height);
      
      // Calculate gradient vector
      let gradDir = gradEndLocal - gradStartLocal;
      let gradLengthSq = dot(gradDir, gradDir);
      
      var t = 0.0;
      // Use a larger epsilon to be extra safe
      if (gradLengthSq > 0.01) {
        // Project current position onto gradient line
        // Convert localPos from centered coordinates to 0,0 origin
        let posFromOrigin = localPos + halfSize;
        let projDot = dot(posFromOrigin - gradStartLocal, gradDir);
        t = projDot / gradLengthSq;
      }
      
      return t;
    }
    

    // Diamond gradient t value (E3.1 P6; Figma's GRADIENT_DIAMOND): placed like the
    // elliptical radial (center, radii, rotation), with the L1 distance in the
    // radii-normalised frame, so the stop ramp runs out in diamond-shaped rings.
    fn calculateDiamondGradientT(
      localPos: vec2<f32>,
      halfSize: vec2<f32>,
      gradCenter: vec2<f32>,
      gradRadiusX: f32,
      gradRadiusY: f32,
      gradRotation: f32,
      width: f32,
      height: f32
    ) -> f32 {
      let gradCenterLocal = (gradCenter * 0.01) * vec2<f32>(width, height);
      let relPos = localPos + halfSize - gradCenterLocal;
      let cosR = cos(-gradRotation);
      let sinR = sin(-gradRotation);
      let rotatedPos = vec2<f32>(
        relPos.x * cosR - relPos.y * sinR,
        relPos.x * sinR + relPos.y * cosR
      );
      let radiusXPixels = max(1.0, (gradRadiusX * 0.01) * width);
      let radiusYPixels = max(1.0, (gradRadiusY * 0.01) * height);
      return abs(rotatedPos.x / radiusXPixels) + abs(rotatedPos.y / radiusYPixels);
    }

    // Calculate elliptical radial gradient t value
    fn calculateEllipticalRadialGradientT(
      localPos: vec2<f32>,
      halfSize: vec2<f32>,
      gradCenter: vec2<f32>,
      gradRadiusX: f32,
      gradRadiusY: f32,
      gradRotation: f32,
      width: f32,
      height: f32
    ) -> f32 {
      
      // Convert gradient center from percentage to local coordinates
      let gradCenterLocal = (gradCenter * 0.01) * vec2<f32>(width, height);
      
      // Convert localPos from centered coordinates to 0,0 origin
      let posFromOrigin = localPos + halfSize;
      
      // Calculate position relative to gradient center
      let relPos = posFromOrigin - gradCenterLocal;
      
      // Apply rotation (negative to match standard rotation direction)
      let cosR = cos(-gradRotation);
      let sinR = sin(-gradRotation);
      let rotatedPos = vec2<f32>(
        relPos.x * cosR - relPos.y * sinR,
        relPos.x * sinR + relPos.y * cosR
      );
      
      // Convert radii from percentage to pixels
      let radiusXPixels = max(1.0, (gradRadiusX * 0.01) * width);
      let radiusYPixels = max(1.0, (gradRadiusY * 0.01) * height);
      
      // Calculate normalized distance in ellipse space
      let normalizedPos = vec2<f32>(
        rotatedPos.x / radiusXPixels,
        rotatedPos.y / radiusYPixels
      );
      
      // Distance from center in ellipse space
      let t = length(normalizedPos);

      return t;
    }

    // Calculate conic (angular) gradient t value.
    // Sweeps the stop ramp around the center: angle 0 points up (12 o'clock) and
    // increases clockwise, matching CSS conic-gradient(from <angle>). The result is
    // wrapped to [0,1) via fract() so the 0deg/360deg seam is continuous in t — any
    // hard color line there is inherent to the user's first/last stops (as in CSS),
    // not the math.
    fn calculateConicGradientT(
      localPos: vec2<f32>,
      halfSize: vec2<f32>,
      gradCenter: vec2<f32>,
      startAngle: f32,
      width: f32,
      height: f32
    ) -> f32 {
      // Convert gradient center from percentage to local coordinates
      let gradCenterLocal = (gradCenter * 0.01) * vec2<f32>(width, height);

      // Convert localPos from centered coordinates to 0,0 origin
      let posFromOrigin = localPos + halfSize;

      // Position relative to gradient center
      let rel = posFromOrigin - gradCenterLocal;

      // atan2(rel.x, -rel.y): 0 at straight up, increasing clockwise in screen
      // space (y points down). Subtract the start angle so the ramp begins there.
      let TWO_PI = 6.28318530717958647692;
      let angle = atan2(rel.x, -rel.y) - startAngle;

      // Wrap to [0,1). fract handles negative angles, so the seam stays seamless.
      return fract(angle / TWO_PI);
    }

    // Apply gradient based on fill type
    fn applyGradient(
      localPos: vec2<f32>,
      halfSize: vec2<f32>,
      width: f32,
      height: f32,
      fillType: f32,
      gradientStart: vec2<f32>,
      gradientEnd: vec2<f32>,
      gradientRotation: f32,
      numStops: u32,
      gradientStops: array<vec4<f32>, 8>,
      flags: f32
    ) -> vec4<f32> {
      // Safety: ensure width and height are positive
      let safeWidth = max(1.0, width);
      let safeHeight = max(1.0, height);
      
      // Safety: clamp gradient points to ±100,000 % of the box (gradient.rs MAX_POINT_PERCENTAGE): they may sit
      // far outside the shape (one gradient spread over several layers)
      let safeGradientStart = clamp(gradientStart, vec2<f32>(-100000.0, -100000.0), vec2<f32>(100000.0, 100000.0));
      let safeGradientEnd = clamp(gradientEnd, vec2<f32>(-100000.0, -100000.0), vec2<f32>(100000.0, 100000.0));
      
      var t = 0.0;
      
      if (fillType == 1.0) {
        // Linear gradient
        t = calculateLinearGradientT(localPos, halfSize, safeGradientStart, safeGradientEnd, safeWidth, safeHeight);
      } else if (fillType == 2.0) {
        // Radial gradient
        // For radial gradients, gradientStart is the center point
        // gradientEnd contains radiusX and radiusY as percentages
        let gradCenter = safeGradientStart;
        let gradRadiusX = max(1.0, min(100000.0, gradientEnd.x)); // A positive radius up to 100,000 %
        let gradRadiusY = max(1.0, min(100000.0, gradientEnd.y)); // A positive radius up to 100,000 %
        t = calculateEllipticalRadialGradientT(localPos, halfSize, gradCenter, gradRadiusX, gradRadiusY, gradientRotation, safeWidth, safeHeight);
      } else if (fillType == 4.0) {
        // Diamond gradient (placed like the radial).
        let dRadiusX = max(1.0, min(100000.0, gradientEnd.x));
        let dRadiusY = max(1.0, min(100000.0, gradientEnd.y));
        t = calculateDiamondGradientT(localPos, halfSize, safeGradientStart, dRadiusX, dRadiusY, gradientRotation, safeWidth, safeHeight);
      } else if (fillType == 3.0) {
        // Conic (angular) gradient.
        // gradientStart is the center point; gradientRotation carries the start angle in radians.
        t = calculateConicGradientT(localPos, halfSize, safeGradientStart, gradientRotation, safeWidth, safeHeight);
      }

      return interpolateGradient(t, numStops, gradientStops, flags);
    }
  

      @vertex
      fn vertexMain(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
        var out: VSOut;
        let s = shapes[iid];

        var pos: vec2<f32>;
        switch(vid) {
          case 0u: { pos = vec2<f32>(-1.0, -1.0); }
          case 1u: { pos = vec2<f32>( 1.0, -1.0); }
          case 2u: { pos = vec2<f32>(-1.0,  1.0); }
          case 3u: { pos = vec2<f32>(-1.0,  1.0); }
          case 4u: { pos = vec2<f32>( 1.0, -1.0); }
          default: { pos = vec2<f32>( 1.0,  1.0); }
        }

        let halfSize = 0.5 * s.dimensions.xy;
        // Grow the quad outward by the stroke's outset so an outer/
        // center stroke band is not clipped (inside-aligned and unstroked bodies
        // grow by 0, leaving the fill quad unchanged).
        let strokeExtend = strokeOutsetExtent(s.strokeColor, s.strokeParams);
        let grownHalf = halfSize + vec2<f32>(strokeExtend);
        let anchored = pos * grownHalf + halfSize;
        let localPos = anchored - halfSize;

        let local = vec4<f32>(anchored, 0.0, 1.0);
        let world = s.model * local;
        var clip = frameUniforms * world;
        clip.z = s.dimensions.z;
        out.position = clip;
        out.worldPos = world.xy;
        out.clip = s.pageClipBounds;
        out.localPos = localPos;
        out.size = halfSize;
        out.radii = s.cornerRadii;
        out.shapeType = s.dimensions.w;
        out.instanceId = iid;

        return out;
      }

      fn sampleGradientColor(inp: VSOut) -> vec4<f32> {
        let s = shapes[inp.instanceId];

        let bounds = inp.clip;
        if (inp.worldPos.x < bounds.x ||
            inp.worldPos.x > bounds.x + bounds.z ||
            inp.worldPos.y < bounds.y ||
            inp.worldPos.y > bounds.y + bounds.w) {
          discard;
        }

        let dist = sdShape(inp.localPos, inp.size, inp.radii, inp.shapeType);
        // fwidth() must run in uniform control flow; derive the edge width and
        // both coverages before the stroke composite (no data-dependent branch).
        let aa = sdfEdgeWidth(fwidth(dist), inp.localPos, dpdx(inp.localPos), dpdy(inp.localPos), inp.size, inp.radii, inp.shapeType);
        let sdfAlpha = sdfCoverage(dist, aa);

        let width = s.dimensions.x;
        let height = s.dimensions.y;
        let halfSize = 0.5 * vec2<f32>(width, height);
        let fillType = s.gradientMeta.x;
        let numStops = u32(s.gradientMeta.y);
        let gradRotation = s.gradientMeta.z;

        let gradColorRaw = applyGradient(
          inp.localPos,
          halfSize,
          width,
          height,
          fillType,
          s.gradientStart,
          s.gradientEnd,
          gradRotation,
          numStops,
          s.gradientStops,
          s.gradientMeta.w
        );
        let gradColor = gsDither(gradColorRaw, inp.position.xy, s.gradientMeta.w);

        // Composite the inner (inset) shadow over the gradient fill (rgb only).
        // No-op when the inner shadow color alpha is 0.
        let fillRgb = applyInnerShadow(
          workingRgb(gradColor.rgb), vec4<f32>(workingRgb(s.innerShadowColor.rgb), s.innerShadowColor.a), s.innerShadowParams,
          inp.localPos, inp.size, inp.radii, inp.shapeType, aa
        );
        // Straight fill alpha = gradient stop alpha × node opacity × SDF coverage.
        let fillAlpha = gradColor.a * s.color.a * sdfAlpha;

        // Solid stroke ring composited OVER the gradient fill. The
        // strokeAlpha == 0 select returns the fill rgb directly (no division), so
        // an unstroked gradient body is byte-identical to the pre-stroke path.
        let strokeDist = strokeRingDistance(dist, s.strokeParams);
        let strokeCoverage = sdfCoverage(strokeDist, aa);
        let strokeAlpha = s.strokeColor.a * strokeCoverage;
        let outAlpha = strokeAlpha + fillAlpha * (1.0 - strokeAlpha);

        if (outAlpha <= 0.001) {
          discard;
        }

        // Straight "stroke over fill"; the premul/A2C entries format from this.
        let outRgb = select(
          (workingRgb(s.strokeColor.rgb) * strokeAlpha + fillRgb * fillAlpha * (1.0 - strokeAlpha)) / outAlpha,
          fillRgb,
          strokeAlpha <= 0.0
        );
        return vec4<f32>(outRgb, outAlpha);
      }

      @fragment
      fn fragmentMainPremul(inp: VSOut) -> @location(0) vec4<f32> {
        return premultiplyAlpha(sampleGradientColor(inp));
      }

      @fragment
      fn fragmentMainA2C(inp: VSOut) -> @location(0) vec4<f32> {
        let color = sampleGradientColor(inp);
        return vec4<f32>(color.rgb, color.a);
      }
