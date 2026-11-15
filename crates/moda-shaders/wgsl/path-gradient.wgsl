// #include "working-color.wgsl"
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
    return vec4<f32>(workingRgb(color.rgb) * color.a, color.a);
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

        // Frame-level uniforms
        @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>; // viewProjection
        // Mask sampling (optional)
        struct MaskUniforms {
          maskActive: u32,
          maskOrigin: vec2<f32>,
          maskInvSize: vec2<f32>,
        }
        @group(2) @binding(0) var maskSampler: sampler;
        @group(2) @binding(1) var maskTexture: texture_2d<f32>;
        @group(2) @binding(2) var<uniform> maskUniforms: MaskUniforms;
        
        // Storage buffer for path gradient instances
        struct PathGradientData {
          model: mat4x4<f32>,      // 16 floats
          color: vec4<f32>,        // 4 floats (base color for gradient)
          pathInfo: vec4<f32>,     // vertexOffset, vertexCount, depth, fillType - 4 floats
          strokeColor: vec4<f32>,  // 4 floats (stroke color)
          gradientStart: vec2<f32>, // Linear gradient start point (percentage 0-100) OR radial gradient center
          gradientEnd: vec2<f32>,   // Linear gradient end point (percentage 0-100) OR radial gradient radiusX, radiusY
          bounds: vec4<f32>,       // minX, minY, width, height - 4 floats
          numStops: u32,
          strokeWidth: f32,
          gradientRotation: f32,   // Rotation angle in radians for elliptical gradients
          gradientFlags: f32,      // Fill interpolation space + 8 × dither (include/gradient-space.wgsl)
          gradientStops: array<vec4<f32>, 8>, // [offset, r, g, b_alpha_packed] for each stop
          pageClipBounds: vec4<f32>, // x, y, width, height - 4 floats
          // Stroke gradient paint set (mirrors the fill gradient fields). When
          // strokeGradientMeta.x == 0 the stroke is solid and uses strokeColor.
          strokeGradientStart: vec2<f32>, // Linear stroke gradient start OR radial center
          strokeGradientEnd: vec2<f32>,   // Linear stroke gradient end OR radial radiusX, radiusY
          strokeGradientMeta: vec4<f32>,  // x=paintType(0=solid,1=linear,2=radial), y=numStops, z=rotation, w=mode (0 within, 1 along, 2 across)
          strokeGradientStops: array<vec4<f32>, 8>, // [offset, r, g, b_alpha_packed] for each stop
          // Freeform gradient fill (fillType 5, E8.18): per point [x%, y%, spread%, link]
          // (link 0 = lone point, 1 = joined to the next point, 2 = a line's last point)
          // and its straight RGBA. numStops counts the points.
          freeformPoints: array<vec4<f32>, 16>,
          freeformColors: array<vec4<f32>, 16>,
          // Each point's colour slope per box px: along x at 2i, along y at 2i + 1.
          freeformSlopes: array<vec4<f32>, 32>,
          // Each point's slope reach (box px), point i at [i / 4][i % 4].
          freeformReach: array<vec4<f32>, 4>,
        }
        
        @group(1) @binding(0) var<storage, read> paths: array<PathGradientData>;
        
        // Stroke depth offset constant
        const STROKE_DEPTH_OFFSET: f32 = -0.000001490116119384765625;
        
        struct VertexInput {
          @location(0) position: vec2<f32>,
          @location(1) vertexType: f32, // 0 = fill, 1 = stroke
          // E8.18: a stroke vertex's (along, across) coordinates (tess::stroke_param); fill vertices carry zeros.
          @location(2) strokeUV: vec2<f32>,
        }
        
        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) @interpolate(flat) instanceId: u32,
          @location(1) localPos: vec2<f32>, // Local position for gradient calculation
          @location(2) @interpolate(flat) isStroke: f32,
          @location(3) worldPos: vec2<f32>, // World position for gradient calculation and clipping
          @location(4) @interpolate(flat) clip: vec4<f32>,
          @location(5) strokeUV: vec2<f32>,
        }
        
        @vertex
        fn vertexMain(
          vertex: VertexInput,
          @builtin(instance_index) instanceIndex: u32
        ) -> VertexOutput {
          var output: VertexOutput;
          let path = paths[instanceIndex];
          
          // Use the vertex position from the buffer
          let localPos = vertex.position;
          
          // Transform to world space
          let worldPos = path.model * vec4<f32>(localPos.x, localPos.y, 0.0, 1.0);
          output.position = frameUniforms * worldPos;
          output.worldPos = worldPos.xy;
          
          // Use pre-calculated depth value
          let depth = path.pathInfo.z;
          // For stroke vertices, add a small offset to render on top of fill
          let depthOffset = select(0.0, STROKE_DEPTH_OFFSET, vertex.vertexType > 0.5);
          output.position.z = depth + depthOffset;
          
          output.instanceId = instanceIndex;
          output.isStroke = vertex.vertexType;
          output.localPos = localPos;
          output.clip = path.pageClipBounds;
          output.strokeUV = vertex.strokeUV;
          
          return output;
        }
        
        
    // SIMPLIFIED GRADIENT TEST - Just 2 colors, no loop
    fn testLinearGradient(
      localPos: vec2<f32>,
      halfSize: vec2<f32>,
      width: f32,
      height: f32,
      gradientStart: vec2<f32>,
      gradientEnd: vec2<f32>,
      numStops: u32,
      gradientStops: array<vec4<f32>, 8>
    ) -> vec4<f32> {
      // Convert gradient points from percentage (0-100) to local coordinates
      let gradStart = (gradientStart * 0.01) * vec2<f32>(width, height);
      let gradEnd = (gradientEnd * 0.01) * vec2<f32>(width, height);
      
      // Calculate gradient vector
      let gradDir = gradEnd - gradStart;
      let gradLengthSq = dot(gradDir, gradDir);
      
      var t = 0.0;
      if (gradLengthSq > 0.0001) {
        // Project current position onto gradient line
        let posFromOrigin = localPos + halfSize;
        t = dot(posFromOrigin - gradStart, gradDir) / gradLengthSq;
      }
      
      // Clamp t to [0, 1] range
      t = clamp(t, 0.0, 1.0);
      
      var color: vec4<f32>;
      
      if (numStops >= 2u) {
        // Use first two stops from the array
        let stop1 = gradientStops[0];
        let stop2 = gradientStops[1];
        
        // Extract offsets and unpack colors from stops
        let offset1 = stop1.x;
        let offset2 = stop2.x;
        
        // Unpack blue and alpha from w component for both stops
        let blueAlpha1 = stop1.w;
        let blue1 = floor(blueAlpha1) / 255.0;
        let alpha1 = fract(blueAlpha1) / 0.99999;
        let color1 = vec4<f32>(stop1.y, stop1.z, blue1, alpha1);
        
        let blueAlpha2 = stop2.w;
        let blue2 = floor(blueAlpha2) / 255.0;
        let alpha2 = fract(blueAlpha2) / 0.99999;
        let color2 = vec4<f32>(stop2.y, stop2.z, blue2, alpha2);
        
        // Map t to the range between stop offsets
        var localT = 0.0;
        let offsetDiff = offset2 - offset1;
        if (abs(offsetDiff) > 0.0001) {
          localT = (t - offset1) / offsetDiff;
        }
        localT = clamp(localT, 0.0, 1.0);
        
        // Interpolate between the two stops
        color = mix(color1, color2, localT);
      } else if (numStops == 1u) {
        // Single color gradient
        let stop1 = gradientStops[0];
        let blueAlpha = stop1.w;
        let blue = floor(blueAlpha) / 255.0;
        let alpha = fract(blueAlpha) / 0.99999;
        color = vec4<f32>(stop1.y, stop1.z, blue, alpha);
      } else {
        // No stops - use white
        color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
      }
      
      return color;
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

    // Freeform gradient colour (E8.18, Illustrator's Freeform). The TS twin, and the model's
    // description, is core/scene/freeformGradient.ts `freeformColorAt`: a weighted mean of the
    // primitives' values, weight 1 / (e³ + ε) with e the distance past the spread radius r, softened
    // over r: e = (d − r + √((d − r)² + r²)) / 2; a lone
    // point's value is its colour plus its slope term (faded past its reach, starting at its spread's
    // edge), a segment's the ends' mix; clamped to the colours' range. `pos` is box-local (origin at
    // the box's top-left), `size` the box size.
    fn freeformWeight(d: f32, r: f32) -> f32 {
      let o = d - r;
      let e = 0.5 * (o + sqrt(o * o + r * r));
      return 1.0 / (e * e * e + 1e-9);
    }

    fn calculateFreeformColor(instanceId: u32, pos: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
      let count = min(paths[instanceId].numStops, 16u);
      if (count == 0u) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
      }
      let spreadUnit = 0.005 * length(size);
      var sum = vec4<f32>(0.0);
      var total = 0.0;
      var lo = vec4<f32>(1.0);
      var hi = vec4<f32>(0.0);
      for (var i = 0u; i < count; i++) {
        let p = paths[instanceId].freeformPoints[i];
        let ca = paths[instanceId].freeformColors[i];
        lo = min(lo, ca);
        hi = max(hi, ca);
        let link = p.w;
        if (link > 1.5) {
          // A line's last point: drawn by the segment before it.
          continue;
        }
        let a = p.xy * 0.01 * size;
        if (link > 0.5 && i + 1u < count) {
          let q = paths[instanceId].freeformPoints[i + 1u];
          let b = q.xy * 0.01 * size;
          let ab = b - a;
          let len2 = dot(ab, ab);
          var t = 0.0;
          if (len2 > 0.0) {
            t = clamp(dot(pos - a, ab) / len2, 0.0, 1.0);
          }
          let w = freeformWeight(length(pos - (a + ab * t)), mix(p.z, q.z, t) * spreadUnit);
          sum += mix(ca, paths[instanceId].freeformColors[i + 1u], t) * w;
          total += w;
        } else {
          let r = p.z * spreadUnit;
          let delta = pos - a;
          let d2 = dot(delta, delta);
          let d = sqrt(d2);
          let reach = paths[instanceId].freeformReach[i / 4u][i % 4u];
          let reach2 = reach * reach;
          var fade = 0.0;
          if (d > 0.0 && reach2 > 0.0) {
            fade = reach2 / (reach2 + d2) * max(0.0, d - r) / d;
          }
          let slope = paths[instanceId].freeformSlopes[2u * i] * delta.x
            + paths[instanceId].freeformSlopes[2u * i + 1u] * delta.y;
          let w = freeformWeight(d, r);
          sum += (ca + fade * slope) * w;
          total += w;
        }
      }
      if (total <= 0.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
      }
      return clamp(sum / total, lo, hi);
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
  
        
        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
          let path = paths[input.instanceId];
          
          // Early page clip check
          let bounds = input.clip;
          if (input.worldPos.x < bounds.x ||
              input.worldPos.x > bounds.x + bounds.z ||
              input.worldPos.y < bounds.y ||
              input.worldPos.y > bounds.y + bounds.w) {
            discard;
          }

          // Precompute mask alpha in uniform control flow
          var maskAlpha: f32 = 1.0;
          if (maskUniforms.maskActive == 1u) {
            let uvMask = (input.worldPos - maskUniforms.maskOrigin) * maskUniforms.maskInvSize;
            if (uvMask.x < 0.0 || uvMask.x > 1.0 || uvMask.y < 0.0 || uvMask.y > 1.0) { discard; }
            maskAlpha = textureSample(maskTexture, maskSampler, uvMask).r;
            if (maskAlpha <= 0.0) { discard; }
          }
          
          // Shared node-local bounds geometry (gradients are sampled in this space
          // for both fill and stroke, matching the existing fill-gradient behavior).
          let width = path.bounds.z;
          let height = path.bounds.w;
          let halfSize = vec2<f32>(width * 0.5, height * 0.5);
          let localPosForGradient = input.localPos - vec2<f32>(path.bounds.x, path.bounds.y) - halfSize;

          // Stroke branch
          if (input.isStroke > 0.5) {
            let strokePaintType = path.strokeGradientMeta.x;
            var scolor: vec4<f32>;
            // meta.w = mode (0 within, 1 along, 2 across) + 4 × the paint's flags (gradient-space.wgsl).
            let strokeMode = f32(u32(path.strokeGradientMeta.w + 0.5) % 4u);
            let strokeFlags = f32(u32(path.strokeGradientMeta.w + 0.5) / 4u);
            if ((strokePaintType == 1.0 || strokePaintType == 2.0 || strokePaintType == 4.0) && strokeMode > 0.5) {
              // Along (1): the stops run from the path's start to its end; a closed path's seam
              // triangles carry along > 1, wrapped here. Across (2): left edge to right edge.
              var t = input.strokeUV.y;
              if (strokeMode < 1.5) {
                t = input.strokeUV.x;
                if (t > 1.0) {
                  t = t - 1.0;
                }
              }
              scolor = gsDither(interpolateGradient(t, u32(path.strokeGradientMeta.y), path.strokeGradientStops, strokeFlags), input.position.xy, strokeFlags);
              scolor.a *= path.strokeColor.a;
            } else if (strokePaintType == 1.0 || strokePaintType == 2.0 || strokePaintType == 4.0) {
              scolor = applyGradient(
                localPosForGradient,
                halfSize,
                width,
                height,
                strokePaintType,
                path.strokeGradientStart,
                path.strokeGradientEnd,
                path.strokeGradientMeta.z,
                u32(path.strokeGradientMeta.y),
                path.strokeGradientStops,
                strokeFlags
              );
              scolor = gsDither(scolor, input.position.xy, strokeFlags);
              // strokeColor.a carries the node opacity multiplier for gradient strokes.
              scolor.a *= path.strokeColor.a;
            } else {
              scolor = path.strokeColor;
            }
            if (scolor.a <= 0.0) { discard; }
            scolor.a *= maskAlpha;
            return premultiplyAlpha(scolor);
          }

          // Fill branch
          let fillType = path.pathInfo.w;
          var color: vec4<f32>;
          // 1.0 = linear, 2.0 = radial, 3.0 = conic, 4.0 = diamond, 5.0 = freeform. Any
          // stop gradient runs the shared evaluator; applyGradient dispatches on the type.
          if (fillType == 5.0) {
            color = calculateFreeformColor(input.instanceId, localPosForGradient + halfSize, vec2<f32>(width, height));
            color.a *= path.color.a;
          } else if (fillType >= 1.0) {
            color = applyGradient(
              localPosForGradient,
              halfSize,
              width,
              height,
              fillType,
              path.gradientStart,
              path.gradientEnd,
              path.gradientRotation,
              path.numStops,
              path.gradientStops,
              path.gradientFlags
            );
            color = gsDither(color, input.position.xy, path.gradientFlags);
            color.a *= path.color.a;
          } else {
            color = path.color;
          }
          if (color.a <= 0.0) { discard; }
          color.a *= maskAlpha;
          return premultiplyAlpha(color);
        }
