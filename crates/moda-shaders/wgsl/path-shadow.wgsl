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


// Common SDF utilities

// SDF for a rounded rectangle with different corner radii
fn sdRoundedBoxNonUniform(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
  // radii: x=top-left, y=top-right, z=bottom-right, w=bottom-left
  
  // Determine which quadrant we're in
  let signX = sign(p.x);
  let signY = sign(p.y);
  
  // Select the appropriate corner radius based on quadrant
  var radius: f32;
  if (signX < 0.0 && signY < 0.0) {
    radius = radii.x; // top-left
  } else if (signX > 0.0 && signY < 0.0) {
    radius = radii.y; // top-right
  } else if (signX > 0.0 && signY > 0.0) {
    radius = radii.z; // bottom-right
  } else {
    radius = radii.w; // bottom-left
  }
  
  // Calculate distance using the selected radius
  let q = abs(p) - halfSize + radius;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// SDF for a rounded rectangle with uniform corner radius
fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
  let q = abs(p) - halfSize + radius;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// SDF for a regular rectangle (no rounded corners)
fn sdBox(p: vec2<f32>, halfSize: vec2<f32>) -> f32 {
  let q = abs(p) - halfSize;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

// SDF for a circle
fn sdCircle(p: vec2<f32>, radius: f32) -> f32 {
  return length(p) - radius;
}

// SDF for an ellipse.
// Guards against degenerate radii and a zero-length p (which would otherwise
// produce NaN from the k1 divide) so callers can safely sample every pixel
// inside the bounding box, including the exact ellipse center.
fn sdEllipse(p: vec2<f32>, radii: vec2<f32>) -> f32 {
  if (radii.x < 1e-4 || radii.y < 1e-4) { return 1.0; }
  let k0 = length(p / radii);
  let k1 = length(p / (radii * radii));
  if (k1 < 1e-4) { return -min(radii.x, radii.y); }
  return k0 * (k0 - 1.0) / k1;
}


// Main shader code

        // Frame-level uniforms (constant for all shapes in a frame)
        @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>; // viewProjection
        
        // Storage buffer for rectangle shadow instances
        struct RectangleShadowData {
          model: mat4x4<f32>,      // 16 floats
          shadowColor: vec4<f32>,  // 4 floats
          shapeData: vec4<f32>,    // width, height, blur radius, depth - 4 floats
          cornerRadii: vec4<f32>,  // top-left, top-right, bottom-right, bottom-left - 4 floats
          shadowOffset: vec4<f32>, // x, y, showBehindNode flag, opaqueSdfCoversShadow flag - 4 floats
          pageClipBounds: vec4<f32>, // x, y, width, height - 4 floats
          innerParams: vec4<f32>,  // isInner flag, spread, reserved, reserved - 4 floats
        }
        
        @group(1) @binding(0) var<storage, read> shadows: array<RectangleShadowData>;
        
        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) @interpolate(flat) instanceId: u32,
          @location(1) localPos: vec2<f32>, // Local position in pixels, centered at rectangle center
          @location(2) worldPos: vec2<f32>, // World position for clipping
        }
        
        @vertex
        fn vertexMain(
          @builtin(vertex_index) vertexIndex: u32,
          @builtin(instance_index) instanceIndex: u32
        ) -> VertexOutput {
          var output: VertexOutput;
          let shadow = shadows[instanceIndex];
          let pos = getNormalizedQuadPosition(vertexIndex);

          // Get dimensions from shapeData
          let width = shadow.shapeData.x;
          let height = shadow.shapeData.y;
          let blurRadius = shadow.shapeData.z;

          // Expand the quad to the Gaussian's reach: 3 sigma = 1.5 × blur per side, plus a
          // drop shadow's spread and a pixel of anti-aliasing. (Inner shadows stay inside the
          // silhouette; the same expansion is harmless there.)
          let growth = select(max(shadow.innerParams.y, 0.0), 0.0, shadow.innerParams.x > 0.5);
          let expansion = blurRadius * 3.0 + 2.0 * growth + 2.0;
          let expandedWidth = width + expansion;
          let expandedHeight = height + expansion;

          // Ellipse sentinel: cornerRadii.w < 0. Center-local mapping for ellipses
          let isEllipse = shadow.cornerRadii.w < 0.0;

          var localX: f32;
          var localY: f32;
          if (isEllipse) {
            // Map [-1,1] to centered coordinates with expansion
            localX = pos.x * expandedWidth * 0.5;
            localY = pos.y * expandedHeight * 0.5;
          } else {
            // Rectangle/rounded-rect: top-left anchored with expansion, will center for SDF later
            localX = (pos.x + 1.0) * expandedWidth * 0.5 - expansion * 0.5;
            localY = (pos.y + 1.0) * expandedHeight * 0.5 - expansion * 0.5;
          }

          // Transform to world space using model matrix (rotation and translation)
          var worldPos = shadow.model * vec4<f32>(localX, localY, 0.0, 1.0);

          // Drop shadows shift the quad by the offset before projection. Inner
          // shadows stay at the shape position (their offset is applied in-SDF in
          // the fragment shader), so skip the world-space shift for them.
          let isInner = shadow.innerParams.x > 0.5;
          if (!isInner) {
            worldPos.x += shadow.shadowOffset.x;
            worldPos.y += shadow.shadowOffset.y;
          }

          output.worldPos = worldPos.xy; // Pass world position to fragment shader
          output.position = frameUniforms * worldPos;
          output.position.z = shadow.shapeData.w; // Use the depth value from shapeData
          output.instanceId = instanceIndex;

          // Local position for SDF: centered for both cases
          if (isEllipse) {
            output.localPos = vec2<f32>(localX, localY);
          } else {
            output.localPos = vec2<f32>(localX - width * 0.5, localY - height * 0.5);
          }

          return output;
        }
        
        
      
      // Gaussian shadow falloff (CSS box-shadow / Figma semantics): a shadow of blur B is the
      // silhouette convolved with a Gaussian of sigma = B / 2. For the rect / rounded-rect /
      // ellipse silhouettes this lane draws, the convolution is evaluated directly (Evan
      // Wallace, "Fast rounded rectangle shadows"): the x integral of one silhouette row is
      // exact (a difference of erfs), and the y integral is a midpoint sum over the rows within
      // 4 sigma (12 rows: a Gaussian's midpoint sum at that spacing is 1 to within 1e-5, so a
      // fully covered point stays exactly opaque). That is exact for a rectangle and matches a
      // real Gaussian blur for rounded
      // corners, ellipses and shapes smaller than the blur, where an SDF falloff
      // (smoothstep of the distance) is too dark.
      const SHADOW_ROWS: i32 = 12;

      // erf for two lanes (Abramowitz & Stegun 7.1.27, |error| < 5e-4).
      fn erf2(x: vec2<f32>) -> vec2<f32> {
        let s = sign(x);
        let a = abs(x);
        var y = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
        y = y * y;
        return s - s / (y * y);
      }

      // Half-extents [left, right] of silhouette row `rowY` (centre-relative), and whether the
      // row is inside the silhouette's height. Rounded rect: radii [tl, tr, br, bl], already
      // clamped to the half size. Ellipse: radii.xy = (rx, ry).
      fn shadowRowExtent(rowY: f32, halfSize: vec2<f32>, radii: vec4<f32>, isEllipse: bool) -> vec2<f32> {
        if (isEllipse) {
          let t = rowY / max(radii.y, 1e-6);
          let w = radii.x * sqrt(max(0.0, 1.0 - t * t));
          return vec2<f32>(w, w);
        }
        let rl = select(radii.w, radii.x, rowY < 0.0);
        let rr = select(radii.z, radii.y, rowY < 0.0);
        let dl = max(abs(rowY) - (halfSize.y - rl), 0.0);
        let dr = max(abs(rowY) - (halfSize.y - rr), 0.0);
        return vec2<f32>(
          halfSize.x - rl + sqrt(max(0.0, rl * rl - dl * dl)),
          halfSize.x - rr + sqrt(max(0.0, rr * rr - dr * dr))
        );
      }

      // The silhouette (centred at the origin; half size `halfSize`) convolved with a Gaussian
      // of `sigma`, at `p`. 0 for an empty silhouette.
      fn gaussianSilhouette(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>, isEllipse: bool, sigma: f32) -> f32 {
        let halfH = select(halfSize.y, radii.y, isEllipse);
        if (halfH <= 0.0 || select(halfSize.x, radii.x, isEllipse) <= 0.0) {
          return 0.0;
        }
        // Kernel offsets k with the row p.y - k inside the silhouette, limited to 4 sigma.
        let lo = clamp(-4.0 * sigma, p.y - halfH, p.y + halfH);
        let hi = clamp(4.0 * sigma, p.y - halfH, p.y + halfH);
        let stepK = (hi - lo) / f32(SHADOW_ROWS);
        let invS = INVERSE_SQRT_2 / sigma;
        let norm = 1.0 / (2.5066282746 * sigma); // 1 / (sqrt(2 pi) sigma)
        var total = 0.0;
        var k = lo + 0.5 * stepK;
        for (var i = 0; i < SHADOW_ROWS; i++) {
          let e = shadowRowExtent(p.y - k, halfSize, radii, isEllipse);
          let span = 0.5 * erf2(vec2<f32>(p.x + e.x, p.x - e.y) * invS);
          total += (span.x - span.y) * exp(-0.5 * k * k / (sigma * sigma)) * norm * stepK;
          k += stepK;
        }
        return clamp(total, 0.0, 1.0);
      }

      // The silhouette grown by `spread` (negative shrinks): half size +spread; a rounded
      // corner r becomes max(0, r + spread) and a square corner stays square (CSS spread);
      // ellipse radii +spread. Radii are then clamped to the half size, as the body draws them.
      fn spreadHalfSize(halfSize: vec2<f32>, spread: f32) -> vec2<f32> {
        return halfSize + vec2<f32>(spread);
      }
      fn spreadRadii(radii: vec4<f32>, halfSize: vec2<f32>, spread: f32, isEllipse: bool) -> vec4<f32> {
        if (isEllipse) {
          return vec4<f32>(radii.xy + vec2<f32>(spread), 0.0, -1.0);
        }
        let grown = select(vec4<f32>(0.0), max(radii + vec4<f32>(spread), vec4<f32>(0.0)), radii > vec4<f32>(0.0));
        return min(grown, vec4<f32>(max(min(halfSize.x, halfSize.y), 0.0)));
      }

        fn sdEllipseDistance(p: vec2<f32>, r: vec2<f32>) -> f32 {
          if (r.x < 0.0001 || r.y < 0.0001) {
            return 1e6;
          }
          let k0 = length(p / r);
          let k1 = length(p / (r * r));
          if (k1 > 1e-6) {
            return k0 * (k0 - 1.0) / k1;
          }
          return -min(r.x, r.y);
        }

        
// Box-filter edge coverage in [0,1]: 1 at half a pixel inside the silhouette,
// 0.5 on the edge, 0 at half a pixel outside. edgeWidth is fwidth(dist) —
// one device pixel expressed in the SDF's units.
fn sdfCoverage(dist: f32, edgeWidth: f32) -> f32 {
  return clamp(0.5 - dist / max(edgeWidth, 1e-6), 0.0, 1.0);
}


        // Silhouette coverage for the shadow lane. This MUST stay the same box
        // filter the shape body uses: the drop-shadow knockout is
        // applied as (1 - coverage), so any disagreement with the body's own
        // coverage leaves the shadow uncancelled under the body's edge, and the
        // inner shadow's inside clip would feather differently from the fill it
        // sits in.
        fn computeCoverageAlpha(distance: f32) -> f32 {
          return sdfCoverage(distance, fwidth(distance));
        }
        
        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
          let shadow = shadows[input.instanceId];
          let width = shadow.shapeData.x;
          let height = shadow.shapeData.y;
          let blurRadius = shadow.shapeData.z;
          let halfSize = vec2<f32>(width * 0.5, height * 0.5);

          // Determine silhouette: rectangle (rounded box) vs ellipse
          // Convention: cornerRadii.w < 0 signals ellipse; x,y carry radiusX, radiusY
          let isEllipse = shadow.cornerRadii.w < 0.0;

          // isInner selects the inset variant: the shadow is clipped to the INSIDE of
          // the silhouette and inverted (dark near the inner edges, falling off inward).
          let isInner = shadow.innerParams.x > 0.5;
          let spread = shadow.innerParams.y;

          // Calculate shadow alpha in shadow-offset space, then evaluate knockout
          // coverage against the original node silhouette in unshifted local space.
          var knockoutDistance: f32;
          var dropAlpha: f32;
          // Signed distance to the silhouette at this fragment (unshifted) — drives
          // the inner shadow's inside clip.
          var shadowDistance: f32;
          let linearCol0 = shadow.model[0].xy;
          let linearCol1 = shadow.model[1].xy;
          // Near-zero determinant (degenerate transform, e.g. zero-scale) falls back to
          // zero offset, which causes full knockout. Acceptable: the node is invisible anyway.
          let linearDet = linearCol0.x * linearCol1.y - linearCol1.x * linearCol0.y;
          let safeInvDet = select(0.0, 1.0 / linearDet, abs(linearDet) > 1e-5);
          let inverseLinear = mat2x2<f32>(
            vec2<f32>(linearCol1.y, -linearCol0.y),
            vec2<f32>(-linearCol1.x, linearCol0.x)
          );
          let localShadowOffset = (inverseLinear * shadow.shadowOffset.xy) * safeInvDet;
          let knockoutLocalPos = input.localPos + localShadowOffset;
          // Inner shadow samples the silhouette on the opposite side of the offset, so a
          // positive offset throws the inset toward the top-left (CSS inset convention).
          let casterLocalPos = input.localPos - localShadowOffset;
          // Corner radii as the body draws them: each clamped to half the shorter side (the
          // node's raw cornerRadius can exceed it). Ellipses carry (rx, ry, 0, -1).
          let radii = select(
            min(max(shadow.cornerRadii, vec4<f32>(0.0)), vec4<f32>(min(halfSize.x, halfSize.y))),
            shadow.cornerRadii,
            isEllipse
          );
          if (isEllipse) {
            let r = radii.xy;
            shadowDistance = sdEllipseDistance(input.localPos, r);
            knockoutDistance = sdEllipseDistance(knockoutLocalPos, r);
          } else {
            shadowDistance = sdRoundedBoxNonUniform(input.localPos, halfSize, radii);
            knockoutDistance = sdRoundedBoxNonUniform(knockoutLocalPos, halfSize, radii);
          }

          // sigma = blur / 2 (CSS / Figma), widened by the pixel footprint so a hard (blur 0)
          // shadow still gets an anti-aliased edge. fwidth stays in uniform control flow.
          let pixelLocal = 0.5 * (fwidth(input.localPos.x) + fwidth(input.localPos.y));
          let sigma = sqrt(0.25 * blurRadius * blurRadius + 0.09 * pixelLocal * pixelLocal) + 1e-6;

          // Keep derivative-based edge coverage outside non-uniform control flow.
          // WGSL forbids calling fwidth() in branches that depend on per-instance data.
          // Safe because all fragments in a quad share the same instance (and isEllipse value).
          let coverageAlpha = computeCoverageAlpha(knockoutDistance);
          let showBehindNode = shadow.shadowOffset.z >= 0.5;
          // shadowOffset.w (opaqueSdfCoversShadow) is set ONLY by the native/WASM
          // SDF body path, where an opaque analytic body occludes its
          // own shadow so the coverage knockout would only carve an AA seam. The
          // browser WebGPU shadow producer always packs 0 here, so this term is a
          // no-op there and WebGPU behavior is unchanged.
          let opaqueSdfCoversShadow = shadow.shadowOffset.w >= 0.5;
          let knockoutMultiplier = select(1.0 - coverageAlpha, 1.0, showBehindNode || opaqueSdfCoversShadow);
          // Inner shadow = inside coverage × (1 − the blurred caster). computeCoverageAlpha
          // (fwidth) is kept at top level for uniformity.
          let insideCoverage = computeCoverageAlpha(shadowDistance);

          // Check page clip bounds
          let bounds = shadow.pageClipBounds;
          if (input.worldPos.x < bounds.x || 
              input.worldPos.x > bounds.x + bounds.z ||
              input.worldPos.y < bounds.y || 
              input.worldPos.y > bounds.y + bounds.w) {
            discard;
          }

          // A fragment whose alpha is already known to be zero is discarded before the
          // Gaussian (12 rows of erf / exp per fragment): a drop shadow under its own knocked-out
          // silhouette (the body's interior, the whole screen when zoomed into a shadowed frame)
          // and an inner shadow outside its node. Both would be discarded below anyway
          // (alpha = blurred × 0), so the output is the same, without the convolution.
          if (select(knockoutMultiplier, insideCoverage, isInner) <= 0.0) {
            discard;
          }

          // Drop shadow: the silhouette grown by the spread, blurred. Inner shadow: the
          // complement of the silhouette shrunk by the spread and moved against the offset
          // (casterLocalPos), blurred, inside the silhouette.
          let casterSpread = select(spread, -spread, isInner);
          let casterHalf = select(spreadHalfSize(halfSize, casterSpread), halfSize, isEllipse);
          let casterRadii = spreadRadii(radii, casterHalf, casterSpread, isEllipse);
          let casterPos = select(input.localPos, casterLocalPos, isInner);
          let blurred = gaussianSilhouette(casterPos, casterHalf, casterRadii, isEllipse, sigma);
          dropAlpha = blurred * knockoutMultiplier;
          let innerAlpha = insideCoverage * (1.0 - blurred);

          let shadowAlpha = select(dropAlpha, innerAlpha, isInner);

          if (shadowAlpha <= 0.001) {
            discard;
          }

          // Return shadow color with calculated alpha
          let shadowColor = vec4<f32>(shadow.shadowColor.rgb, shadow.shadowColor.a * shadowAlpha);
          return premultiplyAlpha(shadowColor);
        }
      