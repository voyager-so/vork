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

        
// Blend mode constants
const BLEND_MODE_NORMAL: f32 = 0.0;
const BLEND_MODE_MULTIPLY: f32 = 1.0;
const BLEND_MODE_SCREEN: f32 = 2.0;
const BLEND_MODE_OVERLAY: f32 = 3.0;

// RGB to HSV conversion
fn rgbToHsv(rgb: vec3<f32>) -> vec3<f32> {
  let cmax = max(max(rgb.r, rgb.g), rgb.b);
  let cmin = min(min(rgb.r, rgb.g), rgb.b);
  let delta = cmax - cmin;

  var hsv: vec3<f32>;

  // Hue calculation
  if (delta == 0.0) {
    hsv.x = 0.0;
  } else if (cmax == rgb.r) {
    hsv.x = fract((rgb.g - rgb.b) / delta / 6.0);
  } else if (cmax == rgb.g) {
    hsv.x = (rgb.b - rgb.r) / delta / 6.0 + 1.0/3.0;
  } else {
    hsv.x = (rgb.r - rgb.g) / delta / 6.0 + 2.0/3.0;
  }

  // Saturation
  hsv.y = select(0.0, delta / cmax, cmax != 0.0);

  // Value
  hsv.z = cmax;

  return hsv;
}

// HSV to RGB conversion
fn hsvToRgb(hsv: vec3<f32>) -> vec3<f32> {
  let h = hsv.x * 6.0;
  let s = hsv.y;
  let v = hsv.z;

  let c = v * s;
  let x = c * (1.0 - abs(fract(h / 2.0) * 2.0 - 1.0));
  let m = v - c;

  var rgb: vec3<f32>;
  let hi = floor(h);

  if (hi == 0.0 || hi == 6.0) {
    rgb = vec3<f32>(c, x, 0.0);
  } else if (hi == 1.0) {
    rgb = vec3<f32>(x, c, 0.0);
  } else if (hi == 2.0) {
    rgb = vec3<f32>(0.0, c, x);
  } else if (hi == 3.0) {
    rgb = vec3<f32>(0.0, x, c);
  } else if (hi == 4.0) {
    rgb = vec3<f32>(x, 0.0, c);
  } else {
    rgb = vec3<f32>(c, 0.0, x);
  }

  return rgb + vec3<f32>(m);
}

// Figma's image adjustments (E7.18), each -1..1 (0 = unchanged), on straight sRGB-encoded colour. The reference
// (and the pixel tests' expectation) is moda-engine `paint::pattern::image_adjust`:
// 1. exposure and white balance in linear light: * 2^(2 exposure); red * (1 + 0.2 temperature),
//    blue * (1 - 0.2 temperature), green * (1 - 0.2 tint);
// 2. contrast about mid grey: (c - 0.5)(1 + contrast) + 0.5;
// 3. highlights and shadows by Rec. 709 luma L: + 0.25 (highlights smoothstep(0.5, 1, L)
//    + shadows (1 - smoothstep(0, 0.5, L)));
// 4. saturation: mix(L, c, 1 + saturation).
// Every step clamps to 0..1.
fn adjustSrgbToLinear(c: vec3<f32>) -> vec3<f32> {
  return select(pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn adjustLinearToSrgb(l: vec3<f32>) -> vec3<f32> {
  return select(1.055 * pow(l, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), l * 12.92, l <= vec3<f32>(0.0031308));
}

fn applyImageAdjust(color: vec4<f32>, a: vec4<f32>, b: vec2<f32>, shadows: f32) -> vec4<f32> {
  if (all(a == vec4<f32>(0.0)) && all(b == vec2<f32>(0.0)) && shadows == 0.0) {
    return color;
  }
  let rec709 = vec3<f32>(0.2126, 0.7152, 0.0722);
  let whiteBalance = vec3<f32>(1.0 + 0.2 * a.w, 1.0 - 0.2 * b.x, 1.0 - 0.2 * a.w);
  let lin = clamp(adjustSrgbToLinear(clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(1.0))) * exp2(2.0 * a.x) * whiteBalance, vec3<f32>(0.0), vec3<f32>(1.0));
  var c = clamp(adjustLinearToSrgb(lin), vec3<f32>(0.0), vec3<f32>(1.0));
  c = clamp((c - vec3<f32>(0.5)) * (1.0 + a.y) + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
  let l = dot(c, rec709);
  let tone = 0.25 * (b.y * smoothstep(0.5, 1.0, l) + shadows * (1.0 - smoothstep(0.0, 0.5, l)));
  c = clamp(c + vec3<f32>(tone), vec3<f32>(0.0), vec3<f32>(1.0));
  let l2 = dot(c, rec709);
  c = clamp(mix(vec3<f32>(l2), c, 1.0 + a.z), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(c, color.a);
}

// Apply color overlay with different blend modes
fn applyColorOverlay(
  originalColor: vec4<f32>,
  overlayColor: vec4<f32>,
  opacity: f32,
  blendMode: f32
) -> vec4<f32> {
  // Preserve original alpha
  let alpha = originalColor.a;
  // Work in straight color space (shader operates on straight colors before final premultiply)
  let base: vec3<f32> = originalColor.rgb;

  var blendedColor: vec3<f32>;

  // Apply blend mode in straight color space
  // Using small epsilon for floating point comparison
  let epsilon = 0.01;

  if (abs(blendMode - BLEND_MODE_NORMAL) < epsilon) {
    blendedColor = overlayColor.rgb;
  } else if (abs(blendMode - BLEND_MODE_MULTIPLY) < epsilon) {
    blendedColor = base * overlayColor.rgb;
  } else if (abs(blendMode - BLEND_MODE_SCREEN) < epsilon) {
    blendedColor = vec3<f32>(1.0) - (vec3<f32>(1.0) - base) * (vec3<f32>(1.0) - overlayColor.rgb);
  } else if (abs(blendMode - BLEND_MODE_OVERLAY) < epsilon) {
    // Overlay blend mode: combines multiply and screen
    blendedColor = mix(
      2.0 * base * overlayColor.rgb,
      vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - base) * (vec3<f32>(1.0) - overlayColor.rgb),
      step(vec3<f32>(0.5), base)
    );
  } else {
    // Default to normal blend mode for unknown values
    blendedColor = overlayColor.rgb;
  }

  // Mix overlay effect independent of alpha to avoid fringe; preserve coverage
  let mixed = mix(base, blendedColor, opacity);
  return vec4<f32>(mixed, alpha);
}

// Apply Color Change (palette-based recoloring)
// - Checks up to 4 source colors and blends towards corresponding target colors based on tolerance/smoothness
// - Params: colorCount in params.x, tolerance in params.y, smoothness in params.z
fn applyColorChange(
  originalColor: vec4<f32>,
  source0: vec4<f32>, source1: vec4<f32>, source2: vec4<f32>, source3: vec4<f32>,
  target0: vec4<f32>, target1: vec4<f32>, target2: vec4<f32>, target3: vec4<f32>,
  params: vec3<f32>
) -> vec4<f32> {
  let colorCount = u32(round(params.x));
  // Scale normalized tolerance [0,1] to full RGB distance range [0, sqrt(3)]
  let tolerance = params.y * 1.7320508;
  let smoothness = params.z;

  var result = originalColor;

  // Inline array-like access
  for (var i = 0u; i < colorCount; i = i + 1u) {
    var src: vec3<f32>;
    var dst: vec3<f32>;
    if (i == 0u) { src = source0.rgb; dst = target0.rgb; }
    else if (i == 1u) { src = source1.rgb; dst = target1.rgb; }
    else if (i == 2u) { src = source2.rgb; dst = target2.rgb; }
    else { src = source3.rgb; dst = target3.rgb; }

    let diff = originalColor.rgb - src;
    let d = length(diff);
    if (d <= tolerance) {
      let w = 1.0 - smoothstep(tolerance * (1.0 - smoothness), tolerance, d);
      // Subtle mode: preserve per-pixel offset relative to src in HSV space
      // Convert to HSV
      let srcHsv = rgbToHsv(clamp(src, vec3<f32>(0.0), vec3<f32>(1.0)));
      let dstHsv = rgbToHsv(clamp(dst, vec3<f32>(0.0), vec3<f32>(1.0)));
      let pixHsv = rgbToHsv(clamp(originalColor.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
      var offsetH = pixHsv.x - srcHsv.x; // hue delta
      // Wrap hue into [-0.5, 0.5]
      offsetH = fract(offsetH + 0.5) - 0.5;
      let offsetS = pixHsv.y - srcHsv.y;
      let offsetV = pixHsv.z - srcHsv.z;
      // Apply offsets to destination hsv and clamp
      var newHsv = vec3<f32>(
        fract(dstHsv.x + offsetH),
        clamp(dstHsv.y + offsetS, 0.0, 1.0),
        clamp(dstHsv.z + offsetV, 0.0, 1.0)
      );
      let subtleRgb = hsvToRgb(newHsv);
      let newRgb = mix(result.rgb, subtleRgb, w);
      result = vec4<f32>(newRgb, result.a);
    }
  }

  return result;
}

        
        // Frame-level uniforms
        @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>; // viewProjection
        // Mask sampling (optional)
        struct MaskUniforms {
          maskActive: u32,
          maskOrigin: vec2<f32>,
          maskInvSize: vec2<f32>,
        }
        @group(3) @binding(0) var maskSampler: sampler;
        @group(3) @binding(1) var maskTexture: texture_2d<f32>;
        @group(3) @binding(2) var<uniform> maskUniforms: MaskUniforms;
        
        // Storage buffer for path pattern instances
        struct PathPatternData {
          model: mat4x4<f32>,      // 16 floats
          fillColor: vec4<f32>,    // 4 floats (mainly for opacity)
          strokeColor: vec4<f32>,  // 4 floats
          pathInfo: vec4<f32>,     // vertexOffset, vertexCount, strokeWidth, depth - 4 floats
          patternData: vec4<f32>,  // repeatMode, scaleX, scaleY, rotation - 4 floats
          textureDims: vec2<f32>,  // texture width, height - 2 floats
          patternOffset: vec2<f32>, // pattern offset x, y - 2 floats
          bounds: vec4<f32>,       // x, y, width, height - 4 floats
          pageClipBounds: vec4<f32>, // x, y, width, height - 4 floats
          adjustA: vec4<f32>,      // image adjustments: exposure, contrast, saturation, temperature - 4 floats
          overlayColor: vec4<f32>, // RGBA overlay color - 4 floats
          overlaySettings: vec2<f32>, // opacity, blendMode - 2 floats
          adjustB: vec2<f32>,      // image adjustments: tint, highlights - 2 floats
          // Color Change data
          colorChangeSourceColor0: vec4<f32>,
          colorChangeSourceColor1: vec4<f32>,
          colorChangeSourceColor2: vec4<f32>,
          colorChangeSourceColor3: vec4<f32>,
          colorChangeTargetColor0: vec4<f32>,
          colorChangeTargetColor1: vec4<f32>,
          colorChangeTargetColor2: vec4<f32>,
          colorChangeTargetColor3: vec4<f32>,
          colorChangeParams: vec3<f32>, // x=count, y=tolerance, z=smoothness
          adjustShadows: f32, // image adjustment: shadows
          perspectiveRow0: vec4<f32>, // xyz = inverse homography row 0, w = enabled flag
          perspectiveRow1: vec4<f32>, // xyz = inverse homography row 1
          perspectiveRow2: vec4<f32>, // xyz = inverse homography row 2
          colorize: vec4<f32>, // enabled, hue turns, saturation, lightness
        }
        
        @group(1) @binding(0) var<storage, read> paths: array<PathPatternData>;
        
        // Texture and sampler
        @group(2) @binding(0) var patternSampler: sampler;
        @group(2) @binding(1) var patternTexture: texture_2d<f32>;
        
        // Stroke depth offset constant
        const STROKE_DEPTH_OFFSET: f32 = -0.000001490116119384765625;
        
        struct VertexInput {
          @location(0) position: vec2<f32>,
          @location(1) vertexType: f32, // 0 = fill, 1 = stroke
        }
        
        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) @interpolate(flat) instanceId: u32,
          @location(1) uv: vec2<f32>,
          @location(2) @interpolate(flat) isStroke: f32,
          @location(3) localPos: vec2<f32>,
          @location(4) worldPos: vec2<f32>, // Add world position for clipping
          @location(5) @interpolate(flat) clip: vec4<f32>,
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
          output.worldPos = worldPos.xy; // Pass world position to fragment shader
          output.position = frameUniforms * worldPos;
          
          // Use pre-calculated depth value (stored in pathInfo.w)
          let depthOffset = select(0.0, STROKE_DEPTH_OFFSET, vertex.vertexType > 0.5);
          output.position.z = path.pathInfo.w + depthOffset;
          
          output.instanceId = instanceIndex;
          output.isStroke = vertex.vertexType;
          output.localPos = localPos;
          output.clip = path.pageClipBounds;
          
          // Calculate UV coordinates with offset
          // Get texture dimensions
          let textureDims = path.textureDims;
          
          // Apply pattern offset
          let offsetPos = localPos - path.patternOffset;
          
          // Calculate UV so that texture maintains its original pixel size
          // offsetPos is in shape space, we want UV in texture space
          output.uv = offsetPos / textureDims;
          
          return output;
        }
        
        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
          let path = paths[input.instanceId];
          
          // Get pattern properties
          let repeatMode = path.patternData.x;
          let scaleX = path.patternData.y;
          let scaleY = path.patternData.z;
          let rotation = path.patternData.w; // rotation in radians
          
          // Start with the UV from vertex shader (already has offset applied)
          var uv = input.uv;
          var perspectiveInvalid = false;

          if (path.perspectiveRow0.w > 0.5) {
            let projected = vec3<f32>(input.localPos.x, input.localPos.y, 1.0);
            let denominator = dot(path.perspectiveRow2.xyz, projected);
            perspectiveInvalid = abs(denominator) < 0.00001;
            let safeDenominator = select(denominator, 1.0, perspectiveInvalid);

            let sourceLocal = vec2<f32>(
              dot(path.perspectiveRow0.xyz, projected) / safeDenominator,
              dot(path.perspectiveRow1.xyz, projected) / safeDenominator
            );
            uv = (sourceLocal - path.patternOffset) / path.textureDims;
          }
          
          // Apply rotation if needed
          if (abs(rotation) > 0.001) {
            // Rotate UV coordinates around (0, 0)
            let cos_r = cos(rotation);
            let sin_r = sin(rotation);
            let rotatedUV = vec2<f32>(
              uv.x * cos_r - uv.y * sin_r,
              uv.x * sin_r + uv.y * cos_r
            );
            uv = rotatedUV;
          }
          
          // Apply scale to UV coordinates (divide to make pattern larger)
          uv = uv / vec2<f32>(scaleX, scaleY);

          // The mip level follows the screen footprint of the continuous UV: taken before the repeat wrap, so a
          // tile seam (where fract jumps by 1) doesn't read as a huge footprint and sample the smallest level.
          let uvDx = dpdx(uv);
          let uvDy = dpdy(uv);
          
          // Apply repeat mode transformations
          if (repeatMode < 0.1) {  // repeat (0.0)
            uv = fract(uv);
          } else if (repeatMode < 0.35) {  // repeat-x (0.25)
            uv.x = fract(uv.x);
            uv.y = clamp(uv.y, 0.0, 1.0);
          } else if (repeatMode < 0.6) {  // repeat-y (0.5)
            uv.x = clamp(uv.x, 0.0, 1.0);
            uv.y = fract(uv.y);
          }
          // Note: for no-repeat (0.75), we keep scaled UV and handle transparency after sampling
          
          // Sample the texture unconditionally (required for uniform control flow)
          // Texture data is premultiplied alpha — un-premultiply so color filters
          // (adjustments, color change, overlay) operate on straight-alpha values.
          var textureColor = textureSampleGrad(patternTexture, patternSampler, uv, uvDx, uvDy);
          if (perspectiveInvalid) {
            discard;
          }
          // Any coverage keeps its colour, however small: a float image (a world's picture) can carry bright
          // colour at very low alpha, and an 8-bit image's contribution there is at most its alpha.
          let texA = textureColor.a;
          let safeInvTexA = select(0.0, 1.0 / texA, texA > 0.0);
          textureColor = vec4<f32>(textureColor.rgb * safeInvTexA, texA);

          // If palette color change is enabled (params.x > 0), apply recoloring first:
          // maps up to 4 source colors to target colors with tolerance/smoothness
          if (path.colorChangeParams.x > 0.0) {
            // Apply color change first (match original image colors)
            textureColor = applyColorChange(
              textureColor,
              path.colorChangeSourceColor0, path.colorChangeSourceColor1, path.colorChangeSourceColor2, path.colorChangeSourceColor3,
              path.colorChangeTargetColor0, path.colorChangeTargetColor1, path.colorChangeTargetColor2, path.colorChangeTargetColor3,
              path.colorChangeParams
            );
          }

          // Figma's image adjustments (E7.18), identity when all are 0.
          textureColor = applyImageAdjust(textureColor, path.adjustA, path.adjustB, path.adjustShadows);
          if (path.colorize.x > 0.5 && textureColor.a > 0.0) {
            let l = dot(textureColor.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            let chroma = (1.0 - abs(2.0 * l - 1.0)) * path.colorize.z;
            let h = fract(path.colorize.y) * 6.0;
            let base = clamp(abs(fract(vec3<f32>(h / 6.0) + vec3<f32>(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
            var c = base * chroma + vec3<f32>(l - chroma * 0.5);
            let light = path.colorize.w;
            c = select(c * (1.0 + light), c * (1.0 - light) + vec3<f32>(light), light > 0.0);
            textureColor = vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), textureColor.a);
          }

          // Apply color overlay if enabled (overlaySettings.y >= 0 means enabled)
          if (path.overlaySettings.y >= 0.0) {
            textureColor = applyColorOverlay(
              textureColor,
              path.overlayColor,
              path.overlaySettings.x, // opacity
              path.overlaySettings.y  // blend mode
            );
          }
          
          // Choose between stroke color and texture based on vertex type
          var color: vec4<f32>;
          if (input.isStroke > 0.5) {
            // For strokes, use the stroke color: authored, so converted to the engine's working colour. The
            // texture is in that encoding already (the pattern cache uploads a decoded image in it).
            color = vec4<f32>(workingRgb(path.strokeColor.rgb), path.strokeColor.a);
          } else {
            // For fills, check if we should show the pattern
            // For no-repeat mode, check against the transformed UV coordinates
            if (repeatMode > 0.6 && (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)) {
              // no-repeat mode and outside bounds
              color = vec4<f32>(0.0);
            } else {
              // Use the texture color with opacity
              color = textureColor;
              color.a *= path.fillColor.a; // Apply opacity
            }
          }
          
          // Discard if fully transparent
          if (color.a <= 0.0) {
            discard;
          }
          
          // Check page clip bounds
          let bounds = input.clip;
          if (input.worldPos.x < bounds.x || 
              input.worldPos.x > bounds.x + bounds.z ||
              input.worldPos.y < bounds.y || 
              input.worldPos.y > bounds.y + bounds.w) {
            discard;
          }
          
          // Optional mask
          if (maskUniforms.maskActive == 1u) {
            let uvMask = (input.worldPos - maskUniforms.maskOrigin) * maskUniforms.maskInvSize;
            if (uvMask.x < 0.0 || uvMask.x > 1.0 || uvMask.y < 0.0 || uvMask.y > 1.0) {
              discard;
            }
            let m = textureSample(maskTexture, maskSampler, uvMask).r;
            if (m <= 0.0) { discard; }
            color.a *= m;
          }
          return premultiplyAlpha(color);
        }
      