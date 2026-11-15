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

        // Frame-level uniforms (constant for all shapes in a frame)
        @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>; // viewProjection
        
        // Page instance data
        struct PageInstance {
          model: mat4x4<f32>,     // Transform matrix
          color: vec4<f32>,       // Fill color
          dimensions: vec4<f32>,  // x, y, width, height
          strokeData: vec4<f32>,  // strokeWidth, padding...
        }
        
        @group(1) @binding(0) var<storage, read> pageInstances: array<PageInstance>;
        
        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) color: vec4<f32>,
          @location(1) localPos: vec2<f32>,
          @location(2) @interpolate(flat) instanceIndex: u32,
        }
        
        @vertex
        fn vertexMain(
          @builtin(vertex_index) vertexIndex: u32,
          @builtin(instance_index) instanceIndex: u32
        ) -> VertexOutput {
          var output: VertexOutput;
          let page = pageInstances[instanceIndex];
          let pos = getNormalizedQuadPosition(vertexIndex);
          
          // Calculate the outer rectangle (includes stroke)
          let strokeWidth = page.strokeData.x;
          let halfStroke = strokeWidth * 0.5;
          let outerX = page.dimensions.x - halfStroke;
          let outerY = page.dimensions.y - halfStroke;
          let outerWidth = page.dimensions.z + strokeWidth;
          let outerHeight = page.dimensions.w + strokeWidth;
          
          // Transform from [-1, 1] to actual rectangle coordinates
          let worldX = outerX + (pos.x + 1.0) * 0.5 * outerWidth;
          let worldY = outerY + (pos.y + 1.0) * 0.5 * outerHeight;
          
          output.position = frameUniforms * vec4<f32>(worldX, worldY, 0.0, 1.0);
          output.position.z = 0.99998; // Page strokes slightly in front of page fills
          output.color = page.color;
          output.localPos = pos;
          output.instanceIndex = instanceIndex;
          
          return output;
        }
        
        @fragment
        fn fragmentMain(
          input: VertexOutput,
          @builtin(front_facing) frontFacing: bool
        ) -> @location(0) vec4<f32> {
          let page = pageInstances[input.instanceIndex];
          
          // Calculate position in rectangle space [0, 1]
          let rectPos = (input.localPos + 1.0) * 0.5;
          
          // Calculate the stroke ratio based on actual stroke width and page dimensions
          let strokeWidth = page.strokeData.x;
          let pageWidth = page.dimensions.z;
          let pageHeight = page.dimensions.w;
          
          // Calculate stroke ratios for x and y separately to handle non-square pages
          let strokeRatioX = strokeWidth / (pageWidth + strokeWidth);
          let strokeRatioY = strokeWidth / (pageHeight + strokeWidth);
          
          // Check if we're inside the inner rectangle (should be discarded)
          if (rectPos.x > strokeRatioX && rectPos.x < (1.0 - strokeRatioX) &&
              rectPos.y > strokeRatioY && rectPos.y < (1.0 - strokeRatioY)) {
            discard;
          }
          
          return premultiplyAlpha(input.color);
        }
      