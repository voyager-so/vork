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


// Main shader code

        // Frame-level uniforms
        @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>; // viewProjection
        // Mask sampling (optional). Group(2)=sampler+texture+uniforms
        struct MaskUniforms {
          maskActive: u32,
          maskOrigin: vec2<f32>,
          maskInvSize: vec2<f32>,
        }
        @group(2) @binding(0) var maskSampler: sampler;
        @group(2) @binding(1) var maskTexture: texture_2d<f32>;
        @group(2) @binding(2) var<uniform> maskUniforms: MaskUniforms;
        
        // Storage buffer for path instances
        struct PathData {
          model: mat4x4<f32>,      // 16 floats
          fillColor: vec4<f32>,    // 4 floats
          strokeColor: vec4<f32>,  // 4 floats
          pathInfo: vec4<f32>,     // triangleOffset, triangleCount, strokeWidth, depth - 4 floats
          pageClipBounds: vec4<f32>, // x, y, width, height - 4 floats
        }
        
        @group(1) @binding(0) var<storage, read> paths: array<PathData>;
        
        // Stroke depth offset constant
        const STROKE_DEPTH_OFFSET: f32 = -0.000001490116119384765625;
        
        struct VertexInput {
          @location(0) position: vec2<f32>,
          @location(1) vertexType: f32, // 0 = fill, 1 = stroke
          @location(2) owner: u32, // phase-local packed instance, carried by every vertex
        }
        
        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) @interpolate(flat) instanceId: u32,
          @location(1) color: vec4<f32>,
          @location(2) @interpolate(flat) isStroke: f32,
          @location(3) worldPos: vec2<f32>, // Add world position for clipping
          @location(4) @interpolate(flat) clip: vec4<f32>,
        }
        
        @vertex
        fn vertexMain(
          vertex: VertexInput,
          @builtin(instance_index) instanceIndex: u32
        ) -> VertexOutput {
          var output: VertexOutput;
          let owner = select(vertex.owner, instanceIndex, vertex.owner == 0xffffffffu);
          let path = paths[owner];
          
          // Use the vertex position from the buffer
          let localPos = vertex.position;
          
          // Transform to world space
          let worldPos = path.model * vec4<f32>(localPos.x, localPos.y, 0.0, 1.0);
          output.worldPos = worldPos.xy; // Pass world position to fragment shader
          output.position = frameUniforms * worldPos;
          
          // Use pre-calculated depth value
          // For stroke vertices, add a small offset to render on top of fill
          let depthOffset = select(0.0, STROKE_DEPTH_OFFSET, vertex.vertexType > 0.5);
          output.position.z = path.pathInfo.w + depthOffset;
          
          output.instanceId = owner;
          output.isStroke = vertex.vertexType;
          
          // Select color based on vertex type
          if (vertex.vertexType > 0.5) {
            output.color = path.strokeColor;
          } else {
            output.color = path.fillColor;
          }
          // Pass page clip bounds to fragment via flat varying
          output.clip = path.pageClipBounds;
          
          return output;
        }
        
        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
          // Discard if color is fully transparent
          if (input.color.a <= 0.0) {
            discard;
          }
          
          // Use passed clip bounds
          let bounds = input.clip;
          
          // Check if fragment is within page bounds
          if (input.worldPos.x < bounds.x || 
              input.worldPos.x > bounds.x + bounds.z ||
              input.worldPos.y < bounds.y || 
              input.worldPos.y > bounds.y + bounds.w) {
            discard;
          }
          
          // Optional mask sample
          if (maskUniforms.maskActive == 1u) {
            let uv = (input.worldPos - maskUniforms.maskOrigin) * maskUniforms.maskInvSize;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
              discard;
            }
            let m = textureSample(maskTexture, maskSampler, uv).r;
            if (m <= 0.0) { discard; }
            var color = input.color;
            color.a *= m;
            return premultiplyAlpha(color);
          }
          return premultiplyAlpha(input.color);
        }
      