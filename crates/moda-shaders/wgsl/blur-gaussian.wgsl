
      struct BlurUniforms {
        // x: direction (0 = horizontal, 1 = vertical)
        // y: kernel size
        // z: texture width
        // w: texture height
        params: vec4<f32>,
        // Packed weights and offsets: xy = weight and offset for sample i
        weightsAndOffsets: array<vec4<f32>, 16>,
      }

      @group(0) @binding(0) var<uniform> blur: BlurUniforms;
      @group(0) @binding(1) var inputTexture: texture_2d<f32>;
      @group(0) @binding(2) var textureSampler: sampler;

      @fragment
      fn gaussianBlur(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let direction = blur.params.x;
        let kernelSize = i32(blur.params.y);
        let texelSize = vec2<f32>(1.0 / blur.params.z, 1.0 / blur.params.w);
        
        var color = vec4<f32>(0.0);
        
        // Sample direction vector
        let sampleDir = select(
          vec2<f32>(texelSize.x, 0.0),
          vec2<f32>(0.0, texelSize.y),
          direction > 0.5
        );
        
        // Center sample
        let centerWeight = blur.weightsAndOffsets[0].x;
        color += textureSample(inputTexture, textureSampler, uv) * centerWeight;
        
        // Bilateral samples using linear filtering
        for (var i = 1; i < kernelSize; i++) {
          let weightOffset = blur.weightsAndOffsets[i];
          let weight = weightOffset.x;
          let offset = weightOffset.y;
          
          // Sample in both directions
          let coord1 = uv + sampleDir * offset;
          let coord2 = uv - sampleDir * offset;
          
          color += textureSample(inputTexture, textureSampler, coord1) * weight;
          color += textureSample(inputTexture, textureSampler, coord2) * weight;
        }
        
        return color;
      }
    