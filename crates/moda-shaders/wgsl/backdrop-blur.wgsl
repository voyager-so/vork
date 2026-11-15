
        struct BlurUniforms {
          // x: direction (0 = horizontal, 1 = vertical)
          // y: kernel size
          // z: texture width
          // w: texture height
          params: vec4<f32>,
          // Packed weights and offsets: xy = weight and offset for sample i
          weightsAndOffsets: array<vec4<f32>, 16>,
        };

        @group(0) @binding(0) var<uniform> blur: BlurUniforms;
        @group(0) @binding(1) var inputSampler: sampler;
        @group(0) @binding(2) var inputTexture: texture_2d<f32>;

        struct VSOut {
          @builtin(position) position: vec4<f32>,
          @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
          var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 3.0, -1.0),
            vec2<f32>(-1.0,  3.0)
          );
          var uv = array<vec2<f32>, 3>(
            vec2<f32>(0.0, 0.0),
            vec2<f32>(2.0, 0.0),
            vec2<f32>(0.0, 2.0)
          );
          var out: VSOut;
          out.position = vec4<f32>(pos[vid], 0.0, 1.0);
          out.uv = uv[vid];
          return out;
        }

        fn gaussianBlurSample(uv: vec2<f32>) -> vec4<f32> {
          let direction = blur.params.x;
          let kernelSize = i32(blur.params.y);
          let texelSize = vec2<f32>(1.0 / blur.params.z, 1.0 / blur.params.w);

          var color = vec4<f32>(0.0);

          let sampleDir = select(
            vec2<f32>(texelSize.x, 0.0),
            vec2<f32>(0.0, texelSize.y),
            direction > 0.5
          );

          let centerWeight = blur.weightsAndOffsets[0].x;
          color += textureSample(inputTexture, inputSampler, uv) * centerWeight;

          for (var i = 1; i < min(kernelSize, 16); i++) {
            let weightOffset = blur.weightsAndOffsets[i];
            let weight = weightOffset.x;
            let offset = weightOffset.y;

            let coord1 = uv + sampleDir * offset;
            let coord2 = uv - sampleDir * offset;

            color += textureSample(inputTexture, inputSampler, coord1) * weight;
            color += textureSample(inputTexture, inputSampler, coord2) * weight;
          }

          return color;
        }

        @fragment
        fn fs(in: VSOut) -> @location(0) vec4<f32> {
          return gaussianBlurSample(in.uv);
        }
      