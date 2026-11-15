
        struct VSOut {
          @builtin(position) position : vec4<f32>,
          @location(0) uv : vec2<f32>,
        };

        @vertex
        fn vs(@builtin(vertex_index) vid : u32) -> VSOut {
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

        @group(0) @binding(0) var uSampler: sampler;
        @group(0) @binding(1) var uTexture: texture_2d<f32>;

        @fragment
        fn fs(in: VSOut) -> @location(0) vec4<f32> {
          // Flip Y to account for render-to-texture vs screen sampling differences
          let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
          return textureSample(uTexture, uSampler, uv);
        }
      