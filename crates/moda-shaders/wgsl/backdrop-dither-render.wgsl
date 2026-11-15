
        struct VSOut {
          @builtin(position) position: vec4<f32>,
        };

        struct DitherUniforms {
          params: vec4<f32>, // x: pixelSizePx; y: brightness offset; z: contrast; w: pattern
          dims: vec4<u32>,   // x: texture width; y: texture height; z: grid width; w: grid height
        };

        @group(0) @binding(0) var<uniform> u: DitherUniforms;
        @group(0) @binding(1) var<storage, read> gray: array<f32>;
        @group(0) @binding(2) var inputTexture: texture_2d<f32>;

        @vertex
        fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
          var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 3.0, -1.0),
            vec2<f32>(-1.0,  3.0)
          );
          var out: VSOut;
          out.position = vec4<f32>(pos[vid], 0.0, 1.0);
          return out;
        }

        @fragment
        fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
          let pixelSize = max(u.params.x, 1.0);
          let pixelX = floor(fragCoord.x);
          let pixelY = floor(fragCoord.y);
          let x = min(u32(floor(pixelX / pixelSize)), u.dims.z - 1u);
          let y = min(u32(floor(pixelY / pixelSize)), u.dims.w - 1u);
          let v = clamp(gray[y * u.dims.z + x], 0.0, 1.0);
          let alpha = textureLoad(inputTexture, vec2<i32>(i32(fragCoord.x), i32(fragCoord.y)), 0).a;
          // Output premultiplied alpha to match the source texture (and the
          // pixelate pass), since the composite blends with srcFactor 'one'.
          // Without the * alpha, translucent/empty backdrop regions inside the
          // shape get an additive grayscale haze.
          return vec4<f32>(vec3<f32>(v) * alpha, alpha);
        }
      