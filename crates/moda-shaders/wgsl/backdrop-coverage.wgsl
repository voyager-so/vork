
        struct CoverageUniforms {
          // x: sample origin X (target px), y: origin Y, z: 1/sampleWidth, w: 1/sampleHeight
          originInvSize: vec4<f32>,
        };

        @group(0) @binding(0) var<uniform> cov: CoverageUniforms;

        @vertex
        fn vs(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
          // Target-pixel -> sample-normalized (0..1), then to clip space with the
          // Y flip so sample texel (0,0) is the top-left, matching the copied
          // backdrop / effect textures the composite samples with the same uv.
          let s = (pos - cov.originInvSize.xy) * cov.originInvSize.zw;
          return vec4<f32>(s.x * 2.0 - 1.0, 1.0 - s.y * 2.0, 0.0, 1.0);
        }

        @fragment
        fn fs() -> @location(0) vec4<f32> {
          return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
      