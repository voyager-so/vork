BC7Encode.hlsl is from microsoft/DirectXTex, at the commit in UPSTREAM, under
its MIT license (LICENSE). Source: DirectXTex/Shaders/BC7Encode.hlsl.

Local adaptations: declare reused loop indices in Slang's lexical scope; add
explicit Vulkan bindings and distinct Metal buffer registers; clamp partial mip
block reads to authored edge texels; round normalized loads to their original
8-bit values; normalize unused alpha to opaque for RGB-only material taps. Suppress Slang's intentional integer-conversion warnings in this
third-party module. Mode-6 endpoint quantization requires odd p-bits for opaque
blocks, preserving alpha 255 independently of RGB error. The host uses modes 4/5/6; exhaustive partitioned-mode searches were slower than the SIMD CPU encoder. Dispatches pad the block count to four, so
all group barriers execute uniformly, including the smallest mip levels.

The host adds a separate principal-axis / least-squares mode5/6 fit in
../../bc7_refine.slangh, retaining the original block unless decoded error
on the sampled channels decreases. See ../ispc_refine for its MIT reference.
