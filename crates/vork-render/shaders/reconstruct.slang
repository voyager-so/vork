// Camera sample placement importance-samples the Gaussian film kernel.
// Normalize that estimate exactly once. Spatial variance of neighbouring
// pixels is image detail, not an estimate of Monte Carlo uncertainty.
import vork_constants;
struct Params { uint width; uint height; uint samples; uint unused; };
[[vk::binding(0,0)]] ConstantBuffer<Params> params : register(b0);
[[vk::binding(1,0)]] StructuredBuffer<float4> accumulation : register(t1);
[[vk::binding(2,0)]] RWStructuredBuffer<float4> destination : register(u2);
[shader("compute")] [numthreads(REG_COMPUTE_TILE,REG_COMPUTE_TILE,1)]
void finish(uint3 id:SV_DispatchThreadID) {
    if(id.x>=params.width || id.y>=params.height) return;
    let i=id.y*params.width+id.x;
    // Preserve premultiplied RGB, alpha, HDR, and empty-background samples.
    // This is the same scene-linear estimate supplied to OIDN when enabled.
    destination[i]=accumulation[i]/float(params.samples);
}
