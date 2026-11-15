// Install-time SH, normalized GGX convolution and Schlick moments.
import vork_constants;
import openpbr_core;
import vork_prelude;
import environment_proposal;

[[vk::binding(0, 0)]] Texture2DArray<uint4> source : register(t0);
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> output : register(u1);
// settings.z is the finite positive reciprocal HDR scale, constructed on the host.
// distribution_size.w offsets bounded dispatches into the output texture.
struct FilterParams { float4 settings; float4 sh[4]; float4 distribution_size; };
[[vk::binding(2, 0)]] ConstantBuffer<FilterParams> params : register(b0);
[[vk::binding(3, 0)]] StructuredBuffer<float2> environment_distribution : register(t1);

float3 half_vector(float2 x,float r) {
    let a=specular_alpha(r);let c=sqrt((1.-x.y)/(1.+(a*a-1.)*x.y));let s=sqrt(max(0.,1.-c*c));
    return float3(s*cos(2.*REG_PI*x.x),s*sin(2.*REG_PI*x.x),c);
}
// Dense near the minimum-alpha grazing transition; lookup uses its inverse.
float brdf_cosine(uint column) {
    let t=float(column)/float(REG_BRDF_SIZE-1u);let t2=t*t;return t2*t2;
}
float brdf_alpha(uint row) {
    if (row==0u) {return REG_GGX_ALPHA_MIN;}
    if (row==REG_BRDF_SIZE-1u) {return 1.;}
    return exp2(log2(REG_GGX_ALPHA_MIN)*(1.-float(row)/float(REG_BRDF_SIZE-1u)));
}
[shader("compute")]
[numthreads(REG_COMPUTE_TILE, REG_COMPUTE_TILE, 1)]
void filter_main(uint3 id : SV_DispatchThreadID) {
    id.y+=uint(params.distribution_size.w);
    uint width, height; output.GetDimensions(width, height);
    let size=uint2(width, height);if (any(id.xy>=size)) {return;}
    let coord=(float2(id.xy)+.5)/float2(size);
    let theta=REG_PI*coord.y;let phi=2.*REG_PI*(coord.x-.5);
    let n=float3(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));
    let t=normalize(cross((abs(n.y)<.999 ? float3(0.,1.,0.) : float3(1.,0.,0.)),n));let b=cross(n,t);
    var sum=float3(0.);var weight=0.;
    let mode=uint(params.settings.y);
    if (mode==2u && id.y>=REG_BRDF_SIZE) {
        let n=REG_BRDF_ALBEDO_SIZE;
        let cell=uint2(id.x%n,(id.y-REG_BRDF_SIZE)%n);
        let slice=((id.y-REG_BRDF_SIZE)/n)*2u+id.x/n;
        if (slice>REG_BRDF_IOR_STEPS) {output[id.xy]=float4(0.);return;}
        let no=max(float(cell.x)/float(n-1u),REG_GGX_ALPHA_MIN);
        let r=float(cell.y)/float(n-1u);
        let q=float(slice)/float(REG_BRDF_IOR_STEPS);
        let eta=select(slice==REG_BRDF_IOR_STEPS,0.,q/max(1.-q,REG_POSITIVE_EPSILON));
        let albedo=dielectric_directional_albedo(no,r,eta,REG_DIELECTRIC_ALBEDO_SAMPLES);
        output[id.xy]=float4(albedo,0.,0.,1.);return;
    }
    if (mode==1u) {
        output[id.xy]=float4(max(params.sh[0].rgb+params.sh[1].rgb*n.x+params.sh[2].rgb*n.y+params.sh[3].rgb*n.z,float3(0.)),1.);
        return;
    }
    if (mode==0u && params.settings.x==0.) {output[id.xy]=float4(environment_radiance(source,coord,0u,true)*params.settings.z,1.);return;}
    for(var i=0u;i<REG_ENVIRONMENT_SAMPLES;i++) {
        let x=float2((float(i)+.5)/float(REG_ENVIRONMENT_SAMPLES),float(reversebits(i))*REG_RADICAL_INVERSE_SCALE);
        if (mode==2u) {
            let nv=brdf_cosine(id.x);let alpha=brdf_alpha(id.y);
            float vh;let g=ggx_visible_reflection(nv,alpha,x,vh);let f=pow(1.-vh,5.);
            // z: the F82-tint metal term's moment, μ(1-μ)^6 (metal_f82).
            sum+=float3((1.-f)*g,f*g,vh*pow(1.-vh,6.)*g);
        } else {
            // Balance heuristic: half radiance proposal, half GGX.
            let random=float2(frac(x.x*2.),x.y);
            let local=reflect(float3(0.,0.,-1.),half_vector(random,params.settings.x));
            var d=t*local.x+b*local.y+n*local.z;
            var environment_density=0.;
            if (x.x<.5) {d=environment_proposal_sample(environment_distribution,uint2(params.distribution_size.xy),random,environment_density);}
            else {environment_density=environment_proposal_pdf(environment_distribution,uint2(params.distribution_size.xy),d);}
            let cosine=max(dot(n,d),0.);
            if (cosine>0.) {
                let nh=sqrt(max(0.,(1.+cosine)*.5));
                let a=specular_alpha(params.settings.x);let a2=a*a;
                let denominator=nh*nh*(a2-1.)+1.;
                let kernel=a2/(4.*REG_PI*denominator*denominator);
                let proposal=kernel;
                let density=.5*(proposal+environment_density);
                let w=kernel/max(density,REG_POSITIVE_EPSILON);
                let radiance=environment_radiance(source,equirect_uv(d),0u,true)*params.settings.z;
                // Seed from the first accepted tap: a reciprocal-lowered w/w
                // need not be exactly one. Subsequent equal taps have zero delta.
                if (weight==0.) {sum=radiance;}
                else {sum+=(radiance-sum)*(w/(weight+w));}
                weight+=w;
            }
        }
    }
    if (mode==2u) {sum/=float(REG_ENVIRONMENT_SAMPLES);}
    var average=mode==2u ? 0. : 1.;
    if (mode==2u && id.x==0u) {
        // The first column's unused alpha stores E_avg(roughness). Compute it
        // once per roughness row, with the same endpoint grid/quadrature as RGB.
        // Other alpha values remain unused and zero.
        average=0.;let alpha=brdf_alpha(id.y);
        for (uint j=0u;j<REG_BRDF_SIZE;j++) {
            let nv=brdf_cosine(j);
            let interval=.5*(brdf_cosine(min(j+1u,REG_BRDF_SIZE-1u))-brdf_cosine(j>0u ? j-1u : 0u));
            for (uint i=0u;i<REG_ENVIRONMENT_SAMPLES;i++) {
                let u=float2((float(i)+.5)/float(REG_ENVIRONMENT_SAMPLES),float(reversebits(i))*REG_RADICAL_INVERSE_SCALE);
                float vh;average+=interval*2.*nv*ggx_visible_reflection(nv,alpha,u,vh);
            }
        }
        average/=float(REG_ENVIRONMENT_SAMPLES);
    }
    output[id.xy]=float4(sum,average);
}
