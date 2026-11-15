// Irradiance volume update (probes.rs). `trace` shades one probe ray per thread
// (spherical Fibonacci, fixed per-probe rotation) through the shared hit
// evaluation (hit.slang, hit_lighting.slang); hits read the previous bounce's
// volume (group 2), so successive bounces gather further light. `gather`
// accumulates one record vec4 or depth-moment texel per thread over a batch of
// rays and writes the probe's L1 record and moments after the last batch.
// No workgroup barriers: each thread owns its output.
import vork_constants;
import vork_prelude;
import openpbr_core;
import frame;
import vork_ray_sampling;
import vork_ray_alpha;
import openpbr_inputs;
import bsdf_response;
import closure;
import variant;
import material;
import surface;
import hit;
import ray_coverage;
import hit_lighting;
import emitter_mesh;
import surface_lighting;
import subsurface_transport;
import probes;
// probes::update_entries: the scene globals (frame.slang) occupy bindings 0-9.
[[vk::binding(11,0)]] RaytracingAccelerationStructure ray_scene : register(t6);
[[vk::binding(12,0)]] StructuredBuffer<uint> ray_alpha : register(t7);
// Per probe and batch ray: radiance/distance, then validity (x: near a
// coverage-accepted boundary). Optional emitter radiance and direction follow.
// Validity has its own record: darkness and zero-distance hits are not sentinels.
[[vk::binding(13,0)]] RWStructuredBuffer<float4> traced : register(u8);
// Per probe: REG_PROBE_VEC4S record sums, then the depth-moment texel sums.
[[vk::binding(14,0)]] RWStructuredBuffer<float4> sums : register(u9);
[[vk::binding(15,0)]] RWStructuredBuffer<float4> probe_output : register(u10);
[[vk::binding(16,0)]] RWStructuredBuffer<float2> visibility_output : register(u11);
// x: this batch's first ray; y: rays per batch; z: gathered slots per probe
// (all records and moments on bounce zero, just records on later bounces).
// w: later lighting generation; invalid preceding support cannot transport light.
struct ProbeBatch {uint4 rays;};
[[vk::binding(17,0)]] ConstantBuffer<ProbeBatch> batch : register(b12);
[[vk::binding(18,0)]] StructuredBuffer<float> probe_emitter_weights : register(t13);
#ifdef VORK_METAL
[[vk::binding(0,3)]] ParameterBlock<RayHitBuffersTable> ray_hit_buffers : register(VORK_HIT_REGISTER);
#else
[[vk::binding(0,3)]] StructuredBuffer<uint> ray_hit_buffers[REG_RAY_HIT_BUFFERS];
#endif

// Interleaved Fibonacci strata jointly cover the full exposure's angular
// quadrature. Each instantaneous field is normalized by its own ray count.
// Zero policy reserves the full budget for an animated shutter. Its current
// exposure selects the same ray count and dimensions as a fixed-width shutter.
uint probe_strata() {let fixed=probe_grid().visibility_policy.x;return fixed==0u ? max(uint(frame.temporal.z),1u) : fixed;}
uint probe_ray_count() {let rays=probe_grid().counts.w;let strata=probe_strata();return probe_grid().visibility_policy.x==0u ? rays/strata+uint(rays%strata!=0u) : rays;}
uint probe_sample_count() {return probe_ray_count()*probe_strata();}
uint probe_sample_index(uint ray) {let strata=probe_strata();return ray*strata+uint(frame.temporal.x)%strata;}
typedef RayGeometry ProbeGeometry;
#ifdef VORK_PROGRAM_TRACE
import geometric_trace;
export TraceHit native_geometric_trace(float3 origin,float3 direction,float minimum,float maximum,uint mask,uint object) {
    return geometric_trace<ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_scene,origin,direction,minimum,maximum,mask,object);
}
#endif

// Capture hits use the shared correlated source/BSDF visibility estimator.
struct ProbeShadows<V:IVariant,P:IProgram> : IShadows {
    static const bool probe_rays=true;
    static bool probe_visible(float3 world,float3 normal,float3 position_error,float3 probe) {return probe_connection_visible<P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,world,normal,position_error,probe);}
    static uint integration_count(uint index,out float3 normalization) {let count=index==0xffffffffu ? sample_count() : 1u;normalization=float3(float(count));return count;}
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {if(index==0xffffffffu) {index=tap;}return float3(1.);}
    static uint sample_count() {return probe_sample_count();}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 position_error,uint index,uint count,bool fibre=false) {return hit_shadow_direction<P,ProbeGeometry,V>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,light,world,normal,position_error,direction,distance,index,count,true);}
    static const bool subsurface=true;
    static float thickness(uint object,float3 world,float3 direction) {return subsurface_exit(ray_scene,object,world,direction);}
};
// Retained transport has a fixed world reference, independent of the camera
// and of unrelated brick allocation. Surface errors use actual hit geometry.
export float3 ray_eye() {return float3(0.);}
static const uint PROBE_TEXELS=REG_PROBE_VISIBILITY_SIDE*REG_PROBE_VISIBILITY_SIDE;
static const uint SOURCE_SLOTS=REG_PROBE_VEC4S+PROBE_TEXELS;
static const uint PROBE_SLOTS=SOURCE_SLOTS+3u;

uint probe_count() {return probe_grid().field.z;}
uint probe_thread(uint3 group,uint thread) {return (group.x+group.y*REG_PROBE_DISPATCH_WIDTH)*REG_PROBE_GROUP+thread;}
float3 spherical_fibonacci(uint i,uint n) {
    let phi=REG_TAU*frac(float(i)*REG_SAMPLE_LATTICE_RATIO);
    let z=1.-(2.*float(i)+1.)/float(n);let r=sqrt(max(1.-z*z,0.));
    return float3(cos(phi)*r,sin(phi)*r,z);
}
// World-lattice identity keeps captures stable when unrelated bricks are added.
uint probe_identity(uint probe) {let p=probe_grid().visibility_policy.y!=0u ? probe_grid_data[probe_grid().field.y+probe].counts.xyz : asuint(probe_position_index(probe));let level=probe_grid_data[probe_grid().field.y+probe].field.w;return pixel_hash(p.x)^pixel_hash(p.y+1u)^pixel_hash(p.z+2u)^(level==0u ? 0u : pixel_hash(level+3u)); }
// A fixed random rotation per probe (Shoemake's uniform quaternion).
float3x3 probe_rotation(uint probe) {
    probe=probe_identity(probe);
    let u=float3(float(screen_ray_hash(uint2(probe,0u),0u,REG_SAMPLING_STREAM_RAY_GI)),float(screen_ray_hash(uint2(probe,1u),0u,REG_SAMPLING_STREAM_RAY_GI)),float(screen_ray_hash(uint2(probe,2u),0u,REG_SAMPLING_STREAM_RAY_GI)))*REG_RADICAL_INVERSE_SCALE;
    let a=sqrt(1.-u.x);let b=sqrt(u.x);
    let x=a*sin(REG_TAU*u.y);let y=a*cos(REG_TAU*u.y);let z=b*sin(REG_TAU*u.z);let w=b*cos(REG_TAU*u.z);
    return float3x3(float3(1.-2.*(y*y+z*z),2.*(x*y+w*z),2.*(x*z-w*y)),
                    float3(2.*(x*y-w*z),1.-2.*(x*x+z*z),2.*(y*z+w*x)),
                    float3(2.*(x*z+w*y),2.*(y*z-w*x),1.-2.*(x*x+y*y)));
}
float3 probe_direction(uint probe,uint ray) {return probe_world_vector(mul(spherical_fibonacci(probe_sample_index(ray),probe_sample_count()),probe_rotation(probe)));}
uint probe_trace_stride() {return probe_grid().visibility_policy.w!=0u ? 4u : 2u;}
float probe_mis(float sampled,float other) {let ratio=other/sampled;return 1./(1.+ratio*ratio);}
float probe_emitter_pdf<P:IProgram>(uint instance,uint primitive,uint object,float2 barycentrics,float3x4 inverse,float3 from) {
    let base=probe_grid().visibility_policy.w;
    if(base==0u) {return 0.;}
    let geometry=hit_emitter<ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),instance,primitive,object,inverse,barycentrics);
    return emitter_mesh_probability<ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),base,instance,primitive,probe_emitter_weights)*emitter_mesh_pdf<ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),base,instance,geometry.area,from,geometry.point,geometry.face);
}
// One source connection at a retained probe vertex. Only emission participates
// in this proposal: reflected light keeps the uniform directional estimator and
// the preceding bounce's field. Adding another hit-vertex NEE would count that
// same previous-bounce source light twice.
float3 probe_source<P:IProgram>(uint probe,uint ray,float3 from,out float3 direction) {
    direction=float3(0.);let base=probe_grid().visibility_policy.w;
    let budget=probe_sample_count();let sample=probe_sample_index(ray);
    let uv=sample_2d(uint2(probe_identity(probe),0u),sample,budget,REG_SAMPLING_STREAM_PROBE_EMITTER);
    let extra=sample_2d(uint2(probe_identity(probe),0u),sample,budget,REG_SAMPLING_STREAM_PROBE_TRIANGLE);
    let selected=emitter_mesh_sample<ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),base,float3(extra.x,uv),probe_emitter_weights);
    if ((objects[selected.object].metadata.y&REG_RAY_DIFFUSE)==0u) {return float3(0.);}
    let inverse=(float3x4)object_normal(selected.object);
    let geometry=hit_emitter<ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),selected.instance,selected.primitive,selected.object,inverse,selected.barycentrics);
    let delta=geometry.point-from;let distance=length(delta);
    if(distance<=0. || distance>probe_spacing(probe).w) {return float3(0.);}
    direction=delta/distance;
    let pdf=probe_emitter_pdf<P>(selected.instance,selected.primitive,selected.object,selected.barycentrics,inverse,from);
    if(pdf<=0.) {return float3(0.);}
    let intersection=RayIntersection(selected.instance,selected.object,selected.primitive,selected.barycentrics);
    if(!candidate_accepts<P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_alpha,intersection,dot(geometry.face,-direction)>0.,inverse,direction,float2(0.))) {return float3(0.);}
    let source=hit_surface<Unspecialized,P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),selected.instance,selected.primitive,selected.object,inverse,selected.barycentrics,direction,float2(0.));
    if(source.s.discarded) {return float3(0.);}
    let emitted=emitter_mesh_radiance<Unspecialized,P>(source);
    if(all(emitted==float3(0.))) {return float3(0.);}
    // End on the receiver-facing side of the sampled triangle. The candidate
    // policy and event mask match directional capture, including cutout programs.
    let end=surface_ray_origin(geometry.point,surface_position_error(selected.object,geometry.point),source.face,-direction);
    let connection=end-from;let extent=length(connection);if(extent==0.) {return float3(0.);}
    RayDesc desc;desc.Origin=from;desc.Direction=connection/extent;desc.TMin=0.;desc.TMax=extent;
    if(diffuse_ray_occluded<P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,desc,REG_RAY_GI_OCCLUDER,float2(0.))) {return float3(0.);}
    let uniform=1./(4.*REG_PI);
    var radiance=emitted;
    return radiance*probe_mis(pdf,uniform)*uniform/pdf;
}
// Scenes with programs compile their own entry with their Programs (probes.rs).
struct ProbeSurfaceHit { RayIntersection hit; float3x4 inverse; float distance; bool found; };
// A touching front/back pair is one boundary within the geometry's arithmetic
// uncertainty. Query each side independently so accepting one cannot prune the
// other, then prefer the incident-facing surface when their intervals overlap.
// Keep hardware closest-hit pruning within each side; no scene-unit epsilon.
ProbeSurfaceHit probe_surface_hit<P:IProgram,let directional:bool>(RayDesc desc,uint probe,uint ray,float origin_error,out bool near_boundary,out float backface) {
    near_boundary=false;backface=REG_FP32_MAX;
    var front=(ProbeSurfaceHit)0;var back=(ProbeSurfaceHit)0;
    front.distance=desc.TMax;back.distance=desc.TMax;
    [loop] for(uint side=0u;side<2u;side++) {
        RayQuery<RAY_FLAG_NONE> query;
        let cull=side==0u ? RAY_FLAG_CULL_BACK_FACING_TRIANGLES : RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
        query.TraceRayInline(ray_scene,RAY_FLAG_FORCE_NON_OPAQUE|cull,directional ? REG_RAY_SPECULAR : REG_RAY_GI_OCCLUDER,desc);
        while(query.Proceed()) {
            // Double-sided instance flags disable hardware culling. These two
            // searches still own disjoint geometric sides for boundary ordering.
            if(query.CandidateTriangleFrontFace()!=(side==0u)) continue;
            let hit=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
            let accepts=directional
                ? reflection_candidate_accepts<P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_alpha,hit,query.CandidateTriangleFrontFace(),query.CandidateWorldToObject3x4(),desc.Direction,float2(0.),uint2(probe_identity(probe),0u),probe_sample_index(ray),probe_sample_count())
                : candidate_coverage<P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),ray_alpha,hit,query.CandidateWorldToObject3x4(),desc.Direction,float2(0.))>=1.;
            if(!accepts) {continue;}
            near_boundary=near_boundary || query.CandidateTriangleRayT()<=origin_error+length(surface_position_error(hit.instance_custom_data,probe_position_index(probe)));
            if(directional || material_covers_side<Unspecialized>(hit.instance_custom_data,query.CandidateTriangleFrontFace())) {query.CommitNonOpaqueTriangleHit();}
            else {backface=min(backface,max(query.CandidateTriangleRayT()-origin_error,0.));}
        }
        if(query.CommittedStatus()!=COMMITTED_NOTHING) {
            let hit=RayIntersection(query.CommittedInstanceIndex(),query.CommittedInstanceID(),query.CommittedPrimitiveIndex(),query.CommittedTriangleBarycentrics());
            let result=ProbeSurfaceHit(hit,query.CommittedWorldToObject3x4(),query.CommittedRayT(),true);
            if(side==0u) {front=result;} else {back=result;}
        }
    }
    if(!front.found) {return back;}
    if(!back.found) {return front;}
    let front_error=length(surface_position_error(front.hit.instance_custom_data,desc.Origin+desc.Direction*front.distance));
    let back_error=length(surface_position_error(back.hit.instance_custom_data,desc.Origin+desc.Direction*back.distance));
    return front.distance<=back.distance+front_error+back_error ? front : back;
}
void probe_trace<V:IVariant,P:IProgram>(uint3 group,uint thread) {
    let index=probe_thread(group,thread);let size=batch.rays.y;
    let probe=index/size;let ray=batch.rays.x+index%size;
    if (probe>=probe_count() || ray>=probe_ray_count()) {return;}
    // Invalid captures cannot contribute to any consumer or later generation.
    // Do not repeat their material/light/source work after validity is known.
    if((batch.rays.w&1u)!=0u && probe_irradiance[probe].support.x<probe_grid().bias.w) {return;}
    let indirect_only=(batch.rays.w&2u)!=0u;
    let position=probe_position_index(probe);
    let direction=probe_direction(probe,ray);let reach=probe_spacing(probe).w;
    // Straddle the lattice position's arithmetic uncertainty interval. A query
    // starting exactly on a triangle may omit T==0 entirely; moving the start
    // to the interval's back endpoint makes that boundary observable. This is
    // a gamma-bound on placement arithmetic, not a physical clearance epsilon.
    let origin_error=length(REG_FP32_GAMMA7*(abs(position)+probe_spacing(probe).xyz));
    RayDesc desc;desc.Origin=position-direction*origin_error;desc.Direction=direction;desc.TMin=0.;desc.TMax=reach+origin_error;
    float backface;bool near_boundary;
    let nearest=probe_surface_hit<P,false>(desc,probe,ray,origin_error,near_boundary,backface);
    var result=float4(indirect_only ? float3(0.) : world_radiance(direction),reach);
    if (nearest.found) {
        let hit=nearest.hit;
        let distance=max(nearest.distance-origin_error,0.);
        result=float4(0.,0.,0.,distance);
        if (!near_boundary && ray_alpha[hit.instance_index*REG_RAY_SLOT_WORDS+2u]!=REG_RAY_NO_ALPHA) {
            let u=select(P.lobes,float(screen_ray_hash(uint2(probe_identity(probe),probe_sample_index(ray)),0u,REG_SAMPLING_STREAM_RAY_GI_LOBE))*REG_RADICAL_INVERSE_SCALE,0.);
            let h=hit_surface<V,P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),hit.instance_index,hit.primitive_index,hit.instance_custom_data,nearest.inverse,hit.barycentrics,direction,float2(0.),Lobe(select(P.lobes,LOBE_PICK,0u),u));
            float3 emission;
            var radiance=float3(0.);
            radiance=finalized_hit_radiance<V,ProbeShadows<V,P>,P>(h,uint2(probe_identity(probe),0u),probe_sample_index(ray),REG_RAY_DIFFUSE,emission,indirect_only,true);
            if(!indirect_only && probe_grid().visibility_policy.w!=0u) {
                let pdf=probe_emitter_pdf<P>(hit.instance_index,hit.primitive_index,hit.instance_custom_data,hit.barycentrics,nearest.inverse,position);
                radiance-=emission*(1.-probe_mis(1./(4.*REG_PI),pdf));
            }
            result=float4(radiance,distance);
        }
    }
    // Buried-probe validity/depth diagnostics retain the nearest boundary even
    // when that single-sided back face did not stop the radiance ray. Candidate
    // traversal already visits it: no second query or geometry representation.
    if (backface<result.w) {result.w=-backface*REG_PROBE_BACKFACE_DISTANCE;}
    let at=(probe*size+index%size)*probe_trace_stride();
    traced[at]=result;
    traced[at+1u]=float4(near_boundary ? 1. : 0.,direction);
    if(probe_grid().visibility_policy.w!=0u) {
        float3 source_direction;
        var source=float3(0.);source_direction=float3(0.);
        if(!near_boundary && !indirect_only) {source=probe_source<P>(probe,ray,position,source_direction);}
        traced[at+2u]=float4(source,0.);traced[at+3u]=float4(source_direction,0.);
    }
}
// One record vec4 (per channel: radiance sum and radiance-weighted direction
// sum; then front and back hit counts) or depth-moment texel per thread.
void probe_gather(uint3 group,uint thread) {
    let index=probe_thread(group,thread);
    let probe=index/batch.rays.z;let slot=index%batch.rays.z;
    if (probe>=probe_count()) {return;}
    if((batch.rays.w&1u)!=0u && probe_irradiance[probe].support.x<probe_grid().bias.w) {
        if(slot<REG_PROBE_VEC4S) {probe_output[probe*REG_PROBE_VEC4S+slot]=float4(0.);}
        return;
    }
    // Scratch retains its full per-probe stride across both dispatch shapes.
    let record=probe*PROBE_SLOTS+slot;
    let size=batch.rays.y;let first=batch.rays.x;let rays=probe_ray_count();
    let count=min(size,rays-first);
    var sum=select(first==0u,float4(0.),sums[record]);
    var axis=float3(0.);
    if (slot>=REG_PROBE_VEC4S) {
        let texel=slot-REG_PROBE_VEC4S;let side=REG_PROBE_VISIBILITY_SIDE;
        axis=decode_octahedron((float2(float(texel%side),float(texel/side))+.5)/float(side)*2.-1.);
    }
    for (uint k=0u;k<count;k++) {
        let at=(probe*size+k)*probe_trace_stride();let t=traced[at];
        // Misses travel exactly the capture distance; back faces are negative.
        if (slot==3u) {sum+=select(t.w<0.,float4(0.,1.,0.,0.),float4(select(t.w<probe_spacing(probe).w,1.,0.),0.,0.,0.));sum.z+=traced[at+1u].x;continue;}
        // The trace already computed this direction. Reusing its spare
        // validity lanes avoids repeating the rotation/hash/trigonometry for
        // every depth texel and RGB channel, without extra capture storage.
        let direction=traced[at+1u].yzw;
        if (slot<3u) {
            sum+=t[slot]*float4(1.,direction);
            if((batch.rays.w&2u)==0u && probe_grid().visibility_policy.w!=0u) {sum+=traced[at+2u][slot]*float4(1.,traced[at+3u].xyz);}
            continue;
        }
        let distance=abs(t.w);
        let w=pow(saturate(dot(axis,direction)),REG_PROBE_DEPTH_SHARPNESS);
        sum+=float4(w*distance,w*distance*distance,w,0.);
    }
    sums[record]=sum;
    if (first+count<rays) {return;}
    let n=1./float(rays);
    if (slot<3u) {
        if((batch.rays.w&1u)==0u) {sums[probe*PROBE_SLOTS+SOURCE_SLOTS+slot]=sum;}
        if((batch.rays.w&2u)!=0u) {sum+=sums[probe*PROBE_SLOTS+SOURCE_SLOTS+slot];}
        // E/pi(n) = mean radiance + 2 mean(radiance direction).n (L1 SH, Lambert-convolved).
        probe_output[probe*REG_PROBE_VEC4S+slot]=float4(sum.x*n,2.*n*sum.yzw);
    } else if (slot==3u) {
        let hits=sum.x+sum.y;
        probe_output[probe*REG_PROBE_VEC4S+3u]=float4(sum.z>0. ? 0. : select(hits>0.,sum.x/max(hits,1.),1.),0.,0.,0.);
    } else {
        let reach=probe_spacing(probe).w;
        visibility_output[probe*PROBE_TEXELS+slot-REG_PROBE_VEC4S]=select(sum.z>0.,sum.xy/max(sum.z,REG_POSITIVE_EPSILON),float2(reach,reach*reach));
    }
}
[shader("compute")]
[numthreads(REG_PROBE_GROUP,1,1)]
void trace(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {probe_trace<Unspecialized,NoPrograms>(group,thread);}
[shader("compute")]
[numthreads(REG_PROBE_GROUP,1,1)]
void gather(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {probe_gather(group,thread);}

// Select capture positions through the installed field's current validity, not
// a CPU visibility proof. A provider is shared by the brick's diffuse vertices.
void probe_choose(uint3 group,uint thread) {
    let provider=probe_thread(group,thread);if(provider>=probe_grid().field.x) {return;}
    let g=probe_grid_data[provider+1u];let center=g.origin.xyz+probe_world_vector(.5*float3(g.counts.xyz-1u)*g.spacing.xyz);
    var best=REG_FP32_MAX;var selected=0xffffffffu;
    let count=g.counts.x*g.counts.y*g.counts.z;
    for(uint local=0u;local<count;local++) {
        let probe=probe_grid_data[g.field.y+local/4u].field[local%4u];
        if(probe==0xffffffffu || probe_irradiance[probe].support.x<g.bias.w) {continue;}
        let delta=probe_position_index(probe)-center;let score=dot(delta,delta);
        if(score<best) {best=score;selected=probe;}
    }
    visibility_output[probe_provider_base()+provider]=selected==0xffffffffu ? float2(-1.,0.) : float2(float(selected),1.);
}
float4 probe_directional_capture<V:IVariant,P:IProgram>(uint probe,uint ray,float3 position,float3 direction,out bool near_boundary,out float3 terminal) {
    near_boundary=false;terminal=float3(0.);
    let reach=probe_spacing(probe).w;let error=length(REG_FP32_GAMMA7*(abs(position)+probe_spacing(probe).xyz));
    RayDesc desc;desc.Origin=position-direction*error;desc.Direction=direction;desc.TMin=0.;desc.TMax=reach+error;
    float backface;
    let nearest=probe_surface_hit<P,true>(desc,probe,ray,error,near_boundary,backface);
    // World remains a terminal query-time provider. Capture only visibility,
    // preserving its original angular detail, rotation, filtering and source split.
    if(!nearest.found) {
        var world=(frame.environment_links.x&REG_RAY_SPECULAR)!=0u ? reflection_fallback(direction,0.,frame,env_specular,env_sampler) : float3(0.);
        terminal=world;world=float3(0.);
        return float4(world,-reach);
    }
    let hit=nearest.hit;
    let distance=max(nearest.distance-error,0.);
    if(near_boundary) {return float4(0.,0.,0.,distance);}
    var radiance=float3(0.);
    if(ray_alpha[hit.instance_index*REG_RAY_SLOT_WORDS+2u]!=REG_RAY_NO_ALPHA) {
    let u=select(P.lobes,float(screen_ray_hash(uint2(probe_identity(probe),probe_sample_index(ray)),0u,REG_SAMPLING_STREAM_RAY_GI_LOBE))*REG_RADICAL_INVERSE_SCALE,0.);
    let h=hit_surface<V,P,ProbeGeometry>(ProbeGeometry(ray_alpha,ray_hit_buffers),hit.instance_index,hit.primitive_index,hit.instance_custom_data,nearest.inverse,hit.barycentrics,direction,float2(0.),Lobe(select(P.lobes,LOBE_PICK,0u),u));
    float3 emission;radiance=finalized_hit_radiance<V,ProbeShadows<V,P>,P>(h,uint2(probe_identity(probe),0u),probe_sample_index(ray),REG_RAY_DIFFUSE|REG_RAY_SPECULAR,emission,false,true);
    }
    return float4(radiance,distance);
}
void probe_directional_trace<V:IVariant,P:IProgram>(uint3 group,uint thread) {
    let index=probe_thread(group,thread);let size=batch.rays.y;let provider=index/size;let ray=batch.rays.x+index%size;
    if(provider>=probe_grid().field.x || ray>=probe_ray_count()) {return;}
    let descriptor=visibility_output[probe_provider_base()+provider];if(descriptor.x<0.) {return;}
    let probe=uint(descriptor.x);let direction=probe_direction(probe,ray);
    let at=2u*(provider*size+index%size);
    bool near_boundary;float3 terminal;
    traced[at]=probe_directional_capture<V,P>(probe,ray,probe_position_index(probe),direction,near_boundary,terminal);
    // Retain the World part separately: its angular detail is queried at the
    // receiver, while the distributed medium source belongs in the local atlas.
    traced[at+1u]=float4(near_boundary ? 1. : 0.,terminal);
}
void probe_directional_gather(uint3 group,uint thread) {
    let texels=REG_PROBE_RADIANCE_SIDE*REG_PROBE_RADIANCE_SIDE;
    let index=probe_thread(group,thread);let provider=index/texels;let texel=index%texels;
    if(provider>=probe_grid().field.x) {return;}
    let pixel=int2(int(texel%REG_PROBE_RADIANCE_SIDE),int(texel/REG_PROBE_RADIANCE_SIDE));
    let atlas=probe_radiance_address(provider,0u,pixel);
    if(visibility_output[probe_provider_base()+provider].x<0.) {
        for(uint i=0u;i<REG_PROBE_RADIANCE_PAIRS;i++) {visibility_output[atlas+i]=float2(0.);}return;
    }
    let first=batch.rays.x;let size=batch.rays.y;let count=min(size,probe_ray_count()-first);
    let record=provider*(texels*2u+3u)+texel;
    let sh_record=provider*(texels*2u+3u)+texels*2u+texel;
    var sh=float4(0.);if(texel<3u && first!=0u) {sh=sums[sh_record];}
    var sum=first==0u ? float4(0.) : sums[record];var depth=first==0u ? float4(0.) : sums[record+texels];
    let axis=decode_octahedron((float2(pixel)+.5)/float(REG_PROBE_RADIANCE_SIDE)*2.-1.);
    for(uint k=0u;k<count;k++) {
        let at=2u*(provider*size+k);let t=traced[at];let direction=probe_direction(uint(visibility_output[probe_provider_base()+provider].x),first+k);
        let c=saturate(dot(axis,direction));let c2=c*c;let c4=c2*c2;let c8=c4*c4;let c16=c8*c8;let w=c16*c16;
        let distance=abs(t.w);sum+=float4(t.rgb*w,w);
        if(texel<3u) {sh+=(t[texel]+traced[at+1u][texel+1u])*float4(1.,direction);}
        depth+=float4(distance*w,distance*distance*w,(t.w<0. ? w : 0.),traced[at+1u].x);
    }
    sums[record]=sum;sums[record+texels]=depth;
    if(texel<3u) {sums[sh_record]=sh;}
    if(first+count<probe_ray_count()) {return;}
    if(texel<3u) {
        // Raw radiance's first SH coefficient is 3*mean(L*direction); the
        // normalized rough GGX endpoint has moment 1/2, hence 1.5 here.
        let value=float4(sh.x,1.5*sh.yzw)/float(probe_ray_count());
        let at=probe_provider_base()+probe_grid().field.x+provider*REG_PROBE_RADIANCE_SH_PAIRS+2u*texel;
        visibility_output[at]=value.xy;visibility_output[at+1u]=value.zw;
    }
    let weight=max(sum.w,REG_POSITIVE_EPSILON);let transport=float4(sum.rgb/weight,depth.z/weight);
    visibility_output[atlas]=transport.xy;visibility_output[atlas+1u]=transport.zw;visibility_output[atlas+2u]=depth.xy/weight;
}
void probe_filter(uint3 group,uint thread) {
    let level=batch.rays.x;let side=REG_PROBE_RADIANCE_SIDE>>level;let count=side*side;
    let index=probe_thread(group,thread);let provider=index/count;let texel=index%count;
    if(provider>=probe_grid().field.x) {return;}
    let pixel=int2(int(texel%side),int(texel/side));let to=probe_radiance_address(provider,level,pixel);
    // Publish capture validity after all ray/gather reads of the descriptor.
    // Only this invocation writes it; other filter invocations read atlas data.
    if(level==1u && texel==0u) {
        let descriptor=visibility_output[probe_provider_base()+provider];
        let bad=sums[provider*(REG_PROBE_RADIANCE_SIDE*REG_PROBE_RADIANCE_SIDE*2u+3u)+REG_PROBE_RADIANCE_SIDE*REG_PROBE_RADIANCE_SIDE].w;
        visibility_output[probe_provider_base()+provider]=float2(descriptor.x,descriptor.x>=0. && bad==0. ? 1. : 0.);
    }
    for(uint pair=0u;pair<REG_PROBE_RADIANCE_PAIRS;pair++) {
        var sum=float2(0.);for(int y=0;y<2;y++) {for(int x=0;x<2;x++) {
            sum+=visibility_output[probe_radiance_address(provider,level-1u,pixel*2+int2(x,y))+pair];
        }}visibility_output[to+pair]=sum*.25;
    }
}
[shader("compute")][numthreads(REG_PROBE_GROUP,1,1)] void choose(uint3 group:SV_GroupID,uint thread:SV_GroupIndex) {probe_choose(group,thread);}
[shader("compute")][numthreads(REG_PROBE_GROUP,1,1)] void directional_trace(uint3 group:SV_GroupID,uint thread:SV_GroupIndex) {probe_directional_trace<Unspecialized,NoPrograms>(group,thread);}
[shader("compute")][numthreads(REG_PROBE_GROUP,1,1)] void directional_gather(uint3 group:SV_GroupID,uint thread:SV_GroupIndex) {probe_directional_gather(group,thread);}
[shader("compute")][numthreads(REG_PROBE_GROUP,1,1)] void filter(uint3 group:SV_GroupID,uint thread:SV_GroupIndex) {probe_filter(group,thread);}
