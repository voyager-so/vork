// Ray-traced reflections whose unseen hits shade their full surface (hit.slang,
// hit_lighting) on the shared Vulkan/Metal globals layout.
import vork_constants;
import frame;
import vork_ray_sampling;
import vork_ray_alpha;
import surface;
import variant;
import hit;
import dielectric_scene;
import emitter_mesh;
import vork_material_table;
import hit_lighting;
import vork_prelude;
import vork_sampling;
import vork_visibility;
import reflection_environment;
import reflection_receiver;
import reflection_bsdf;
import ray_coverage;
import surface_lighting;
import subsurface_transport;
// rt::reflections_hits_entries: the scene globals (frame.slang) occupy bindings 0-9.
[[vk::binding(11,0)]] Texture2D<float4> color : register(t4);
[[vk::binding(12,0)]] Texture2D<float4> receivers : register(t5);
[[vk::binding(13,0)]] Texture2D<float4> normals : register(t6);
[[vk::binding(14,0)]] Texture2D<float4> ambient : register(t7);
[[vk::binding(15,0)]] [[vk::image_format("rgba16f")]] RWTexture2DArray<float4> output : register(u8);
[[vk::binding(16,0)]] RaytracingAccelerationStructure ray_scene : register(t6);
[[vk::binding(17,0)]] StructuredBuffer<uint> ray_alpha : register(t7);
[[vk::binding(18,0)]] StructuredBuffer<uint> sampling : register(t8);
[[vk::binding(19,0)]] StructuredBuffer<uint> reflection_slots : register(t9);
[[vk::binding(20,0)]] StructuredBuffer<uint> reflection_guides : register(t10);
#ifdef VORK_METAL
struct ReflectionFields {StructuredBuffer<uint> buffers[REG_VISIBILITY_REFLECTION_WORDS];};
[[vk::binding(1,3)]] ParameterBlock<ReflectionFields> reflection_fields : register(VORK_REFLECTION_FIELDS_REGISTER);
#define reflection_records reflection_fields.buffers
#else
[[vk::binding(1,3)]] StructuredBuffer<uint> reflection_fields[REG_VISIBILITY_REFLECTION_WORDS];
#define reflection_records reflection_fields
#endif
struct ReflectionDispatch {uint4 work;uint4 packet;};
[[vk::binding(21,0)]] ConstantBuffer<ReflectionDispatch> reflection_dispatch : register(b11);
#define reflection_range reflection_dispatch.work
[[vk::binding(22,0)]] RWStructuredBuffer<float4> reflection_accumulation : register(u12);
[[vk::binding(23,0)]] StructuredBuffer<float> reflection_emitter_weights : register(t13);
// Raster geometry buffers ray hits read in place (rt::hit_record, rt::hits_layout).
#ifdef VORK_METAL
[[vk::binding(0,3)]] ParameterBlock<RayHitBuffersTable> ray_hit_buffers : register(VORK_HIT_REGISTER);
#else
[[vk::binding(0,3)]] StructuredBuffer<uint> ray_hit_buffers[REG_RAY_HIT_BUFFERS];
#endif

export float3 ray_eye() {return frame.eye.xyz;}
typedef RayGeometry ReflectionGeometry;
#ifdef VORK_PROGRAM_TRACE
import geometric_trace;
export TraceHit native_geometric_trace(float3 origin,float3 direction,float minimum,float maximum,uint mask,uint object) {
    return geometric_trace<ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),ray_scene,origin,direction,minimum,maximum,mask,object);
}
#endif

// Sample the physical emitter and its visibility together at reflection hits.
struct ReflectionShadows<V:IVariant,P:IProgram> : IShadows {
    static const bool probe_rays=true;
    static bool probe_visible(float3 world,float3 normal,float3 position_error,float3 probe) {return probe_connection_visible<P,ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,world,normal,position_error,probe);}
    static uint integration_count(uint index,out float3 normalization) {let count=index==0xffffffffu ? sample_count() : 1u;normalization=float3(float(count));return count;}
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {if(index==0xffffffffu) {index=tap;}return float3(1.);}
    static uint sample_count() {return uint(frame.temporal.y)*max(frame.sampling.w,1u);}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 position_error,uint index,uint count,bool fibre=false) {return hit_shadow_direction<P,ReflectionGeometry,V>(ReflectionGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,light,world,normal,position_error,direction,distance,index,count,false);}
    static const bool subsurface=true;
    static float thickness(uint object,float3 world,float3 direction) {return subsurface_exit(ray_scene,object,world,direction);}
};
struct SurfaceHits<V:IVariant,P:IProgram> : IReflectionTrace {
    static float connection_probability() {return ray_alpha[reflection_range.z+1u]>0u ? .5 : 0.;}
    static float emitter_pdf(RayIntersection hit,float3x4 inverse,float3 from) {
        let base=reflection_range.z;let count=ray_alpha[base+1u];
        if(count==0u || ray_alpha[base+3u+hit.instance_index]==0xffffffffu) {return 0.;}
        let geometry=hit_emitter<ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),hit.instance_index,hit.primitive_index,hit.instance_custom_data,inverse,hit.barycentrics);
        return emitter_mesh_pdf<ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),base,hit.instance_index,geometry.area,from,geometry.point,geometry.face)/float(count);
    }
    // A separate source connection competes only with emitted radiance. Reflected
    // surface lighting has no source proposal and retains its ordinary weight.
    static float3 direction(ReflectionSite site,uint2 cell,uint index,float3 incident,float3 ray,float pdf,bool connection,EmitterTriangle selected) {
        let size=vork_dimensions(receivers);let footprint=connection ? float2(0.) : float2(1.)/float2(size);let receiver=site.receiver;
        let n=site.normal;
        let position=site.position;
        let geometric=site.geometric;
        let error=site.error;
        let origin=ray_origin(position,geometric,ray,error);
        RayDesc desc;
        desc.Origin=origin;desc.Direction=ray;
        desc.TMin=0.;desc.TMax=REG_FP32_MAX;
#ifdef VORK_METAL
        // Metal is the selection path because no per-device HW-RT capability gate exists: two queries in one kernel change the first hit on Apple hardware ray tracing (observed M5, 0.5.0); remove when a per-device gate exists or Apple fixes it.
        // Never commit: hardware closest-hit pruning can hide equal-distance
        // candidates. One full traversal selects the stable accepted minimum.
        DielectricHit closest={};closest.t=-1.;
        var closest_coverage=false;
        {
            RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
            query.TraceRayInline(ray_scene,RAY_FLAG_NONE,REG_RAY_SPECULAR,desc);
            while(query.Proceed()) {
                if(query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE) {continue;}
                let t=query.CandidateTriangleRayT();
                if(closest.t>=0. && t>closest.t) {continue;}
                let candidate=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
                let record=material_records[objects[candidate.instance_custom_data].material_index];
                let coverage=record.tag.x!=0u || record.extra.y>0.;
                if(t==closest.t) {
                    if(closest_coverage && !coverage) {continue;}
                    if(closest_coverage==coverage && (candidate.instance_custom_data>closest.object
                        || (candidate.instance_custom_data==closest.object && candidate.primitive_index>=closest.primitive))) {continue;}
                }
                let inverse=query.CandidateWorldToObject3x4();
                if(!reflection_candidate_accepts<P,ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),ray_alpha,candidate,query.CandidateTriangleFrontFace(),inverse,ray,footprint,cell,index,ReflectionShadows<V,P>.sample_count())) {continue;}
                closest.t=t;closest.instance=candidate.instance_index;closest.object=candidate.instance_custom_data;
                closest.primitive=candidate.primitive_index;closest.barycentrics=candidate.barycentrics;closest.world_to_object=inverse;
                closest_coverage=coverage;
            }
        }
#else
        // Find the nearest accepted distance with hardware pruning, then resolve
        // equal-distance coverage in a bounded interval. Committing the first
        // query may prune ties, so only the second query chooses their owner.
        DielectricHit closest={};closest.t=-1.;
        {
            // Instance flags preserve double-sided and solid-boundary rules.
            RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
            query.TraceRayInline(ray_scene,RAY_FLAG_NONE,REG_RAY_SPECULAR,desc);
            while (query.Proceed()) {
                if (query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE) {continue;}
                let candidate=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
                if (reflection_candidate_accepts<P,ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),ray_alpha,candidate,query.CandidateTriangleFrontFace(),query.CandidateWorldToObject3x4(),ray,footprint,cell,index,ReflectionShadows<V,P>.sample_count())) {query.CommitNonOpaqueTriangleHit();}
            }
            if(query.CommittedStatus()==COMMITTED_TRIANGLE_HIT) {
                closest.t=query.CommittedRayT();closest.instance=query.CommittedInstanceIndex();closest.object=query.CommittedInstanceID();
                closest.primitive=query.CommittedPrimitiveIndex();closest.barycentrics=query.CommittedTriangleBarycentrics();closest.world_to_object=query.CommittedWorldToObject3x4();
            }
        }
        if(closest.t>=0.) {
            let record=material_records[objects[closest.object].material_index];
            var closest_coverage=record.tag.x!=0u || record.extra.y>0.;
            RayDesc ties=desc;
            // Adjacent representable endpoints keep the computed hit strictly
            // inside the interval; only exact equal-t candidates can replace it.
            ties.TMin=closest.t>0. ? asfloat(asuint(closest.t)-1u) : 0.;
            ties.TMax=min(asfloat(asuint(closest.t)+1u),REG_FP32_MAX);
            RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
            query.TraceRayInline(ray_scene,RAY_FLAG_NONE,REG_RAY_SPECULAR,ties);
            while(query.Proceed()) {
                if(query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE || query.CandidateTriangleRayT()!=closest.t) {continue;}
                let candidate=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
                let record=material_records[objects[candidate.instance_custom_data].material_index];
                let coverage=record.tag.x!=0u || record.extra.y>0.;
                if(closest_coverage && !coverage) {continue;}
                if(closest_coverage==coverage && (candidate.instance_custom_data>closest.object
                    || (candidate.instance_custom_data==closest.object && candidate.primitive_index>=closest.primitive))) {continue;}
                let inverse=query.CandidateWorldToObject3x4();
                if(!reflection_candidate_accepts<P,ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),ray_alpha,candidate,query.CandidateTriangleFrontFace(),inverse,ray,footprint,cell,index,ReflectionShadows<V,P>.sample_count())) {continue;}
                closest.instance=candidate.instance_index;closest.object=candidate.instance_custom_data;
                closest.primitive=candidate.primitive_index;closest.barycentrics=candidate.barycentrics;closest.world_to_object=inverse;
                closest_coverage=coverage;
            }
        }
#endif
        closest=merge_field_hits<P,ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),closest,ray_fields(ray_alpha),origin,ray,desc.TMax,REG_RAY_SPECULAR);
        var transmission=float3(1.);var source=float3(0.);
        let world_specular=site.world_specular;
        if (closest.t<0.) {
            // An escaped hardware ray sees the World rather than a local probe.
            return connection ? float3(0.) : (source+transmission*select(world_specular,directional_radiance(position,n,-incident,ray,0.,true,frame,env_specular,env_sampler),float3(0.)))*receiver.reflected*frame.radiance.w;
        }
        let hit=RayIntersection(closest.instance,closest.object,closest.primitive,closest.barycentrics);
        if(connection && (closest.field || hit.instance_index!=selected.instance || hit.primitive_index!=selected.primitive)) {return float3(0.);}
        if (ray_alpha[hit.instance_index*REG_RAY_SLOT_WORDS+2u]==REG_RAY_NO_ALPHA) {return source*receiver.reflected*frame.radiance.w;}
        let inverse=closest.world_to_object;
        let u=select(P.lobes,float(screen_ray_hash(cell,index,REG_SAMPLING_STREAM_RAY_GI_LOBE))*REG_RADICAL_INVERSE_SCALE,0.);
        var h=dielectric_surface<V,P,ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),closest,ray,footprint,Lobe(select(P.lobes,LOBE_PICK,0u),u));
        if(connection) {
            if(h.s.discarded || !material_covers_side<V>(h.v,h.entering)) {return float3(0.);}
            let emitted=emitter_mesh_radiance<V,P>(h);
            return transmission*emitted*receiver.reflected*frame.radiance.w;
        }
        // Non-source instances have no competing proposal. Builtin black
        // emission also needs no MIS geometry; retain conservative program support.
        let builtin=material_records[objects[h.v.object].material_index].tag.x==0u;
        let light_pdf=(!closest.field && (!builtin || any(h.s.closure.emission_color*h.s.closure.emission_luminance!=float3(0.)))) ? emitter_pdf(hit,inverse,position) : 0.;
        float3 emitted;
        let radiance=finalized_hit_radiance<V,ReflectionShadows<V,P>,P>(h,cell,index,REG_RAY_DIFFUSE|REG_RAY_SPECULAR,emitted);
        let mis=pdf/(pdf+light_pdf);
        return (source+transmission*(radiance-emitted+emitted*mis))*receiver.reflected*frame.radiance.w;
    }
    static float3 trace(ReflectionSite site,uint2 cell,uint index,float3 incident,float3 ray,float pdf) {
        EmitterTriangle selected={};
        return direction(site,cell,index,incident,ray,pdf,false,selected);
    }
    static float3 direct(ReflectionSite site,uint2 cell,uint index,float3 incident) {
        EmitterTriangle selected={};var value=float3(0.);
        let base=reflection_range.z;let count=ray_alpha[base+1u];
        if(count==0u) {return value;}
        // Discrete triangle choice and its residual are independent under the
        // uniform square. The residual and second coordinate sample its area.
        var u=sample_2d_installed(sampling,frame.sampling.x,cell,index,sampling[frame.sampling.x],REG_SAMPLING_STREAM_REFLECTION_EMITTER);
        let pick=min(uint(u.x*float(count)),count-1u);u.x=u.x*float(count)-float(pick);
        selected=emitter_mesh_triangle<ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),base,pick);
        let radius=sqrt(u.x);selected.barycentrics=float2(radius*(1.-u.y),radius*u.y);
        let inverse=(float3x4)object_normal(selected.object);
        let geometry=hit_emitter<ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),selected.instance,selected.primitive,selected.object,inverse,selected.barycentrics);
        let axis=normalize(geometry.point-site.position);
        if(dot(axis,site.geometric)<=0.) {return value;}
        let light_pdf=emitter_mesh_pdf<ReflectionGeometry>(ReflectionGeometry(ray_alpha,ray_hit_buffers),base,selected.instance,geometry.area,site.position,geometry.point,geometry.face)/float(count);
        if(light_pdf<=0.) {return value;}
        float bsdf_pdf;let response=reflection_bsdf_evaluate(site.bsdf,-incident,axis,bsdf_pdf);
        let mixture_pdf=bsdf_pdf;
        site.receiver.reflected=response/(light_pdf+mixture_pdf);
        if(any(site.receiver.reflected!=float3(0.))) {value+=direction(site,cell,index,incident,axis,mixture_pdf,true,selected);}
        return value;
    }
};
// Scenes with programs compile their own entry with their Programs (SlangForward::program_kernel).
public void reflections_pixel<V:IVariant,P:IProgram>(uint3 gid) {
    let size=vork_dimensions(receivers);
    if(gid.x>=REG_REFLECTION_DISPATCH_PIXELS) {return;}
    let pixel=(reflection_range.w&~REG_REFLECTION_MERGED_SLOTS)+gid.x;
    if(pixel>=size.x*size.y || (reflection_dispatch.packet.x!=0u && pixel>=reflection_dispatch.packet.y)) {return;}
    gid.xy=uint2(pixel%size.x,pixel/size.x);
    reflection_integrate<SurfaceHits<V,P>>(gid,frame,vork_dimensions(receivers),reflection_slots,reflection_records,reflection_guides,sampling,reflection_range.xy,reflection_accumulation,output,(reflection_range.w&REG_REFLECTION_MERGED_SLOTS)!=0u,reflection_dispatch.packet.x,reflection_dispatch.packet.z);
}
[shader("compute")]
[numthreads(REG_COMPUTE_TILE*REG_COMPUTE_TILE,1,1)]
void compute(uint3 gid : SV_DispatchThreadID) {reflections_pixel<Unspecialized,NoPrograms>(gid);}
