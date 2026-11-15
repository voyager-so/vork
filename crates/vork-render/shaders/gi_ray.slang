// Preparation is retained with the incident accumulator for this exposure.
struct GiQuery {
    float4 accumulation;
    float4 position_depth;
    float4 normal_view;
    float4 geometric_active;
    float4 error;
    float4 moments;
};
#ifdef VORK_METAL
[[vk::binding(1,3)]] RWStructuredBuffer<GiQuery> gi_accumulation : register(VORK_GI_QUERY_REGISTER);
#else
[[vk::binding(1,3)]] RWStructuredBuffer<GiQuery> gi_accumulation;
#endif
// Ray diffuse indirect shades off-screen hits through the raster surface
// evaluation (hit.slang) and the shared BSDF evaluator.
import vork_constants;
import openpbr_core;
import frame;
import vork_ray_sampling;
import vork_sampling;
import vork_ray_alpha;
import openpbr_inputs;
import bsdf_response;
import closure;
import variant;
import material;
import surface;
import hit;
import dielectric_scene;
import ray_coverage;
import hit_lighting;
import light_tree;
import emitter;
import gi_proposal;
import emitter_mesh;
import emitter_implicit;
import implicit_scene;
import surface_lighting;
import subsurface_transport;
#ifdef VORK_PROBES
import probes;
#endif
// rt::gi_entries: the scene globals (frame.slang) occupy bindings 0-9.
[[vk::binding(11,0)]] Texture2D<float4> receivers : register(t4);
[[vk::binding(12,0)]] Texture2D<float4> normals : register(t5);
[[vk::binding(13,0)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> output : register(u6);
[[vk::binding(14,0)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> incident_output : register(u7);
[[vk::binding(15,0)]] RaytracingAccelerationStructure ray_scene : register(t6);
[[vk::binding(16,0)]] StructuredBuffer<uint> ray_alpha : register(t7);
[[vk::binding(17,0)]] [[vk::image_format("rgba32f")]] RWTexture2D<float4> cell_receivers : register(u8);
struct GiSources {uint4 topology;uint4 batch;uint4 tile;};
[[vk::binding(18,0)]] ConstantBuffer<GiSources> gi_source_data : register(b8);
#define gi_sources gi_source_data.topology
[[vk::binding(19,0)]] StructuredBuffer<float> gi_emitter_weights : register(t9);

struct GiGuide {float4 point;float4 moments;};
#ifdef VORK_METAL
[[vk::binding(2,3)]] RWStructuredBuffer<GiGuide> gi_guide : register(VORK_GI_GUIDE_REGISTER);
#else
[[vk::binding(2,3)]] RWStructuredBuffer<GiGuide> gi_guide;
#endif
#ifdef VORK_METAL
[[vk::binding(3,3)]] StructuredBuffer<LightTreeNode> gi_light_nodes : register(VORK_GI_LIGHT_REGISTER);
#else
[[vk::binding(3,3)]] StructuredBuffer<LightTreeNode> gi_light_nodes;
#endif
// Discovery already distinguishes bounded infinite-only source sets. Keep
// their cheap exact beauty lighting too: a discrete source lottery adds
// variance without spatial selection benefit. The static kind is a fast proof;
// conservative variants prove the same property from the current records.
bool gi_small_infinite_set<V:IVariant>(uint count) {
    if(count>REG_GI_PILOT_RAYS) return false;
    if(V.light_types==1u) return true;
    for(uint i=0u;i<count;i++) {if(lights[i].position.w>0.) return false;}
    return true;
}
float2 gi_hit_light<V:IVariant>(float3 world,uint2 pixel,uint index,uint count,bool pilot,bool lit_material) {
    let lights_count=uint(frame_light_count(frame));
    // Authored Shader-to-RGB can apply nonlinear functions to `lit`.
    // Preserve its complete input lighting; sampling before that function
    // would change authored semantics rather than merely estimator variance.
    if(lights_count<=1u || (!pilot && lit_material)) return float2(-1.,1.);
    if(!pilot && gi_small_infinite_set<V>(lights_count)) return float2(-1.,1.);
    // One spatial-tree connection, divided by its full selection probability
    // in shared hit lighting. Beauty stratifies over its actual ray budget;
    // the four-ray pilot sequence stays independent and unchanged.
    let u=sample_2d(pixel,index,pilot ? REG_GI_PILOT_RAYS : count,REG_SAMPLING_STREAM_MEDIUM_LIGHT);
    let pick=light_tree_pick(gi_light_nodes,2u*lights_count-1u,world,u);
    return float2(float(pick.identity),pick.probability);
}
// Independent pilots learn where reflected illumination is visible. The guide
// changes only the proposal: every beauty sample still traces and shades its hit.
// Respect render-call seed without changing the default-zero sequence.
uint2 gi_pixel_key(uint2 cell) {
    let seed=pixel_hash(asuint(frame.display.x));
    return cell^uint2(seed,pixel_hash(seed));
}
int2 gi_guide_cell(uint2 pixel,uint tap,int2 size) {
    let maximum=((size-1)/2)*2;
    return clamp(int2((pixel/2u)*2u)+(int2(int(tap%3u),int(tap/3u))-1)*4,int2(0),maximum);
}
GiProposal gi_proposal(uint2 pixel,float3 position,float3 normal) {
    GiProposal result={};uint w,h;output.GetDimensions(w,h);let size=int2(w,h);
    for(uint i=0u;i<9u;i++) {
        let q=gi_guide_cell(pixel,i,size);
        let guide=gi_guide[uint(q.y)*uint(size.x)+uint(q.x)].point;
        let delta=guide.xyz-position;let distance2=dot(delta,delta);
        result.axis[i]=distance2>0. ? delta*rsqrt(distance2) : normal;
        result.weight[i]=guide.w*max(dot(normal,result.axis[i]),0.)/max(distance2,0.01);
        result.total+=result.weight[i];
    }
    let count=uint(frame_light_count(frame));
    if(count>0u) {
        var identity=0u;
        if(count>1u) {
            let u=sample_2d(gi_pixel_key(pixel)^uint2(0x4a39b70du,0x12fad5c9u),0u,1u,REG_SAMPLING_STREAM_MEDIUM_LIGHT);
            identity=light_tree_pick(gi_light_nodes,2u*count-1u,position,u).identity;
        }
        let source=lights[identity];
        let delta=source.position.xyz-position;let distance2=dot(delta,delta);
        if(source.position.w>0. && distance2>0. && any(source.radiance.rgb>float3(0.))) {
            result.axis[9]=delta*rsqrt(distance2);
            result.weight[9]=max(result.total,1.e-20)*max(dot(normal,result.axis[9]),0.);
            result.total+=result.weight[9];
        }
    }
    return result;
}
float gi_unoccluded_signal<V:IVariant>(uint identity,float3 position,float3 normal) {
    let source=emitter_center_li<V>(specialized_light<V>(lights[identity]),identity,position);
    let density=source.delta ? 1. : source.pdf;
    return density>0. ? dot(source.radiance,float3(.2126,.7152,.0722))*max(dot(normal,source.direction),0.)*
        frame.radiance.w/(REG_PI*density) : 0.;
}
// A guide-concentration heuristic allocates work; it is not an energy estimate.
// Ordinary broad bounce uses reconstruction. Rare intense patches receive rays.
bool gi_needs_samples<V:IVariant>(uint2 pixel,float3 position,float3 normal) {
    uint w,h;output.GetDimensions(w,h);let size=int2(w,h);
    var mean=0.;var peak=0.;
    for(uint i=0u;i<9u;i++) {
        let q=gi_guide_cell(pixel,i,size);
        let index=uint(q.y)*w+uint(q.x);
        let point=gi_guide[index].point;
        mean+=gi_guide[index].moments.x;
        let delta=point.xyz-position;let distance2=dot(delta,delta);
        if(distance2>0.) peak=max(peak,point.w*max(dot(normal,delta*rsqrt(distance2)),0.));
    }
    // The final signal is at least the pilot mean. Once that alone rules out
    // concentration, world/light proxies cannot change the budget decision.
    if(peak<=REG_GI_CONCENTRATION*max(mean/9.,1.e-12)) return false;
    let world_signal=max(dot(world_radiance(normal)*frame.radiance.w,float3(.2126,.7152,.0722)),0.);
    var direct_signal=0.;let count=uint(frame_light_count(frame));
    // Up to the discovery budget, exact cheap proxies avoid lottery-driven
    // oversampling of ordinary broad bounce. Many-source views retain the tree.
    if(count<=REG_GI_PILOT_RAYS) {
        for(uint i=0u;i<count;i++) direct_signal+=gi_unoccluded_signal<V>(i,position,normal);
    } else {
        let u=sample_2d(gi_pixel_key(pixel)^uint2(0x4a39b70du,0x12fad5c9u),0u,1u,REG_SAMPLING_STREAM_MEDIUM_LIGHT);
        let pick=light_tree_pick(gi_light_nodes,2u*count-1u,position,u);
        if(pick.probability>0.) direct_signal=gi_unoccluded_signal<V>(pick.identity,position,normal)/pick.probability;
    }
    let signal=max(mean/9.,world_signal+direct_signal);
    return peak>REG_GI_CONCENTRATION*max(signal,1.e-12);
}
// Raster geometry buffers ray hits read in place (rt::hit_record, rt::hits_layout).
#ifdef VORK_METAL
[[vk::binding(0,3)]] ParameterBlock<RayHitBuffersTable> ray_hit_buffers : register(VORK_HIT_REGISTER);
#else
[[vk::binding(0,3)]] StructuredBuffer<uint> ray_hit_buffers[REG_RAY_HIT_BUFFERS];
#endif

typedef RayGeometry GiGeometry;
#ifdef VORK_PROGRAM_TRACE
import geometric_trace;
export TraceHit native_geometric_trace(float3 origin,float3 direction,float minimum,float maximum,uint mask,uint object) {
    return geometric_trace<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),ray_scene,origin,direction,minimum,maximum,mask,object);
}
#endif

export float3 ray_eye() {return frame.eye.xyz;}
uint2 image_size(Texture2D<float4> t) {uint w,h;t.GetDimensions(w,h);return uint2(w,h);}
uint2 image_size(RWTexture2D<float4> t) {uint w,h;t.GetDimensions(w,h);return uint2(w,h);}


float gi_mis(float sampled,float other) {
    if(other<=0.) {return 1.;}if(sampled<=0.) {return 0.;}
    let ratio=other/sampled;return 1./(1.+ratio*ratio);
}
float gi_emitter_pdf<P:IProgram>(RayIntersection hit,float3x4 inverse,float3 from) {
    let base=gi_sources.x;
    if(ray_alpha[base]==0u) {return 0.;}
    let probability=emitter_mesh_probability<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),base,hit.instance_index,hit.primitive_index,gi_emitter_weights);
    if(probability<=0.) {return 0.;}
    let geometry=hit_emitter<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),hit.instance_index,hit.primitive_index,hit.instance_custom_data,inverse,hit.barycentrics);
    return probability*emitter_mesh_pdf<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),base,hit.instance_index,geometry.area,from,geometry.point,geometry.face);
}
// A physical source connection complements the cosine continuation. Only source
// emission receives MIS; reflected radiance retains the hemisphere estimator.
float3 gi_source<V:IVariant,P:IProgram>(float3 from,float3 normal,float3 geometric,float3 error,uint2 pixel,uint index,uint count,GiProposal guide) {
    let base=gi_sources.x;if(ray_alpha[base]==0u) {return float3(0.);}
    let uv=sample_2d(pixel,index,count,REG_SAMPLING_STREAM_PROBE_EMITTER);
    let extra=sample_2d(pixel,index,count,REG_SAMPLING_STREAM_PROBE_TRIANGLE);
    let selected=emitter_mesh_sample<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),base,float3(extra.x,uv),gi_emitter_weights);
    if((objects[selected.object].metadata.y&REG_RAY_DIFFUSE)==0u) {return float3(0.);}
    let inverse=(float3x4)object_normal(selected.object);
    let geometry=hit_emitter<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),selected.instance,selected.primitive,selected.object,inverse,selected.barycentrics);
    let delta=geometry.point-from;let distance=length(delta);if(distance<=0.) {return float3(0.);}
    let direction=delta/distance;let cosine=max(dot(normal,direction),0.);
    if(cosine<=0. || dot(geometric,direction)<0.) {return float3(0.);}
    let hit=RayIntersection(selected.instance,selected.object,selected.primitive,selected.barycentrics);
    let pdf=gi_emitter_pdf<P>(hit,inverse,from);if(pdf<=0.) {return float3(0.);}
    if(!diffuse_candidate_accepts<P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),ray_alpha,hit,dot(geometry.face,-direction)>0.,inverse,direction,float2(0.))) {return float3(0.);}
    let source=hit_surface<V,P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),selected.instance,selected.primitive,selected.object,inverse,selected.barycentrics,direction,float2(0.));
    if(source.s.discarded) {return float3(0.);}
    let emitted=emitter_mesh_radiance<V,P>(source);if(!any(emitted!=float3(0.))) {return float3(0.);}
    let start=ray_origin(from,geometric,direction,error);
    let end=surface_ray_origin(geometry.point,surface_position_error(selected.object,geometry.point),source.face,-direction);
    let connection=end-start;let extent=length(connection);if(extent<=0.) {return float3(0.);}
    RayDesc ray;ray.Origin=start;ray.Direction=connection/extent;ray.TMin=0.;ray.TMax=extent;
    if(diffuse_ray_occluded<P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,ray,REG_RAY_GI_OCCLUDER,float2(0.))) {return float3(0.);}
    let continuation=cosine/REG_PI;var continuation_pdf=gi_proposal_pdf(guide,normal,direction);
    var radiance=emitted;
    return radiance*(continuation/pdf)*gi_mis(pdf,continuation_pdf)*frame.radiance.w;
}

// The domain cone proposes a direction; only the exact field root emits.
uint4 gi_fields() {return uint4(frame.field_geometry.xy,gi_sources.yz);}
float3 gi_field_source<V:IVariant,P:IProgram>(float3 from,float3 normal,float3 geometric,float3 error,uint2 pixel,uint index,uint count,GiProposal guide) {
    let table=gi_fields();if(table.w==0u) return float3(0.);
    let uv=sample_2d(pixel,index,count,REG_SAMPLING_STREAM_PROBE_EMITTER);
    let extra=sample_2d(pixel,index,count,REG_SAMPLING_STREAM_PROBE_TRIANGLE);
    let selected=field_emitter<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),table,min(uint(extra.x*float(table.w)),table.w-1u));
    if((objects[selected.object].metadata.y&REG_RAY_DIFFUSE)==0u || collapsed_endpoint(objects[selected.object].model)) return float3(0.);
    float density;let direction=field_emitter_direction<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),selected,from,uv,density);
    let cosine=max(dot(normal,direction),0.);let pdf=density/float(table.w);
    if(pdf<=0. || cosine<=0. || dot(geometric,direction)<0.) return float3(0.);
    var minimum=0.;
    for(;;) {
        let root=field_intersection<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),selected.descriptor,selected.object,from,direction,minimum,REG_FP32_MAX);
        if(!root.found) return float3(0.);
        let source=field_surface<V,P>(selected.object,root,direction,float2(0.),Lobe(0u,0.));
        if(source.s.discarded || source.s.closure.geometry_opacity<=0.) {
            minimum=root.t+REG_FP32_GAMMA7*max(1.,abs(root.t));continue;
        }
        if(!material_covers_side<V>(source.v,source.entering)) return float3(0.);
        let emitted=emitter_mesh_radiance<V,P>(source)*source.s.closure.geometry_opacity;
        if(!any(emitted>float3(0.))) return float3(0.);
        let start=ray_origin(from,geometric,direction,error);
        let end=surface_ray_origin(source.v.world,surface_position_error(selected.object,source.v.world),source.face,-direction);
        let delta=end-start;let extent=length(delta);if(extent<=0.) return float3(0.);
        RayDesc ray;ray.Origin=start;ray.Direction=delta/extent;ray.TMin=0.;ray.TMax=extent;
        if(diffuse_ray_occluded<P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,ray,REG_RAY_GI_OCCLUDER,float2(0.))) return float3(0.);
        var continuation=gi_proposal_pdf(guide,normal,direction);var radiance=emitted;
        return radiance*(cosine/(REG_PI*pdf))*gi_mis(pdf,continuation)*frame.radiance.w;
    }
}

// Shared surface shadows at GI hits: one ray toward each light (hit_lighting).
struct GiShadows<V:IVariant,P:IProgram> : IShadows {
    static const bool probe_rays=true;
    static bool probe_visible(float3 world,float3 normal,float3 position_error,float3 probe) {return probe_connection_visible<P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,world,normal,position_error,probe);}
    static uint integration_count(uint index,out float3 normalization) {let count=index==0xffffffffu ? sample_count() : 1u;normalization=float3(float(count));return count;}
    static float3 integration_sample(uint tap,inout uint index,inout float3 world) {if(index==0xffffffffu) {index=tap;}return float3(1.);}
    static uint sample_count() {return uint(frame.temporal.y)*max(frame.sampling.w,1u)*uint(frame.ao.z);}
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 position_error,uint index,uint count,bool fibre=false) {return hit_shadow_direction<P,GiGeometry,V>(GiGeometry(ray_alpha,ray_hit_buffers),ray_scene,ray_alpha,light,world,normal,position_error,direction,distance,index,count,false);}
    static const bool subsurface=true;
    static float thickness(uint object,float3 world,float3 direction) {return subsurface_exit(ray_scene,object,world,direction);}
};
// Outgoing diffuse radiance at a hit the camera cannot see (hit_lighting).
float3 hit_shading<V:IVariant,P:IProgram>(RayIntersection hit,float3x4 world_to_object,float3 world,float3 direction,float3 from,float continuation_pdf,uint2 pixel,uint index,uint count,bool pilot) {
    if (ray_alpha[hit.instance_index*REG_RAY_SLOT_WORDS+2u]==REG_RAY_NO_ALPHA) {return float3(0.);}
    // Hits filter textures with the camera pixel's footprint at the hit.
    // A mixed surface (Blend programs) contributes one lobe picked per ray.
    let u=select(P.lobes,float(screen_ray_hash(pixel,index,REG_SAMPLING_STREAM_RAY_GI_LOBE))*REG_RADICAL_INVERSE_SCALE,0.);
    var h=hit_surface<V,P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),hit.instance_index,hit.primitive_index,hit.instance_custom_data,world_to_object,hit.barycentrics,direction,float2(2.)/float2(image_size(receivers)),Lobe(select(P.lobes,LOBE_PICK,0u),u));
    float3 emitted;
    let radiance=finalized_hit_radiance<V,GiShadows<V,P>,P>(h,pixel,index,REG_RAY_DIFFUSE,emitted,false,false,gi_hit_light<V>(h.v.world,pixel,index,count,pilot,P.lit && P.lit_material(h.v)));
    // Source MIS weights emission only. Nonemitting hits need no source-tree
    // probability, triangle reconstruction or solid-angle PDF in either pass.
    let source_pdf=any(emitted!=float3(0.)) ? gi_emitter_pdf<P>(hit,world_to_object,from) : 0.;
    return (radiance-emitted*(1.-gi_mis(continuation_pdf,source_pdf)))*frame.radiance.w;
}
// Absolute cell addressing: the mask contains the exact full-support OR.
[[vk::binding(22,0)]] Texture2D<uint> demand_mask : register(t10);
bool gi_consumed(uint2 cell) {return demand_mask.Load(int3(cell,0))!=0u;}
// Scenes with programs compile their own entry with their Programs (SlangForward::program_kernel).
void gi_pixel<V:IVariant,P:IProgram>(uint3 gid) {
    if(any(gid.xy>=gi_source_data.tile.zw)) return;
    gid.xy+=gi_source_data.tile.xy;
    if(gid.y>=gi_source_data.batch.z) return;
    let at=gid.y*image_size(output).x+gid.x;
    gid.y+=gi_source_data.batch.y;
    if (any(gid.xy>=image_size(output))) {return;}
    let begin=gi_source_data.batch.x;
    let pilot=gi_source_data.batch.w!=0u;
    // Four discovery rays per 2x2 query tile; beauty and reconstruction stay at
    // their original receiver resolution. Every guide record is initialized.
    if(pilot && ((gid.x|gid.y)&1u)!=0u) {
        GiGuide empty={};gi_guide[gid.y*image_size(output).x+gid.x]=empty;return;
    }
    var rays=REG_GI_PILOT_RAYS;
    if(pilot) {
        let count=uint(frame_light_count(frame));
        // A small infinite-only source set has no light origins to explore.
        // Keep the independent hemisphere pilots; omit the two unavailable
        // source-origin pilots instead of replacing them with redundant rays.
        if(gi_small_infinite_set<V>(count)) rays=REG_GI_PILOT_RAYS/2u;
    }
    if(!pilot) {
        rays=begin==0u ? uint(frame.ao.z)*max(frame.sampling.w,1u) : uint(gi_accumulation[at].error.w);
    }
    if(begin>=rays) return;
    GiQuery prepared;
    if(begin==0u) {
    let surface=ao_surface(receivers,normals,gid.xy,int(frame.ao.w),frame.view);
    if(!pilot && surface.depth<1.) {cell_receivers[gid.xy]=float4(surface.position,surface.depth);}
    let active=surface.depth<1. && gi_consumed(gid.xy);
    if(active && !pilot) {
        let maximum=uint(frame.ao.z);
        let budget=gi_needs_samples<V>(gid.xy,surface.position,surface.normal) ? maximum : max(maximum/REG_GI_ADAPTIVE_RATIO,1u);
        rays=budget*max(frame.sampling.w,1u);
    }
    let ng=active ? screen_geometric_normal(receivers,surface.pixel,surface.position,surface.normal) : surface.normal;
    let position_error=active ? surface_position_error(surface.object,surface.position) : float3(0.);
    prepared.accumulation=float4(0.);
    prepared.position_depth=float4(surface.position,surface.depth);
    prepared.normal_view=float4(surface.normal,surface.view_depth);
    prepared.geometric_active=float4(ng,active ? 1. : 0.);
    prepared.error=float4(position_error,float(rays));
    prepared.moments=float4(0.);
    if(pilot) {GiGuide empty={};gi_guide[gid.y*image_size(output).x+gid.x]=empty;}
    // Pilots publish only gi_guide; beauty initializes its own query and guides.
    if(!pilot) gi_accumulation[at]=prepared;
    if(!active) {
        if(pilot) return;
        output[gid.xy]=surface.depth>=1. ? float4(1.,0.,0.,0.) : float4(1.,surface.view_depth,encode_octahedron(surface.normal));
        incident_output[gid.xy]=float4(0.);return;
    }
    } else {prepared=gi_accumulation[at];}
    // Inactive cells skip every later batch before receiver reconstruction or
    // the support-aware demand scan. Their first batch published filter guides.
    if(prepared.geometric_active.w==0.) {return;}
    let position=prepared.position_depth.xyz;let n=prepared.normal_view.xyz;
    let view_depth=prepared.normal_view.w;
    let ng=prepared.geometric_active.xyz;let position_error=prepared.error.xyz;
    let count=rays*uint(max(frame.temporal.y,1.));let first=uint(frame.temporal.x)*rays;
    // Replacement and radiance estimate the same clipped cosine measure.
    // The signed residual may be negative at finite sample counts.
    var hits=0.;var incident=float3(0.);
    let end=pilot ? rays : min(begin+REG_GI_RAY_BATCH,rays);
    // Narrow caps help concentrated secondary patches. Broad illumination keeps
    // the cheaper cosine proposal and its full support. The independent pilot
    // work decision is retained in rays before any beauty values are observed.
    GiProposal guide={};
    if(!pilot && rays>max(uint(frame.ao.z)/REG_GI_ADAPTIVE_RATIO,1u)*max(frame.sampling.w,1u))
        guide=gi_proposal(gid.xy,position,n);
    var best=float4(0.);var moments=prepared.moments;var pilot_stats=float2(0.);
    // Pilot and beauty streams must be independent for conditional unbiasedness.
    let sample_pixel=pilot ? gi_pixel_key(gid.xy)^uint2(0x91e10da5u,0x6c8e9cf5u) : gi_pixel_key(gid.xy);
    incident=prepared.accumulation.xyz;hits=prepared.accumulation.w;
    for(var k=begin;k<end;k++) {
        let before=incident;let before_hits=hits;
        if(!pilot) {
            incident+=gi_source<V,P>(position,n,ng,position_error,sample_pixel,first+k,count,guide);
            incident+=gi_field_source<V,P>(position,n,ng,position_error,sample_pixel,first+k,count,guide);
        }
        let u=sample_2d(sample_pixel,first+k,count,REG_SAMPLING_STREAM_RAY_GI);
        let strategy=sample_2d(sample_pixel,first+k,count,REG_SAMPLING_STREAM_PROBE_TRIANGLE);
        uint sampled_cap;
        var direction=gi_proposal_direction(guide,n,u,strategy,sampled_cap);
        let density=gi_proposal_pdf(guide,n,direction,sampled_cap);
        var direction_weight=density>0. ? max(dot(n,direction),0.)/(REG_PI*density) : 0.;
        var light_pilot=false;var light_origin=float3(0.);
        if(pilot && k>=REG_GI_PILOT_RAYS/2u) {
            let light_count=uint(frame_light_count(frame));
            if(light_count>0u) {
                let pick=light_tree_pick(gi_light_nodes,2u*light_count-1u,position,strategy);
                let source=lights[pick.identity];
                if(source.position.w>0.) {
                    light_pilot=true;light_origin=source.position.xyz;
                    if(source.position.w==3.) light_origin+=(u.x-.5)*source.emitter_u.xyz+(u.y-.5)*source.emitter_v.xyz;
                    let uv=sample_2d(sample_pixel,first+k,count,REG_SAMPLING_STREAM_SECONDARY_EMITTER);
                    let z=source.position.w==3. ? sqrt(uv.x) : 1.-2.*uv.x;
                    let radius=sqrt(max(0.,1.-z*z));let phi=2.*REG_PI*uv.y;
                    direction=float3(radius*cos(phi),radius*sin(phi),z);
                    if(source.position.w==3.) direction=mul(direction,ray_basis(normalize(source.direction.xyz)));
                    light_origin+=direction*(REG_FP32_GAMMA7*max(1.,length(light_origin)));
                    direction_weight=1.;
                }
            }
        }
        // Directions below the geometric plane would hit the receiver itself: misses.
        // Tangent ones (the sample square's edge maps to the horizon) are traced.
        if (!light_pilot && (dot(direction,ng)<0. || direction_weight<=0.)) {
            let value=dot(incident-before,float3(.2126,.7152,.0722));
            if(!pilot) moments+=float4((incident-before)*(incident-before),(hits-before_hits)*(hits-before_hits));
            if(pilot && !light_pilot) pilot_stats+=float2(value,1.);continue;
        }
        let origin=light_pilot ? light_origin : ray_origin(position,ng,direction,position_error);
        RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0.;ray.TMax=REG_FP32_MAX;
        // Explicit candidates avoid Metal's repeat drift in automatic opaque
        // closest-hit traversal. Keep one admission/commit path on every GPU;
        // hardware still culls single-sided back faces and selects the closest hit.
        RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
        query.TraceRayInline(ray_scene,RAY_FLAG_CULL_BACK_FACING_TRIANGLES,light_pilot ? REG_RAY_SHADOW : REG_RAY_GI_OCCLUDER,ray);
        while (query.Proceed()) {
            let candidate=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
            if (diffuse_candidate_accepts<P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),ray_alpha,candidate,query.CandidateTriangleFrontFace(),query.CandidateWorldToObject3x4(),direction,float2(2.)/float2(image_size(receivers)))) {query.CommitNonOpaqueTriangleHit();}
        }
        DielectricHit closest={};closest.t=-1.;
        if(query.CommittedStatus()==COMMITTED_TRIANGLE_HIT) {
            closest.t=query.CommittedRayT();closest.instance=query.CommittedInstanceIndex();closest.object=query.CommittedInstanceID();
            closest.primitive=query.CommittedPrimitiveIndex();closest.barycentrics=query.CommittedTriangleBarycentrics();closest.world_to_object=query.CommittedWorldToObject3x4();
        }
        closest=merge_field_hits<P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),closest,ray_fields(ray_alpha),origin,direction,ray.TMax,light_pilot ? REG_RAY_SHADOW : REG_RAY_GI_OCCLUDER);
        if (closest.t<0.) {
#if defined(VORK_PROBES)
            // Integrate the complete escaping segment, including medium sources.
            hits+=direction_weight;var radiance=world_radiance(direction);
            incident+=radiance*frame.radiance.w*direction_weight;
#endif
            let value=dot(incident-before,float3(.2126,.7152,.0722));
            if(!pilot) moments+=float4((incident-before)*(incident-before),(hits-before_hits)*(hits-before_hits));
            if(pilot && !light_pilot) pilot_stats+=float2(value,1.);
            continue;
        }
        if(closest.field) {
            let shading_direction=light_pilot ? normalize(origin+direction*closest.t-position) : direction;
            let h=dielectric_surface<V,P,GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),closest,shading_direction,float2(2.)/float2(image_size(receivers)),Lobe(0u,0.));
            hits+=direction_weight;
            float3 emitted;
            var radiance=finalized_hit_radiance<V,GiShadows<V,P>,P>(h,sample_pixel,first+k,REG_RAY_DIFFUSE,emitted,false,false,gi_hit_light<V>(h.v.world,sample_pixel,first+k,count,pilot,P.lit && P.lit_material(h.v)));
            let continuation=direction_weight>0. ? max(dot(n,direction),0.)/(REG_PI*direction_weight) : 0.;
            if(any(emitted!=float3(0.)))
                radiance-=emitted*(1.-gi_mis(continuation,field_emitter_pdf<GiGeometry>(GiGeometry(ray_alpha,ray_hit_buffers),gi_fields(),closest.instance,position)));
            incident+=radiance*frame.radiance.w*direction_weight;
            let value=dot(incident-before,float3(.2126,.7152,.0722));
            if(!pilot) moments+=float4((incident-before)*(incident-before),(hits-before_hits)*(hits-before_hits));
            if(pilot && !light_pilot) pilot_stats+=float2(value,1.);
            if(pilot && value>best.w) best=float4(origin+direction*closest.t,value);
            continue;
        }
        let hit=RayIntersection(query.CommittedInstanceIndex(),query.CommittedInstanceID(),query.CommittedPrimitiveIndex(),query.CommittedTriangleBarycentrics());
        hits+=direction_weight;
        let world=origin+direction*query.CommittedRayT();
        // Evaluate outgoing diffuse radiance at the actual hit. Camera colour
        // contains a different view's glossy and transmission response.
        let cosine_pdf=max(dot(n,direction),0.)/REG_PI;
        let continuation_pdf=direction_weight>0. ? cosine_pdf/direction_weight : 0.;
        let shading_direction=light_pilot ? normalize(world-position) : direction;
        var radiance=hit_shading<V,P>(hit,query.CommittedWorldToObject3x4(),world,shading_direction,position,continuation_pdf,sample_pixel,first+k,count,pilot);
        incident+=radiance*direction_weight;
        let value=dot(incident-before,float3(.2126,.7152,.0722));
            if(!pilot) moments+=float4((incident-before)*(incident-before),(hits-before_hits)*(hits-before_hits));
            if(pilot && !light_pilot) pilot_stats+=float2(value,1.);
        if(pilot && value>best.w) best=float4(world,value);
    }
    if(pilot) {
        GiGuide learned;learned.point=best;
        learned.moments=float4(pilot_stats.x/max(pilot_stats.y,1.),0.,0.,0.);
        gi_guide[gid.y*image_size(output).x+gid.x]=learned;return;
    }
    if(end<rays) {gi_accumulation[at].accumulation=float4(incident,hits);gi_accumulation[at].moments=moments;return;}
    let total=float(max(rays,1u));
    var estimate=incident/total;
    output[gid.xy]=float4(1.-hits/total,view_depth,encode_octahedron(n));
    let variance=dot(max(float3(0.),moments.xyz/total-estimate*estimate),float3(1./3.))/max(total-1.,1.);
    incident_output[gid.xy]=float4(estimate,log2(1.+variance));
    let visibility_mean=hits/total;
    gi_guide[gid.y*image_size(output).x+gid.x].moments.y=max(0.,moments.w/total-visibility_mean*visibility_mean)/max(total-1.,1.);
    gi_guide[gid.y*image_size(output).x+gid.x].moments.z=total;
}
[shader("compute")]
[numthreads(REG_COMPUTE_TILE,REG_COMPUTE_TILE,1)]
void compute(uint3 gid : SV_DispatchThreadID) {gi_pixel<Unspecialized,NoPrograms>(gid);}
