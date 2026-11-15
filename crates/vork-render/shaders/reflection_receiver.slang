// Coverage-aware receivers for scene ray reflections.
import vork_constants;
import vork_prelude;
import vork_sampling;
import vork_visibility;
import decoupled_slots;
import reflection_bsdf;


uint reflection_guide_at(uint pixel,uint world_class) {
    return (pixel*REG_REFLECTION_CLASSES+world_class)*REG_VISIBILITY_REFLECTION_GUIDE_WORDS;
}

struct ReflectionLobe { float3 reflected; float roughness; };
ReflectionLobe reflection_response(float2 payload) {
    let packed=float2(surface_specular_payload(payload.x),surface_reflection_payload(payload.y));
    let radix=REG_REFLECTION_PACK_BASE;let levels=REG_REFLECTION_PACK_LEVELS;
    let encoded=float3(packed.x%radix,floor(packed.x/radix),packed.y%radix)/levels;
    let reflected=encoded*encoded;let roughness=surface_reflection_roughness(payload.y);
    return ReflectionLobe(reflected,roughness);
}
struct ReflectionGuide { float4 position; float3 normal; float coverage; float roughness; };
ReflectionGuide reflection_guide(StructuredBuffer<uint> guides,uint2 size,int2 pixel,uint world_class) {
    let at=reflection_guide_at(uint(pixel.y)*size.x+uint(pixel.x),world_class);
    return ReflectionGuide(asfloat(uint4(guides[at],guides[at+1u],guides[at+2u],guides[at+3u])),
        decode_octahedron(unpackHalf2x16ToFloat(guides[at+4u])),asfloat(guides[at+5u]),asfloat(guides[at+6u]));
}
struct ReflectionSelection { int2 pixel; float source_tag; bool eligible; };
ReflectionSelection reflection_receiver(StructuredBuffer<uint> guides,uint2 size,uint2 cell,uint world_class) {
    let pixel=min(int2(cell),int2(size)-1);
    let guide=reflection_guide(guides,size,pixel,world_class);
    return ReflectionSelection(pixel,1.,guide.position.w<1. && guide.coverage>0.);
}

// A traced receiver and its geometric ray origin.
struct ReflectionSite { ReflectionBsdf bsdf; ReflectionLobe receiver; float3 normal; float3 position; float3 geometric; float3 error; bool world_specular; };
// Frame-invariant camera ray basis. Plane intersections retain every valid
// positive distance without an anchor-dependent projective clamp.
struct ReflectionCamera {
    float3 right;
    float3 up;
    float3 forward;
    float2 center;
    float2 scale;
    bool ortho;
    __init(Frame frame) {
        right=float3(frame.view[0].x,frame.view[1].x,frame.view[2].x);
        up=float3(frame.view[0].y,frame.view[1].y,frame.view[2].y);
        forward=-float3(frame.view[0].z,frame.view[1].z,frame.view[2].z);
        ortho=frame.flags.x>.5;
        let clip=mul(float4(select(ortho,frame.eye.xyz,forward),select(ortho,1.,0.)),frame.vp);
        center=clip.xy/clip.w;scale=1./frame.projection.yx;
    }
    float3 point(float2 pixel,uint2 size,float3 normal,float distance) {
        let ndc=pixel/float2(size)*float2(2.,-2.)+float2(-1.,1.);
        let lateral=right*((ndc.x-center.x)*scale.x)+up*((ndc.y-center.y)*scale.y);
        let origin=select(ortho,lateral,float3(0.));let direction=forward+select(ortho,float3(0.),lateral);
        return origin+direction*((distance-dot(origin,normal))/dot(direction,normal));
    }
}
// Each full-resolution pixel supplies its own estimate.
// Decode a slot once, then visit only its covered indices. Overflow slots use
// exactly the same iterator, without a local list or another capacity bound.
struct ReflectionSlots {
    SlotLayout layout;
    uint pixel;
    int2 selected;
    uint2 size;
    uint dense;
    uint pool;
    uint at;
    uint word;
    uint bits;
    ReflectionSite site;
    float distance;
    ReflectionCamera camera;
    uint world_class;
    bool merged;
    uint first_word;
    uint last_word;
    __init(StructuredBuffer<uint> slots,uint2 size,int2 selected,Frame frame,uint world_class,uint begin=0u,uint end=0xffffffffu,bool merged=true,uint packet_pixels=0u,uint first_pixel=0u) {
        this.merged=merged;
        first_word=begin/32u;last_word=min((end==0xffffffffu ? frame.sampling.w : end)+31u,frame.sampling.w+31u)/32u;
        this.layout=SlotLayout(size.x*size.y,frame.sampling.w,packet_pixels,first_pixel);
        this.pixel=uint(selected.y)*size.x+uint(selected.x);this.selected=selected;this.size=size;this.world_class=world_class;
        // Only opaque single-sample views omit the slot buffer. Transparent
        // views retain all merged receivers even with a single camera sample.
        dense=0u;pool=0u;if (merged) {pool=slots[layout.head(pixel)];}at=0u;word=layout.masks;bits=0u;
        site=(ReflectionSite)0;distance=0.;camera=ReflectionCamera(frame);
    }
    [mutating] bool next(StructuredBuffer<uint> slots,StructuredBuffer<uint> records[REG_VISIBILITY_REFLECTION_WORDS],StructuredBuffer<uint> sampling,Frame frame,out ReflectionSite receiver,out uint index) {
        receiver=(ReflectionSite)0;index=0u;
        for (;;) {
            if (bits!=0u) {
                let sample=(word-1u)*32u+firstbitlow(bits);bits&=bits-1u;
                let at=frame.sampling.z+sample*2u;
                let offset=asfloat(uint2(sampling[at],sampling[at+1u]))*float2(-1.,1.);
                receiver=site;receiver.position=frame.eye.xyz+camera.point(float2(selected)+.5+offset,size,site.geometric,distance);
                receiver.error=screen_world_error(receiver.position,frame);
                index=uint(frame.temporal.x)*frame.sampling.w+sample;return true;
            }
            // A traceable slot's mask words, then the next slot.
            if (word<last_word) {
                bits=1u;if (merged) {bits=slots[at+REG_VISIBILITY_KEY_WORDS+word];}
                word++;continue;
            }
            if (dense<(merged ? REG_VISIBILITY_SLOTS : 1u)) {
                at=layout.dense(pixel,dense++);
                if (merged && slots[at]==0u) {dense=REG_VISIBILITY_SLOTS;continue;}
            } else if (pool!=0u) {
                let entry=layout.entry(pool-1u);at=entry+1u;pool=slots[entry];
            } else {return false;}
            let base=layout.reflection(at,!merged);
            let payload=unpackHalf2x16ToFloat(records[5u][base]);
            if (surface_world_specular(payload.x)!=(world_class==0u) || !surface_reflection_traceable(payload)) {continue;}
            site.receiver=reflection_response(payload);
            for(uint i=0u;i<REG_REFLECTION_BSDF_VECTORS-1u;i++) {site.bsdf.data[i]=float4(unpackHalf2x16ToFloat(records[6u+i*2u][base]),unpackHalf2x16ToFloat(records[7u+i*2u][base]));}
            site.bsdf.data[11]=float4(unpackHalf2x16ToFloat(records[REG_VISIBILITY_REFLECTION_WORDS-1u][base]),site.bsdf.data[9].w,site.bsdf.data[10].w);
            site.bsdf.data[9].w=0.;site.bsdf.data[10].w=0.;
            site.geometric=asfloat(uint3(records[0u][base],records[1u][base],records[2u][base]));
            distance=asfloat(records[3u][base]);
            site.normal=decode_octahedron(unpackHalf2x16ToFloat(records[4u][base]));
            site.world_specular=world_class==0u;
            word=first_word;
        }
    }
}

// Each receiver traces one lobe ray above its facet.
interface IReflectionTrace { static float connection_probability(); static float3 direct(ReflectionSite site,uint2 cell,uint index,float3 incident); static float3 trace(ReflectionSite site,uint2 cell,uint index,float3 incident,float3 ray,float pdf); }
// One strategy consumes one camera sample. Equal selection probabilities keep
// the source/direction balance weights symmetric; only emitted radiance competes.
float3 reflection_sample<T:IReflectionTrace>(ReflectionSite site,uint2 cell,uint index,float3 incident,float2 u,
    float connection_probability) {
    if(u.x<connection_probability) {return T.direct(site,cell,index,incident)/connection_probability;}
    u.x=(u.x-connection_probability)/(1.-connection_probability);
    let ray=reflection_bsdf_sample(site.bsdf,-incident,u);
    float pdf;let response=reflection_bsdf_evaluate(site.bsdf,-incident,ray,pdf);
    let weight=pdf>0. ? response/pdf : float3(0.);
    if (dot(ray,site.geometric)<=0. || all(weight==float3(0.))) {return float3(0.);}
    site.receiver.reflected=weight/(1.-connection_probability);
    return T.trace(site,cell,index,incident,ray,pdf);
}
void reflection_integrate<T:IReflectionTrace>(uint3 gid,Frame frame,uint2 size,
    StructuredBuffer<uint> reflection_slots,StructuredBuffer<uint> reflection_records[REG_VISIBILITY_REFLECTION_WORDS],StructuredBuffer<uint> reflection_guides,StructuredBuffer<uint> sampling,uint2 range,RWStructuredBuffer<float4> accumulation,RWTexture2DArray<float4> output,bool merged=true,uint packet_pixels=0u,uint first_pixel=0u) {
    if (any(gid.xy>=size) || gid.z>=REG_REFLECTION_CLASSES) {return;}
    let receiver=reflection_receiver(reflection_guides,size,gid.xy,gid.z);
    if (!receiver.eligible) {output[gid]=float4(0.);return;}
    let connection_probability=T.connection_probability();
    var sum=float3(0.);
    var slots=ReflectionSlots(reflection_slots,size,receiver.pixel,frame,gid.z,range.x,range.y,merged,packet_pixels,first_pixel);
    ReflectionSite site;uint index;
    let count=sampling[frame.sampling.x];
    while (slots.next(reflection_slots,reflection_records,sampling,frame,site,index)) {
        // Remove exactly what this covered sample received from camera shading,
        // even when the sampled direction is below the geometric hemisphere.
        sum-=reflection_bsdf_environment(site.bsdf);
        let incident=-camera_view(site.position,frame);
        let u=sample_2d_installed(sampling,frame.sampling.x,gid.xy,index,count,REG_SAMPLING_STREAM_REFLECTION);
        sum+=reflection_sample<T>(site,gid.xy,index,incident,u,connection_probability);
    }
    let at=(gid.y*size.x+gid.x)*REG_REFLECTION_CLASSES+gid.z;
    let previous=range.x==0u ? float3(0.) : accumulation[at].rgb;
    let total=previous+sum;
    accumulation[at]=float4(total,0.);
    if(range.y>=frame.sampling.w) {output[gid]=float4(total/float(frame.sampling.w),receiver.source_tag);}
}
