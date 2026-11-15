import ray_coverage;
import frame;
import surface;
import material;
import vork_ray_alpha;
import vork_sampling;
import vork_prelude;
import bsdf;
import hit;
import variant;
import vork_constants;
import vork_ray_sampling;
import openpbr_core;

// Decorrelate the two VNDF coordinates from alpha/lobe selection and from
// other sheets on the connection. Ray origin/direction distinguish connections
// even when they hit the same point; distance distinguishes repeated field roots.
float2 shadow_sheet_sample(RayDesc ray,float distance,uint object,uint primitive,uint sample_index) {
    var key=pixel_hash(asuint(frame.display.x)^asuint(ray.Origin.x)^pixel_hash(asuint(ray.Origin.y))^pixel_hash(asuint(ray.Origin.z)));
    key=pixel_hash(key^asuint(ray.Direction.x)^pixel_hash(asuint(ray.Direction.y))^pixel_hash(asuint(ray.Direction.z)));
    let site=uint2(key,pixel_hash(object^pixel_hash(primitive)^asuint(distance)));
    return float2(screen_ray_hash(site,sample_index,REG_SAMPLING_STREAM_SHADOW_SHEET_U)>>8u,
        screen_ray_hash(site,sample_index,REG_SAMPLING_STREAM_SHADOW_SHEET_V)>>8u)*(256.*REG_RADICAL_INVERSE_SCALE);
}

// Final's transparent-shadow approximation: preserve the source connection,
// reserve interface reflection, and absorb over the actual straight chord.
// Committed closest-boundary walks already own their hit; only unordered
// candidate walks need half-open triangle edge ownership.
// Path NEE deliberately continues to use connection_candidate_transmittance.
public float3 shadow_candidate_transmittance<P:IProgram,G:IHitGeometry,let Initial:bool=false,let CandidateWalk:bool=true,V:IVariant=Unspecialized>(G g,RaytracingAccelerationStructure scene,StructuredBuffer<uint> words,RayIntersection hit,float3x4 inverse,RayDesc ray,float distance,float2 pixel,uint sample_index) {
    let beyond=distance>ray.TMax;
    if (CandidateWalk && !Initial && !beyond && !connection_owns_hit<G>(g,hit,inverse,ray.Direction)) {return float3(1.);}
    let opacity=candidate_coverage<P,G>(g,words,hit,inverse,ray.Direction,pixel);
    if (opacity<=0.) {return float3(1.);}
    let world=ray.Origin+ray.Direction*distance;
    let spatial=pixel_hash(asuint(world.x)^asuint(world.y)^asuint(world.z)^hit.instance_custom_data^hit.primitive_index);
    let u=float(screen_ray_hash(uint2(spatial,hit.instance_custom_data),sample_index,REG_SAMPLING_STREAM_SHADOW_ALPHA)>>8u)*(256.*REG_RADICAL_INVERSE_SCALE);
    let h=hit_surface<Unspecialized,P,G>(g,hit.instance_index,hit.primitive_index,hit.instance_custom_data,inverse,hit.barycentrics,ray.Direction,pixel,Lobe(select(P.lobes,LOBE_PICK,0u),u));
    let c=h.s.closure;
    if(Initial && (h.entering || h.s.is_hair || c.geometry_thin_walled || c.transmission_weight<=0.
        || c.transmission_depth<=0. || !any(c.transmission_color<float3(1.)))) {return float3(1.);}
    if (h.s.is_hair) {return beyond ? float3(1.) : float3(1.-opacity);}
    if (c.geometry_thin_walled) {
        if (beyond) {return float3(1.);}
        return (1.-opacity)+opacity*transport_shadow_transmittance<V>(c,-ray.Direction,h.s.geometric,h.face,shadow_sheet_sample(ray,distance,hit.instance_custom_data,hit.primitive_index,sample_index))*h.s.weight;
    }
    if (c.transmission_weight<=0.) {return beyond ? float3(1.) : float3(1.-opacity);}
    // Both sides use the exterior angle: this reciprocal approximation does
    // not apply total internal reflection to an intentionally unbent ray.
    var transmission=beyond ? float3(1.) : transport_shadow_transmittance<V>(c,-ray.Direction,h.s.geometric,h.face,shadow_sheet_sample(ray,distance,hit.instance_custom_data,hit.primitive_index,sample_index))*h.s.weight;
    if (c.transmission_depth>0. && any(c.transmission_color<float3(1.))) {
        // A receiver initially inside this solid has one owner independent of
        // endpoint rounding: the nearest outward boundary found by occupancy.
        if(Initial) {
            if(h.entering) {return float3(1.);}
            transmission*=transmission_attenuation(c.transmission_color,c.transmission_depth,min(distance,ray.TMax));
            return (1.-opacity)+opacity*transmission;
        }
        RayDesc chord=ray;
        if(!h.entering) {chord.Origin=world;chord.Direction=-ray.Direction;chord.TMin=0.;chord.TMax=distance;}
        // Entry discovery keeps the finite walk's complete parameterization,
        // including TMin. Changing it can round an exit on the emitter to the
        // other side and make entry and exit both claim absorption.
        // Material draws are not solid identities. Pair every interval with
        // the same owning mesh used by initial occupancy.
        let table=g.word(REG_RAY_SHADOW_TABLE_WORD);
        let identities=table==0xffffffffu ? 0u : g.word(table+2u);
        let solid=table==0xffffffffu ? 0xffffffffu : g.word(identities+hit.instance_index);
        RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
        query.TraceRayInline(scene,RAY_FLAG_NONE,REG_RAY_SHADOW,chord);
        while (query.Proceed()) {
            if ((solid==0xffffffffu ? query.CandidateInstanceIndex()==hit.instance_index : g.word(identities+query.CandidateInstanceIndex())==solid)
                && query.CandidateTriangleRayT()>(h.entering ? distance : 0.) && !query.CandidateTriangleFrontFace()) {query.CommitNonOpaqueTriangleHit();}
        }
        let found=query.CommittedStatus()==COMMITTED_TRIANGLE_HIT;
        // Initial intervals (including their exit interface coverage) belong
        // wholly to occupancy, even when that exit lies on the finite endpoint.
        if(!h.entering && !found) {return float3(1.);}
        let thickness=h.entering ? (found ? 0. : max(ray.TMax-distance,0.)) : query.CommittedRayT();
        transmission*=transmission_attenuation(c.transmission_color,c.transmission_depth,thickness);
    }
    return (1.-opacity)+opacity*transmission;
}


import implicit_scene;
#ifdef VORK_FIELD_OUTLINE
[noinline]
#endif
public float3 field_shadow_transmittance<P:IProgram,G:IHitGeometry,V:IVariant=Unspecialized>(G g,StructuredBuffer<uint> words,RayDesc ray,uint groups,bool all_casters,uint sample_index) {
    let fields=ray_fields(words);var result=float3(1.);
    for(uint i=0u;i<fields.y;i++) {
        let at=fields.x+i*3u;let object=g.word(at+1u);let descriptor=g.word(at+2u);
        if((objects[object].metadata.y&REG_RAY_SHADOW)==0u || collapsed_endpoint(objects[object].model)
            || (!all_casters && !links_intersect(objects[object].metadata.x,groups))) continue;
        var minimum=ray.TMin;
        for(;;) {
            let root=field_intersection<G>(g,descriptor,object,ray.Origin,ray.Direction,minimum,ray.TMax);
            if(!root.found) break;
            let h=field_surface<Unspecialized,P>(object,root,ray.Direction,float2(0.),Lobe(0u,0.));
            let c=h.s.closure;let opacity=h.s.discarded ? 0. : c.geometry_opacity*objects[object].opacity.x;
            var transmission=float3(0.);
            if(c.geometry_thin_walled) transmission=transport_shadow_transmittance<V>(c,-ray.Direction,h.s.geometric,h.face,shadow_sheet_sample(ray,root.t,object,descriptor,sample_index))*h.s.weight;
            else if(c.transmission_weight>0.) {
                transmission=transport_shadow_transmittance<V>(c,-ray.Direction,h.s.geometric,h.face,shadow_sheet_sample(ray,root.t,object,descriptor,sample_index))*h.s.weight;
                if(c.transmission_depth>0. && any(c.transmission_color<float3(1.))) {
                    let direction=h.entering ? ray.Direction : -ray.Direction;
                    let start=ray.Origin+ray.Direction*root.t;
                    let maximum=h.entering ? max(ray.TMax-root.t,0.) : root.t;
                    let chord=field_intersection<G>(g,descriptor,object,start,direction,REG_FP32_GAMMA7*max(1.,abs(root.t)),maximum);
                    let thickness=h.entering && chord.found ? 0. : (chord.found ? chord.t : maximum);
                    transmission*=transmission_attenuation(c.transmission_color,c.transmission_depth,thickness);
                }
            }
            result*=(1.-opacity)+opacity*transmission;
            if(!any(result>float3(0.))) return float3(0.);
            minimum=root.t+REG_FP32_GAMMA7*max(1.,abs(root.t));
            if(minimum>=ray.TMax) break;
        }
    }
    return result;
}

// The same conservative overlap proof as camera occupancy sizes per-ray
// records. This is scene-derived storage, never a crossing budget.
#ifndef VORK_SHADOW_CAPACITY
#define VORK_SHADOW_CAPACITY 1
#endif
bool shadow_bounds_hold<G:IHitGeometry>(G g,uint solid,uint bounds,float3 origin) {
    let at=bounds+solid*REG_PATH_SOLID_WORDS;
    let inverse=(float3x4)object_normal(g.word(at+6u));
    let lo=asfloat(uint3(g.word(at),g.word(at+1u),g.word(at+2u)));
    let hi=asfloat(uint3(g.word(at+3u),g.word(at+4u),g.word(at+5u)));
    let local=mul(inverse,float4(origin,1.));
    let error=REG_FP32_GAMMA7*mul(abs(inverse),float4(abs(origin),1.));
    return !any(local+error<lo) && !any(local-error>hi);
}

// Discover domains through placement-refitted bounds, then resolve all their
// nearest boundaries in ONE shared-AS traversal. Distant solids cost no linear
// discovery scan; overlapping/unbounded domains never repeat the shared walk.
public float3 shadow_contained_transmittance<P:IProgram,G:IHitGeometry,V:IVariant=Unspecialized>(G g,RaytracingAccelerationStructure scene,StructuredBuffer<uint> words,RayDesc ray,bool all_casters,uint groups,float2 pixel,uint sample_index) {
#ifdef VORK_NO_SHADOW_CONTAINMENT
    return float3(1.);
#else
    let table=g.word(REG_RAY_SHADOW_TABLE_WORD);
    if(table==0xffffffffu || VORK_SHADOW_CAPACITY==0) {return float3(1.);}
    let unbounded=g.word(table);let bounds=g.word(table+1u);let identities=g.word(table+2u);
    uint selected[VORK_SHADOW_CAPACITY>0 ? VORK_SHADOW_CAPACITY : 1];
    float nearest[VORK_SHADOW_CAPACITY>0 ? VORK_SHADOW_CAPACITY : 1];
    RayIntersection hits[VORK_SHADOW_CAPACITY>0 ? VORK_SHADOW_CAPACITY : 1];
    float3x4 inverses[VORK_SHADOW_CAPACITY>0 ? VORK_SHADOW_CAPACITY : 1];
    var count=0u;var maximum=0.;
    let buffer=g.word(table+3u);let end=g.word(table+4u)*REG_SHADOW_BOUND_NODE_WORDS;
    var node=0u;var unknown=0u;
    while(unknown<unbounded || node<end) {
        uint solid;var extent=REG_FP32_MAX;
        if(unknown<unbounded) {
            solid=g.word(table+REG_RAY_SHADOW_TABLE_HEADER_WORDS+unknown++);
        } else {
            let lo=asfloat(uint3(g.load(buffer,node),g.load(buffer,node+1u),g.load(buffer,node+2u)));
            let hi=asfloat(uint3(g.load(buffer,node+4u),g.load(buffer,node+5u),g.load(buffer,node+6u)));
            let escape=g.load(buffer,node+3u);
            if(any(ray.Origin<lo) || any(ray.Origin>hi)) {node=escape;continue;}
            solid=g.load(buffer,node+7u);
            if(solid==0xffffffffu) {node+=REG_SHADOW_BOUND_NODE_WORDS;continue;}
            node=escape;
            for(uint axis=0u;axis<3u;axis++) {
                let d=ray.Direction[axis];if(d==0.) {continue;}
                let boundary=d>0. ? hi[axis] : lo[axis];
                let t=(boundary-ray.Origin[axis])/d;
                let error=REG_FP32_GAMMA7*(abs(boundary)+abs(ray.Origin[axis]))/abs(d);
                extent=min(extent,t*(1.+REG_FP32_GAMMA7)+error);
            }
        }
        if(!shadow_bounds_hold<G>(g,solid,bounds,ray.Origin)) {continue;}
        // A violated host overlap proof is a render error, never a silent
        // crossing cutoff or an out-of-bounds local-memory write.
        if(count>=VORK_SHADOW_CAPACITY) {
            InterlockedMax(shader_errors[0],REG_PATH_BOUNDARY_OVERFLOW);
            InterlockedMin(shader_errors[1],g.word(bounds+solid*REG_PATH_SOLID_WORDS+6u));
            return float3(0.);
        }
        selected[count]=solid;nearest[count]=REG_FP32_MAX;count++;
        maximum=max(maximum,extent);
    }
    if(count==0u) {return float3(1.);}
    RayDesc occupancy=ray;occupancy.TMin=0.;occupancy.TMax=maximum;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(scene,RAY_FLAG_NONE,REG_RAY_SHADOW_SOLID,occupancy);
    while(query.Proceed()) {
        let identity=g.word(identities+query.CandidateInstanceIndex());
        var slot=0u;while(slot<count && selected[slot]!=identity) {slot++;}
        if(slot==count || query.CandidateTriangleRayT()>=nearest[slot]) {continue;}
        if(!all_casters && !links_intersect(objects[query.CandidateInstanceID()].metadata.x,groups)) {continue;}
        nearest[slot]=query.CandidateTriangleRayT();
        hits[slot]=RayIntersection(query.CandidateInstanceIndex(),query.CandidateInstanceID(),query.CandidatePrimitiveIndex(),query.CandidateTriangleBarycentrics());
        inverses[slot]=query.CandidateWorldToObject3x4();
    }
    var result=float3(1.);
    for(uint slot=0u;slot<count;slot++) {
        if(nearest[slot]==REG_FP32_MAX) {continue;}
        result*=shadow_candidate_transmittance<P,G,true,true,V>(g,scene,words,hits[slot],inverses[slot],ray,nearest[slot],pixel,sample_index);
        if(!any(result>float3(0.))) {return float3(0.);}
    }
    return result;
#endif
}
