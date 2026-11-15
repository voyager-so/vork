import catcher;
// Scene boundary services; callers supply the shared ray table and hit geometry.
import vork_constants;
import frame;
import material;
import hit;
import implicit_scene;
import hit_geometry;
#ifndef VORK_NO_GAUSSIANS
import gaussian_geometry;
#endif
import variant;
import surface;
import stream;
import closure;
import path_boundary;
#include "path_work_arguments.slang"
struct DielectricHit { float t; uint instance; uint primitive; uint object; float2 barycentrics; float3x4 world_to_object; bool field; FieldHit field_hit;
#ifndef VORK_NO_GAUSSIANS
    float3 gaussian_point;
#endif
};
// Merge field zero sets with an existing hardware triangle hit. Both camera
// path transport and raster dielectric continuation use the same acceptance,
// visibility masks and contact tie rule. Domain-box triangles never shade.
#ifdef VORK_FIELD_OUTLINE
[noinline]
#endif
DielectricHit merge_field_hits<P:IProgram,G:IHitGeometry>(G g,DielectricHit h,uint2 fields,float3 origin,float3 direction,float distance,uint mask,uint only_object=0xffffffffu) {
    for(uint i=0u;i<fields.y;i++) {
        let at=fields.x+i*3u;let instance=g.word(at);let object=g.word(at+1u);
        if((only_object!=0xffffffffu && object!=only_object) || !catcher_keeps(object) || (objects[object].metadata.y&mask)==0u || collapsed_endpoint(objects[object].model)) continue;
        var minimum=0.;let maximum=h.t>=0. ? h.t : distance;
        for(;;) {
            let root=field_intersection<G>(g,g.word(at+2u),object,origin,direction,minimum,maximum);
            if(!root.found) break;
            if(h.t>=0. && !h.field) {
                // Resolve a numerically coincident contact in favour of the
                // triangle. Accepting the field exit instead can offset the
                // continuation through an opaque receiver touching the liquid.
                // Use the existing world-position bounds, not a scene epsilon.
                let field_error=surface_position_error(object,origin+direction*root.t);
                let triangle_error=surface_position_error(h.object,origin+direction*h.t);
                if(h.t-root.t<=length(field_error)+length(triangle_error)) break;
            }
            let surface=field_surface<Unspecialized,P>(object,root,direction,float2(0.),Lobe(0u,0.));
            if(!surface.s.discarded && surface.s.closure.geometry_opacity>0.) {
                h.field=true;h.field_hit=root;h.t=root.t;h.instance=instance;h.object=object;
                h.world_to_object=(float3x4)object_normal(object);break;
            }
            minimum=root.t+REG_FP32_GAMMA7*max(1.,abs(root.t));
            if(minimum>=maximum) break;
        }
    }
    return h;
}
HitSurface dielectric_surface<V:IVariant,P:IProgram,G:IHitGeometry>(G g,DielectricHit h,float3 direction,float2 pixel,Lobe lobe) {
    if(h.field) return field_surface<V,P>(h.object,h.field_hit,direction,pixel,lobe);
#ifndef VORK_NO_GAUSSIANS
    if(gaussian_geometry<G>(g,h.instance)) return gaussian_surface<V,P,G>(g,h.instance,h.primitive,h.object,h.world_to_object,h.gaussian_point,direction,pixel,lobe);
#endif
    return hit_surface<V,P,G>(g,h.instance,h.primitive,h.object,h.world_to_object,h.barycentrics,direction,pixel,lobe);
}
void path_boundary_error(uint code, uint object) {
    InterlockedMax(shader_errors[0],code);InterlockedMin(shader_errors[1],object);
}
void boundary_overflow<let N:uint>(PathOccupancy<N> state,uint object) {
    path_boundary_error(REG_PATH_BOUNDARY_OVERFLOW,object);
    InterlockedMin(shader_errors[16],object);
    if(state.count>0u) InterlockedMin(shader_errors[17],state.at(state.count-1u).object);
    if(state.count>1u) InterlockedMin(shader_errors[18],state.at(state.count-2u).object);
    InterlockedMax(shader_errors[19],state.count);
}
void path_cross<let N:uint>(inout PathOccupancy<N> state, PathBoundary boundary, bool entering) {
    if (!state.cross(boundary,entering)) {boundary_overflow(state,boundary.object);}
}
void plain_path_cross<let N:uint>(inout PathOccupancy<N> state,PathBoundary boundary,bool entering) {
    if(!state.cross<EqualBoundaryPriority>(boundary,entering)) boundary_overflow(state,boundary.object);
}
// The solid a ray instance bounds, or 0xffffffff.
uint instance_solid<G:IHitGeometry>(G g,uint instance,uint2 tables) {
    let identity=g.word(tables.x+instance);
    return identity==REG_RAY_OPAQUE_SOLID_OWNER ? 0xffffffffu : identity;
}
// Side admission retains every patch of an installed possible solid, even if
// its evaluated transmission is zero. Occupancy still uses boundary IDs above.
bool instance_has_solid_owner<G:IHitGeometry>(G g,uint instance,uint2 tables) {return g.word(tables.x+instance)!=0xffffffffu;}
// A solid's words (REG_PATH_SOLID_WORDS): object-space lo.xyz, hi.xyz, then
// the object row whose transform (the TLAS instance's) places them.
uint solid_words(uint solid,uint2 tables) {return tables.y+solid*REG_PATH_SOLID_WORDS;}
float3 solid_lo<G:IHitGeometry>(G g,uint at) {return asfloat(uint3(g.word(at),g.word(at+1u),g.word(at+2u)));}
float3 solid_hi<G:IHitGeometry>(G g,uint at) {return asfloat(uint3(g.word(at+3u),g.word(at+4u),g.word(at+5u)));}
float3x4 solid_world_to_object<G:IHitGeometry>(G g,uint at) {return (float3x4)object_normal(g.word(at+6u));}
// Whether a solid's bounds, placed by world_to_object (camera containment has
// it from the ray query), can hold world, with the f32 error of that affine
// transform. passes::path::placed_bounds (and its test's copy of this margin)
// pads install bounds to hold every point this lets in. Unbounded solids
// (±inf) hold every point; a placement without an inverse (NaN) is no
// evidence against holding it.
bool solid_bounds_hold<G:IHitGeometry>(G g,uint solid, float3x4 world_to_object, float3 world,uint2 tables) {
    let at=solid_words(solid,tables);
    let local=mul(world_to_object,float4(world,1.));
    let error=REG_FP32_GAMMA7*mul(abs(world_to_object),float4(abs(world),1.));
    return !any(local+error<solid_lo<G>(g,at)) && !any(local-error>solid_hi<G>(g,at));
}
// The tracked set can hold a solid the path left without crossing its
// boundary, where rounding loses an exit hit. A path is never inside a solid
// beyond its bounds, so at each hit, before its segment is attenuated or its
// interface resolved, drop the tracked solids whose bounds cannot hold the hit
// point (a hit on a solid's own surface is within them). The set keeps its
// order, so the owner among the others is unchanged. This is a conservative
// rejection only; actual membership is established by geometric crossings.
void reconcile_occupancy<G:IHitGeometry,let N:uint>(G g,inout PathOccupancy<N> state, float3 world, uint hit_solid,uint2 tables) {
    for (var slot=state.count;slot>0u;slot--) {
        let solid=state.at(slot-1u).object;
        if (solid!=hit_solid && !solid_bounds_hold<G>(g,solid,solid_world_to_object<G>(g,solid_words(solid,tables)),world,tables)) {state.remove(slot-1u);}
    }
}
// Occupancy describes the volume, independently of the stochastic surface lobe.
// A heterogeneous interface mixture with conflicting volumes is ambiguous;
// reject it instead of making camera-inside state depend on the selected lobe.
void validate_boundary<V:IVariant,P:IProgram>(Varying v,bool entering,Coordinates k,Surface s) {
    let c=s.closure;
    if (P.lobes && s.lobes>1u) {
        for (var i=0u;i<s.lobes;i++) {
            let other=surface_at<V,P>(v,entering,s.view,k,Lobe(i,0.)).closure;
            if (other.geometry_thin_walled!=c.geometry_thin_walled || other.specular_ior!=c.specular_ior || other.specular_weight!=c.specular_weight || other.transmission_dispersion_scale!=c.transmission_dispersion_scale || other.transmission_dispersion_abbe_number!=c.transmission_dispersion_abbe_number || any(other.transmission_color!=c.transmission_color) || other.transmission_depth!=c.transmission_depth) {path_boundary_error(REG_PATH_BOUNDARY_AMBIGUOUS,v.object);}
        }
    }
}
PathBoundary boundary_descriptor<V:IVariant,P:IProgram,G:IHitGeometry>(G g,DielectricHit hit, float3 direction, HitSurface h,uint2 tables,float2 pixel) {
    validate_boundary<V,P>(h.v,h.entering,h.k,h.s);
    return path_boundary(instance_solid<G>(g,hit.instance,tables),h.s.closure,g.word(hit.instance*REG_RAY_SLOT_WORDS+1u));
}
// A transaction in the existing set: retain exited descriptors until the BSDF
// decides, and provisionally insert entries. Only interiors and contacts at
// this point need storage; scene-wide candidate counts never size this set.
// Transaction tags live beside the existing descriptors, including spills.
// 0: untouched, 1: exit, 2: provisional entry, 3: absent exit, 4: unchanged.
// The absent-exit marker deduplicates open-surface contacts without making that
// solid an optical owner. It is removed on both transmission and reflection.
struct BoundaryEvent<let N:uint> {
    uint count;
    uint change(PathOccupancy<N> state,uint object) {
        let slot=state.find(object);return slot<state.count ? state.tag(slot)&3u : 0u;
    }
    bool contains(PathOccupancy<N> state,uint object) {
        let slot=state.find(object);return slot<state.count && state.tag(slot)!=0u;
    }
    [mutating] void add(inout PathOccupancy<N> state,PathBoundary b,bool entering) {
        let existed=state.find(b.object)<state.count;
        if(!existed && !state.cross(b,true)) {boundary_overflow(state,b.object);return;}
        state.set_tag(state.find(b.object),entering ? (existed ? 4u : 2u) : (existed ? 1u : 3u));count++;
    }
    void order_entries(inout PathOccupancy<N> state) {
        // Absent exits were retained only for candidate deduplication. Remove
        // them before ordering entries so they cannot split an equal-priority
        // run and change the deterministic last-entered owner.
        for(var slot=state.count;slot>0u;slot--) {if(state.tag(slot-1u)==3u) state.remove(slot-1u);}
        if(count<=1u) return;
        for(uint i=1u;i<state.count;i++) {
            let e=state.entry(i);if(e.tag!=2u) continue;
            var slot=i;
            while(slot>0u) {
                let previous=state.entry(slot-1u);
                if(boundary_priority(previous.boundary)!=boundary_priority(e.boundary) || previous.boundary.object<e.boundary.object || previous.tag!=2u) break;
                state.set_entry(slot,previous);slot--;
            }
            state.set_entry(slot,e);
        }
        state.resolve_iors();
    }
    PathBoundary owner(PathOccupancy<N> state) {
        for(var slot=state.count;slot>0u;slot--) {
            let e=state.entry(slot-1u);if(e.tag!=1u && e.tag!=3u) return e.boundary;
        }
        return path_air();
    }
    void finish(inout PathOccupancy<N> state,bool transmitted) {
        for(var slot=state.count;slot>0u;slot--) {
            let kind=state.tag(slot-1u);
            if(kind==3u || kind==(transmitted ? 1u : 2u)) state.remove(slot-1u);
            else state.set_tag(slot-1u,0u);
        }
    }
};
// Resume within the existing hit-position uncertainty, excluding processed
// primitives. Native intersections of a shared face can differ by many t ULPs
// (large triangles, oblique rays); never offset through that interval.
// Every gathered solid must hold the same hit point under the existing bounds
// test, rejecting separated boxes without inferring geometric containment.
// Deduplicate solids and order simultaneous entries
// by solid identity so last-entered ties do not depend on hardware traversal.
float boundary_distance_error(float3 error,float3 face,float3 direction) {
    return dot(abs(face),error)/max(abs(dot(face,direction)),REG_FP32_GAMMA7);
}
// Only reject traversal outside conservative static surface bounds. Closure
// uncertainty can make a medium unbounded without making its surface infinite.
// Moving/unknown geometry and larger sets always retain the normal AS query.
bool boundary_contact_possible<G:IHitGeometry>(G g,uint solid,float3 world,float3 interval,uint2 tables) {
    let count=g.word(tables.y-1u);
    if(count>8u) return true;
    for(uint i=0u;i<count;i++) {
        if(i==solid) continue;
        let at=tables.y+count*REG_PATH_SOLID_WORDS+i*6u;
        if(!any(world+interval<solid_lo<G>(g,at)) && !any(world-interval>solid_hi<G>(g,at))) return true;
    }
    return false;
}
BoundaryEvent<N> boundary_event<V:IVariant,P:IProgram,G:IHitGeometry,let N:uint>(G g,
    RaytracingAccelerationStructure scene,float3 origin,float3 direction,DielectricHit first,float3 world,float3 error,float3 face,
    PathBoundary boundary,bool entering,inout PathOccupancy<N> state,uint2 tables,float2 pixel,uint mask VORK_WORK_PARAMETER) {
    BoundaryEvent<N> event={};
    if(N==0u) {path_cross(state,boundary,entering);return event;}
    event.add(state,boundary,entering);
    if(N<=1u || first.t<0.) return event;
    let distance_error=boundary_distance_error(error,face,direction);
    if(!boundary_contact_possible<G>(g,boundary.object,world,error+abs(direction)*distance_error,tables)) return event;
    let minimum=max(0.,first.t-distance_error);
    let maximum=min(REG_FP32_MAX,first.t+distance_error);
    RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=minimum;
    ray.TMax=asfloat(min(asuint(REG_FP32_MAX),asuint(maximum)+1u));
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene,RAY_FLAG_FORCE_NON_OPAQUE,mask,ray);
    while(query.Proceed()) {
#ifdef VORK_MEDIUM_WORK_LIMIT
        if(!medium_work_step(medium_work)) {break;}
#endif
        if(query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE || query.CandidateTriangleRayT()<minimum || query.CandidateTriangleRayT()>maximum) continue;
        let instance=query.CandidateInstanceIndex();let primitive=query.CandidatePrimitiveIndex();
        if(instance==first.instance && primitive==first.primitive) continue;
        let identity=instance_solid<G>(g,instance,tables);
        if(identity==0xffffffffu || !catcher_keeps(query.CandidateInstanceID())) continue;
        if(event.contains(state,identity) || !solid_bounds_hold<G>(g,identity,query.CandidateWorldToObject3x4(),world,tables)) continue;
        DielectricHit hit={};hit.t=first.t;hit.instance=instance;hit.primitive=primitive;hit.object=query.CandidateInstanceID();
        hit.barycentrics=query.CandidateTriangleBarycentrics();hit.world_to_object=query.CandidateWorldToObject3x4();
#ifdef VORK_MEDIUM_WORK_LIMIT
        if(!medium_work_step(medium_work)) {break;}
#endif
        let h=dielectric_surface<V,P,G>(g,hit,direction,pixel,Lobe(0u,0.));
        if(h.s.discarded || h.s.closure.geometry_opacity<=0. || h.s.closure.geometry_thin_walled || h.s.closure.transmission_weight<=0.) continue;
        let b=boundary_descriptor<V,P,G>(g,hit,direction,h,tables,pixel);
        event.add(state,b,h.entering);
    }
    event.order_entries(state);
    return event;
}
// Retain only proven interiors, in deterministic priority/exit order.
void camera_interior<let N:uint>(inout PathOccupancy<N> state,PathBoundary b,float distance) {
    BoundaryEntry e={};e.boundary=b;e.exit_distance=distance;
    if(!state.append(e)) {boundary_overflow(state,b.object);return;}
    var slot=state.count-1u;
    while(slot>0u) {
        let previous=state.entry(slot-1u);let p=previous.boundary;
        if(boundary_priority(p)<boundary_priority(b) ||
           (boundary_priority(p)==boundary_priority(b) &&
            (previous.exit_distance>distance || (previous.exit_distance==distance && p.object<b.object)))) break;
        state.set_entry(slot,previous);slot--;
    }
    state.set_entry(slot,e);
    state.resolve_iors();
}
PathOccupancy<N> camera_occupancy<V:IVariant,P:IProgram,G:IHitGeometry,let N:uint>(G g,RaytracingAccelerationStructure scene,float3 origin, float3 direction,uint2 tables,float2 pixel,uint field_offset,uint field_count VORK_WORK_PARAMETER) {
    PathOccupancy<N> state={};
    // Installation can prove that no camera sample starts inside a solid
    // (passes::path::camera_exterior); the query then compiles away.
    if(N==0u || !V.has_camera_interior) return state;
    // Bounds reject impossible interiors; they never establish containment.
    // Resolve each solid's nearest actual boundary before retaining a descriptor.
    // In particular, arbitrarily many separated deforming/open solids may have
    // infinite conservative bounds without consuming any interior-list slots.
    var first_identity=0u;
    for(;;) {
        // Enumerate only identities encountered by the containment ray. A ray
        // outside all finite bounds completes in one query, regardless of the
        // scene's solid count. Increasing identity guarantees progress without
        // retaining a scene-sized candidate set.
        var selected=0xffffffffu;
        RayDesc ray;ray.Origin=origin;ray.Direction=direction;ray.TMin=0.;ray.TMax=REG_FP32_MAX;
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(scene,RAY_FLAG_FORCE_NON_OPAQUE,255u,ray);
        DielectricHit nearest={};nearest.t=-1.;
        while(query.Proceed()) {
#ifdef VORK_MEDIUM_WORK_LIMIT
            if(!medium_work_step(medium_work)) break;
#endif
            if(query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE || !catcher_keeps(query.CandidateInstanceID())) continue;
            let instance=query.CandidateInstanceIndex();
            let identity=instance_solid<G>(g,instance,tables);
            if(identity==0xffffffffu || identity<first_identity || identity>selected) continue;
            if(!solid_bounds_hold<G>(g,identity,query.CandidateWorldToObject3x4(),origin,tables)) continue;
            let distance=query.CandidateTriangleRayT();
            if(identity==selected && nearest.t>=0. && distance>=nearest.t) continue;
            selected=identity;
            nearest.t=distance;nearest.instance=instance;nearest.object=query.CandidateInstanceID();
            nearest.primitive=query.CandidatePrimitiveIndex();nearest.barycentrics=query.CandidateTriangleBarycentrics();
            nearest.world_to_object=query.CandidateWorldToObject3x4();
        }
        if(nearest.t<0.) break;
        first_identity=selected+1u;
#ifdef VORK_MEDIUM_WORK_LIMIT
        if(!medium_work_step(medium_work)) break;
#endif
        let h=dielectric_surface<V,P,G>(g,nearest,direction,pixel,Lobe(0u,0.));
        if(h.entering || h.s.closure.geometry_thin_walled || h.s.closure.transmission_weight<=0.) continue;
        camera_interior(state,boundary_descriptor<V,P,G>(g,nearest,direction,h,tables,pixel),nearest.t);
    }
    for(uint i=0u;i<field_count;i++) {
        let at=field_offset+i*3u;let instance=g.word(at);let object=g.word(at+1u);
        let identity=instance_solid<G>(g,instance,tables);
        if(!catcher_keeps(object) || (objects[object].metadata.y&~REG_OBJECT_IMPLICIT)==0u || collapsed_endpoint(objects[object].model) || identity==0xffffffffu || !solid_bounds_hold<G>(g,identity,(float3x4)object_normal(object),origin,tables)) continue;
        let root=field_intersection<G>(g,g.word(at+2u),object,origin,direction,0.,REG_FP32_MAX);
        if(!root.found) continue;
        let h=field_surface<V,P>(object,root,direction,pixel,Lobe(0u,0.));
        if(h.entering || h.s.closure.geometry_thin_walled || h.s.closure.transmission_weight<=0.) continue;
        DielectricHit boundary_hit={};boundary_hit.instance=instance;boundary_hit.object=object;
        camera_interior(state,boundary_descriptor<V,P,G>(g,boundary_hit,direction,h,tables,pixel),root.t);
    }
    return state;
}
