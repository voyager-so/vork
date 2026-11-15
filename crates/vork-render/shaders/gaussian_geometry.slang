// Point attributes and analytic Gaussian intersections; no traversal state.
import hit_geometry;
import hit;
import variant;
import surface;
import vork_constants;
import std.gaussian_math;
import vork_sampling;
#ifndef VORK_ATTRIBUTES
#define VORK_ATTRIBUTES 0
#endif
public bool gaussian_geometry<G:IHitGeometry>(G g,uint instance) {
#ifdef VORK_NO_GAUSSIANS
    return false;
#else
    return (g.word(g.word(instance*REG_RAY_SLOT_WORDS+2u)+8u)&(1u<<27u))!=0u;
#endif
}
public GaussianHit gaussian_intersection<G:IHitGeometry>(G g,uint instance,uint primitive,float3 origin,float3 direction,float minimum,float maximum) {
    let record=g.word(instance*REG_RAY_SLOT_WORDS+2u);let buffer=g.word(record);
    let at=g.word(record+1u)+primitive*g.word(record+2u);
    return gaussian_hit(hit_vec3<G>(g,buffer,at),hit_vec3<G>(g,buffer,at+4u),hit_vec4<G>(g,buffer,at+8u),asfloat(g.load(buffer,at+3u)),origin,direction,minimum,maximum,3.);
}
// Preserve per-(pixel,sample) identity and make each decision independent of
// native BVH traversal order. Ray coordinates distinguish continuation rays.
public bool gaussian_accept<G:IHitGeometry>(G g,uint instance,uint primitive,uint object,GaussianHit h,float3 origin,float3 direction,uint seed,uint2 launch_point) {
    let record=g.word(instance*REG_RAY_SLOT_WORDS+2u);
    if(object==launch_point.x && primitive+g.word(record+10u)==launch_point.y) return false;
    // Fold the tuple in order: a commutative XOR would give swapped object /
    // point indices the same coin and bias overlapping captures.
    var hash=pixel_hash(seed);
    hash=pixel_hash(hash^object);
    hash=pixel_hash(hash^(primitive+g.word(record+10u)));
    [unroll] for(uint i=0u;i<3u;i++) {hash=pixel_hash(hash^asuint(origin[i]));hash=pixel_hash(hash^asuint(direction[i]));}
    return h.found && float(hash>>8u)*(1./16777216.)<h.opacity;
}
public HitTriangle gaussian_triangle<G:IHitGeometry>(G g,uint instance,uint primitive,uint object,float3x4 world_to_object,float3 world) {
    let record=g.word(instance*REG_RAY_SLOT_WORDS+2u);let buffer=g.word(record);
    let at=g.word(record+1u)+primitive*g.word(record+2u);
    HitTriangle t={};t.object=object;t.primitive_offset=g.word(record+10u);t.normal_matrix=(float3x3)world_to_object;
    t.face=gaussian_normal(hit_vec3<G>(g,buffer,at+4u),hit_vec4<G>(g,buffer,at+8u));
    HitCorner c={};c.POSITION=mul(world_to_object,float4(world,1.));c.rest=c.POSITION;c.NORMAL=t.face;c.COLOR_0=float4(1.);
#if VORK_ATTRIBUTES > 0
    let offsets=g.word(record+3u);
    [unroll] for(uint k=0u;k<VORK_ATTRIBUTES;k++) {
        let descriptor=g.load(offsets,k);var value=float4(0.);
        if(descriptor!=0xffffffffu) {
            let offset=descriptor&0xffffffu;let width=descriptor>>24u;
            [unroll] for(uint channel=0u;channel<4u;channel++) {if(channel<width) value[channel]=asfloat(g.load(buffer,at+offset+channel));}
        }
        c.attributes[k]=value;
    }
#endif
    t.c[0]=c;t.c[1]=c;t.c[2]=c;return t;
}

public HitSurface gaussian_surface<V:IVariant,P:IProgram,G:IHitGeometry>(G g,uint instance,uint primitive,uint object,float3x4 inverse,float3 world,float3 direction,float2 pixel,Lobe lobe) {
    return triangle_surface<V,P>(gaussian_triangle<G>(g,instance,primitive,object,inverse,world),primitive,float2(0.),direction,pixel,lobe);
}
