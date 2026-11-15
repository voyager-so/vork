// Vertex streams, raster varyings and the camera/depth vertex stages; entry
// modules (depth.slang, generated forward variants) declare the entry points.
// The forward pass tests depth Equal against the depth passes, so every camera
// position comes from clip_position(): Slang has no Invariant qualifier, and one
// compiler evaluating one expression is what keeps the depths bit-identical.
import vork_constants;
import frame;
// Named geometry attributes (attributes.rs): their count, defined by the compile of
// scenes that bind materials to them; zero for the precompiled modules.
#ifndef VORK_ATTRIBUTES
#define VORK_ATTRIBUTES 0
#endif

// Vertex attribute locations match geometry::VertexLayout::attributes().
struct VertexIn {
    uint vertex_index : SV_VertexID;
    [[vk::location(15)]] float3 rest : REST;
    [[vk::location(0)]] float3 POSITION : POSITION;
    [[vk::location(1)]] float3 NORMAL : NORMAL;
    [[vk::location(2)]] float2 TEXCOORD_0 : TEXCOORD_0;
    [[vk::location(6)]] float2 TEXCOORD_1 : TEXCOORD_1;
    [[vk::location(3)]] float3 previous_p : PREVIOUS;
    [[vk::location(16)]] float3 next_p : NEXT;
    [[vk::location(4)]] float4 COLOR_0 : COLOR_0;
    [[vk::location(5)]] float4 TANGENT : TANGENT;
#if VORK_ATTRIBUTES > 0
    [[vk::location(7)]] float4 attr0 : ATTRIBUTE0;
#endif
#if VORK_ATTRIBUTES > 1
    [[vk::location(8)]] float4 attr1 : ATTRIBUTE1;
#endif
#if VORK_ATTRIBUTES > 2
    [[vk::location(9)]] float4 attr2 : ATTRIBUTE2;
#endif
#if VORK_ATTRIBUTES > 3
    [[vk::location(10)]] float4 attr3 : ATTRIBUTE3;
#endif
#if VORK_ATTRIBUTES > 4
    [[vk::location(11)]] float4 attr4 : ATTRIBUTE4;
#endif
#if VORK_ATTRIBUTES > 5
    [[vk::location(12)]] float4 attr5 : ATTRIBUTE5;
#endif
#if VORK_ATTRIBUTES > 6
    [[vk::location(13)]] float4 attr6 : ATTRIBUTE6;
#endif
#if VORK_ATTRIBUTES > 7
    [[vk::location(14)]] float4 attr7 : ATTRIBUTE7;
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    // Metal accepts no array as a stage input (a vertex attribute or a fragment
    // varying), so there the attributes after the eighth are separate members at
    // the array's locations. Eleven reach Metal's last vertex attribute location.
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+1)]] float4 attr8 : ATTRIBUTE8;
#if VORK_ATTRIBUTES > 9
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+2)]] float4 attr9 : ATTRIBUTE9;
#endif
#if VORK_ATTRIBUTES > 10
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+3)]] float4 attr10 : ATTRIBUTE10;
#endif
#if VORK_ATTRIBUTES > 11
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+4)]] float4 attr11 : ATTRIBUTE11;
#endif
#if VORK_ATTRIBUTES > 12
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+5)]] float4 attr12 : ATTRIBUTE12;
#endif
#if VORK_ATTRIBUTES > 13
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+6)]] float4 attr13 : ATTRIBUTE13;
#endif
#if VORK_ATTRIBUTES > 14
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+7)]] float4 attr14 : ATTRIBUTE14;
#endif
#if VORK_ATTRIBUTES > 15
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+8)]] float4 attr15 : ATTRIBUTE15;
#endif
#if VORK_ATTRIBUTES > 16
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+9)]] float4 attr16 : ATTRIBUTE16;
#endif
#if VORK_ATTRIBUTES > 17
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+10)]] float4 attr17 : ATTRIBUTE17;
#endif
#if VORK_ATTRIBUTES > 18
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+11)]] float4 attr18 : ATTRIBUTE18;
#endif
#else
    [[vk::location(REG_VISIBILITY_LENS_LOCATION+1)]] float4 attr_extra[VORK_ATTRIBUTES-8] : ATTRIBUTE_EXTRA;
#endif
#endif

};
struct Varying {
    // The decoupled visibility pass's sample (decoupled_visibility.slang masked_vs), for hashed alpha.
    [[vk::location(20+(VORK_ATTRIBUTES>8 ? VORK_ATTRIBUTES-8 : 0))]] nointerpolation uint sample : VISIBILITY_SAMPLE;
    [[vk::location(19+(VORK_ATTRIBUTES>8 ? VORK_ATTRIBUTES-8 : 0))]] nointerpolation uint primitive : AUTHORED_PRIMITIVE;
    [[vk::location(18+(VORK_ATTRIBUTES>8 ? VORK_ATTRIBUTES-8 : 0))]] nointerpolation uint curve_strand : CURVE_STRAND;
    [[vk::location(17+(VORK_ATTRIBUTES>8 ? VORK_ATTRIBUTES-8 : 0))]] float3 rest : REST;
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world : WORLD;
    [[vk::location(1)]] float3 NORMAL : NORMAL;
    [[vk::location(2)]] float2 TEXCOORD_0 : TEXCOORD_0;
    [[vk::location(7)]] float2 TEXCOORD_1 : TEXCOORD_1;
    [[vk::location(3)]] nointerpolation uint object : OBJECT;
    [[vk::location(4)]] float4 previous_clip : PREVIOUS;
    [[vk::location(8+VORK_ATTRIBUTES)]] float4 next_clip : NEXT;
    [[vk::location(5)]] float4 COLOR_0 : COLOR_0;
    [[vk::location(6)]] float4 TANGENT : TANGENT;
#if VORK_ATTRIBUTES > 0
    [[vk::location(8)]] float4 attr0 : VARYING_ATTRIBUTE0;
#endif
#if VORK_ATTRIBUTES > 1
    [[vk::location(9)]] float4 attr1 : VARYING_ATTRIBUTE1;
#endif
#if VORK_ATTRIBUTES > 2
    [[vk::location(10)]] float4 attr2 : VARYING_ATTRIBUTE2;
#endif
#if VORK_ATTRIBUTES > 3
    [[vk::location(11)]] float4 attr3 : VARYING_ATTRIBUTE3;
#endif
#if VORK_ATTRIBUTES > 4
    [[vk::location(12)]] float4 attr4 : VARYING_ATTRIBUTE4;
#endif
#if VORK_ATTRIBUTES > 5
    [[vk::location(13)]] float4 attr5 : VARYING_ATTRIBUTE5;
#endif
#if VORK_ATTRIBUTES > 6
    [[vk::location(14)]] float4 attr6 : VARYING_ATTRIBUTE6;
#endif
#if VORK_ATTRIBUTES > 7
    [[vk::location(15)]] float4 attr7 : VARYING_ATTRIBUTE7;
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    [[vk::location(16)]] float4 attr8 : VARYING_ATTRIBUTE8;
#if VORK_ATTRIBUTES > 9
    [[vk::location(17)]] float4 attr9 : VARYING_ATTRIBUTE9;
#endif
#if VORK_ATTRIBUTES > 10
    [[vk::location(18)]] float4 attr10 : VARYING_ATTRIBUTE10;
#endif
#if VORK_ATTRIBUTES > 11
    [[vk::location(19)]] float4 attr11 : VARYING_ATTRIBUTE11;
#endif
#if VORK_ATTRIBUTES > 12
    [[vk::location(20)]] float4 attr12 : VARYING_ATTRIBUTE12;
#endif
#if VORK_ATTRIBUTES > 13
    [[vk::location(21)]] float4 attr13 : VARYING_ATTRIBUTE13;
#endif
#if VORK_ATTRIBUTES > 14
    [[vk::location(22)]] float4 attr14 : VARYING_ATTRIBUTE14;
#endif
#if VORK_ATTRIBUTES > 15
    [[vk::location(23)]] float4 attr15 : VARYING_ATTRIBUTE15;
#endif
#if VORK_ATTRIBUTES > 16
    [[vk::location(24)]] float4 attr16 : VARYING_ATTRIBUTE16;
#endif
#if VORK_ATTRIBUTES > 17
    [[vk::location(25)]] float4 attr17 : VARYING_ATTRIBUTE17;
#endif
#if VORK_ATTRIBUTES > 18
    [[vk::location(26)]] float4 attr18 : VARYING_ATTRIBUTE18;
#endif
#else
    [[vk::location(16)]] float4 attr_extra[VORK_ATTRIBUTES-8] : VARYING_ATTRIBUTE_EXTRA;
#endif
#endif

};
#if VORK_ATTRIBUTES > 8 && defined(VORK_STAGE_SCALARS)
// The separate members by their index in the other targets' attr_extra array.
float4 attribute_extra(VertexIn v, uint i) {
    if(i==0u) return v.attr8;
#if VORK_ATTRIBUTES > 9
    if(i==1u) return v.attr9;
#endif
#if VORK_ATTRIBUTES > 10
    if(i==2u) return v.attr10;
#endif
#if VORK_ATTRIBUTES > 11
    if(i==3u) return v.attr11;
#endif
#if VORK_ATTRIBUTES > 12
    if(i==4u) return v.attr12;
#endif
#if VORK_ATTRIBUTES > 13
    if(i==5u) return v.attr13;
#endif
#if VORK_ATTRIBUTES > 14
    if(i==6u) return v.attr14;
#endif
#if VORK_ATTRIBUTES > 15
    if(i==7u) return v.attr15;
#endif
#if VORK_ATTRIBUTES > 16
    if(i==8u) return v.attr16;
#endif
#if VORK_ATTRIBUTES > 17
    if(i==9u) return v.attr17;
#endif
#if VORK_ATTRIBUTES > 18
    if(i==10u) return v.attr18;
#endif
    return float4(0.);
}
void set_attribute_extra(inout VertexIn v, uint i, float4 value) {
    if(i==0u) v.attr8=value;
#if VORK_ATTRIBUTES > 9
    if(i==1u) v.attr9=value;
#endif
#if VORK_ATTRIBUTES > 10
    if(i==2u) v.attr10=value;
#endif
#if VORK_ATTRIBUTES > 11
    if(i==3u) v.attr11=value;
#endif
#if VORK_ATTRIBUTES > 12
    if(i==4u) v.attr12=value;
#endif
#if VORK_ATTRIBUTES > 13
    if(i==5u) v.attr13=value;
#endif
#if VORK_ATTRIBUTES > 14
    if(i==6u) v.attr14=value;
#endif
#if VORK_ATTRIBUTES > 15
    if(i==7u) v.attr15=value;
#endif
#if VORK_ATTRIBUTES > 16
    if(i==8u) v.attr16=value;
#endif
#if VORK_ATTRIBUTES > 17
    if(i==9u) v.attr17=value;
#endif
#if VORK_ATTRIBUTES > 18
    if(i==10u) v.attr18=value;
#endif
}
float4 attribute_extra(Varying v, uint i) {
    if(i==0u) return v.attr8;
#if VORK_ATTRIBUTES > 9
    if(i==1u) return v.attr9;
#endif
#if VORK_ATTRIBUTES > 10
    if(i==2u) return v.attr10;
#endif
#if VORK_ATTRIBUTES > 11
    if(i==3u) return v.attr11;
#endif
#if VORK_ATTRIBUTES > 12
    if(i==4u) return v.attr12;
#endif
#if VORK_ATTRIBUTES > 13
    if(i==5u) return v.attr13;
#endif
#if VORK_ATTRIBUTES > 14
    if(i==6u) return v.attr14;
#endif
#if VORK_ATTRIBUTES > 15
    if(i==7u) return v.attr15;
#endif
#if VORK_ATTRIBUTES > 16
    if(i==8u) return v.attr16;
#endif
#if VORK_ATTRIBUTES > 17
    if(i==9u) return v.attr17;
#endif
#if VORK_ATTRIBUTES > 18
    if(i==10u) return v.attr18;
#endif
    return float4(0.);
}
void set_attribute_extra(inout Varying v, uint i, float4 value) {
    if(i==0u) v.attr8=value;
#if VORK_ATTRIBUTES > 9
    if(i==1u) v.attr9=value;
#endif
#if VORK_ATTRIBUTES > 10
    if(i==2u) v.attr10=value;
#endif
#if VORK_ATTRIBUTES > 11
    if(i==3u) v.attr11=value;
#endif
#if VORK_ATTRIBUTES > 12
    if(i==4u) v.attr12=value;
#endif
#if VORK_ATTRIBUTES > 13
    if(i==5u) v.attr13=value;
#endif
#if VORK_ATTRIBUTES > 14
    if(i==6u) v.attr14=value;
#endif
#if VORK_ATTRIBUTES > 15
    if(i==7u) v.attr15=value;
#endif
#if VORK_ATTRIBUTES > 16
    if(i==8u) v.attr16=value;
#endif
#if VORK_ATTRIBUTES > 17
    if(i==9u) v.attr17=value;
#endif
#if VORK_ATTRIBUTES > 18
    if(i==10u) v.attr18=value;
#endif
}
#endif

float4 world_position(uint object, float3 p) {return mul(float4(p,1.),object_model(object));}
float4 clip_position(uint object, float3 p) {
    var clip:float4;
    if(camera_needs_relative_clip(frame)) {
        let model=object_model(object);
        // Preserve local detail before adding a potentially large world translation.
        let relative=mul(float4(p,0.),model).xyz+(model[3].xyz-frame.eye.xyz);
        clip=camera_relative_clip(relative,frame);
    } else {
        clip=mul(world_position(object,p),frame.vp);
    }
    // Preserve projected x/y and w (including eye-plane clipping), but put
    // field carriers on the near plane. Near/far clipping must act on roots,
    // not remove the exit face that launches the root search.
    if((objects[object].metadata.y&REG_OBJECT_IMPLICIT)!=0u) clip.z=0.;
    return clip;
}

// Motion is needed only for exported controls and compositor motion channels.
bool motion_required() {
    return (frame.temporal.x+1.>=frame.temporal.y &&
         (uint(frame.projection.w)&(REG_OUTPUT_MOTION|REG_OUTPUT_FLOAT_MOTION|REG_COMPOSITOR_MOTION))!=0u);
}

// The varyings at one clip position: the camera's, or a light's for shadow casters.
Varying varying(VertexIn v, uint object, float4 position, float3 world) {
    Varying o={};o.position=position;o.world=world;o.rest=v.rest;
    if (motion_required()) {o.previous_clip=mul(float4(collapsed_endpoint(objects[object].previous_model) ? v.POSITION : v.previous_p,1.),mul(object_previous_model(object),frame.previous_vp));o.next_clip=mul(float4(collapsed_endpoint(objects[object].next_model) ? v.POSITION : v.next_p,1.),mul(object_next_model(object),frame.next_vp));}
    let model=object_model(object);
    o.TANGENT=float4(mul(float4(v.TANGENT.xyz,0.),model).xyz,v.TANGENT.w*sign(dot(cross(model[0].xyz,model[1].xyz),model[2].xyz)));
    o.curve_strand=abs(v.TANGENT.w)>=2. ? uint(v.TEXCOORD_1.x) : 0u;
    o.NORMAL=mul(float4(v.NORMAL,0.),object_normal(object)).xyz;o.TEXCOORD_0=v.TEXCOORD_0;o.TEXCOORD_1=v.TEXCOORD_1;o.COLOR_0=v.COLOR_0;o.object=object;
#if VORK_ATTRIBUTES > 0
    o.attr0=v.attr0;
#endif
#if VORK_ATTRIBUTES > 1
    o.attr1=v.attr1;
#endif
#if VORK_ATTRIBUTES > 2
    o.attr2=v.attr2;
#endif
#if VORK_ATTRIBUTES > 3
    o.attr3=v.attr3;
#endif
#if VORK_ATTRIBUTES > 4
    o.attr4=v.attr4;
#endif
#if VORK_ATTRIBUTES > 5
    o.attr5=v.attr5;
#endif
#if VORK_ATTRIBUTES > 6
    o.attr6=v.attr6;
#endif
#if VORK_ATTRIBUTES > 7
    o.attr7=v.attr7;
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    [unroll] for(uint i=0u;i<VORK_ATTRIBUTES-8;i++) set_attribute_extra(o,i,attribute_extra(v,i));
#else
    [unroll] for(uint i=0u;i<VORK_ATTRIBUTES-8;i++) o.attr_extra[i]=v.attr_extra[i];
#endif
#endif
    return o;
}

Varying vertex(VertexIn v, uint object) {
    if ((objects[object].metadata.y&REG_RAY_CAMERA)==0u) {Varying hidden={};hidden.position=REG_CULLED_CLIP;return hidden;}
    return varying(v,object,clip_position(object,v.POSITION),world_position(object,v.POSITION).xyz);
}
// Fibre capture: shadow-visible objects in the light's linked groups,
// projected into the retained fibre field (frame.light_vp).
bool shadow_hidden(uint object) {
    return (objects[object].metadata.y&REG_RAY_SHADOW)==0u
        || (frame.shadow_links.y==0u && !links_intersect(objects[object].metadata.x,frame.shadow_links.x));
}
Varying shadow_vertex(VertexIn v, uint object) {
    if (shadow_hidden(object)) {Varying hidden={};hidden.position=REG_CULLED_CLIP;return hidden;}
    let world=world_position(object,v.POSITION);
    return varying(v,object,mul(world,frame.light_vp),world.xyz);
}


uint camera_object(uint direct,uint mapped) {return mapped!=0u ? mapped-1u : direct;}

// Opaque double-sided depth coverage reads the position-only stream.
float4 opaque_depth(float3 p, uint object) {
    if ((objects[object].metadata.y&REG_RAY_CAMERA)==0u) {return REG_CULLED_CLIP;}
    return clip_position(object,p);
}
// Single-sided opaque depth needs position and winding, not material/UV shading.
struct OpaqueDepth {
    float4 position : SV_Position;
    [[vk::location(0)]] nointerpolation uint mirrored : MIRRORED;
};
OpaqueDepth opaque_single_depth(float3 p, uint object) {
    OpaqueDepth o={};
    let model=object_model(object);
    o.position=clip_position(object,p);
    if ((objects[object].metadata.y&REG_RAY_CAMERA)==0u) {o.position=REG_CULLED_CLIP;}
    o.mirrored=select(dot(cross(model[0].xyz,model[1].xyz),model[2].xyz)<0.,1u,0u);
    return o;
}

// Supplied smooth normals belong to the triangle's winding hemisphere.
// Coverage and dielectric entry/exit continue to use winding, not authored normals.
// Scale first: finite but tiny/large authored vectors must not under/overflow.
float3 finite_unit_normal(float3 normal,float3 fallback) {
    let scale=max(abs(normal.x),max(abs(normal.y),abs(normal.z)));
    if(all(isfinite(normal)) && scale>0.) {return normalize(normal/scale);}
    let fallback_scale=max(abs(fallback.x),max(abs(fallback.y),abs(fallback.z)));
    if(all(isfinite(fallback)) && fallback_scale>0.) {return normalize(fallback/fallback_scale);}
    // A fully collapsed primitive has no face; retain a finite placeholder.
    return float3(0.,0.,1.);
}
float3 winding_normal(float3 normal,float3 face) {
    let unit=finite_unit_normal(normal,face);
    return dot(unit,face)<0. ? -unit : unit;
}
