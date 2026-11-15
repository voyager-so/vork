// Raster-only compact point vertex pulling. Ray readers use their hit buffers.
import frame;
import stream;
#ifndef VORK_ATTRIBUTES
#define VORK_ATTRIBUTES 0
#endif
#ifndef VORK_POINT_WORDS_REGISTER
#define VORK_POINT_WORDS_REGISTER t29
#endif
[[vk::binding(11,0)]] StructuredBuffer<uint> point_words : register(VORK_POINT_WORDS_REGISTER);
uint point_header(uint object) {if(point_words[0]==0u) return 0u;return point_words[object+1u];}
float3 point_vec3(uint at) {return asfloat(uint3(point_words[at],point_words[at+1u],point_words[at+2u]));}
float4 point_vec4(uint at) {return float4(point_vec3(at),asfloat(point_words[at+3u]));}
float3 point_position(float3 fallback,uint object,uint vertex_index) {
    let header=point_header(object);if(header==0u) return fallback;
    let base=point_words[header];let count=point_words[base];let shape_count=point_words[base+1u];
    let row=vertex_index/shape_count;let shape=base+2u+count*10u+(vertex_index%shape_count)*5u;
    let pose=base+2u+row*10u;
    return point_vec3(pose)+point_vec3(shape)*asfloat(point_words[pose+3u])*0.5;
}
VertexIn point_vertex(VertexIn v,uint object) {
    let header=point_header(object);if(header==0u) return v;
    let base=point_words[header];let count=point_words[base];let shape_count=point_words[base+1u];
    let row=v.vertex_index/shape_count;let shape=base+2u+count*10u+(v.vertex_index%shape_count)*5u;
    let pose=base+2u+row*10u;let a=point_words[header+1u]+row*point_words[header+3u];
    let history=point_words[header+2u]+row*20u;
    v.NORMAL=point_vec3(shape);v.TANGENT=float4(0.);
    v.POSITION=point_vec3(pose)+v.NORMAL*asfloat(point_words[pose+3u])*0.5;
    v.previous_p=point_vec3(history)+v.NORMAL*asfloat(point_words[history+3u])*0.5;
    v.next_p=point_vec3(history+10u)+v.NORMAL*asfloat(point_words[history+13u])*0.5;
    v.rest=point_vec3(a)+v.NORMAL*asfloat(point_words[a+3u])*0.5;
    v.TEXCOORD_0=asfloat(uint2(point_words[shape+3u],point_words[shape+4u]));v.TEXCOORD_1=float2(0.);
    v.COLOR_0=point_vec4(a+7u);
#if VORK_ATTRIBUTES > 0
    v.attr0=point_vec4(a+11u);
#endif
#if VORK_ATTRIBUTES > 1
    v.attr1=point_vec4(a+15u);
#endif
#if VORK_ATTRIBUTES > 2
    v.attr2=point_vec4(a+19u);
#endif
#if VORK_ATTRIBUTES > 3
    v.attr3=point_vec4(a+23u);
#endif
#if VORK_ATTRIBUTES > 4
    v.attr4=point_vec4(a+27u);
#endif
#if VORK_ATTRIBUTES > 5
    v.attr5=point_vec4(a+31u);
#endif
#if VORK_ATTRIBUTES > 6
    v.attr6=point_vec4(a+35u);
#endif
#if VORK_ATTRIBUTES > 7
    v.attr7=point_vec4(a+39u);
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    [unroll] for(uint i=0u;i<VORK_ATTRIBUTES-8u;i++) set_attribute_extra(v,i,point_vec4(a+43u+i*4u));
#else
    [unroll] for(uint i=0u;i<VORK_ATTRIBUTES-8u;i++) v.attr_extra[i]=point_vec4(a+43u+i*4u);
#endif
#endif
    return v;
}
