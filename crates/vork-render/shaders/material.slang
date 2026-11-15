import vork_constants;
import frame;
import stream;
import material_layout;
import variant;
// Named geometry attributes (stream.slang); each module sees only its own defines.
#ifndef VORK_ATTRIBUTES
#define VORK_ATTRIBUTES 0
#endif
__exported import vork_material_table;
uint material_record(uint object) {return objects[object].material_index;}
MaterialUv material_uv(Varying v) {
    let uv1=select(abs(v.TANGENT.w)>=2.,float2(0.),v.TEXCOORD_1);
    return material_uv(v,ddx(v.TEXCOORD_0),ddy(v.TEXCOORD_0),ddx(uv1),ddy(uv1));
}
// Explicit footprints: raster derivatives or a ray hit's pixel footprint.
MaterialUv material_uv(Varying v, float2 dx0, float2 dy0, float2 dx1, float2 dy1, float3 filter=float3(0.)) {
#if VORK_ATTRIBUTES > 0
    material_attributes[0]=v.attr0;
#endif
#if VORK_ATTRIBUTES > 1
    material_attributes[1]=v.attr1;
#endif
#if VORK_ATTRIBUTES > 2
    material_attributes[2]=v.attr2;
#endif
#if VORK_ATTRIBUTES > 3
    material_attributes[3]=v.attr3;
#endif
#if VORK_ATTRIBUTES > 4
    material_attributes[4]=v.attr4;
#endif
#if VORK_ATTRIBUTES > 5
    material_attributes[5]=v.attr5;
#endif
#if VORK_ATTRIBUTES > 6
    material_attributes[6]=v.attr6;
#endif
#if VORK_ATTRIBUTES > 7
    material_attributes[7]=v.attr7;
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    [unroll] for(uint i=8u;i<VORK_ATTRIBUTES;i++) material_attributes[i]=attribute_extra(v,i-8u);
#else
    [unroll] for(uint i=8u;i<VORK_ATTRIBUTES;i++) material_attributes[i]=v.attr_extra[i-8u];
#endif
#endif
    material_instance_attributes(objects[v.object].reserved);
    let uv1=select(abs(v.TANGENT.w)>=2.,float2(0.),v.TEXCOORD_1);
    return MaterialUv(v.TEXCOORD_0,uv1,dx0,dy0,dx1,dy1,filter);
}
float material_opacity<V:IVariant>(Varying v, MaterialUv m) {return material_alpha<V>(material_record(v.object),m,v.COLOR_0.a);}
bool material_varying_opacity<V:IVariant>(uint material, float3 alpha) {
    return material_alpha_bound<V>(material) ||
        ((material_records[material].tag.z&REG_FLAG_VERTEX_ALPHA)!=0u && (alpha.x!=alpha.y || alpha.x!=alpha.z));
}
// glTF occlusion: linear red-channel visibility at the record's strength, indirect light only.
float material_occlusion<V:IVariant>(Varying v, MaterialUv m) {
    let r=material_record(v.object);let b=material_byte<V>(r,MATERIAL_OCCLUSION);
    if ((b&15u)==0u) {return 1.;}
    return lerp(1.,material_sample(material_tap(r,b&15u,m)).r,material_records[r].extra.z);
}

// Current local bounds enclose every triangle used by raster and ray queries.
// gamma(7) covers non-FMA interpolation, triangle intersection (NVIDIA's 3u
// extent bound), and each affine transform. Keep absolute products before sums
// so translated geometry does not lose its uncertainty through cancellation.
float3 surface_position_error(uint object,float3 world) {
    let base=objects[object].reserved+6u;
    let lo=float3(asfloat(material_words[base]),asfloat(material_words[base+1u]),asfloat(material_words[base+2u]));
    let hi=float3(asfloat(material_words[base+3u]),asfloat(material_words[base+4u]),asfloat(material_words[base+5u]));
    let magnitude=max(abs(lo),abs(hi));let span=abs(hi-lo);
    let extent=max(span.x,max(span.y,span.z));
    let model=object_model(object);let inverse=transpose(object_normal(object));
    let local_error=REG_FP32_GAMMA7*(magnitude+extent+mul(float4(abs(world),1.),abs(inverse)).xyz);
    return mul(float4(local_error,0.),abs(model)).xyz
        +REG_FP32_GAMMA7*mul(float4(magnitude,1.),abs(model)).xyz;
}
