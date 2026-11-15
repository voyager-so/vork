// High-bit primitive keys name field crossings only in field-capable kernels.
public bool field_primitive(uint primitive) {
#if defined(VORK_NO_IMPLICIT) || defined(VORK_CAMERA_FUSED)
    return false;
#else
    return (primitive&0x80000000u)!=0u;
#endif
}
// Shared current-pose hit geometry; no material or surface evaluation.
import vork_constants;
import frame;
import stream;

// Metal binds a pass's hit buffer array through an argument buffer of this type.
public struct RayHitBuffersTable { public StructuredBuffer<uint> buffers[REG_RAY_HIT_BUFFERS]; };
// The consuming pass's ray table and hit buffer array (rt::hit_buffers_entry).
public interface IHitGeometry {
    uint word(uint index);
    uint load(uint buffer, uint index);
}
// Entry-owned immutable geometry snapshot, forwarded through every reader.
public struct RayGeometry : IHitGeometry {
    public StructuredBuffer<uint> words;
#ifdef VORK_METAL
    public ParameterBlock<RayHitBuffersTable> buffers;
    public __init(StructuredBuffer<uint> words,ParameterBlock<RayHitBuffersTable> buffers) {this.words=words;this.buffers=buffers;}
#else
    public StructuredBuffer<uint> buffers[REG_RAY_HIT_BUFFERS];
    public __init(StructuredBuffer<uint> words,StructuredBuffer<uint> buffers[REG_RAY_HIT_BUFFERS]) {this.words=words;this.buffers=buffers;}
#endif
    public uint word(uint index) {return words[index];}
    public uint load(uint buffer,uint index) {
#ifdef VORK_METAL
        return buffers.buffers[buffer][index];
#else
        return buffers[NonUniformResourceIndex(buffer)][index];
#endif
    }
};
// Named geometry attributes (stream.slang); per-scene hit kernels define the count.
#ifndef VORK_ATTRIBUTES
#define VORK_ATTRIBUTES 0
#endif
// POSITION, NORMAL and TANGENT are the current pose; `rest` is the installed position.
public struct HitCorner {
    public float3 POSITION; public float3 rest; public float3 NORMAL; public float2 TEXCOORD_0; public float2 TEXCOORD_1; public float4 COLOR_0; public float4 TANGENT;
#if VORK_ATTRIBUTES > 0
    public float4 attributes[VORK_ATTRIBUTES];
#endif
};
// Object-space corners, the instancing object and its normal transform (the
// ray query's world-to-object rows).
public struct HitTriangle { public HitCorner c[3]; public float3 face; public uint object; public uint primitive_offset; public float3x3 normal_matrix; };
float hit_float<G:IHitGeometry>(G g,uint buffer, uint i) {return asfloat(g.load(buffer,i));}
public float3 hit_vec3<G:IHitGeometry>(G g,uint buffer, uint i) {return float3(hit_float<G>(g,buffer,i),hit_float<G>(g,buffer,i+1u),hit_float<G>(g,buffer,i+2u));}
public float4 hit_vec4<G:IHitGeometry>(G g,uint buffer, uint i) {return float4(hit_vec3<G>(g,buffer,i),hit_float<G>(g,buffer,i+3u));}
// An optional attribute: 0xff offsets read the raster default stream's value.
float2 hit_attribute2<G:IHitGeometry>(G g,uint buffer, uint vertex, uint offset) {
    return offset==0xffu ? float2(0.) : float2(hit_float<G>(g,buffer,vertex+offset),hit_float<G>(g,buffer,vertex+offset+1u));
}
float4 hit_attribute4<G:IHitGeometry>(G g,uint buffer, uint vertex, uint offset, float4 fallback) {
    return offset==0xffu ? fallback : hit_vec4<G>(g,buffer,vertex+offset);
}
public HitTriangle hit_triangle<G:IHitGeometry>(G g,uint instance, uint primitive, uint object, float3x4 world_to_object) {
    let r=g.word(instance*REG_RAY_SLOT_WORDS+2u);
    let positions=g.word(r);let base=g.word(r+1u);let stride=g.word(r+2u);
    let indices=g.word(r+3u);let first=g.word(r+4u)+primitive*3u;
    let attributes=g.word(r+5u);let attribute_base=g.word(r+6u);let attribute_stride=g.word(r+7u);
    let fields=g.word(r+8u);let channels=g.word(r+9u);
    // Deformed draws read normals/tangents from the pose, static ones from attributes.
    let deformed=(fields&(1u<<24u))!=0u;
    let surface=deformed ? positions : attributes;
    HitTriangle t;t.object=object;
#ifdef VORK_NO_POINTS
    t.primitive_offset=0u;t.normal_matrix=(float3x3)world_to_object;
#else
    t.primitive_offset=g.word(r+10u);t.normal_matrix=(fields&(1u<<25u))!=0u ? (float3x3)object_normal(object) : (float3x3)world_to_object;
#endif
    [unroll] for (uint corner=0u;corner<3u;corner++) {
        let index=g.load(indices,first+corner);
        #ifdef VORK_NO_POINTS
        let compact=false;
#else
        let compact=(fields&(1u<<26u))!=0u;
#endif
        var p=base+index*stride;var a=attribute_base+index*attribute_stride;
        var c:HitCorner;
        if(compact) {
            let count=g.load(positions,base);let shape_count=g.load(positions,base+1u);
            let row=index/shape_count;let shape=base+2u+count*10u+(index%shape_count)*5u;
            p=base+2u+row*10u;a=attribute_base+row*attribute_stride;
            c.NORMAL=hit_vec3<G>(g,positions,shape);c.TANGENT=float4(0.);
            c.POSITION=hit_vec3<G>(g,positions,p)+c.NORMAL*hit_float<G>(g,positions,p+3u)*0.5;
            c.rest=hit_vec3<G>(g,attributes,a)+c.NORMAL*hit_float<G>(g,attributes,a+3u)*0.5;
            c.TEXCOORD_0=hit_attribute2<G>(g,positions,shape,3u);c.TEXCOORD_1=float2(0.);
        } else {
            let s=deformed ? p : a;
            c.POSITION=hit_vec3<G>(g,positions,p);
            c.rest=deformed ? hit_vec3<G>(g,attributes,a+((fields>>16u)&0xffu)) : c.POSITION;
            c.NORMAL=hit_vec3<G>(g,surface,s+(fields&0xffu));
            c.TANGENT=hit_attribute4<G>(g,surface,s,(fields>>8u)&0xffu,float4(0.));
            c.TEXCOORD_0=hit_attribute2<G>(g,attributes,a,channels&0xffu);
            c.TEXCOORD_1=hit_attribute2<G>(g,attributes,a,(channels>>8u)&0xffu);
        }
        c.COLOR_0=hit_attribute4<G>(g,attributes,a,(channels>>16u)&0xffu,float4(1.));
#if VORK_ATTRIBUTES > 0
        // Named attributes follow colour, one float4 each (VertexLayout::pack).
        let custom=channels>>24u;
        [unroll] for (uint k=0u;k<VORK_ATTRIBUTES;k++) {
            c.attributes[k]=custom==0xffu || custom+4u*k+4u>attribute_stride ? float4(0.) : hit_vec4<G>(g,attributes,a+custom+4u*k);
        }
#endif
        t.c[corner]=c;
    }
    // The BLAS winding: object-space corners in index order.
    let face=cross(t.c[1].POSITION-t.c[0].POSITION,t.c[2].POSITION-t.c[0].POSITION);
    t.face=finite_unit_normal(face,t.c[0].NORMAL);
    return t;
}
// stream.slang varying() at barycentrics `b`: the same object transforms.
public float3 hit_point(HitTriangle t, float3 b) {
    let p=t.c[0].POSITION*b.x+t.c[1].POSITION*b.y+t.c[2].POSITION*b.z;
    return mul(float4(p,1.),object_model(t.object)).xyz;
}
public Varying hit_varying(HitTriangle t, float3 b) {
    let model=object_model(t.object);
    Varying o={};
    let tangent=t.c[0].TANGENT*b.x+t.c[1].TANGENT*b.y+t.c[2].TANGENT*b.z;
    o.rest=t.c[0].rest*b.x+t.c[1].rest*b.y+t.c[2].rest*b.z;o.world=hit_point(t,b);
    o.NORMAL=mul(t.c[0].NORMAL*b.x+t.c[1].NORMAL*b.y+t.c[2].NORMAL*b.z,t.normal_matrix);
    o.NORMAL=winding_normal(o.NORMAL,mul(t.face,t.normal_matrix));
    o.TEXCOORD_0=t.c[0].TEXCOORD_0*b.x+t.c[1].TEXCOORD_0*b.y+t.c[2].TEXCOORD_0*b.z;
    o.TEXCOORD_1=t.c[0].TEXCOORD_1*b.x+t.c[1].TEXCOORD_1*b.y+t.c[2].TEXCOORD_1*b.z;
    o.COLOR_0=t.c[0].COLOR_0*b.x+t.c[1].COLOR_0*b.y+t.c[2].COLOR_0*b.z;
    o.TANGENT=float4(mul(float4(tangent.xyz,0.),model).xyz,tangent.w*sign(dot(cross(model[0].xyz,model[1].xyz),model[2].xyz)));
    o.object=t.object;o.position=float4(0.,0.,0.,1.);o.previous_clip=float4(0.,0.,0.,1.);
    // stream.slang: curve strands carry their strand index in TEXCOORD_1 (tangent w flag).
    o.curve_strand=abs(o.TANGENT.w)>=2. ? uint(round(o.TEXCOORD_1.x)) : 0u;
#if VORK_ATTRIBUTES > 0
    o.attr0=t.c[0].attributes[0]*b.x+t.c[1].attributes[0]*b.y+t.c[2].attributes[0]*b.z;
#endif
#if VORK_ATTRIBUTES > 1
    o.attr1=t.c[0].attributes[1]*b.x+t.c[1].attributes[1]*b.y+t.c[2].attributes[1]*b.z;
#endif
#if VORK_ATTRIBUTES > 2
    o.attr2=t.c[0].attributes[2]*b.x+t.c[1].attributes[2]*b.y+t.c[2].attributes[2]*b.z;
#endif
#if VORK_ATTRIBUTES > 3
    o.attr3=t.c[0].attributes[3]*b.x+t.c[1].attributes[3]*b.y+t.c[2].attributes[3]*b.z;
#endif
#if VORK_ATTRIBUTES > 4
    o.attr4=t.c[0].attributes[4]*b.x+t.c[1].attributes[4]*b.y+t.c[2].attributes[4]*b.z;
#endif
#if VORK_ATTRIBUTES > 5
    o.attr5=t.c[0].attributes[5]*b.x+t.c[1].attributes[5]*b.y+t.c[2].attributes[5]*b.z;
#endif
#if VORK_ATTRIBUTES > 6
    o.attr6=t.c[0].attributes[6]*b.x+t.c[1].attributes[6]*b.y+t.c[2].attributes[6]*b.z;
#endif
#if VORK_ATTRIBUTES > 7
    o.attr7=t.c[0].attributes[7]*b.x+t.c[1].attributes[7]*b.y+t.c[2].attributes[7]*b.z;
#endif
#if VORK_ATTRIBUTES > 8
#ifdef VORK_STAGE_SCALARS
    [unroll] for(uint i=8u;i<VORK_ATTRIBUTES;i++) set_attribute_extra(o,i-8u,t.c[0].attributes[i]*b.x+t.c[1].attributes[i]*b.y+t.c[2].attributes[i]*b.z);
#else
    [unroll] for(uint i=8u;i<VORK_ATTRIBUTES;i++) o.attr_extra[i-8u]=t.c[0].attributes[i]*b.x+t.c[1].attributes[i]*b.y+t.c[2].attributes[i]*b.z;
#endif
#endif
    return o;
}
// A local smooth receiver envelope. The quadratic normal variation estimates
// curvature; the supporting planes also cover concave triangles. All positions
// are relative to one corner, so world translation does not enter the products.
public float4 shadow_terminator_geometry(float3 p[3],float3 n[3],float3 face,float3 b) {
    var smooth=n[0]*b.x+n[1]*b.y+n[2]*b.z;
    if(dot(smooth,face)<0.) {
        [unroll] for(uint i=0u;i<3u;i++) n[i]=-n[i];
        smooth=-smooth;
    }
    let point=p[0]*b.x+p[1]*b.y+p[2]*b.z;
    var envelope=0.;var quadratic=0.;var cone=0.;
    [unroll] for(uint i=0u;i<3u;i++) {
        var support=0.;
        [unroll] for(uint j=0u;j<3u;j++) support=max(support,dot(p[j]-point,n[i]));
        envelope=i==0u ? support : min(envelope,support);
        cone=max(cone,length(n[i]-face));
        [unroll] for(uint j=i+1u;j<3u;j++)
            quadratic+=b[i]*b[j]*dot(n[j]-n[i],p[j]-p[i]);
    }
    return float4(smooth*max(envelope,.5*quadratic),cone);
}
public float4 hit_shadow_offset(HitTriangle t,float3 b,bool front) {
    let model=object_model(t.object);let side=select(front,1.,-1.);
    let face=hit_face(t)*side;
    float3 p[3];float3 n[3];
    [unroll] for(uint i=0u;i<3u;i++) {
        p[i]=mul(float4(t.c[i].POSITION-t.c[0].POSITION,0.),model).xyz;
        let raw=mul(t.c[i].NORMAL,t.normal_matrix);
        n[i]=finite_unit_normal(raw*side,face);
    }
    return shadow_terminator_geometry(p,n,face,b);
}
public struct SolidExit { public float distance; public float3 normal; public bool found; };
// Normalize geometry before products so equivalent object/local scales do not
// overflow or underflow a boundary normal.
public float3 exit_unit(float3 v) {
    let scale=max(abs(v.x),max(abs(v.y),abs(v.z)));
    return scale>0. ? normalize(v/scale) : float3(0.);
}
public float3 exit_edge(float3 a,float3 b) {
    let edge=b-a;
    // Opposite finite endpoints overflow by at most a factor of two. Halving
    // both endpoints is exact here and preserves the edge direction.
    return all(isfinite(edge)) ? edge : b*.5-a*.5;
}
public float3 exit_face(float3 a,float3 b,float3 c,float3x3 model) {
    // Transform differences before normalization: a tiny local component can
    // become significant under nonuniform scale. Translation never enters it.
    let x=mul(exit_edge(a,b),model);let y=mul(exit_edge(a,c),model);
    let raw=cross(x,y);
    let face=all(isfinite(raw)) && any(raw!=float3(0.)) ? exit_unit(raw) : exit_unit(cross(exit_unit(x),exit_unit(y)));
    // Exit queries already establish an outward crossing. Transport orients
    // this unoriented world plane against its ray, including mirrored objects.
    return face;
}
// Both exit queries use smooth raster normals when their chosen transport
// event crosses the geometric boundary in the right direction. A tilted smooth
// normal must not reflect a TIR ray outside, or refract an escaped ray inside.
public float3 solid_exit_normal(float3 shading,float3 geometric,float3 direction,float eta) {
    let face=select(dot(geometric,direction)<0.,geometric,-geometric);
    let n=select(dot(shading,direction)<0.,shading,-shading);
    let transmitted=refract(direction,n,eta);
    let escapes=dot(transmitted,transmitted)>0.;
    let valid=escapes ? dot(transmitted,face)<0. : dot(reflect(direction,n),face)>0.;
    return valid ? n : face;
}
public float3 solid_exit_normal(float3 interpolated,float3 world_face,float3x3 matrix,float3 direction,float eta) {
    let geometric=exit_unit(world_face);
    let transformed=mul(interpolated,matrix);
    let normal=all(isfinite(transformed)) ? transformed : mul(exit_unit(interpolated),matrix);
    let shading=any(interpolated!=float3(0.)) ? exit_unit(normal) : geometric;
    return solid_exit_normal(shading,geometric,direction,eta);
}
// The world face normal on the object's front (winding) side.
public float3 hit_face(HitTriangle t) {return finite_unit_normal(mul(t.face,t.normal_matrix),float3(0.,0.,1.));}

// Current emitter geometry shares the same transformed/deformed corners as hits.
public struct HitEmitterGeometry { public float3 point; public float3 face; public float area; };
public HitEmitterGeometry hit_emitter<G:IHitGeometry>(G g,uint instance, uint primitive, uint object, float3x4 inverse, float2 barycentrics) {
    let t=hit_triangle<G>(g,instance,primitive,object,inverse);let m=object_model(object);
    let a=mul(float4(t.c[1].POSITION-t.c[0].POSITION,0.),m).xyz;
    let b=mul(float4(t.c[2].POSITION-t.c[0].POSITION,0.),m).xyz;
    let point=hit_point(t,float3(1.-barycentrics.x-barycentrics.y,barycentrics));
    return HitEmitterGeometry(point,hit_face(t),length(cross(a,b))*.5);
}

// A camera triangle and its world-space corners, shared by retained evaluation
// and the collector's zero-opacity proof.
public struct SurfaceTriangle { public HitTriangle t; public uint primitive; public float3 p0; public float3 p1; public float3 p2; };
public SurfaceTriangle surface_triangle<G:IHitGeometry>(G g,uint draw, uint primitive, uint object) {
    if(field_primitive(primitive)) {
        SurfaceTriangle st={};st.t.object=object;st.primitive=primitive;
        for(uint i=0u;i<3u;i++) st.t.c[i].COLOR_0=float4(1.);
        return st;
    }
    let normal=object_normal(object);
    let t=hit_triangle<G>(g,draw,primitive,object,float3x4(normal[0],normal[1],normal[2]));
    let model=object_model(object);
    return SurfaceTriangle(t,primitive,mul(float4(t.c[0].POSITION,1.),model).xyz,mul(float4(t.c[1].POSITION,1.),model).xyz,mul(float4(t.c[2].POSITION,1.),model).xyz);
}
