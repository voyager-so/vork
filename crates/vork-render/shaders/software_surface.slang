// Software Vulkan coverage may include samples outside the unsnapped triangle.
// Reconstruct exactly the same bounded point as retained camera shading before
// depth tests, identity selection or material evaluation can consume that sample.
import frame;
import stream;
import vork_constants;
import transparent_opacity;
import camera_ray;
import hit;

public Varying software_surface(uint object,uint primitive,float4 raster,float2 position,float2 lens) {
    if ((objects[object].metadata.y&REG_OBJECT_IMPLICIT)!=0u) {
        Varying v={};v.object=object;v.primitive=primitive;v.position=raster;return v;
    }
    let draw=fragment_object_draws[object]&0x7fffffffu;
    let st=surface_triangle<FragmentGeometry>(FragmentGeometry(fragment_draw_table,fragment_hit_buffers),draw,primitive,object);
    float3 origin,direction;
    let size=uint2(round(2./frame.lens.zw));
    camera_ray(position,size,lens,origin,direction);
    let b=triangle_barycentrics(origin,direction,st.p0,st.p1,st.p2);
    var v=hit_varying(st.t,float3(1.-b.x-b.y,b));
    if(motion_required()) {
        let g=FragmentGeometry(fragment_draw_table,fragment_hit_buffers);
        let r=g.word(draw*REG_RAY_SLOT_WORDS+2u);
        let history=r+REG_RAY_HIT_RECORD_WORDS;
        let buffer=g.word(history);let base=g.word(history+1u);let stride=g.word(history+2u);let next=g.word(history+3u);
        let indices=g.word(r+3u);let first=g.word(r+4u)+primitive*3u;
        let compact=(g.word(r+8u)&(1u<<26u))!=0u;
        float3 previous=float3(0.);float3 following=float3(0.);
        let weights=float3(1.-b.x-b.y,b);
        for(uint corner=0u;corner<3u;corner++) {
            let index=g.load(indices,first+corner);
            var row=index;
            if(compact) row/=g.load(g.word(r),g.word(r+1u)+1u);
            let at=base+row*stride;
            var p=hit_vec3<FragmentGeometry>(g,buffer,at);
            var n=hit_vec3<FragmentGeometry>(g,buffer,at+next);
            if(compact) {
                p+=st.t.c[corner].NORMAL*asfloat(g.load(buffer,at+3u))*.5;
                n+=st.t.c[corner].NORMAL*asfloat(g.load(buffer,at+next+3u))*.5;
            }
            previous+=p*weights[corner];following+=n*weights[corner];
        }
        let current=st.t.c[0].POSITION*weights.x+st.t.c[1].POSITION*weights.y+st.t.c[2].POSITION*weights.z;
        if(collapsed_endpoint(objects[object].previous_model)) previous=current;
        if(collapsed_endpoint(objects[object].next_model)) following=current;
        v.previous_clip=mul(float4(previous,1.),mul(object_previous_model(object),frame.previous_vp));
        v.next_clip=mul(float4(following,1.),mul(object_next_model(object),frame.next_vp));
    }
    // Project the bounded object-space point through the vertex stage's
    // camera-relative path; adding a large world translation first loses depth.
    let local=st.t.c[0].POSITION*(1.-b.x-b.y)+st.t.c[1].POSITION*b.x+st.t.c[2].POSITION*b.y;
    let clip=clip_position(object,local);
    v.position=raster;v.position.z=clip.z/clip.w;v.primitive=primitive;
    // A bounded point can lie outside the camera segment even though the
    // clipped raster carrier covered this sample. Never commit such a point.
    if (!(v.position.z>=0. && v.position.z<=1.)) discard;
    return v;
}
public Varying software_visibility_surface(uint object,uint primitive,float4 position,uint sample) {
    let offset=asfloat(fragment_camera_samples[sample*2u+3u].zw);
    var v=software_surface(object,primitive,position,sample_position(uint2(position.xy),offset),camera_sample_lens(fragment_camera_samples,sample));
    v.sample=sample;return v;
}
