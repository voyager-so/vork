// Frame, light and object ABI shared by every pass; group 0 bindings.
// ABI structs avoid float3 members: Slang's Metal layout gives them 16 bytes in
// constant buffers (native::tests compares the Metal and Vulkan field offsets).
// Each binding names its Vulkan (binding, set) and its Metal index (buffers use
// t/b registers, Metal [[buffer(n)]]; textures t, [[texture(n)]]; samplers s), which is
// wgpu-hal's per-stage counter for the forward and depth layouts (native::tests).
import vork_constants;
import variant;
import vork_visibility;
import vork_sampling;
// Shared CPU/WGSL frame and light ABI for forward shading and cluster culling.
// shadow_links: caster mask/all, receiver-link enable; w is reserved.
// sampling: x/y are sampling-table offsets (render samples, motion taps); z is the visibility film-offset table;
// w is the visibility samples per exposure sample (0 outside decoupled colour).
struct Frame { float4x4 vp; float4x4 light_vp; float4 eye; float4 background; float4 radiance; float4 params; float4 flags; float4x4 previous_vp; float4x4 next_vp; float4x4 inverse_vp; float4x4 view; float4 ao; float4 projection; float4 lens; float4 aperture; float4 temporal; float4 environment; float4x4 environment_rotation; float4 environment_color; float4 environment_mean; float4 scene; uint4 shadow_links; uint4 integrator; uint4 sampling; uint4 environment_links; uint4 medium; float4 display; float4 post; float4x4 composite_projection; float4 composite_optics; float4 composite_sensor; float4 composite_shutter; float4 composite_step; float4 camera_sample; uint4 camera_program; uint4 field_geometry; };
// Subtract the eye before projection: the z/w cancellation in a combined
// world-to-clip matrix is amplified when device depth is converted to metres.
bool camera_needs_relative_clip(Frame f) {
    // At 512 m a world-coordinate ULP reaches 2^-14 m. Below this
    // conservative boundary retain the combined transform: reassociating its
    // arithmetic changes raster interpolation and can amplify through glass.
    return any(abs(f.eye.xyz)>=512.);
}
float4 camera_relative_clip(float3 relative,Frame f) {
    let view=mul(float4(relative,0.),f.view).xyz;
    return mul(float4(view,1.),f.composite_projection);
}
float4 camera_world_clip(float3 world,Frame f) {
    if(!camera_needs_relative_clip(f)) return mul(float4(world,1.),f.vp);
    return camera_relative_clip(world-f.eye.xyz,f);
}
// WGSL `background: vec3<f32>, light_count: f32` is one float4 here.
float frame_light_count(Frame f) {return f.background.w;}
struct ClusterConfig { uint4 grid; uint4 screen; };
uint cluster_base(float2 pixel, float depth, Frame f, ClusterConfig config) {
    let tile=min(uint2(pixel)/config.screen.w,config.grid.xy-uint2(1u));
    depth=clamp(depth,f.params.x,f.params.y);
    let slice=min(uint(log(depth/f.params.x)/log(f.params.y/f.params.x)*float(config.screen.z)),config.screen.z-1u);
    return ((slice*config.grid.y+tile.y)*config.grid.x+tile.x)*config.grid.z;
}
// Medium and secondary receivers can leave the camera frustum. The complete
// light batch remains the conservative list there, never an edge cluster.
bool camera_cluster_base(float3 position, Frame f, ClusterConfig config, out uint base) {
    base=0u;
    if (config.grid.w==0u) {return false;}
    let clip=mul(float4(position,1.),f.vp);
    let depth=-mul(float4(position,1.),f.view).z;
    if (clip.w<=0. || any(abs(clip.xy)>clip.w) || depth<f.params.x || depth>f.params.y) {return false;}
    let pixel=(clip.xy/clip.w*float2(.5,-.5)+float2(.5))*float2(config.screen.xy);
    base=cluster_base(pixel,depth,f,config);
    return true;
}
struct Light { uint shadow_offsets[REG_LIGHT_SHADOW_OFFSET_WORDS]; float4 position; float4 direction; float4 radiance; float4 params; float4 emitter_u; float4 emitter_v; float4 emission; float4 projection; };

uint2 vork_dimensions(Texture2D<float4> t) { uint w,h;t.GetDimensions(w,h);return uint2(w,h); }
struct AoSurface { int2 pixel; uint object; float depth; float3 position; float3 normal; float view_depth; };
struct LightValues { float4 position; float4 direction; float4 radiance; float4 params; float4 emitter_u; float4 emitter_v; float4 emission; float4 projection; };
__exported import vork_object;
float4x4 object_normal_matrix(Object object) {
    let m=transpose(float4x4(object.model[0],object.model[1],object.model[2],float4(0.,0.,0.,1.))); let a=m[0].xyz; let b=m[1].xyz; let c=m[2].xyz;
    let det=dot(a,cross(b,c));
    if (det==0. || (object.metadata.y&~REG_OBJECT_IMPLICIT)==0u) {return float4x4(float4(1.,0.,0.,0.),float4(0.,1.,0.,0.),float4(0.,0.,1.,0.),float4(0.,0.,0.,1.));}
    let x=cross(b,c)/det; let y=cross(c,a)/det; let z=cross(a,b)/det;
    return float4x4(float4(x,-dot(x,m[3].xyz)),float4(y,-dot(y,m[3].xyz)),float4(z,-dot(z,m[3].xyz)),float4(0.,0.,0.,1.));
}
float2 encode_octahedron(float3 normal) {
    let n=normal/(abs(normal.x)+abs(normal.y)+abs(normal.z));
    return select(n.z>=0.,n.xy,(1.-abs(n.yx))*select(n.xy>=float2(0.),float2(1.),float2(-1.)));
}
float3 decode_octahedron(float2 v) {
    var n=float3(v,1.-abs(v.x)-abs(v.y));
    if (n.z<0.) {n=float3((1.-abs(n.yx))*select(n.xy>=float2(0.),float2(1.),float2(-1.)),n.z);}
    return normalize(n);
}

// One environment convention for filtering, shading and camera backgrounds.
// The environment light's orientation is its sampled native node rotation.
float2 equirect_uv(float3 d) {
    // Longitude is undefined at either pole. Give it a canonical meridian
    // before atan2: Metal may return NaN for atan2(0,0). Proposal inversion
    // reaches the north pole exactly when its latitude sample is zero.
    if (d.x==0. && d.z==0.) {return float2(.5,d.y>0. ? 0. : 1.);}
    return float2(atan2(d.z,d.x)/REG_TAU+.5,atan2(length(d.xz),d.y)/REG_PI);
}
float2 environment_uv(float3 direction, float4x4 rotation) {
    return equirect_uv(normalize(mul(float4(direction,0.),rotation).xyz));
}
// Outgoing camera direction. Orthographic rays share an axis at every pixel.
float3 camera_view(float3 world, Frame f) {
    return select(f.flags.x>.5,float3(f.view[0].z,f.view[1].z,f.view[2].z),normalize(f.eye.xyz-world));
}
// A camera ray from its view-depth-zero plane, with lens/visibility shifts.
// Perspective construction is independent of world translation and far-plane
// projection precision. Orthographic rays unproject only a finite origin.
void camera_ray_line(float2 ndc,Frame f,out float3 origin,out float3 direction) {
    let right=float3(f.view[0].x,f.view[1].x,f.view[2].x);
    let up=float3(f.view[0].y,f.view[1].y,f.view[2].y);
    let forward=-float3(f.view[0].z,f.view[1].z,f.view[2].z);
    origin=f.eye.xyz;direction=forward;
    if(f.flags.x>.5) {
        let h=mul(float4(ndc,0.,1.),f.inverse_vp);origin=h.xyz/h.w-forward*f.params.x;
    } else {
        let shift=float2(dot(float3(f.vp[0].x,f.vp[1].x,f.vp[2].x),forward),
            dot(float3(f.vp[0].y,f.vp[1].y,f.vp[2].y),forward));
        let camera=(ndc-shift)/f.projection.yx;
        direction=normalize(forward+right*camera.x+up*camera.y);
    }
}
float3 reflection_fallback(float3 ray, float roughness, Frame frame, Texture2D environment, SamplerState environment_sampler) {
    var radiance=frame.radiance.xyz;
    if (frame.environment.x>0.) {
        let uv=environment_uv(ray,frame.environment_rotation);
        radiance+=environment.SampleLevel(environment_sampler,uv,roughness*frame.environment.z).rgb*frame.environment.x*frame.environment.y*frame.environment_color.rgb;
    }
    return radiance;
}
// The fixed golden-disk table covers every dynamic quadrature budget; constants.wgsl asserts it.

float4 ao_receiver(Texture2D<float4> receivers,uint2 cell,int divisor,out int2 pixel) {
    let size=int2(vork_dimensions(receivers));
    let origin=int2(cell)*divisor;pixel=min(origin,size-1);var best=float4(0.,0.,0.,1.);
    for(var y=0;y<divisor;y++) {for(var x=0;x<divisor;x++) {
        let p=min(origin+int2(x,y),size-1);let tap=receivers.Load(int3(p,0));
        if (tap.w<best.w) {best=tap;pixel=p;}
    }}
    return best;
}
AoSurface ao_surface(Texture2D<float4> receivers,Texture2D<float4> normals,uint2 cell,int divisor,float4x4 view) {
    int2 pixel;let receiver=ao_receiver(receivers,cell,divisor,pixel);
    var s=AoSurface(pixel,0u,receiver.w,receiver.xyz,float3(0.,0.,1.),0.);
    if (receiver.w>=1.) {return s;}
    let normal=normals.Load(int3(pixel,0));
    s.normal=normalize(normal.xyz);s.object=asuint(normal.w);
    s.view_depth=-(mul(float4(s.position,1.),view)).z;
    return s;
}

// Shared light-value specialization.
LightValues specialized_light<V:IVariant>(Light record) {
    var position=record.position;
    // A single-kind scene makes all emitter and shadow helpers constant-fold that kind.
    if (V.light_types==1u) {position.w=0.;}
    if (V.light_types==2u) {position.w=1.;}
    if (V.light_types==4u) {position.w=2.;}
    if (V.light_types==8u) {position.w=3.;}
    return LightValues(position,record.direction,record.radiance,
        record.params,record.emitter_u,record.emitter_v,record.emission,
        record.projection);
}

// Adjacent IEEE-754 value in the requested direction. Used for conservative
// grid-boundary progression instead of a fixed fraction of a screen pixel.
float screen_next_float(float value,bool up) {
    if (value==0.) {return asfloat(up ? 1u : 0x80000001u);}
    let increasing=(value>0.)==up;
    return asfloat(asuint(value)+(increasing ? 1u : 0xffffffffu));
}

// Moving a smooth receiver is larger than a numerical ray-origin nudge.
// Preserve a finite light's sampled endpoint instead of shortening the original
// parallel ray by the offset length. Directional lights retain their axis.
float4 surface_ray_connection(float3 origin,float3 world,float3 direction,float distance) {
    if(distance==REG_FP32_MAX || all(origin==world)) {return float4(direction,distance);}
    let delta=direction*distance-(origin-world);let remaining=length(delta);
    return float4(remaining>0. ? delta/remaining : direction,remaining);
}
// PBRT's interval origin construction: leave the position uncertainty box
// along the geometric normal, then round each moved component away from it.
// No distance in scene units is introduced. Non-unit normals are supported.
float3 surface_ray_origin(float3 world,float3 error,float3 geometric,float3 direction) {
    let scale=max(abs(geometric.x),max(abs(geometric.y),abs(geometric.z)));
    if (scale==0.) {return world;}
    let n=geometric/scale;
    let amount=screen_next_float(dot(abs(n),error)/dot(n,n),true);
    let offset=n*amount*select(dot(n,direction)>=0.,1.,-1.);
    var origin=world+offset;
    for (uint axis=0u;axis<3u;axis++) {
        if (offset[axis]!=0.) {origin[axis]=screen_next_float(origin[axis],offset[axis]>0.);}
    }
    return origin;
}

// Arithmetic uncertainty given the stored camera matrices and depth. The
// four-term homogeneous product uses gamma(7); include a depth-storage ULP.
// Matrix construction/inversion and rasterization error are not bounded here.
float3 screen_world_error(float4 clip,float4x4 inverse_vp) {
    let h=mul(clip,inverse_vp);let p=h.xyz/h.w;
    let error=REG_FP32_GAMMA7*mul(abs(clip),abs(inverse_vp))
        +abs(inverse_vp[2])*abs(clip.z)*REG_FP32_EPSILON;
    if (abs(h.w)<=error.w) {return float3(REG_FP32_MAX);}
    return (error.xyz+abs(p)*error.w)/(abs(h.w)-error.w)+REG_FP32_GAMMA7*abs(p);
}
float3 screen_world_error(float2 uv,float z,float4x4 inverse_vp) {
    return screen_world_error(float4(uv*float2(2.,-2.)+float2(-1.,1.),z,1.),inverse_vp);
}
float3 screen_world_error(float3 world,Frame frame) {
    let clip=mul(float4(world,1.),frame.vp);
    return screen_world_error(float4(clip.xyz/clip.w,1.),frame.inverse_vp);
}
float2 screen_projection_error(float3 world,float3 error,float4x4 vp) {
    let h=mul(float4(world,1.),vp);let ndc=h.xy/h.w;
    let e=mul(float4(error,0.),abs(vp))+REG_FP32_GAMMA7*mul(float4(abs(world),1.),abs(vp));
    if (abs(h.w)<=e.w) {return float2(REG_FP32_MAX);}
    return .5*((e.xy+abs(ndc)*e.w)/(abs(h.w)-e.w)+REG_FP32_GAMMA7*abs(ndc));
}

// Clip the complete world ray against the camera frustum before dividing by w.
// Interpolating in homogeneous clip space avoids catastrophic world-coordinate
// cancellation at the near plane. Explicit view-depth planes preserve the
// authored finite range even when the FP32 projection rounds to infinite-far.
bool screen_ray_segment(float3 origin,float3 direction,Frame frame,out float3 start,out float3 end) {
    start=float3(0.);end=float3(0.);
    let a=mul(float4(origin,1.),frame.vp);
    let b=mul(float4(direction,0.),frame.vp);
    let depth=-mul(float4(origin,1.),frame.view).z;
    let slope=-mul(float4(direction,0.),frame.view).z;
    let planes=float2[8](
        float2(a.w+a.x,b.w+b.x),float2(a.w-a.x,b.w-b.x),
        float2(a.w+a.y,b.w+b.y),float2(a.w-a.y,b.w-b.y),
        float2(a.z,b.z),float2(a.w-a.z,b.w-b.z),
        float2(depth-frame.params.x,slope),float2(frame.params.y-depth,-slope));
    var lo=0.;var hi=REG_FP32_MAX;
    for (uint plane=0u;plane<8u;plane++) {
        let value=planes[plane].x;let derivative=planes[plane].y;
        if (derivative==0.) {if (value<0.) {return false;}}
        else if (derivative>0.) {lo=max(lo,-value/derivative);}
        else {hi=min(hi,-value/derivative);}
    }
    if (hi<=lo) {return false;}
    let first=a+b*lo;let last=a+b*hi;
    if (first.w<=0. || last.w<=0.) {return false;}
    start=saturate(float3(first.xy/first.w*float2(.5,-.5)+.5,first.z/first.w));
    end=saturate(float3(last.xy/last.w*float2(.5,-.5)+.5,last.z/last.w));
    return true;
}

// Recover the local plane from the flatter depth neighbour on each axis.
// Normalize tangents before their cross product: a metre-independent normal
// must not vanish merely because its geometry is small (or overflow if large).
float3 screen_geometric_normal(Texture2D<float4> receivers,int2 pixel,float3 position,float3 shading) {
    uint width,height;receivers.GetDimensions(width,height);let size=int2(width,height);
    float3 tangents[2]={};
    for (int axis=0;axis<2;axis++) {
        var best=float3(0.);var best_length=REG_FP32_MAX;var found=false;
        for (int side=-1;side<=1;side+=2) {
            var offset=int2(0);offset[axis]=side;let p=pixel+offset;
            if (any(p<int2(0)) || any(p>=size)) {continue;}
            let tap=receivers.Load(int3(p,0));if(tap.w>=1.) {continue;}
            let point=tap.xyz;
            let d=(point-position)*float(side);
            let scale=max(abs(d.x),max(abs(d.y),abs(d.z)));
            let distance=scale==0. ? 0. : scale*length(d/scale);
            if (!found || distance<best_length) {best=d;best_length=distance;found=true;}
        }
        let scale=max(abs(best.x),max(abs(best.y),abs(best.z)));
        if (!found || scale==0.) {return shading;}
        tangents[axis]=best/scale;
    }
    let g=cross(tangents[0],tangents[1]);let scale=max(abs(g.x),max(abs(g.y),abs(g.z)));
    if (scale==0.) {return shading;}
    let n=normalize(g/scale);return select(dot(n,shading)>=0.,n,-n);
}

// Authored environment images occupy a packed float32 buffer at their own extents.
// Decode before reconstruction, preserving HDR and both authored address modes.
int environment_image_address(int p,int n,uint mode) {
    if(mode==1u) return clamp(p,0,n-1);
    if(mode==2u) {let q=(p+2*n)%(2*n);return min(q,2*n-1-q);}
    return (p+n)%n;
}
float4 environment_image_load(StructuredBuffer<uint> image,int2 p,int2 size,uint2 mode,uint slot) {
    let q=int2(environment_image_address(p.x,size.x,mode.x),environment_image_address(p.y,size.y,mode.y));
    let base=image[slot];let pixel=uint(q.y)*uint(size.x)+uint(q.x);uint4 bits;
    [unroll] for(uint c=0u;c<4u;c++) {
        let descriptor=image[base+6u+c];
        if((image[base+5u]&(1u<<c))!=0u) {bits[c]=descriptor;}
        else {bits[c]=image[base+descriptor+pixel];}
    }
    return asfloat(bits);
}
float4 environment_image_sample(StructuredBuffer<uint> image,float2 uv,uint2 extent,uint3 mode,uint slot) {
    let base=image[slot];
    if(image[base+5u]==15u) {
        return asfloat(uint4(image[base+6u],image[base+7u],image[base+8u],image[base+9u]));
    }
    var bounded=uv;
    for(uint k=0;k<2u;k++) {
        if(mode[k]==1u) bounded[k]=clamp(uv[k],0.,1.);
        else {let period=mode[k]==2u ? 2. : 1.;bounded[k]=uv[k]-floor(uv[k]/period)*period;}
    }
    let size=int2(extent);let p=bounded*float2(size)-.5;let q=int2(floor(p));let f=p-floor(p);
    if(mode.z!=0u) return environment_image_load(image,int2(floor(p+.5)),size,mode.xy,slot);
    let a=environment_image_load(image,q,size,mode.xy,slot);
    let b=environment_image_load(image,q+int2(1,0),size,mode.xy,slot);
    let c=environment_image_load(image,q+int2(0,1),size,mode.xy,slot);
    let d=environment_image_load(image,q+int2(1,1),size,mode.xy,slot);
    return (a+(b-a)*f.x)+((c+(d-c)*f.x)-(a+(b-a)*f.x))*f.y;
}

// Float32 source bits live in an integer texture: no optional float filtering.
// Decode each explicit load before masking and bilinear interpolation.
// Longitude wraps and latitude clamps, matching the environment image sampler.
float3 environment_texel(Texture2DArray<uint4> image, int2 cell, bool residual) {
    return asfloat(image.Load(int4(cell,residual ? 1 : 0,0))).rgb;
}
float3 environment_radiance(Texture2DArray<uint4> image, float2 uv, uint mode=0u, bool residual=false) {
    uint width,height,layers; image.GetDimensions(width,height,layers);
    // Address in floating point before converting to texel indices. In
    // particular, negative/large image UVs must never overflow an int cast.
    var u=frac(uv.x);
    if ((mode&3u)==1u) {u=clamp(uv.x,0.,1.);}
    if ((mode&3u)==2u) {u=1.-abs(2.*frac(uv.x*.5)-1.);}
    let size=int2(width,height);let p=float2(u,clamp(uv.y,0.,1.))*float2(size)-.5;
    let cell=int2(floor(p));let f=frac(p);
    // u is in [0,1], so cell.x is in [-1,width-1]. Keep both % operands
    // nonnegative: MSL (and C++) leave a negative remainder's result undefined.
    var x0=(cell.x+size.x)%size.x;var x1=(x0+1)%size.x;
    if ((mode&3u)==1u) {x0=clamp(cell.x,0,size.x-1);x1=clamp(cell.x+1,0,size.x-1);}
    if ((mode&3u)==2u) {x0=clamp(cell.x,0,size.x-1);x1=clamp(cell.x+1,0,size.x-1);}
    let y0=clamp(cell.y,0,size.y-1);let y1=clamp(cell.y+1,0,size.y-1);
    if ((mode&4u)!=0u) {return environment_texel(image,int2(f.x<.5 ? x0 : x1,f.y<.5 ? y0 : y1),residual);}
    // Difference interpolation preserves a constant image exactly. Weighted
    // sums of equal HDR texels can round down before half-float filtering.
    let a=environment_texel(image,int2(x0,y0),residual);
    let b=environment_texel(image,int2(x1,y0),residual);
    let c=environment_texel(image,int2(x0,y1),residual);
    let d=environment_texel(image,int2(x1,y1),residual);
    let top=a+(b-a)*f.x;
    let bottom=c+(d-c)*f.x;
    let result=top+(bottom-top)*f.y;
    return result;
}

// Invocation-local scene wavelength. Non-path evaluations retain RGB (zero).
static float vork_path_wavelength=0.;

// Fade receiver reconstruction over the triangle's normal cone. Flat faces
// have no cone and no correction. Away from the smooth horizon ordinary cast
// shadows retain the actual triangle origin, including concave contact shadows.
float3 surface_shadow_offset(float4 geometry,float3 face,float3 smooth,float3 direction) {
    let cone=geometry.w;if(cone<=0.) return float3(0.);
    let nl=dot(smooth,direction);let ng=dot(face,direction);
    let amount=nl<cone ? saturate(2.-(ng+nl)/cone) : saturate(1.-ng/cone);
    return geometry.xyz*amount;
}
