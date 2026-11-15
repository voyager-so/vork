// Texture functions in scene-linear space. No lights, resources or kernel state.
// Independent implementations of the node definitions documented in Blender's
// manual; no Blender implementation source is incorporated.
// Integer mixing uses Bob Jenkins' public-domain lookup3 finalizer.
static const float TEX_PI = 3.14159265358979323846;
uint tex_rot(uint x, uint k) { return (x << k) | (x >> (32-k)); }
uint tex_hash(uint x, uint y, uint z) {
    uint a=0xdeadbeefu+13u+x, b=0xdeadbeefu+13u+y, c=0xdeadbeefu+13u+z;
    c^=b; c-=tex_rot(b,14); a^=c; a-=tex_rot(c,11);
    b^=a; b-=tex_rot(a,25); c^=b; c-=tex_rot(b,16);
    a^=c; a-=tex_rot(c,4); b^=a; b-=tex_rot(a,14);
    c^=b; c-=tex_rot(b,24); return c;
}
public float tex_random(int3 p) { return float(tex_hash(uint(p.x),uint(p.y),uint(p.z)))/4294967295.0; }
float3 tex_random3(int3 p) {
    return float3(tex_random(p),tex_random(p.yzx+int3(19,47,101)),tex_random(p.zxy+int3(73,11,29)));
}
float tex_grad(uint h, float3 p) {
    h &= 15u; float u=h<8u?p.x:p.y;
    float v=h<4u?p.y:((h==12u||h==14u)?p.x:p.z);
    return ((h&1u)!=0u?-u:u)+((h&2u)!=0u?-v:v);
}
// Improved Perlin noise, signed, with quintic interpolation.
float tex_perlin(float3 p) {
    int3 cell=int3(floor(p)); float3 f=frac(p), t=f*f*f*(f*(f*6.-15.)+10.);
    float values[8];
    for(int z=0;z<2;z++) for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
        int3 q=cell+int3(x,y,z);
        values[x+2*y+4*z]=tex_grad(tex_hash(uint(q.x),uint(q.y),uint(q.z)),f-float3(x,y,z));
    }
    return 0.982*lerp(lerp(lerp(values[0],values[1],t.x),lerp(values[2],values[3],t.x),t.y),
                      lerp(lerp(values[4],values[5],t.x),lerp(values[6],values[7],t.x),t.y),t.z);
}
// Improved gradient noise in 1D--4D. Unused coordinates do not affect a sample.
uint tex_hash4(int4 p) { return tex_hash(tex_hash(uint(p.x),uint(p.y),uint(p.z)),uint(p.w),0u); }
float tex_perlin_nd(float4 p, int dimensions) {
    dimensions=clamp(dimensions,1,4);
    if(dimensions==3) return tex_perlin(p.xyz);
    for(int axis=dimensions;axis<4;axis++) p[axis]=0.;
    int4 cell=int4(floor(p)); float4 f=frac(p), t=f*f*f*(f*(f*6.-15.)+10.);
    float sum=0.;
    for(int corner=0;corner<(1<<dimensions);corner++) {
        int4 delta=int4(corner&1,(corner>>1)&1,(corner>>2)&1,(corner>>3)&1);
        uint h=tex_hash4(cell+delta); float weight=1., gradient=0.;
        for(int axis=0;axis<dimensions;axis++) {
            weight*=delta[axis]!=0?t[axis]:1.-t[axis];
            gradient+=((h&(1u<<axis))!=0u?1.:-1.)*(f[axis]-float(delta[axis]));
        }
        sum+=gradient*weight;
    }
    // Each signed-gradient component contributes at most 1/2 after interpolation.
    return sum*(2./float(dimensions));
}
public enum NoiseType { FBM, Multifractal, Hybrid_Multifractal, Ridged_Multifractal, Hetero_Terrain };
float tex_fractal(float4 p, int dimensions, float detail, float roughness,
                  float lacunarity, bool normalize, NoiseType type, float offset, float gain) {
    detail=clamp(detail,0.,15.); roughness=max(roughness,0.);
    float sum=0., amplitude=1., amplitude_sum=0., previous=1., weight=1.;
    if(int(type) == int(NoiseType.Multifractal)) sum=1.;
    int whole=int(floor(detail));
    for(int i=0;i<=whole+1;i++) {
        float n=tex_perlin_nd(p,dimensions), next=sum;
        if(int(type) == int(NoiseType.FBM)) next=sum+n*amplitude;
        else if(int(type) == int(NoiseType.Multifractal)) next=sum*(1.+n*amplitude);
        else if(int(type) == int(NoiseType.Hetero_Terrain)) next=i==0?offset+n:sum+(offset+n)*amplitude*sum;
        else if(int(type) == int(NoiseType.Hybrid_Multifractal)) {
            float signal=(n+offset)*amplitude;
            if(i==0) { next=signal; weight=gain*signal; }
            else { weight=saturate(weight); next=sum+weight*signal; weight*=gain*signal; }
        } else {
            float signal=offset-abs(n); signal*=signal;
            if(i>0) signal*=saturate(previous*gain);
            next=sum+signal*amplitude; previous=signal;
        }
        if(i==whole+1) {
            if(int(type) == int(NoiseType.FBM) && normalize)
                return lerp(0.5+0.5*sum/amplitude_sum,0.5+0.5*next/(amplitude_sum+amplitude),frac(detail));
            return lerp(sum,next,frac(detail));
        }
        sum=next; amplitude_sum+=amplitude; amplitude*=roughness; p*=lacunarity;
    }
    return sum;
}
float tex_noise(float3 vector, float scale, float detail, float roughness,
    float lacunarity, float distortion, bool normalize, NoiseType type,
    int dimensions, float w, float offset, float gain, float4 seed) {
    dimensions=clamp(dimensions,1,4);
    float4 p=float4(vector,w)*scale;
    if(dimensions==1) p=float4(w*scale,0.,0.,0.);
    if(dimensions==2) p.zw=0.;
    if(dimensions==3) p.w=0.;
    // Channel offsets are independent of spatial scale, including scale zero.
    p+=seed;
    if(distortion!=0.) {
        float4 warped=p;
        for(int axis=0;axis<dimensions;axis++)
            warped[axis]+=distortion*tex_perlin_nd(p+float4(13.5+axis*19.1,7.2+axis*3.4,19.1+axis*11.2,5.7+axis*8.3),dimensions);
        p=warped;
    }
    return tex_fractal(p,dimensions,detail,roughness,lacunarity,normalize,type,offset,gain);
}
// In 1D, W supplies the coordinate; in 4D it supplies the fourth coordinate.
// Detail counts layers beyond the base octave; fractional detail blends the last.
public float noise(float3 vector, float scale=5., float detail=2., float roughness=0.5,
                   float lacunarity=2., float distortion=0., bool normalize=true,
                   NoiseType type=NoiseType.FBM, int dimensions=3, float w=0.,
                   float offset=0., float gain=1.) {
    return tex_noise(vector,scale,detail,roughness,lacunarity,distortion,normalize,type,dimensions,w,offset,gain,float4(0.));
}
public float3 noise_color(float3 vector, float scale=5., float detail=2., float roughness=0.5,
                   float lacunarity=2., float distortion=0., bool normalize=true,
                   NoiseType type=NoiseType.FBM, int dimensions=3, float w=0.,
                   float offset=0., float gain=1.) {
    return float3(tex_noise(vector,scale,detail,roughness,lacunarity,distortion,normalize,type,dimensions,w,offset,gain,float4(0.)),
        tex_noise(vector,scale,detail,roughness,lacunarity,distortion,normalize,type,dimensions,w,offset,gain,float4(19.1,7.3,41.7,17.1)),
        tex_noise(vector,scale,detail,roughness,lacunarity,distortion,normalize,type,dimensions,w,offset,gain,float4(73.2,11.8,29.6,53.7)));
}
public enum VoronoiFeature { F1, F2, Smooth_F1, Distance_To_Edge, N_Sphere_Radius };
public enum DistanceMetric { Euclidean, Manhattan, Chebychev, Minkowski };
public struct VoronoiResult { public float distance; public float3 color; public float3 position; public float w; public float radius; };
float4 tex_random4(int4 p) {
    return float4(float(tex_hash4(p))/4294967295.,float(tex_hash4(p+int4(19,47,101,73)))/4294967295.,
        float(tex_hash4(p+int4(73,11,29,53)))/4294967295.,float(tex_hash4(p+int4(31,67,7,97)))/4294967295.);
}
float4 tex_point_random(int4 p,int dimensions) {
    // Preserve the existing 3D point process.
    return dimensions==3?float4(tex_random3(p.xyz),0.):tex_random4(p);
}
float tex_distance_nd(float4 v,int dimensions,DistanceMetric metric,float exponent) {
    float sum=0.,hi=0.;
    for(int i=0;i<dimensions;i++) {
        float a=abs(v[i]);hi=max(hi,a);
        sum+=int(metric) == int(DistanceMetric.Manhattan)?a:int(metric) == int(DistanceMetric.Minkowski)?pow(a,exponent):a*a;
    }
    if(int(metric) == int(DistanceMetric.Chebychev)) return hi;
    if(int(metric) == int(DistanceMetric.Manhattan)) return sum;
    if(int(metric) == int(DistanceMetric.Minkowski)) return pow(sum,1./exponent);
    return sqrt(sum);
}
VoronoiResult tex_voronoi(float4 p,float scale,int dimensions,float randomness,
    VoronoiFeature feature,DistanceMetric metric,float smoothness,float exponent) {
    int4 cell=int4(floor(p));float4 local=frac(p);
    float d1=1.e20,d2=1.e20;float4 pos1=0.,pos2=0.;float3 col1=0.,col2=0.;
    int extent=int(feature) == int(VoronoiFeature.Smooth_F1)?2:1;
    int ey=dimensions>1?extent:0,ez=dimensions>2?extent:0,ew=dimensions>3?extent:0;
    for(int w=-ew;w<=ew;w++) for(int z=-ez;z<=ez;z++) for(int y=-ey;y<=ey;y++) for(int x=-extent;x<=extent;x++) {
        int4 offset=int4(x,y,z,w);float4 point=float4(offset)+randomness*tex_point_random(cell+offset,dimensions);
        for(int axis=dimensions;axis<4;axis++) point[axis]=0.;
        float d=tex_distance_nd(point-local,dimensions,metric,max(exponent,0.000001));
        float3 color=tex_point_random(cell+offset,dimensions).xyz;
        if(int(feature) == int(VoronoiFeature.Smooth_F1) && smoothness>0.) {
            // d1 starts as a sentinel, not a distance, so it never enters lerp.
            if(d1==1.e20) {d1=d;pos1=point;col1=color;continue;}
            float h=smoothstep(0.,1.,0.5+0.5*(d1-d)/smoothness),correction=smoothness*h*(1.-h);
            d1=lerp(d1,d,h)-correction;correction/=1.+3.*smoothness;
            col1=lerp(col1,color,h)-correction;pos1=lerp(pos1,point,h)-correction;
        } else if(d<d1) {d2=d1;pos2=pos1;col2=col1;d1=d;pos1=point;col1=color;}
        else if(d<d2) {d2=d;pos2=point;col2=color;}
    }
    VoronoiResult result;result.distance=d1;result.color=col1;result.radius=0.;
    if(int(feature) == int(VoronoiFeature.F2)) {result.distance=d2;result.color=col2;pos1=pos2;}
    float4 position=scale==0.?float4(0.):(pos1+float4(cell))/scale;
    result.position=position.xyz;result.w=dimensions==1?position.x:position.w;
    if(dimensions==1) result.position=float3(0.);
    if(int(feature) == int(VoronoiFeature.Distance_To_Edge) || int(feature) == int(VoronoiFeature.N_Sphere_Radius)) {
        float edge=1.e20,radius=1.e20;int4 nearest=int4(floor(pos1));
        for(int w=dimensions>3?-1:0;w<=(dimensions>3?1:0);w++)
        for(int z=dimensions>2?-1:0;z<=(dimensions>2?1:0);z++)
        for(int y=dimensions>1?-1:0;y<=(dimensions>1?1:0);y++) for(int x=-1;x<=1;x++) {
            int4 offset=nearest+int4(x,y,z,w);
            float4 point=float4(offset)+randomness*tex_point_random(cell+offset,dimensions);
            for(int axis=dimensions;axis<4;axis++) point[axis]=0.;
            float4 between=point-pos1;float len=length(between);
            if(len>0.0001) {radius=min(radius,len*0.5);edge=min(edge,dot((point+pos1)*0.5-local,between/len));}
        }
        result.radius=radius;if(int(feature) == int(VoronoiFeature.Distance_To_Edge)) result.distance=edge;
    }
    return result;
}
public VoronoiResult voronoi(float3 vector,float scale=5.,float randomness=1.,
    VoronoiFeature feature=VoronoiFeature.F1,DistanceMetric metric=DistanceMetric.Euclidean,
    float smoothness=1.,float exponent=0.5,int dimensions=3,float w=0.,
    float detail=0.,float roughness=0.5,float lacunarity=2.,bool normalize=false) {
    dimensions=clamp(dimensions,1,4);float4 p=float4(vector,w)*scale;
    if(dimensions==1) p=float4(w*scale,0.,0.,0.);
    for(int axis=dimensions;axis<4;axis++) p[axis]=0.;
    detail=clamp(detail,0.,15.);roughness=saturate(roughness);
    VoronoiResult result={};float amplitude=1.,weight=0.,frequency=1.;
    int whole=int(floor(detail));
    if(int(feature) == int(VoronoiFeature.N_Sphere_Radius)) whole=0;
    for(int octave=0;octave<=whole+1;octave++) {
        float blend=octave==whole+1?frac(detail):1.;
        if(int(feature) == int(VoronoiFeature.N_Sphere_Radius) && octave>0) break;
        if(blend==0.) break;
        VoronoiResult sample=tex_voronoi(p*frequency,scale*frequency,dimensions,saturate(randomness),feature,metric,clamp(smoothness*0.5,0.,0.5),exponent);
        float a=amplitude*blend;
        result.distance+=sample.distance*a;result.color+=sample.color*a;
        result.position+=sample.position*a;result.w+=sample.w*a;result.radius+=sample.radius*a;
        weight+=a;amplitude*=roughness;frequency*=lacunarity;
    }
    if(normalize && weight>0.) {
        // Normalize Euclidean distance to the diagonal of the jitter cell.
        float maximum=int(metric) == int(DistanceMetric.Manhattan)?float(dimensions):int(metric) == int(DistanceMetric.Chebychev)?1.:
            int(metric) == int(DistanceMetric.Minkowski)?pow(float(dimensions),1./max(exponent,0.000001)):sqrt(float(dimensions));
        result.distance/=weight*maximum;
    }
    // Color/position identify the average feature independently of normalization.
    if(weight>0.) {result.color/=weight;result.position/=weight;result.w/=weight;result.radius/=weight;}
    return result;
}
// White noise hashes IEEE coordinate bits: continuous input, no implicit cells.
// Signed zero is canonicalized so equal coordinates always hash equally.
public struct WhiteNoiseResult { public float value; public float3 color; };
public WhiteNoiseResult white_noise(float3 vector,int dimensions=3,float w=0.) {
    dimensions=clamp(dimensions,1,4);float4 p=float4(vector,w);
    if(dimensions==1) p=float4(w,0.,0.,0.);
    for(int axis=dimensions;axis<4;axis++) p[axis]=0.;
    int4 bits;for(int axis=0;axis<4;axis++) bits[axis]=int(asuint(p[axis]==0.?0.:p[axis]));
    float4 random=tex_random4(bits);WhiteNoiseResult result;result.value=random.x;result.color=random.xyz;return result;
}
public struct GaborResult { public float value; public float phase; public float intensity; };
public GaborResult gabor(float3 vector,float scale=5.,float frequency=2.,float anisotropy=1.,
    int dimensions=2,float orientation=0.7853981633974483,float3 direction=float3(1.41421356237,1.41421356237,0.)) {
    dimensions=dimensions==2?2:3;float3 p=vector*scale;
    if(dimensions==2) p.z=0.;
    int3 cell=int3(floor(p));float3 local=frac(p);float2 sum=0.;
    float3 axis=dimensions==2?float3(cos(orientation),sin(orientation),0.):direction;
    axis=dot(axis,axis)>0.?normalize(axis):float3(1.,0.,0.);
    // Compact Gaussian-windowed sparse convolution, independent impulse process.
    for(int z=dimensions==3?-1:0;z<=(dimensions==3?1:0);z++)
    for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
        int3 offset=int3(x,y,z);
        for(int impulse=0;impulse<8;impulse++) {
            int3 key=cell+offset+int3(impulse*101,impulse*47,impulse*19);
            float3 random=tex_random3(key),delta=float3(offset)+random-local;
            if(dimensions==2) delta.z=0.;
            float r2=dot(delta,delta);if(r2>=1.) continue;
            float3 random_axis=tex_random3(key+int3(73,29,11))*2.-1.;
            if(dimensions==2) {float angle=random_axis.x*TEX_PI;random_axis=float3(cos(angle),sin(angle),0.);}
            float3 dir=lerp(random_axis,axis,saturate(anisotropy));
            dir=dot(dir,dir)>0.?normalize(dir):axis;
            float phase=2.*TEX_PI*(max(frequency,0.)*dot(dir,delta)+tex_random(key+int3(17,59,83)));
            float envelope=exp(-TEX_PI*3.*r2)-exp(-TEX_PI*3.);
            sum+=envelope*float2(cos(phase),sin(phase));
        }
    }
    GaborResult result;
    // Display value is a centered, inverted sine of the complex phase.
    // L2 energy of exp(-3*pi*r^2), with eight impulses per cell.
    float kernel_energy=pow(1./6.,float(dimensions)*0.5);
    float normalization=0.25*rsqrt(8.*kernel_energy);
    result.intensity=saturate(length(sum)*normalization);result.phase=frac(atan2(sum.y,sum.x)/(2.*TEX_PI));
    result.value=0.5-0.5*sin(2.*TEX_PI*result.phase)*result.intensity;return result;
}

// Blender Wave, Brick and Magic textures in scene-linear space. Independent
// implementations of the node definitions documented in Blender's manual; no
// Blender implementation source is incorporated.
static const float BLENDER_TEX_PI = 3.14159265358979323846;
public enum WaveType { Bands, Rings };
public enum WaveDirection { X, Y, Z, Diagonal };
public enum WaveProfile { Sin, Saw, Tri };
public float wave(float3 vector, float scale=5., float distortion=0., float detail=2.,
    float detail_scale=1., float detail_roughness=0.5, float phase_offset=0.,
    WaveType type=WaveType.Bands, WaveDirection direction=WaveDirection.X,
    WaveProfile profile=WaveProfile.Sin) {
    float3 p=vector*scale; float n;
    if(int(type) == int(WaveType.Bands)) n=(int(direction) == int(WaveDirection.X)?p.x:int(direction) == int(WaveDirection.Y)?p.y:int(direction) == int(WaveDirection.Z)?p.z:p.x+p.y+p.z)*20.;
    else n=length(int(direction) == int(WaveDirection.X)?p.yz:int(direction) == int(WaveDirection.Y)?p.xz:p.xy)*20.;
    if(int(type) == int(WaveType.Rings) && int(direction) == int(WaveDirection.Diagonal)) n=length(p)*20.;
    n+=phase_offset;
    if(distortion!=0.) n+=distortion*(noise(p*detail_scale,1.,detail,detail_roughness)*2.-1.);
    if(int(profile) == int(WaveProfile.Saw)) return frac(n/(2.*BLENDER_TEX_PI));
    if(int(profile) == int(WaveProfile.Tri)) return abs(frac(n/(2.*BLENDER_TEX_PI)+0.5)*2.-1.);
    return 0.5+0.5*sin(n-BLENDER_TEX_PI*0.5);
}
public struct BrickResult { public float3 color; public float factor; };
public BrickResult brick(float3 vector, float3 color1=float3(0.8), float3 color2=float3(0.2),
    float3 mortar=float3(0.), float scale=5., float mortar_size=0.02, float mortar_smooth=0.,
    float bias=0., float brick_width=0.5, float row_height=0.25,
    float offset=0.5, int offset_frequency=2, float squash=1., int squash_frequency=2) {
    float2 p=vector.xy*scale; float height=max(row_height,0.000001);int row=int(floor(p.y/height));
    float width=max(brick_width,0.000001);
    if(squash_frequency>0 && row%squash_frequency==0) width*=max(squash,0.000001);
    if(offset_frequency>0 && row%offset_frequency==0) p.x+=width*offset;
    int column=int(floor(p.x/width));float2 local=float2(p.x-float(column)*width,p.y-float(row)*height);
    float dist=min(min(local.x,width-local.x),min(local.y,height-local.y));
    float factor=dist<mortar_size?1.:0.;
    if(mortar_smooth>0. && mortar_size>0.) factor=1.-smoothstep(mortar_size*(1.-mortar_smooth),mortar_size,dist);
    BrickResult result;result.factor=factor;
    result.color=lerp(lerp(color1,color2,saturate(tex_random(int3(column,row,0))+bias)),mortar,factor);return result;
}
public struct MagicResult { public float3 color; public float factor; };
public MagicResult magic(float3 vector,float scale=5.,float distortion=1.,int depth=2) {
    // Coupled trigonometric iteration; each pass feeds the next channel.
    float3 p=vector*scale*5.;
    float3 v=float3(sin(p.x+p.y+p.z),cos(-p.x+p.y-p.z),-cos(-p.x-p.y+p.z));
    for(int i=0;i<clamp(depth,0,10);i++) {
        int axis=i%3;
        v[axis]=sin((v[(axis+1)%3]+v[(axis+2)%3])*distortion);
    }
    MagicResult result;result.color=0.5-0.5*v;result.factor=(result.color.x+result.color.y+result.color.z)/3.;return result;
}
