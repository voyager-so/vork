// Blender Bloom arithmetic, expressed as ordinary image-graph nodes.
import vork;
public enum BloomStage { Prepare, Downsample, Upsample, Combine };
public struct Inputs {
    BloomStage stage = BloomStage::Combine;
    Image source;
    Image original;
    Image fine;
    uint divisor = 1;
    bool karis = false;
    uint levels = 1;
    [InputRange(0.0, 3.402823466e+38, true, true)] float threshold = 1.;
    [InputRange(0.0, 1.0, true, true)] float smoothness = 0.1;
    bool clamp = false;
    [InputRange(0.0, 3.402823466e+38, true, true)] float maximum = 10.;
    [InputRange(0.0, 3.402823466e+38, true, true)] float strength = 1.;
    [InputRange(0.0, 3.402823466e+38, true, true)] float saturation = 1.;
    [InputRange(0.0, 3.402823466e+38, true, true)] float3 tint = float3(1);
};
float4 intermediate(float3 c) { return float4(c,1.); }
float3 filtered(Image source,float2 uv) { return compositor_sample(source,uv).rgb; }
float3 quad(Image source,float2 uv,float2 center,bool karis) {
    float3 sum=0.;float weights=0.;
    for(int y=-1;y<=1;y+=2) for(int x=-1;x<=1;x+=2) {
        float3 c=filtered(source,uv+(center+float2(x,y))/compositor_size(source));
        float w=karis ? 1./(1.+max(c.r,max(c.g,c.b))) : 1.;
        sum+=c*w;weights+=w;
    }
    return sum/weights;
}
float3 box(Image original,int2 pixel,int2 extent,uint divisor) {
    int2 p=clamp(pixel,int2(0),extent-1)*int(divisor);
    float3 c=0.;
    for(uint y=0;y<divisor;y++) for(uint x=0;x<divisor;x++)
        c+=compositor_load(original,p+int2(x,y)).rgb;
    return c/float(divisor*divisor);
}
public float4 composite(CompositeContext ctx, Inputs inputs) {
    if(int(inputs.stage) == int(BloomStage::Prepare)) {
        // Box reduction precedes the nonlinear highlight threshold.
        uint divisor=max(inputs.divisor,1u);
        int2 extent=int2(ceil(compositor_size(inputs.original)/float(divisor)));
        float2 p=ctx.uv*float2(extent)-.5;int2 q=int2(floor(p));float2 f=frac(p);
        float3 a=lerp(box(inputs.original,q,extent,divisor),box(inputs.original,q+int2(1,0),extent,divisor),f.x);
        float3 b=lerp(box(inputs.original,q+int2(0,1),extent,divisor),box(inputs.original,q+int2(1,1),extent,divisor),f.x);
        float3 c=max(lerp(a,b,f.y),0.);
        float peak=max(c.r,max(c.g,c.b));
        float width=min(inputs.threshold,inputs.smoothness);
        float h=max(peak-inputs.threshold,0.);
        if(width>0.) {float d=max(width-abs(peak-inputs.threshold),0.);h+=d*d/(4.*width);}
        if(inputs.clamp) h=min(h,inputs.maximum);
        return intermediate(peak>0. ? c*(h/peak) : float3(0.));
    }
    if(int(inputs.stage) == int(BloomStage::Downsample)) {
        float3 c=.5*quad(inputs.source,ctx.uv,0.,inputs.karis);
        for(int y=-1;y<=1;y+=2) for(int x=-1;x<=1;x+=2)
            c+=.125*quad(inputs.source,ctx.uv,float2(x,y),inputs.karis);
        return intermediate(c);
    }
    if(int(inputs.stage) == int(BloomStage::Upsample)) {
        float3 c=0.;
        for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++)
            c+=float((2-abs(x))*(2-abs(y)))/16.*filtered(inputs.source,ctx.uv+float2(x,y)/ctx.size);
        return intermediate(compositor_load(inputs.fine,int2(ctx.uv*ctx.size)).rgb+c);
    }
    float4 original=max(compositor_sample(inputs.original,ctx.uv),0.);
    float3 bloom=filtered(inputs.source,ctx.uv);
    float peak=max(bloom.r,max(bloom.g,bloom.b));
    float minimum=min(bloom.r,min(bloom.g,bloom.b));
    if(peak>minimum) {
        float saturation=clamp((peak-minimum)/peak*inputs.saturation,0.,1.);
        bloom=lerp(float3(peak),bloom,saturation*peak/(peak-minimum));
    }
    return float4(original.rgb+bloom*inputs.tint*inputs.strength/float(max(inputs.levels,1u)),original.a);
}
