// A physical escape terminates at World. Retained directional radiance supplies
// secondary endpoints; unsupported local data remains explicit.
import vork_prelude;
import vork_constants;
#ifdef VORK_PROBES
import probes;
import probe_visibility_policy;
#endif
struct DirectionalRadianceSample {float3 radiance;bool local;bool supported;};
DirectionalRadianceSample directional_provider(float3 world,float3 normal,float3 view,float3 direction,float roughness,bool escaped,Frame frame,Texture2D environment,SamplerState environment_sampler,bool world_permitted=true) {
    var terminal=world_permitted ? reflection_fallback(direction,roughness,frame,environment,environment_sampler) : float3(0.);
#ifdef VORK_PROBES
    if(!escaped) {
        let sample=probe_directional_field(world,normal,view,direction,roughness,terminal);
        if(sample.owned) {return DirectionalRadianceSample(sample.radiance,true,sample.valid);}
    }
#endif
    return DirectionalRadianceSample(terminal,false,true);
}
float3 directional_radiance(float3 world,float3 normal,float3 view,float3 direction,float roughness,bool escaped,Frame frame,Texture2D environment,SamplerState environment_sampler,bool world_permitted=true) {
    return directional_provider(world,normal,view,direction,roughness,escaped,frame,environment,environment_sampler,world_permitted).radiance;
}
DirectionalRadianceSample receiver_provider(float3 world,float3 normal,float3 view,float3 direction,float roughness,Frame frame,Texture2D environment,SamplerState environment_sampler,bool capture_endpoint=false,bool world_permitted=true) {
#ifdef VORK_PROBES
    if(capture_endpoint) {
        if(!world_permitted) {return DirectionalRadianceSample(float3(0.),false,true);}
        var terminal=reflection_fallback(direction,1.,frame,environment,environment_sampler);
        let endpoint=probe_field_sample<ProbeMomentVisibility>(world,normal,view,direction,.75,normal,float3(0.));
        return DirectionalRadianceSample(endpoint.owned ? endpoint.radiance : terminal,endpoint.owned,!endpoint.owned || endpoint.valid);
    }
#endif
    return directional_provider(world,normal,view,direction,roughness,false,frame,environment,environment_sampler,world_permitted);
}
// A black terminal with no local provider has no reflected-radiance integral.
// Test source availability before evaluating an expensive directional BSDF.
bool receiver_reflection_active(Frame frame,bool world_permitted) {
#ifdef VORK_PROBES
    if(probe_grid().field.x!=0u) return true;
#endif
    return world_permitted && (any(frame.radiance.xyz!=float3(0.)) || frame.environment.x>0.);
}
float3 receiver_reflection(float3 world,float3 normal,float3 view,float3 direction,float roughness,Frame frame,Texture2D environment,SamplerState environment_sampler,bool capture_endpoint=false,bool world_permitted=true) {
    return receiver_provider(world,normal,view,direction,roughness,frame,environment,environment_sampler,capture_endpoint,world_permitted).radiance;
}
