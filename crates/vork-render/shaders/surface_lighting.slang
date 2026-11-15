__exported import surface;
#ifdef VORK_FIBRE
import fibre_transport;
#endif
// Surface lighting independent of raster attachments.
// Raster and secondary transport supply the shadow/thickness query policy.
import hair_lighting;
import probe_visibility_policy;
import vork_constants;
import vork_prelude;
import openpbr_core;
import dielectric_interface;
import frame;
import closure;
import material_layout;
import material;
import stream;
import bsdf;
import emitter;
import light_tree;
// Shared scene-global slot, declared only by lighting consumers. Geometry-only
// programs use their own compact buffer ABI and do not read this tree.
#ifndef VORK_ANALYTIC_TREE_REGISTER
#define VORK_ANALYTIC_TREE_REGISTER t5
#endif
[[vk::binding(10,0)]] StructuredBuffer<LightTreeNode> analytic_light_nodes : register(VORK_ANALYTIC_TREE_REGISTER);
import vork_sampling;
import reflection_environment;
import reflection_bsdf;
#ifdef VORK_PROBES
// Queued camera work supplies completed incident irradiance before evaluating
// a nonlinear lit hook. Secondary hits never consume this camera-local input.
static bool camera_indirect_override=false;
static float3 camera_indirect_irradiance=float3(0.);
#endif
import subsurface_transport;
import variant;
#ifdef VORK_PROBES
import probes;
#endif
// One closure's lighting and the responses replaced by screen passes.

// How a shading point queries triangle visibility, including a ray tracer's
// own visibility (path.slang PathShadows).
// Consumers own visibility and thickness queries, without raster bindings.
interface IShadows : IProbeVisibility {
    static const bool subsurface;
    static uint integration_count(uint index,out float3 normalization);
    static float3 integration_sample(uint tap,inout uint index,inout float3 world);
    static uint sample_count();
    static float3 visibility_direction(float3 world,float3 normal,LightValues record,uint light,float3 direction,float distance,float3 position_error,uint index,uint count,bool fibre=false);
    static float thickness(uint object,float3 world,float3 direction);
}

// sample_index selects one exposure-wide sample; the sentinel integrates a shared slot's quadrature.
struct LightSite { MaterialUv m; bool camera_clusters; bool secondary; float3 facet; uint2 pixel; float3 position_error; uint sample_index; float4 shadow_offset; bool exact_sources; };

// Analytic-source lighting shared by primary and secondary surfaces. The
// receiver owns its covered-sample quadrature and visibility; every source
// direction retains its exact BSDF response through that visibility query.
struct AnalyticRadiance {
    int selected_source=-1;float source_probability=1.;
    uint lobes;bool exposed;bool indirect_only;bool capture_endpoint;bool glossy_ambient;bool exact_sources;
    __init(uint lobes,bool exposed,bool indirect_only=false,bool capture_endpoint=false,bool glossy_ambient=true,bool exact_sources=false) {this.selected_source=-1;this.source_probability=1.;this.lobes=lobes;this.exposed=exposed;this.indirect_only=indirect_only;this.capture_endpoint=capture_endpoint;this.glossy_ambient=glossy_ambient;this.exact_sources=exact_sources;}
};

// One connection samples the complete emitter/visible-normal mixture. The
// light proposal retains support for diffuse, fuzz and transmitted response;
// the GGX proposal resolves narrow base/coat highlights without an energy cap.
LightConnection surface_light_sample<V:IVariant>(LightValues record,uint identity,float3 world,
    Layers layers,Closure closure,float3 view,bool hair,uint mask,float2 u,uint index) {
    // Previs deliberately uses central quadrature, including the BRDF response:
    // fixing visibility alone leaves four-sample direct-light grain on lit surfaces.
    if((frame.integrator.w&REG_INTEGRATOR_PREVIS_CENTER)!=0u) {return emitter_center_li<V>(record,identity,world,index);}
    if(hair || (mask&REG_RAY_SPECULAR)==0u || emitter_delta(record)
        || (layers.base.specular_weight==0. && dielectric_ior(layers.base)==1. && layers.base.thin_film_weight==0. && layers.base.base_metalness==0. && closure.coat_weight==0.)) {
        return emitter_sample_li<V>(record,identity,world,u,index);
    }
    let bsdf=reflection_bsdf(layers.base,closure,layers.scale,layers.throughput,1.,transport_metal_compensation(layers.base,view));
    LightConnection source={};source.identity=identity;source.sample_index=index;
    if(u.x<.5) {
        u.x*=2.;source=emitter_sample_li<V>(record,identity,world,u,index);
    } else {
        u.x=2.*u.x-1.;source.direction=reflection_bsdf_sample(bsdf,view,u);
        if(dot(source.direction,source.direction)==0.) {return source;}
        let value=emitter_eval_li<V>(record,world,source.direction);
        source.radiance=value.radiance;source.pdf=value.pdf;source.distance=value.distance;
        source.endpoint=world+source.direction*source.distance;
    }
    let pdf=reflection_bsdf_pdf(bsdf,view,source.direction);
    source.pdf=.5*(source.pdf+pdf);
    return source;
}

// Shared finite-light, ambient and emission evaluation for one closure.
Response shade_terms_at<V:IVariant, S:IShadows>(Varying v, Surface s, LightSite site, out float3 scattering, out float3 indirect_scattering,bool interior_transport=false,bool decode_table=true) {
    return analytic_radiance_at<V,S>(v,s,site,AnalyticRadiance(REG_RAY_DIFFUSE|REG_RAY_SPECULAR,true),scattering,indirect_scattering,interior_transport,decode_table);
}
// Shipped Vulkan surface entries retain one lighting graph through Slang, then restore
// driver inlining so closure payloads need not spill across a native call.
#if defined(VORK_SURFACE_SLANG_OUTLINE)
[noinline]
#endif
Response analytic_radiance_at<V:IVariant, S:IShadows>(Varying v, Surface s, LightSite site, AnalyticRadiance policy, out float3 scattering, out float3 indirect_scattering,bool interior_transport=false,bool decode_table=true) {
    let exposure=select(policy.exposed,frame.radiance.w,1.);
    s=transmission_interface<V>(s,site.facet);
    // Interior continuation already transports the transmitting share of TIR.
    // Scale reflected response explicitly; changing specular_weight changes IOR.
    let exit_direction=refract(-s.view,s.closure.geometry_normal,1./dielectric_ior(s.closure));
    let reflection_scale=interior_transport && !s.front_facing && !s.closure.geometry_thin_walled
        && dot(exit_direction,exit_direction)==0. ? 1.-s.closure.transmission_weight : 1.;
    var l=layers<V>(s.closure,s.view);
    let original=s.closure;
    // Finish deterministic material reads before shadow queries. Only the scalar
    // visibility, not its UV/filter payload, needs to survive the light loop.
    let material_visibility=material_occlusion<V>(v,site.m);
    var thickness=-1.;
    if (S.subsurface && (frame.integrator.w&REG_INTEGRATOR_SUBSURFACE_THICKNESS)!=0u && thickness_active(original) && !s.is_hair) {
        thickness=S.thickness(v.object,v.world,-s.geometric);
        if (thickness>=0.) {s.closure.subsurface_color=subsurface_slab(original,thickness).reflection;l=layers<V>(s.closure,s.view);}
    }
    let spatial=thickness>=0.;
    scattering=float3(0.);
    // Isolate the substrate, retaining its Fresnel and coat attenuation. Fuzz and
    // coat reflection stay sharp. This is the same evaluator, with a diffuse mask.
    var sl=l;sl.base.base_color=float3(1.);sl.base.subsurface_weight=0.;
    let sw=s.closure.geometry_thin_walled ? 0. : s.closure.subsurface_weight;
    let thin=original.geometry_thin_walled && original.subsurface_weight>0.;
    let c=s.closure;
    let geometric=s.geometric;let view=s.view;
    let specular=specular_response<V>(l.base,view);
    let diffuse_albedo=diffuse_quadrature(l.base) ? specular.directional_albedo : float3(-1.);
    let facet=site.facet;
    var direct=float3(0.);
    var list_base=0u;
    // Reconstructed camera receivers can leave their nominal pixel's ray on a
    // degenerate facet. Reproject the actual receiver and retain the complete
    // light list outside the camera frustum.
    // Directional-only variants have no spatially culled lights. Their complete
    // batch is already the exact cluster list, without a receiver lookup.
    // Source selection is a discrete outer proposal over the existing
    // emitter/GGX directional MIS. Small sets retain the exact traversal.
    let source_count=uint(frame_light_count(frame));
    let tree_nodes=source_count==0u ? 0u : 2u*source_count-1u;
    let sampled=(frame.integrator.w&REG_INTEGRATOR_PREVIS_CENTER)==0u && !policy.exact_sources && !site.exact_sources && policy.selected_source<0
        && source_count>REG_ANALYTIC_LIGHT_SAMPLES && analytic_light_nodes[0].links.w==source_count;
    let clustered=!sampled && policy.selected_source<0 && V.light_types>1u && site.camera_clusters && camera_cluster_base(v.world,frame,clusters,list_base);
    // Cluster lists are counted and sorted by light kind at construction.
    let count=uint(frame_light_count(frame));
    var covered_normalization=float3(1.);var covered_taps=0u;
    if(sampled && !policy.indirect_only) {covered_taps=S.integration_count(site.sample_index,covered_normalization);}
    let visible_count=policy.indirect_only ? 0u : sampled ? covered_taps*REG_ANALYTIC_LIGHT_SAMPLES : clustered ? cluster_lights[list_base] : count;
    var next=0u;
    Sample2DSequence source_sequence;
    for (;;) {
    if (next>=visible_count) { break; }
    let pick_index=next%REG_ANALYTIC_LIGHT_SAMPLES;
    var covered_sample=site.sample_index;var covered_world=v.world;var covered_weight=float3(0.);
    var light_weight=policy.selected_source>=0 ? 1./policy.source_probability : 1.;
    var i=policy.selected_source>=0 ? uint(policy.selected_source) : next;
    if(sampled) {
        covered_weight=S.integration_sample(next/REG_ANALYTIC_LIGHT_SAMPLES,covered_sample,covered_world)/covered_normalization;
        next++;
        if(all(covered_weight==float3(0.))) {continue;}
        let source_seed=pixel_hash(asuint(frame.display.x));
        let source_key=site.pixel^uint2(source_seed,pixel_hash(source_seed));
        // Each covered tap owns contiguous source picks. Advance before any
        // per-light rejection so rejected connections keep their RNG identity.
        if(pick_index==0u) {
            source_sequence=sample_2d_sequence(source_key,covered_sample*REG_ANALYTIC_LIGHT_SAMPLES,
                S.sample_count()*REG_ANALYTIC_LIGHT_SAMPLES,REG_SAMPLING_STREAM_ANALYTIC_SOURCE);
        }
        var u=sample_2d_next(source_sequence);
        // Complete the source rotation with the remaining hash bits. A 16-bit
        // rotation alone cannot reach every leaf in larger balanced trees.
        let rotation_bits=screen_ray_hash(source_key,0u,REG_SAMPLING_STREAM_ANALYTIC_SOURCE);
        u.x=frac(u.x+(float(rotation_bits>>16u)+.5)*(1./4294967296.));
        let pick=light_tree_pick(analytic_light_nodes,tree_nodes,covered_world,u);
        // Rejected links stay zero; never redraw or condition this probability.
        i=pick.identity;light_weight=1./(float(REG_ANALYTIC_LIGHT_SAMPLES)*pick.probability);
    } else {
        if(clustered) {i=cluster_lights[list_base+REG_CLUSTER_LIST_HEADER+next];}
        next=policy.selected_source>=0 ? visible_count : next+1u;
    }
    if (i>=count) { break; }
    let record=specialized_light<V>(lights[i]);
    var contribution_mask=REG_RAY_DIFFUSE|REG_RAY_SPECULAR;
    if (frame.shadow_links.z!=0u) {
        contribution_mask=lights[i].shadow_offsets[REG_LIGHT_CONTRIBUTION_WORD];
        if ((contribution_mask&(REG_RAY_DIFFUSE|REG_RAY_SPECULAR))==0u) {continue;}
        if (!light_receives(contribution_mask,lights[i].shadow_offsets[REG_LIGHT_LINK_WORD],objects[v.object].metadata.x)) {continue;}
    }
    contribution_mask&=policy.lobes;
    if (contribution_mask==0u) {continue;}
    if (all(record.radiance.rgb==float3(0.))) { continue; }
    // Integrate response and visibility together, including each covered
    // camera sample's origin and front-to-back transmission weight.
    var normalization=covered_normalization;var taps=1u;
    if(!sampled) {taps=S.integration_count(site.sample_index,normalization);}
    for(uint tap=0u;tap<taps;tap++) {
        var source_sample=sampled ? covered_sample : site.sample_index;var source_world=sampled ? covered_world : v.world;
        var sample_weight=covered_weight;
        if(!sampled) {sample_weight=S.integration_sample(tap,source_sample,source_world)/normalization;}
        if(all(sample_weight==float3(0.))) {continue;}
        // Source selection estimates the existing per-covered-sample sum.
        // Keep its conditional emitter coordinates, including repeated picks:
        // changing this quadrature would move another sampling dimension.
        let source_u=sample_2d(site.pixel,source_sample,S.sample_count(),REG_SAMPLING_STREAM_SECONDARY_EMITTER);
        let rotation=float2(float(pixel_hash(i)&65535u),float(pixel_hash(i^REG_SAMPLE_HASH_MIX)&65535u))/65536.;
        let u=frac(source_u+rotation);
        let source=surface_light_sample<V>(record,i,source_world,l,c,view,s.is_hair,contribution_mask,u,source_sample);
        if ((!source.delta && source.pdf<=0.) || all(source.radiance==float3(0.))) {continue;}
        let axis=source.direction;
        let weight=light_weight*sample_weight*source.radiance/select(source.delta,1.,source.pdf);
        var response=float3(0.);
        if (s.is_hair) {
            float pdf;
            response=select((contribution_mask&REG_RAY_SPECULAR)!=0u,strand_direct(s.hair0,s.hair1,s.strand,geometric,view,axis,pdf),float3(0.));
#ifdef VORK_FIBRE
            let behind=fibre_neighbours(i,v.world,record.position.xyz,v.object,v.curve_strand);
            response+=select((contribution_mask&REG_RAY_SPECULAR)!=0u,strand_neighbours(s.hair0,s.hair1,s.strand,view,axis,behind),float3(0.));
#endif
        } else {response=layered_direct<V>(l,c,direct_brdf(l.base,view,axis,contribution_mask,diffuse_albedo,reflection_scale),view,axis,contribution_mask);}
        var scattered=float3(0.);
        if (sw>0.) {scattered=direct_brdf(sl.base,view,axis,contribution_mask&REG_RAY_DIFFUSE,diffuse_albedo,reflection_scale);}
        if (any(response!=float3(0.)) || any(scattered!=float3(0.))) {
            var shadow_world=source_world+select(s.is_hair || dot(geometric,axis)<=0.,float3(0.),surface_shadow_offset(site.shadow_offset,site.facet,geometric,axis));var shadow_normal=geometric;
            if(s.is_hair) {
                let model=object_model(v.object);let inverse=transpose(object_normal(v.object));
                let offset=strand_shadow(mul(float4(source_world,1.),inverse).xyz,mul(float4(facet,0.),transpose(model)).xyz,mul(float4(axis,0.),inverse).xyz,objects[v.object].padding.x,v.curve_strand);
                shadow_world+=mul(float4(offset,0.),model).xyz;shadow_normal=axis;
            }
            if (thin && dot(geometric,axis)<0.) {shadow_normal=-geometric;}
            let connection=surface_ray_connection(shadow_world,source_world,axis,source.distance);
            var visibility=S.visibility_direction(shadow_world,shadow_normal,record,i,connection.xyz,connection.w,site.position_error,source_sample,S.sample_count(),s.is_hair);
            direct+=response*weight*visibility;scattering+=scattered*weight*visibility;
        }
        if (spatial && dot(geometric,axis)<0. && (contribution_mask&REG_RAY_DIFFUSE)!=0u) {
            let chord=S.thickness(v.object,source_world,axis);
            if (chord>=0. && chord<source.distance) {
                let boundary=source_world+axis*chord;
                var visibility=S.visibility_direction(boundary,axis,record,i,axis,source.distance-chord,site.position_error,source_sample,S.sample_count());
                direct+=subsurface_transmission(original,geometric,view,axis,thickness)*max(-dot(geometric,axis),0.)/REG_PI*l.throughput*l.scale*weight*visibility;
            }
        }
    }
    }
    if (spatial) {scattering*=s.closure.subsurface_color/max(original.subsurface_color,float3(1e-20));}
    scattering*=l.throughput*l.scale;
    var o=surface_radiance_at<V,S>(v,s,l,specular,material_visibility,policy,site.position_error,indirect_scattering,site.sample_index,site.secondary,reflection_scale);
    if (spatial) {let ratio=s.closure.subsurface_color/max(original.subsurface_color,float3(1e-20));indirect_scattering*=ratio;o.missing_scattering*=ratio;}
    scattering=scattering*exposure+indirect_scattering;
    o.color+=direct;o.direct=direct;o.radiance.direct=direct;
    return o;
}

Response surface_terms_at<V:IVariant,S:IShadows>(Varying v, Surface s, Layers l, SpecularResponse specular, MaterialUv m, out float3 scattering) {
    return surface_radiance_at<V,S>(v,s,l,specular,material_occlusion<V>(v,m),AnalyticRadiance(REG_RAY_DIFFUSE|REG_RAY_SPECULAR,true),surface_position_error(v.object,v.world),scattering);
}
Response surface_radiance_at<V:IVariant,S:IShadows>(Varying v, Surface s, Layers l, SpecularResponse specular, float material_visibility, AnalyticRadiance policy, float3 position_error, out float3 scattering,uint quadrature_sample=0xffffffffu,bool secondary=false,float reflection_scale=1.) {
    let exposure=select(policy.exposed,frame.radiance.w,1.);
    let c=s.closure;
    let geometric=s.geometric;let n=c.geometry_normal;let view=s.view;
    // The base lobes under any coat use the layered closure (roughened, IOR ratio).
    let layered_material=V.layered && (c.coat_weight>0. || c.fuzz_weight>0.);
    let roughness=specular_roughness(l.base.specular_roughness);
    var packed_roughness=roughness;
    // A constant colour is a uniform world: its cosine integral is
    // independent of the surface normal (the constant-environment case).
    // Use the same material response for a constant world as for an HDR.
    // Dielectrics reflect the world too; metals have no diffuse lobe.
    let nv=max(dot(n,view),.0001);
    let moments=brdf_moments(float2(nv,roughness));
    var reflected=reflection_scale*specular_reflectance(specular,moments,internal_reflection_quadrature<V>(c));
    // A cosine hemisphere integrates the same diffuse response used by direct
    // and path lighting. Reserve dielectric directional albedo only: the metal
    // mixture (and its compensation) belongs to glossy reflection, not to the
    // remaining dielectric substrate.
    let diffuse_albedo=diffuse_quadrature(l.base) ? specular.directional_albedo : float3(-1.);
    var diffuse=direct_diffuse(l.base,view,diffuse_albedo,reflection_scale)*REG_PI;
    let linked=light_receives(frame.environment_links.x,frame.environment_links.y,objects[v.object].metadata.x);
    let world_diffuse=linked && (frame.environment_links.x&policy.lobes&REG_RAY_DIFFUSE)!=0u;
    let glossy_enabled=policy.glossy_ambient && (policy.lobes&REG_RAY_SPECULAR)!=0u && (secondary || frame.projection.z!=REG_REFLECTIONS_OFF);
    let world_specular=linked && (frame.environment_links.x&REG_RAY_SPECULAR)!=0u && glossy_enabled;
    let world_response=select(world_diffuse,diffuse,float3(0.));
    let back_diffuse=thin_diffuse(l.base,view,diffuse_albedo,reflection_scale)*REG_PI;
    let back_response=select(world_diffuse,back_diffuse,float3(0.));
    let world_reflected=select(world_specular && !policy.indirect_only,reflected,float3(0.));
    let indirect_reflected=select(glossy_enabled && !policy.indirect_only,reflected,float3(0.));
    var missing_diffuse=false;
    // Diffusion consumes irradiance before substrate colour, just as the
    // direct-light scattering closure does; do not divide by possibly black RGB.
    var scattering_closure=l.base;scattering_closure.base_color=float3(1.);scattering_closure.subsurface_weight=0.;
    let scattering_consumer=direct_diffuse(scattering_closure,view,diffuse_albedo,reflection_scale)*REG_PI*l.throughput*l.scale*material_visibility;
    var irradiance=policy.indirect_only ? float3(0.) : frame.radiance.xyz;
    var ambient=(world_response+back_response+world_reflected)*irradiance;
    var back_ambient=back_response*irradiance;
    var reflection_environment=world_reflected*irradiance;
    if (frame.environment.x>0. && !policy.indirect_only) {
        let env_irradiance=env_diffuse.SampleLevel(env_sampler,environment_uv(n,frame.environment_rotation),0.).rgb;
        irradiance+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_irradiance;
        let radiance=env_specular.SampleLevel(env_sampler,environment_uv(reflect(-view,n),frame.environment_rotation),(policy.capture_endpoint ? 1. : roughness)*frame.environment.z).rgb;
        ambient+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*(world_response*env_irradiance+world_reflected*radiance);
        if (any(back_response>float3(0.))) {
            let back_irradiance=env_diffuse.SampleLevel(env_sampler,environment_uv(-n,frame.environment_rotation),0.).rgb;
            let received=frame.environment.x*frame.environment.y*frame.environment_color.rgb*back_response*back_irradiance;
            ambient+=received;back_ambient+=received;
        }
        reflection_environment+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*world_reflected*radiance;
    }
#ifdef VORK_PROBES
    // World permission belongs to the source, not the material response to the
    // captured local lights, emission and bounce light. Only remove the World
    // baseline this receiver actually received, including at the volume edge.
    irradiance=select(world_diffuse,irradiance,float3(0.));
    var probed=irradiance;
    let diffuse_consumer=layered_material ? l.throughput*l.scale*diffuse+c.fuzz_weight*c.fuzz_color*l.fuzz_albedo : diffuse;
    let diffusion_consumer=c.subsurface_weight>0. && any(c.subsurface_radius*c.subsurface_radius_scale>float3(0.)) && any(scattering_consumer!=float3(0.));
    if(any(diffuse_consumer!=float3(0.)) || diffusion_consumer || s.is_hair) {
        if(camera_indirect_override && !secondary) {probed=camera_indirect_irradiance;}
        else {
            let field=probe_field_sample<S>(v.world,n,view,n,1.,geometric,position_error);
            probed=field.owned ? field.radiance : (policy.indirect_only ? float3(0.) : irradiance);
            // Uncovered computed/moving receivers use the same explicit
            // connection work as invalid owned stencils. Leave no World
            // baseline behind: the connection integrates its misses too.
            // Without an installed volume there is no probe work to replace
            // the baseline, exactly as in kernels compiled without probes.
            missing_diffuse=!secondary && probe_volume_installed() && !field.valid;
            if(missing_diffuse) {probed=float3(0.);}
        }
    }
    ambient+=diffuse*(probed-irradiance);irradiance=probed;
    if (any(back_diffuse>float3(0.))) {
        var back_world=policy.indirect_only ? float3(0.) : frame.radiance.xyz;
        if (frame.environment.x>0. && !policy.indirect_only) {back_world+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_diffuse.SampleLevel(env_sampler,environment_uv(-n,frame.environment_rotation),0.).rgb;}
        back_world=select(world_diffuse,back_world,float3(0.));
        let field=probe_field_sample<S>(v.world,-n,-view,-n,1.,-geometric,position_error);
        let back_probed=field.owned ? field.radiance : back_world;
        let received=back_diffuse*(back_probed-back_world);
        ambient+=received;back_ambient+=received;
    }
    if(probe_grid().field.x!=0u && glossy_enabled && !policy.indirect_only && any(reflected!=float3(0.))) {
        let mirror=reflect(-view,n);
        let provider=receiver_provider(v.world,n,view,mirror,policy.capture_endpoint ? 1. : roughness,frame,env_specular,env_sampler,policy.capture_endpoint,world_specular);
        if(provider.local) {
            var world_reflection=world_specular ? reflection_fallback(mirror,policy.capture_endpoint ? 1. : roughness,frame,env_specular,env_sampler) : float3(0.);
            let reflection_delta=indirect_reflected*provider.radiance-world_reflected*world_reflection;
            ambient+=reflection_delta;reflection_environment+=reflection_delta;
        }
    }
#endif
    if (reflection_quadrature<V>(c) && glossy_enabled && !policy.indirect_only && receiver_reflection_active(frame,world_specular)) {
        var integrated=float3(0.);
        let samples=reflection_quadrature_samples<V>(l.base);
        for (uint j=0u;j<quadrature_rotations(quadrature_sample);j++) {
            let rotation=quadrature_rotation(j,quadrature_sample,S.sample_count());
            for (uint i=0u;i<samples;i++) {
                float3 direction;
                let weight=specular_quadrature_sample(l.base,view,specular_quadrature_point(i,rotation,samples),direction,specular.metal_compensation);
                integrated+=reflection_scale*weight*receiver_reflection(v.world,n,view,direction,policy.capture_endpoint ? 1. : 0.,frame,env_specular,env_sampler,policy.capture_endpoint,world_specular);
            }
        }
        let original_world=receiver_reflection(v.world,n,view,reflect(-view,n),policy.capture_endpoint ? 1. : roughness,frame,env_specular,env_sampler,policy.capture_endpoint,world_specular);
        let reflection_delta=integrated/float(samples*quadrature_rotations(quadrature_sample))-indirect_reflected*original_world;
        ambient+=reflection_delta;reflection_environment+=reflection_delta;
    }
    if (layered_material) {
        // Coat reflection and fuzz under the same constant or HDR world.
        var coat_reflected=0.;var coat_ambient=float3(0.);
        if (V.has_coat && c.coat_weight>0. && policy.glossy_ambient) {
            let cn=c.geometry_coat_normal;let coat_roughness=specular_roughness(c.coat_roughness);
            let coat_moments=brdf_moments(float2(max(dot(cn,view),.0001),coat_roughness)).rg;
            coat_reflected=l.throughput*c.coat_weight*(l.coat_f0*coat_moments.x+coat_moments.y);
            var received=world_specular;
            var world=frame.radiance.xyz;
            if (frame.environment.x>0.) { world+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_specular.SampleLevel(env_sampler,environment_uv(reflect(-view,cn),frame.environment_rotation),(policy.capture_endpoint ? 1. : coat_roughness)*frame.environment.z).rgb; }
#ifdef VORK_PROBES
            if(probe_grid().field.x!=0u) {
                let provider=receiver_provider(v.world,cn,view,reflect(-view,cn),policy.capture_endpoint ? 1. : coat_roughness,frame,env_specular,env_sampler,policy.capture_endpoint,world_specular);
                if(provider.local) {world=provider.radiance;received=glossy_enabled && !policy.indirect_only;}
            }
#endif
            coat_ambient=select(received,coat_reflected*world,float3(0.));
            if(V.has_coat_anisotropy && c.coat_roughness_anisotropy>0. && glossy_enabled && !policy.indirect_only) {
                let coat=transport_coat_closure(c);
                var integrated=float3(0.);let samples=reflection_quadrature_samples<V>(coat);
                for(uint j=0u;j<quadrature_rotations(quadrature_sample);j++) {
                    let rotation=quadrature_rotation(j,quadrature_sample,S.sample_count());
                    for(uint i=0u;i<samples;i++) {
                        float3 direction;
                        let weight=specular_quadrature_sample(coat,view,specular_quadrature_point(i,rotation,samples),direction,float3(0.));
                        integrated+=weight*receiver_reflection(v.world,cn,view,direction,policy.capture_endpoint ? 1. : 0.,frame,env_specular,env_sampler,policy.capture_endpoint,world_specular);
                    }
                }
                coat_ambient=l.throughput*c.coat_weight*integrated/float(samples*quadrature_rotations(quadrature_sample));
                coat_reflected=l.throughput*c.coat_weight*coat_directional_albedo<V>(c,view);
            }
        }
        let fuzz=c.fuzz_weight*c.fuzz_color*l.fuzz_albedo;
        var world_irradiance=frame.radiance.xyz;
        if (frame.environment.x>0.) { world_irradiance+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_diffuse.SampleLevel(env_sampler,environment_uv(n,frame.environment_rotation),0.).rgb; }
#ifdef VORK_PROBES
        world_irradiance=irradiance;
#endif
#ifdef VORK_PROBES
        ambient=l.throughput*l.scale*ambient+coat_ambient+fuzz*world_irradiance;
#else
        ambient=l.throughput*l.scale*ambient+coat_ambient+select(world_diffuse,fuzz*world_irradiance,float3(0.));
#endif
        back_ambient*=l.throughput*l.scale;
        reflection_environment=l.throughput*l.scale*reflection_environment+coat_ambient;
        // The packed roughness guides reconstruction and eligibility only.
        // Tracing samples both lobes; its guide must not switch at a dominant-lobe threshold.
        let base_reflected=l.throughput*l.scale*reflected;
        let base_strength=max(base_reflected.r,max(base_reflected.g,base_reflected.b));
        packed_roughness=(base_strength*roughness+coat_reflected*specular_roughness(c.coat_roughness))/max(base_strength+coat_reflected,REG_POSITIVE_EPSILON);
        reflected=base_reflected+float3(coat_reflected);
        diffuse=l.throughput*l.scale*diffuse+fuzz;
    }
    if (s.is_hair) {
        var world=frame.radiance.xyz;
        if (frame.environment.x>0.) {world+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_diffuse.SampleLevel(env_sampler,environment_uv(n,frame.environment_rotation),0.).rgb;}
#if defined(VORK_PROBES)
        world=irradiance;
#endif
        ambient=select(world_specular,strand_world(s.hair0,s.hair1,s.strand,geometric,view,world),float3(0.));
        reflection_environment=ambient;reflected=float3(0.);diffuse=float3(0.);packed_roughness=1.;
    }
    // glTF occlusion is linear red-channel visibility, applied to indirect light only.
    ambient *= material_visibility;back_ambient*=material_visibility;
    var diffuse_received=world_diffuse;
#ifdef VORK_PROBES
    diffuse_received=true;
#endif
    scattering=select(diffuse_received,scattering_consumer*irradiance*exposure,float3(0.));
    var emissive=policy.indirect_only ? float3(0.) : surface_emission(c,l);
    Response r;r.reflection=reflection_bsdf(l.base,c,l.scale*reflection_scale,l.throughput,material_visibility,specular.metal_compensation);reflection_bsdf_set_environment(r.reflection,reflection_environment*material_visibility*exposure);r.radiance=RadianceTerms(float3(0.),emissive*exposure,(ambient-reflection_environment*material_visibility)*exposure,reflection_environment*material_visibility*exposure);
    // Front-hemisphere GI can replace only its own ambient estimate. The
    // sheet's opposite environment/probe response remains represented.
    r.baseline=ReplaceableBaselines(r.radiance.diffuse_indirect-back_ambient*exposure,r.radiance.glossy_indirect);
    r.color=r.radiance.total();r.reflected=reflected*material_visibility;
    r.diffuse=diffuse*material_visibility;r.missing_diffuse=missing_diffuse ? r.diffuse : float3(0.);r.missing_scattering=missing_diffuse ? scattering_consumer : float3(0.);r.missing_lit=false;r.roughness=packed_roughness;r.normal=n;r.direct=float3(0.);
    return r;
}
float reflects(Response r) {return max(r.reflected.r,max(r.reflected.g,r.reflected.b));}
Response weighted(Response r, float weight) {var o=r;o.color*=weight;o.reflected*=weight;o.reflection.data[8].xyz*=weight;reflection_bsdf_set_environment(o.reflection,reflection_bsdf_environment(r.reflection)*weight);o.diffuse*=weight;o.missing_diffuse*=weight;o.missing_scattering*=weight;o.direct*=weight;
    o.radiance.direct*=weight;o.radiance.emission*=weight;o.radiance.diffuse_indirect*=weight;o.radiance.glossy_indirect*=weight;
    o.baseline.diffuse*=weight;o.baseline.glossy*=weight;return o;}

// Blender's Shader to RGB, shared by raster fragments and ray hits: a material's
// lit stage receives its surface's shared evaluation `r` (all lobes summed) and
// returns the closure shaded in its place; emission-only results skip the lights.
// The result is final: screen passes replace none of its terms.
Response lit_terms<V:IVariant, S:IShadows, P:IProgram,let decode_table:bool=true>(Varying v, Surface s, Coordinates k, LightSite site, Response r) {
    let lit=surface_response<V>(P.apply_lit(v,s,k,SurfaceLighting(r.color,s.closure.geometry_opacity,r.direct)));
    var o=r;
    if (P.lit_reflects) {
        // The nonlinear hook has consumed its complete input. Its returned
        // physical closure can use the ordinary unbiased source estimator.
        site.exact_sources=false;
        float3 scattering;float3 indirect_scattering;
        o=shade_terms_at<V,S>(v,lit,site,scattering,indirect_scattering,false,decode_table);
    } else {
        o.color=lit.closure.emission_color*lit.closure.emission_luminance*frame.radiance.w;o.direct=float3(0.);o.radiance=RadianceTerms(float3(0.),o.color,float3(0.),float3(0.));
    }
    let pending=any(r.missing_diffuse!=float3(0.)) || any(r.missing_scattering!=float3(0.)) || any(o.missing_diffuse!=float3(0.)) || any(o.missing_scattering!=float3(0.));
    var result=unreplaced(o,r);result.missing_lit=pending;return result;
}
// A lit result's control payloads: the surface's normal, nothing for screen passes to replace.
Response unreplaced(Response o, Response surface) {
    var r=o;r.baseline=ReplaceableBaselines(float3(0.),float3(0.));r.reflected=float3(0.);r.diffuse=float3(0.);r.missing_diffuse=float3(0.);r.missing_scattering=float3(0.);
    r.normal=surface.normal;r.roughness=surface.roughness;
    return r;
}
// A surface's subsurface diffusion inputs (subsurface.slang filters them): its
// scattered light and the ambient part of it, diffusion radius and albedo.

// Christensen/Burley, diffuse transmission MFP fit (Pixar technical memo 15-04);
// forward scattering increases transport distance (transport similarity).
float3 diffusion_radius(Closure c) {
    let a=c.subsurface_color;
    let shape=1.9-a+3.5*(a-.8)*(a-.8);
    let radius=c.subsurface_radius*c.subsurface_radius_scale/(shape*max(1.-c.subsurface_scatter_anisotropy,.001));
    return select(c.subsurface_weight>0. && !c.geometry_thin_walled,radius,float3(0.));
}
// A hit surface's lighting over every lobe of a mixed surface (Blend
// programs), each weighted, then its program's lit stage: the decoupled shade.
// `first` is lobe 0, as hit reconstruction returns it.
// `d` gathers the lobes' diffusion: light, ambient and albedo weighted by each
// lobe, the radius averaged over their subsurface weight; a lit result is final.
Response surface_radiance<V:IVariant,S:IShadows,P:IProgram,let decode_table:bool=true>(Varying v, bool entering, Coordinates k, Surface first, LightSite site, out Diffusion d,bool interior_transport=false) {
    site.exact_sources=P.lit && P.lit_material(v);
    Response r=(Response)0;d=(Diffusion)0;
    var best=0.;var radius=float3(0.);var total=0.;
    // One lighting call site, including the first lobe. Table materials have a
    // constant trip count of one; authored mixtures use the same loop as forward.
    for(uint i=0u;i<(P.lobes ? max(first.lobes,1u) : 1u);i++) {
        var lobe=first;
        if(i>0u) {lobe=surface_at<V,P,decode_table>(v,entering,first.view,k,Lobe(i,0.));}
        float3 scattering;float3 indirect_scattering;
        let response=shade_terms_at<V,S>(v,lobe,site,scattering,indirect_scattering,interior_transport,decode_table);
        let lobe_radius=diffusion_radius(lobe.closure);
        let lobe_share=select(any(lobe_radius>0.),lobe.closure.subsurface_weight,0.);
        if(i==0u) {
            r=response;
            let albedo=lobe.closure.subsurface_color*lobe_share;
            d=Diffusion(scattering,indirect_scattering,lobe_radius,albedo,scattering*albedo,indirect_scattering*albedo,response.missing_scattering,response.missing_scattering*albedo);
            if(P.lobes) {
                r=weighted(r,lobe.weight);best=reflects(r);
                let share=lobe.weight*lobe_share;
                radius=d.radius*share;total=share;
                d.light*=share;d.ambient*=share;d.missing*=share;d.missing_albedo*=lobe.weight;d.albedo*=lobe.weight;d.light_albedo*=lobe.weight;d.ambient_albedo*=lobe.weight;
            }
        } else {
            let terms=weighted(response,lobe.weight);
            r.color+=terms.color;r.missing_diffuse+=terms.missing_diffuse;r.missing_scattering+=terms.missing_scattering;
            r.radiance.direct+=terms.radiance.direct;r.radiance.emission+=terms.radiance.emission;r.radiance.diffuse_indirect+=terms.radiance.diffuse_indirect;r.radiance.glossy_indirect+=terms.radiance.glossy_indirect;
            r.baseline.diffuse+=terms.baseline.diffuse;r.baseline.glossy+=terms.baseline.glossy;r.reflected+=terms.reflected;r.diffuse+=terms.diffuse;r.direct+=terms.direct;
            // Keep the retained BSDF and its weight with the selected reflective controls.
            if(reflects(terms)>best) {best=reflects(terms);r.roughness=terms.roughness;r.normal=terms.normal;r.reflection=terms.reflection;}
            let share=select(any(lobe_radius>0.),lobe.weight*lobe.closure.subsurface_weight,0.);
            let albedo=lobe.closure.subsurface_color*share;
            d.light_albedo+=scattering*albedo;d.ambient_albedo+=indirect_scattering*albedo;d.missing_albedo+=response.missing_scattering*albedo;
            d.light+=scattering*share;d.ambient+=indirect_scattering*share;d.missing+=response.missing_scattering*share;d.albedo+=albedo;
            radius+=lobe_radius*share;total+=share;
        }
    }
    if(P.lobes) {
        d.radius=select(total>0.,radius/max(total,1e-30),float3(0.));
        d.light/=max(total,1e-30);d.ambient/=max(total,1e-30);d.missing/=max(total,1e-30);
    }
    if (P.lit && P.lit_material(v)) {let emitted=r.radiance.emission;r.color-=emitted;r.radiance.emission=float3(0.);r=lit_terms<V,S,P,decode_table>(v,first,k,site,r);r.color+=emitted;r.radiance.emission+=emitted;d.radius=float3(0.);d.albedo=float3(0.);d.missing=float3(0.);d.missing_albedo=float3(0.);}
    return r;
}
// A ray hit's lit stage sees every lobe before replacing its chosen surface.
Surface lit_surface<V:IVariant,S:IShadows, P:IProgram>(Varying v, bool entering, Coordinates k, Surface picked, LightSite site) {
    site.exact_sources=true;
    var first=picked;
    if (P.lobes) {first=surface_at<V,P>(v,entering,picked.view,k,Lobe(0u,0.));}
    first.closure.emission_luminance=0.;
    float3 scattering;float3 indirect_scattering;
    Response r={};
    // One call site for every lobe, including the first.
    for (uint i=0u;i<max(first.lobes,1u);i++) {
        var lobe=first;
        if (i>0u) {lobe=surface_at<V,P>(v,entering,picked.view,k,Lobe(i,0.));}
        lobe.closure.emission_luminance=0.;
        let terms=weighted(shade_terms_at<V,S>(v,lobe,site,scattering,indirect_scattering),lobe.weight);
        if(i==0u) {r=terms;}
        else {
            r.color+=terms.color;r.missing_diffuse+=terms.missing_diffuse;r.missing_scattering+=terms.missing_scattering;r.direct+=terms.direct;
            r.radiance.direct+=terms.radiance.direct;r.radiance.emission+=terms.radiance.emission;
            r.radiance.diffuse_indirect+=terms.radiance.diffuse_indirect;r.radiance.glossy_indirect+=terms.radiance.glossy_indirect;
            r.baseline.diffuse+=terms.baseline.diffuse;r.baseline.glossy+=terms.baseline.glossy;
        }
    }
    return surface_response<V>(P.apply_lit(v,first,k,SurfaceLighting(r.color,first.closure.geometry_opacity,r.direct)));
}
// A lighting site: its own footprint, its sample (~0: the exposure sample), whether it lights a
// secondary (ray) hit, and optional conservative camera clusters for reconstructed camera slots.
LightSite hit_site(Varying v, Coordinates k, float3 face, uint2 seed, uint sample_index=0xffffffffu, bool secondary=false, bool camera_clusters=false) {return LightSite(material_uv(v,k.uv_dx,k.uv_dy,k.uv1_dx,k.uv1_dy,k.material_filter),camera_clusters,secondary,face,seed,surface_position_error(v.object,v.world),sample_index,k.shadow_offset,false);}
