// Environment/screen transport through a transparent surface:
// the solid's interior (analytic thickness or mesh exits) and what it refracts
// (the world or an opaque image). The decoupled shade's transparent slots
// (decoupled.slang) supply the resources through ITransmission.
import vork_constants;
__exported import hit_geometry;
import openpbr_core;
import dielectric_interface;
import frame;
import closure;
import material;
import stream;
import surface;
import bsdf;
import variant;

// What a transparent surface refracts: the opaque image it sees through
// (its size, each pixel's receiver normal and depth, negative for background,
// its colour pyramid with opaque coverage in alpha, and the camera background
// at a screen uv), its solid's exit along a ray, and the exposure's exhaustion
// word. Hardware refraction uses the shared dielectric interface walk instead.
interface ITransmission {
    static uint2 size();
    static float4 receiver(int2 pixel);
    static float4 color(float2 uv, float lod);
    static float3 background(float2 uv);
    static SolidExit solid_exit(uint object, float3 origin, float3 direction,float eta);
    // A solid ray left `object` before escaping it.
    static void exhausted(uint object);
}


// Screen uv (0..1, y down) and NDC depth to world.
float3 unproject(float2 uv, float z) {
    let h=mul(float4(uv*float2(2.,-2.)+float2(-1.,1.),z,1.),frame.inverse_vp);
    return h.xyz/h.w;
}
float2 opaque_sample<O:ITransmission>(int2 pixel) {
    let receiver=O.receiver(pixel);
    return float2(select(receiver.w>=0.,receiver.w,1.),select(receiver.w>=0.,1.,0.));
}
// The resolved centre/covered-centroid point; w is one for a receiver, zero
// for background, keeping the normal and depth on the same receiver.
float4 opaque_point<O:ITransmission>(float2 pixel) {
    let size=int2(O.size());
    let sample=opaque_sample<O>(clamp(int2(pixel),int2(0),size-int2(1)));
    return float4(unproject((floor(pixel)+.5)/float2(size),sample.x),sample.y);
}
// A screen hit must be a surface crossed after the solid exit. Opaque foreground
// pixels (labels/caps) cannot serve as a refracted background merely by projecting there.
struct TransmissionHit { float2 uv; bool found; };
TransmissionHit transmission_hit<O:ITransmission>(float3 origin, float3 direction) {
    let missed=TransmissionHit(float2(0.),false);
    float3 start,end;if (!screen_ray_segment(origin,direction,frame,start,end)) {return missed;}
    let delta=end-start;let begin=0.;let finish=1.;
    let size=float2(O.size());
    let length_squared=dot(delta,delta);
    if (length_squared==0.) {return missed;}
    // Amanatides-Woo in projected pixel coordinates: visit every crossed cell,
    // with complete traversal by default. Integer cell progression also handles rounded ties
    // without an arbitrary epsilon or a float cursor that can stop advancing.
    var cell=clamp(int2(floor((start+delta*begin).xy*size)),int2(0),int2(size)-1);
    let step=int2(select(delta.xy>float2(0.),float2(1.),float2(-1.)));
    var cursor=begin;
    while (cursor<finish && all(cell>=int2(0)) && all(cell<int2(size))) {
        let receiver_pixel=cell;
        let edge=float2(cell)+select(delta.xy>float2(0.),float2(1.),float2(0.));
        let crossing=select(delta.xy!=float2(0.),(edge-start.xy*size)/(delta.xy*size),float2(1.));
        let hi=min(finish,max(cursor,min(crossing.x,crossing.y)));
        cursor=hi;
        if (crossing.x<=hi) {cell.x+=step.x;}
        if (crossing.y<=hi) {cell.y+=step.y;}
        let receiver=O.receiver(receiver_pixel);
        let receiver_sample=float2(receiver.w,select(receiver.w>=0.,1.,0.));
        if (receiver_sample.y==0.) {continue;}
        var actual=float4(unproject((float2(receiver_pixel)+.5)/size,receiver_sample.x),receiver_sample.y);
        let receiver_normal=receiver.xyz;
        let denominator=dot(direction,receiver_normal);
        if (denominator==0.) {continue;}
        let ray_distance=dot(actual.xyz-origin,receiver_normal)/denominator;
        if (ray_distance<=0.) {continue;}
        let refined=mul(float4(origin+direction*ray_distance,1.),frame.vp);
        if (refined.w<=0.) {continue;}
        let at=float3(refined.xy/refined.w*float2(.5,-.5)+.5,refined.z/refined.w);
        if (any(at<=float3(0.)) || any(at>=float3(1.))) {continue;}
        // Bound the conditioning of ray/receiver-plane intersection. At grazing
        // incidence depth roundoff can move the refined UV across a pixel edge.
        let sample_depth=receiver_sample.x;
        let point_error=screen_world_error((float2(receiver_pixel)+.5)/size,sample_depth,frame.inverse_vp);
        let distance_error=(dot(abs(receiver_normal),point_error)
            +REG_FP32_GAMMA7*dot(abs(receiver_normal),abs(actual.xyz)+abs(origin)))/abs(denominator);
        let ray_error=abs(direction)*distance_error+REG_FP32_GAMMA7*(abs(origin)+abs(direction*ray_distance));
        let pixel_error=screen_projection_error(origin+direction*ray_distance,ray_error,frame.vp)*size;
        if (any(at.xy*size<float2(receiver_pixel)-pixel_error)
            || any(at.xy*size>float2(receiver_pixel)+1.+pixel_error)) {continue;}
        actual=opaque_point<O>(at.xy*size);
        let expected=unproject((floor(at.xy*size)+.5)/size,at.z);
        let pixel_width=distance(expected,unproject((floor(at.xy*size)+float2(1.5,.5))/size,at.z));
        let pixel=floor(at.xy*size)+.5;
        let neighbour_x=opaque_point<O>(pixel+float2(1.,0.));let neighbour_y=opaque_point<O>(pixel+float2(0.,1.));
        var footprint=pixel_width;
        if (neighbour_x.w>0.) {footprint=max(footprint,distance(actual.xyz,neighbour_x.xyz));}
        if (neighbour_y.w>0.) {footprint=max(footprint,distance(actual.xyz,neighbour_y.xyz));}
        // Depth is measured at pixel centres. A sloping receiver covers a world-space
        // footprint along its own surface, not just the camera's image plane.
        let tolerance=max(footprint*2.,REG_TRANSMISSION_DEPTH_TOLERANCE*max(1.,distance(expected,frame.eye.xyz)));
        if (actual.w>0. && dot(actual.xyz-origin,direction)>0.
            && distance(actual.xyz,expected)<=tolerance) {
            return TransmissionHit(at.xy,true);
        }
    }
    return missed;
}
float3 transmission_radiance<V:IVariant,O:ITransmission,P:IProgram>(Varying v,Surface s,float3 facet) {
    let c=s.closure;
    var color=float3(0.);
    if (V.has_transmission && c.transmission_weight>0.) {
        let interface_closure=transmission_interface<V>(s,facet).closure;
        let n=interface_closure.geometry_normal;
        let view=s.view;
        let eta=dielectric_ior(c);
        // A visible rear face is the current exit (an open mesh or a camera
        // already inside), not another entry followed by a coincident exit.
        let entering=c.geometry_thin_walled || s.front_facing;
        // Thin sheets have no interior. Solids intersect their raster mesh in local space,
        // using the same per-frame transform and deformed positions as the camera pass.
        let into=refract(-view,n,1./dielectric_ior(interface_closure));
        let rear_tir=!entering && dot(into,into)==0.;
        let interface_fresnel=rear_tir ? float3(0.) : interface_reflection<V>(interface_closure,view);
        var chord=0.;var exit=v.world;var out_dir=entering ? -view : into;
        var escaped=c.geometry_thin_walled || (!entering && dot(into,into)>0.);
        var exit_transmission=float3(1.);
        var exit_tint=float3(1.);
        if (V.has_solid_transmission && !c.geometry_thin_walled && ((entering && dot(into,into)>0.) || rear_tir)) {
            var inside=rear_tir ? reflect(-view,n) : into;
            {
            // Entry consumes one boundary event. Reserve subsequent events for
            // each internal reflection and the exit, like the authored path budget.
            for (uint bounce=1u;frame.environment_links.z>0u;bounce++) {
                let boundary=O.solid_exit(v.object,exit,inside,eta);
                // An open volume boundary has no exit interface. Continue its interior
                // ray into the scene; do not invent an exit normal, thickness, or black wall.
                if (!boundary.found) {out_dir=inside;escaped=true;break;}
                if (bounce>=frame.environment_links.z) {break;}
                chord+=boundary.distance;exit+=inside*boundary.distance;
                out_dir=refract(inside,boundary.normal,eta);
                if (dot(out_dir,out_dir)>0.) {
                    exit_transmission=1.-interface_reflection<V>(exiting_closure(c,boundary.normal),-inside);
                    // Tint belongs only to a transmitted interface.
                    // Internal reflection and an absent exit do not cross one.
                    exit_tint=surface_transmission_tint(c);
                    escaped=true;break;
                }
                // Total internal reflection stays inside the mesh and continues to its
                // next boundary. It never samples the unrelated camera pixel.
                inside=reflect(inside,boundary.normal);
            }
            if (!escaped) {O.exhausted(v.object);}
            }
        }
        let size=float2(O.size());
        // Beer-Lambert absorption over transmission_depth (metres; glTF attenuation distance).
        // OpenPBR transmission_depth 0 is a surface tint; otherwise Beer-Lambert over the path.
        let entry_tint=rear_tir ? float3(1.) : surface_transmission_tint(c);
        let segment=DielectricSegment(c.transmission_color,c.transmission_depth,dielectric_ior(c),dielectric_ior(c));
        let attenuation=entry_tint*exit_tint*segment_attenuation(segment,chord);
        let roughness=c.specular_roughness;
        let world_allowed=light_receives(frame.environment_links.x,frame.environment_links.y,objects[v.object].metadata.x)
            && (frame.environment_links.x&REG_RAY_TRANSMISSION)!=0u;
        var transmitted_scene=float3(0.);
        var found=false;
        if ((frame.environment_links.w&1u)==0u) {
            var hit=TransmissionHit(v.position.xy/size,c.geometry_thin_walled);
            if (V.has_solid_transmission && !c.geometry_thin_walled && escaped) {hit=transmission_hit<O>(exit,out_dir);}
            found=hit.found;
            if (found) {
                let lod=log2(1.+roughness*roughness*REG_TRANSMISSION_BLUR_DIAMETER);
                // The capture includes the camera background over its uncovered part (1-alpha).
                // Glass that does not receive the world removes it, estimated at the hit's centre.
                let captured=O.color(hit.uv,lod);
                transmitted_scene=captured.rgb;
                if (!world_allowed && captured.a<1.) {transmitted_scene=max(captured.rgb-O.background(hit.uv)*(1.-captured.a),float3(0.));}
            }
        }
        if (!found && world_allowed && escaped) {
            transmitted_scene=frame.radiance.xyz;
            if (frame.environment.x>0.) {
                transmitted_scene+=frame.environment.x*frame.environment.y*frame.environment_color.rgb*env_specular.SampleLevel(env_sampler,environment_uv(out_dir,frame.environment_rotation),roughness*frame.environment.z).rgb;
            }
            transmitted_scene*=frame.radiance.w;
        }
        // Exit-interface transmission belongs to the escaped scene radiance;
        // interior sources are integrated before that boundary.
        var behind=transmitted_scene*attenuation*exit_transmission;
        // Light from behind passes the base substrate, so the coat and fuzz attenuate it too.
        if (V.layered && (c.coat_weight>0. || c.fuzz_weight>0.)) { let l=layers<V>(interface_closure,view);behind*=l.throughput*l.scale; }
        let transmission=c.transmission_weight*(1.-interface_fresnel);
        color=behind*transmission;
    }
    return color;
}
