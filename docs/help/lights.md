LIGHTS
Area lights emit along local -Z; use target to aim them at the subject.
Aiming a finite area light rotates its whole emitting surface. Check its transformed
extent against walls and glazing; its centre being outside does not ensure clearance.
Inspect direct and reflected source visibility before attributing bright shapes to transport.
Use one environment light for sky or HDR lighting; RGB render.background is only a backdrop.
Without an environment light the world is dim grey (0.05).
std/environment_image takes an Image input. vork std environment preprocesses HDR
lighting into an environment and a sun; vork std sky provides a sky and sun.
For `vork std environment`, fps is a positive number or rational string. Use a separate output directory/stem: generated EXR frames must
not alias input frames, including through symlinks. Invalid rates and output/source
collisions fail before frame writes. SDK helper fps options remain numeric.
Sun angle is the full angular diameter in radians; 0 gives sharp shadows.
Field reference: vork help field light.angle.

Light energy is watts for local lights, W/m² for sun, and a multiplier for environment.
OpenPBR emission_luminance is cd/m², not light watts; emission_color is linear RGB.
AgX exposure is stops (radiance multiplied by 2^exposure), default 0; there is no auto-exposure.
The stdout exposure summary and meta.json report clipped, near-black and mean values.
<!-- all -->
LIGHTING REFERENCE
std/sky sun direction: [cos(el)*sin(rot),sin(el),-cos(el)*cos(rot)] (radians;
rot 0 = -Z, pi/2 = +X, matching Blender).
Use illuminates and shadowed_by for light linking; use shadowed_by: [] for a light without shadows.
Use std/path for refracted caustics. Final's glass shadows attenuate along straight
connections; roughness does not spread them with distance.
