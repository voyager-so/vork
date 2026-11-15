SPRINGS
Bake motion in authoring code; the renderer reads ordinary animation tracks.
For FPS p/q, frame f is time f*q/p seconds. Write that time as a rational string
(e.g. "1001/30000"), not a rounded decimal. Values remain ordinary numbers.
Evaluate the easing or spring at each frame, map its result to the property's
units, then write [time,value] keys. LINEAR joins samples; STEP holds each sample.
Use LINEAR for motion between samples, including shutter samples. A bake matches
its source at keyed instants only; increase sampling density for curved motion
between frames, and rebake when changing FPS or motion parameters.

Springs can overshoot. Keep the overshoot for translation/scale; clamp to the
property's range before writing bounded values such as opacity. Choose an
explicit end time: tracks hold their last value, not the spring's eventual rest.
CUBICSPLINE tangents are per second; do not pass per-frame derivatives unchanged.

Example: examples/animation-recipes/spring.py bakes an initially resting,
underdamped unit-step spring and smoothstep easing, using Python's math module.
For another library's exact motion, evaluate that project's spring/easing with
its own version, FPS and parameters, then serialize those samples using the same
key convention. The analytic example does not promise that library's motion.
Generation and render commands: examples/animation-recipes/README.md.
Track semantics: vork help animation. Frame windows: vork help sequences.
