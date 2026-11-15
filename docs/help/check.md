CHECK
vork check SCENE|- [--profile P] [--shaders]
Validate without opening a GPU driver or rendering frames. The default profile is
std/final; select the same profile as the intended render. File-relative assets
resolve beside SCENE; stdin assets resolve from the current directory.

Check collects independent errors at each stage; untyped scenes stop after expansion.
Missing sampled images retain their magenta substitution. File-backed resources
must be readable regular files; decoding can fail on later sequence frames.

Runs the render loader and installation: JSON decoding and schema constraints,
std expansion, program resolution and input validation, feature and fog admission,
asset loading, animation binding and timeline/shutter sanity. Program reflection and effect proofs need the same Slang toolchain as rendering.
Compilation caches are shared. Default checks omit native backend shader generation.
--shaders also compiles portable shaders; it stays CPU-only but can take minutes.

Success prints `Scene checked successfully` and exits 0. Errors use render's
messages and exit categories (see errors). A successful check does not guarantee
that a particular GPU has enough memory or supports the required device features,
or validate encoder setup or effects that depend on executing a frame.
