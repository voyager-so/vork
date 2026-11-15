Setup and render progress goes to stderr: a plain line every 5 seconds in terminals and redirected logs. Redirect both streams (`>run.log 2>&1`) for a complete log. Counts cover completed configurations/frames and compiler calls discovered so far; elapsed time is not an ETA.
TIMINGS: every written render writes NAME.timings.json (adapter, cache, GPU, encoder timings)
with `stages`: where the wall time went (setup, Slang front end and code generation, driver
pipelines, first frame, GPU time per pass, denoise, encode).
DIAGNOSTICS: --diagnostics adds memory snapshots and non-finite pixel counts; with it,
--no-write (no files) and --gpu-only (no readback) benchmark. It also writes each Slang
compile, driver pipeline and 3D shader module to stderr as it begins and ends, one
`creation: {json}` line each (event, kind, label, and ms on end), so a render stopped before
it finishes still shows where its setup time went and what was in flight. Not announced:
the 2D canvas engine's own shader modules (its pipelines are) and denoiser setup.
Environment: VORK_CACHE_DIR, VORK_CACHE_BYTES (default 16 GiB; 0 = no disk cache), VORK_SLANGC,
VORK_OIDN, VORK_BIN (SDK executable), VORK_GPU (hardware GPU index or cpu, as --gpu; CPU rendering is explicit, see limits).

Diagnostics JSON uses version 3 and numeric evaluation seconds for frame times;
output FPS remains number/rational string. See vork help outputs for completion
metadata and overwrite ownership.
