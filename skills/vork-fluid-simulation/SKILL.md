---
name: vork-fluid-simulation
description: Liquid simulation in vork using libraries/fluid - APIC/FLIP solver, implicit liquid surface, wetting film, whitewater, Slang callbacks (container, initial field, boundary velocity), budgets and timestep rules. Use when adding water, splashes or other liquids to a vork scene.
---

# Fluid simulation

Advanced. Read `vork help fluid --all` and `examples/water/` (README, `bowl.slang`, `make_scene.py`, `prepare_snapshot.py`) before authoring.

## What it is

GPU APIC/FLIP liquid implemented as Slang compute programs, plus Yu-Turk-style surface reconstruction, a conservative wetting film and a one-way whitewater population. Authoring is `libraries.fluid.build` (Python, **source checkout**), which returns ordinary scene maps (buffers, geometry, programs). Materials shade the resulting surfaces. There is no fluid scheduler or kernel primitive in the renderer.

GPU modules: `#include "fluid/config.slang"`, `"fluid/solver.slang"`, `"fluid/surface.slang"`, `"fluid/whitewater.slang"`. Declare `struct Inputs {int stage=0;};` in your wrapper before including them.

## Controls

Physical: `extent` (domain half-size, m), `scale` (m per solver unit), `gravity`, `surface_tension` (N/m), `density` (kg/m^3), `particle_budget`, `spacing`, `whitewater_budget`. Numerics: `surface_grid`, bins, XZ `film_grid`, `dt`, `speed` (max expected m/s, for timestep validation), `pressure_iterations`, `flip` (0 APIC .. 1 FLIP), `seed`. Defaults: terrestrial gravity, room-temperature water, APIC. Defaults live in `build`'s signature.

**No automatic CFL adaptation.** You must bound velocities and refine `dt`/grid yourself.

## Callbacks (Slang source, solver units)

`float container(float3 p)` (positive solid, negative free; signed distance) - `float initial_field(float3 p)` (includes solid clipping) - `bool initial_marker(uint id, out float3 p)` - `float3 initial_velocity(...)` - `float initial_pressure(...)` - `bool film_accept(float2 xz)` / `float film_height(float2 xz)` - `float3 boundary_velocity(float3 p, float time, float3 v)`. Initial bulk mass = accepted marker slots x spacing^3. Changing inflow mass needs your own mass-accounting/recycling pass.

## Outputs to render

- `surface`: implicit scalar grid in local [-1,1]^3; node translation = origin, scale = extent (m). Shade with an optical water material (transmission; use `std/path`).
- `wetness`, `floor_film`/`film_surface`: closed implicit height surface on a flat opaque floor; floor and film must share the authored height.
- `white_draw_position/white_width`: compact points; node scale = scale; `white_count` is the active prefix; width is not compute-bound.

## Cost and workflow

- All mutable buffers are retained simulation state; seeking replays from checkpoints, so **motion blur multiplies cost** and a later `--at` takes longer.
- Resolution does not shrink the simulation. Start with the closed-shutter diagnostic: `examples/water/prepare_snapshot.py` then `vork render snapshot.json --profile std/path --at 0 --out water.png`, then increase `--at`.
- The example is authored-impulse, not a resolved rigid-body impact. Do not promise impact physics.
- Check VRAM per-buffer binding limits on large particle counts; follow the error's advice or reduce budgets.
