# vork agent skills

Skills that teach a coding agent (Claude Code, Codex, any agent that reads `SKILL.md`)
how to drive vork: author scenes, render, inspect, refine.

Each skill is a folder with one `SKILL.md`. They are short routing and workflow guides.
The source of truth is the binary's own help (`vork help`, `vork help TOPIC --all`,
`vork help field PATH`), and every skill points back to it. Skills never restate
field-level reference that the help already generates.

## Skills

| Skill | Use it for |
|---|---|
| [`vork`](vork/SKILL.md) | Entry point: first image, the render-inspect-refine loop, which skill to read next |
| [`vork-scene-authoring`](vork-scene-authoring/SKILL.md) | Scene JSON, nodes, std shapes, glTF assets, instances, buffers, coordinate conventions |
| [`vork-materials-lighting`](vork-materials-lighting/SKILL.md) | OpenPBR materials, glass, subsurface, textures, lights, sky, HDR environments, profiles |
| [`vork-cameras-animation`](vork-cameras-animation/SKILL.md) | Physical cameras, depth of field, motion blur, animation tracks and clips, camera export |
| [`vork-render-outputs`](vork-render-outputs/SKILL.md) | `vork render` flags, stills, frames, EXR, video, control passes, sidecars, exit codes |
| [`vork-compositing-vfx`](vork-compositing-vfx/SKILL.md) | Compositor graph, bloom, glare, display transforms, shadow catcher, VFX on real footage |
| [`vork-slang-programs`](vork-slang-programs/SKILL.md) | Custom Slang: surface, displacement, environment, camera, sensor, compute, light paths |
| [`vork-fluid-simulation`](vork-fluid-simulation/SKILL.md) | APIC/FLIP liquid, surfaces, wetting, whitewater |
| [`vork-sdk`](vork-sdk/SKILL.md) | TypeScript and Python scene authoring and render calls |
| [`vork-blender-bridge`](vork-blender-bridge/SKILL.md) | Exporting `.blend` scenes, Blender-to-vork coordinate and material conversion |
| [`vork-setup-troubleshooting`](vork-setup-troubleshooting/SKILL.md) | Install, source builds, GPU support, limits, remote GPU, upgrades, failures |

## Install

Claude Code (project-level):

```sh
mkdir -p .claude/skills && cp -R skills/vork* .claude/skills/
```

Personal (all projects): copy the folders into `~/.claude/skills/`.
Other agents: point their skills directory at `skills/`.

## Conventions

- Frontmatter has only `name` (matches the folder) and `description` (what it does and when to trigger).
- Facts come from `docs/help/*.md`, `libraries/*/HELP.txt` and `examples/`. When a skill and
  `vork help` disagree, the help wins; fix the skill.
- Skills were written against vork 0.2. After a release, re-check `vork help upgrade`.
