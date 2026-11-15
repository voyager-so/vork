SDK
Start with vork init --language typescript or --language python.
scene.json renders immediately; scene.ts/scene.py regenerates it with the matching SDK.
Scenes are plain Python dicts or TypeScript objects with the JSON fields.
Python: pip install vork in a virtual environment (Python 3.11+); from vork import std, write_scene, render.
The Python SDK contains no renderer. Install it with npm install -g @voyager-so/vork,
or use the curl installer or platform archive: https://github.com/voyager-so/vork#install
Use matching SDK and renderer versions.
TypeScript (Node 22.18+): @voyager-so/vork; import builders, writeScene and render.

TypeScript requires npm; older Node needs --experimental-strip-types.
Python std(function, **options); TS roundedBox, extrude and text. Contracts: vork help std --all.
Builders return buffers and geometry maps to merge into the scene.
write_scene(scene, path, base_dir) / writeScene(scene, path, baseDir) rebase paths
and externalize large buffers. base_dir/baseDir defaults to the working directory;
when rewriting a loaded scene elsewhere, pass its original directory explicitly.
Keep the JSON with its referenced files.
render(scene_path, out, at=0) / render(path, options) return exposure summaries
and warnings for omitted effects.
Session keeps the GPU and installed scene between calls: open installs or replaces
a whole scene, render returns revision, reuse, installs and provisional with its results,
and close releases resources. Edit and provisional semantics: vork help render.
VORK_BIN selects the executable. Types: vork.types, TS Scene; schema: sdk/scene.schema.json in the archive (packages/ts/scene.schema.json in source).

SDK render helpers require current completion metadata, including warnings.
Writers preserve an explicit $schema but do not insert one. Use matching SDK and executable versions.
Regenerate SDK-written scenes with the matching SDK after upgrading.

Bundled SDK locations (not registry installs):
  Linux / Mac: ~/.local/lib/vork/python and ~/.local/lib/vork/sdk.
  Windows: %LOCALAPPDATA%\Programs\vork\python and %LOCALAPPDATA%\Programs\vork\sdk.
For custom installations, use the python/sdk directories in the install folder.
The starter README has platform-specific install and virtualenv commands.
Ubuntu needs the venv package matching your Python (python3-venv for the system Python).

Writers retire obsolete generated buffer siblings referenced by the previous JSON
at that destination. Copy JSON and all referenced buffers for a lasting snapshot;
serialize saves and buffer reads at one destination. Unrelated files and symlinks
are preserved. SDK CLI phases forward native stderr progress.

Render at and environmentImage/environment_image fps are numeric SDK options
rather than rational strings. Rational strings remain valid in scene clock
fields, CLI --at and render --set fps. New writer files use 0666 subject to umask;
replacement JSON preserves its mode.
