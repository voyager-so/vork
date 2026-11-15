INIT
vork init [dir] [--language json|typescript|python]
Create a starter project in dir (default: current directory), without network or GPU access.
Writes scene.json and README.md. The scene uses std sphere, box and grid builders.
Existing output files cause a refusal before writing;
unrelated files are preserved. scene.json renders immediately (vork help).

JSON is the default. TypeScript adds scene.ts and a package.json with type: module;
Python adds scene.py. README.md explains how to install the matching bundled SDK
and regenerate scene.json. See vork help sdk for SDK details.

Requires Node 22.18+ and npm. On older Node, upgrade (recommended), or run `node --experimental-strip-types scene.ts` instead of `node scene.ts`.
