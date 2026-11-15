import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { compile } from 'json-schema-to-typescript';

execFileSync(process.platform === 'win32' ? 'python' : 'python3', ['scripts/sync-version', ...(process.argv.includes('--check') ? ['--check'] : [])], {cwd: new URL('../..', import.meta.url), stdio:'inherit'});

// Regenerate the shared std implementation and helper signatures from SDK owners.
execFileSync(process.execPath, ['packages/ts/generate-std.mjs', ...(process.argv.includes('--check') ? ['--check'] : [])], {
  cwd: new URL('../..', import.meta.url), stdio:'inherit',
});

const schemaText = execFileSync('cargo', ['run', '--quiet', '-p', 'vork-scene', '--example', 'generate-docs', '--', '--schema'], {
  cwd: new URL('../..', import.meta.url), encoding: 'utf8',
});
const schema = JSON.parse(schemaText);
// Defaults and help annotations on a reference are not distinct TypeScript types
// (they would emit Visibility1, Render1, ...).
const typeSchema = structuredClone(schema);
function sharedRefs(value) {
  if (!value || typeof value !== 'object') return;
  if (value.$ref) for (const key of Object.keys(value)) if (key !== '$ref' && key !== 'description') delete value[key];
  for (const child of Object.values(value)) sharedRefs(child);
}
sharedRefs(typeSchema);
let types = await compile(typeSchema, 'Scene', {
  bannerComment: '/** Generated from vork-scene Rust input types. Run npm run generate; do not edit. */',
  additionalProperties: false,
});
types += '\n' + await compile({ ...typeSchema.$defs.Profile, $defs: typeSchema.$defs }, 'Profile', {bannerComment:'', additionalProperties:false});
// Output descriptors share WorkingColorSpace with composition; reference that declaration once.
const outputTypes = await compile({ ...typeSchema.$defs.OutputDestination, $defs: typeSchema.$defs }, 'OutputDestination', {bannerComment:'', additionalProperties:false});
types += '\n' + (types.includes('export type WorkingColorSpace =') ? outputTypes.replace(/export type WorkingColorSpace = [^;]+;\n/, '') : outputTypes);
for (const [file, value] of [['scene.schema.json', schemaText], ['src/schema.ts', types]]) {
  if (process.argv.includes('--check')) {
    if (readFileSync(new URL(file, import.meta.url), 'utf8') !== value) {
      throw new Error(`${file} is stale; run npm run generate`);
    }
  } else writeFileSync(new URL(file, import.meta.url), value);
}

execFileSync(process.execPath, ['packages/ts/generate-resource-slots.mjs', ...(process.argv.includes('--check') ? ['--check'] : [])], {cwd: new URL('../..', import.meta.url), stdio: 'inherit'});

// Rust owns field documentation and defaults as well as the input schema.
execFileSync('cargo', ['run', '--quiet', '-p', 'vork-scene', '--example', 'generate-docs', ...(process.argv.includes('--check') ? ['--', '--check'] : [])], {
  cwd: new URL('../..', import.meta.url), stdio: 'inherit',
});

execFileSync(process.platform === 'win32' ? 'python' : 'python3', ['packages/python/generate.py', ...(process.argv.includes('--check') ? ['--check'] : [])], {cwd: new URL('../..', import.meta.url), stdio:'inherit'});
