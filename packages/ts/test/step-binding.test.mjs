import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import Ajv2020 from 'ajv/dist/2020.js';

test('compositor inputs accept only the step binding in JSON Schema', () => {
  const schema = JSON.parse(readFileSync(new URL('../scene.schema.json', import.meta.url)));
  const validate = new Ajv2020({strict:false, validateFormats:false}).compile(schema);
  const scene = binding => ({version:2, nodes:[{type:'camera'}], composition:{space:'3d', compositor:[
    {name:'saved', material:{program:'copy'}},
    {material:{program:'copy', inputs:{source:binding}}},
  ]}});
  assert.equal(validate(scene({step:'saved'})), true, JSON.stringify(validate.errors));
  for (const binding of [{node:'saved'}, {step:'saved',node:'saved'}, {step:1}])
    assert.equal(validate(scene(binding)), false, JSON.stringify(binding));
  assert.ok(schema.$defs.StepBinding);
  assert.equal(schema.$defs.NodeBinding, undefined);
});

test('TypeScript exposes StepBinding and rejects the removed node spelling', () => {
  const directory = mkdtempSync(join(tmpdir(), 'vork-step-types-'));
  try {
    const source = join(directory, 'binding.mts');
    const schema = JSON.stringify(fileURLToPath(new URL('../src/schema.js', import.meta.url)));
    writeFileSync(source, `import type {Input, StepBinding, Material} from ${schema};
// @ts-expect-error the compositor NodeBinding export was removed
import type {NodeBinding} from ${schema};
const step: StepBinding = {step:'saved'};
const input: Input = step;
const material: Material = {program:'copy', inputs:{source:input}};
// @ts-expect-error an earlier compositor entry uses step
const old: Input = {node:'saved'};
// @ts-expect-error no second spelling on StepBinding
const both: StepBinding = {step:'saved', node:'saved'};
void material;
`);
    execFileSync(process.execPath, [fileURLToPath(new URL('../node_modules/typescript/bin/tsc', import.meta.url)),
      '--noEmit', '--strict', '--skipLibCheck', '--target', 'ES2022', '--module', 'NodeNext', source], {stdio:'pipe'});
  } finally { rmSync(directory, {recursive:true, force:true}); }
});
