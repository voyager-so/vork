import {writeExecutable} from './executable-fixture.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, writeFile, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFileSync} from 'node:child_process';

test('exportAsset forwards selection/time/loss codes and returns the complete report', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-export-'));
  try {
    const asset = join(directory, 'hero.glb'), executable = join(directory, 'vork');
    const report = {version:1, status:'complete', issues:[], counts:{nodes:1,meshes:1,materials:1,images:0}, source_hash:'test'};
    await writeFile(join(directory, 'hero.export.json'), JSON.stringify(report));
    writeExecutable(executable, '#!/usr/bin/env node\nrequire("fs").writeFileSync(process.env.CAPTURE,JSON.stringify(process.argv.slice(2)));\n', {mode:0o700});
    const module = new URL('../dist/index.js', import.meta.url).href;
    const output = execFileSync(process.execPath, ['--input-type=module','-e', `import {exportAsset} from ${JSON.stringify(module)}; console.log(JSON.stringify(exportAsset('scene.json',{out:${JSON.stringify(asset)},root:'hero',at:1.25,allowLosses:['camera_dof']})));`], {encoding:'utf8', env:{...process.env,VORK_BIN:executable,CAPTURE:join(directory,'args.json')}});
    assert.deepEqual(JSON.parse(output), report);
    assert.deepEqual(JSON.parse(await readFile(join(directory,'args.json'),'utf8')), ['export','scene.json','--out',asset,'--root','hero','--at','1.25','--allow-loss','camera_dof']);
    execFileSync(process.execPath, ['--input-type=module','-e', `import {exportAsset} from ${JSON.stringify(module)}; exportAsset('scene.json',{out:${JSON.stringify(asset)},frameStart:12,frameEnd:36});`], {env:{...process.env,VORK_BIN:executable,CAPTURE:join(directory,'args.json')}});
    assert.deepEqual(JSON.parse(await readFile(join(directory,'args.json'),'utf8')), ['export','scene.json','--out',asset,'--frame-start','12','--frame-end','36']);
  } finally {await rm(directory,{recursive:true,force:true});}
});
