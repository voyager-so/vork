import {writeExecutable} from './executable-fixture.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, writeFile, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname} from 'node:path';
import {execFileSync} from 'node:child_process';

test('render preserves exposure and warnings', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-report-'));
 const summary={frames:2,coverage:0.5,clipped:0.123456789,near_black:0.2,mean:0.41,worst_clipped:{share:0.25,frame:7},worst_near_black:{share:0.3,frame:8}};
 const empty={frames:1,coverage:0,clipped:null,near_black:null,mean:null,worst_clipped:null,worst_near_black:null};
 const summaries=[summary,empty,summary];
 const degradation={code:"fog_watchdog",requested:{frame:795,terminated_medium_paths:2,discarded_medium_radiance:1.25},dropped:"ended fog paths; continuation energy unknown; simplify fog and rerender"};
 const warning={code:"additional_environment",requested:{node:{name:"second",energy:2}},dropped:"second environment; retaining first"};
 try {
  const outputs=['still.png','clip.MOV','frames.png/','controls','exr/%06d.exr'].map(p=>directory+'/'+p);
  const metas=['still.png.meta.json','clip.MOV.meta.json','frames.png/meta.json','controls/meta.json','exr/meta.json'];
  for(let i=0;i<metas.length;i++) {
   const path=join(directory,metas[i]);await mkdir(dirname(path),{recursive:true});
   await writeFile(path,JSON.stringify({...(i<3?{exposure:summaries[i]}:{}), warnings:[warning, degradation]}));
  }
  const executable=join(directory,'vork');
  writeExecutable(executable,'#!/usr/bin/env node\nconsole.log("exposure: 12% clipped, 20% near-black, mean 0.41");\n',{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  const source=`import {render} from ${JSON.stringify(module)}; console.log(JSON.stringify(render('shot.json',{out:${JSON.stringify(outputs)}})));`;
  const output=execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable},encoding:'utf8'});
  assert.match(output,/exposure: 12% clipped/);
  assert.deepEqual(JSON.parse(output.trim().split('\n').at(-1)),{exposure:outputs.slice(0,3).map((output,i)=>({output,exposure:summaries[i]})),warnings:[warning, degradation]});
 } finally {await rm(directory,{recursive:true,force:true});}
});

test('render preserves rational output rates in the child arguments', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-clocks-'));
 try {
  const output=join(directory,'frame.png'),capture=join(directory,'args.json'),executable=join(directory,'vork');
  await writeFile(output+'.meta.json',JSON.stringify({warnings:[]}));
  writeExecutable(executable,'#!/usr/bin/env node\nrequire("fs").writeFileSync(process.env.VORK_ARGS_CAPTURE,JSON.stringify(process.argv.slice(2)));\n',{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  const source=`import {render} from ${JSON.stringify(module)}; render('shot.json',{out:${JSON.stringify(output)},at:0.5,set:{fps:'30000/1001'}});`;
  execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable,VORK_ARGS_CAPTURE:capture}});
  assert.deepEqual(JSON.parse(await readFile(capture,'utf8')),
   ['render','shot.json','--out',output,'--at','0.5','--set','fps="30000/1001"']);
 } finally {await rm(directory,{recursive:true,force:true});}
});
