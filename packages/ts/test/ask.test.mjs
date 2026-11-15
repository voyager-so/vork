import {writeExecutable} from './executable-fixture.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFileSync} from 'node:child_process';

const ENGINE = `#!/usr/bin/env node
const readline=require('readline');
function answers(q) {if(q.some(x=>'bad' in x)) throw Error('ask: unknown address'); return q.map(x=>x.at??0);}
if(process.argv[2]==='ask') {
 if(JSON.stringify(process.argv.slice(3))!==JSON.stringify(['scene with spaces.json','-'])) process.exit(9);
 let input='';process.stdin.on('data',c=>input+=c);process.stdin.on('end',()=>{
 try{console.log(JSON.stringify(answers(JSON.parse(input))));}catch(e){console.error(e.message);process.exitCode=3;}});
} else {
 let revision=0;
 readline.createInterface({input:process.stdin}).on('line',line=>{
 const r=JSON.parse(line), reply={id:r.id,ok:true};
 if(r.op==='open')Object.assign(reply,{revision:++revision,reuse:'updated',changed:[]});
 if(r.op==='ask')try{Object.assign(reply,{answers:answers(r.questions),revision});}catch(e){Object.assign(reply,{ok:false,kind:'scene',error:e.message});}
 console.log(JSON.stringify(reply));if(r.op==='close')process.exit(0);
 });
}
`;

test('ask preserves batches and exact times through CLI and session, then recovers from refusal', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-ask-'));
 try {
  const executable=join(directory,'vork');
  writeExecutable(executable,ENGINE,{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  const source=`import {ask,Session} from ${JSON.stringify(module)};
    import assert from 'node:assert/strict';
    const q=[{field:'nodes.hero.translation',at:['1/3',2]},{buffer:'p'}];
    assert.deepEqual(ask('scene with spaces.json',q),[['1/3',2],0]);
    assert.throws(()=>ask('scene with spaces.json',[{bad:true}]),/ask: unknown address/);
    const s=new Session();await s.open('scene with spaces.json');
    assert.deepEqual(await s.ask(q),[['1/3',2],0]);
    await assert.rejects(s.ask([{bad:true}]),/ask: unknown address/);
    assert.deepEqual(await s.ask([]),[]);await s.close();
    await assert.rejects(s.ask(q),/closed/);`;
  execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable},stdio:'pipe'});
 } finally {await rm(directory,{recursive:true,force:true});}
});
