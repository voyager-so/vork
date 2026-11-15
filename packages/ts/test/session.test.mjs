import {writeExecutable} from './executable-fixture.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFileSync} from 'node:child_process';

// A stand-in worker: records every request, answers in protocol shape, exits on close.
const WORKER = `#!/usr/bin/env node
const fs=require("fs");const requests=[];let revision=0,buffer="";
process.stdin.setEncoding("utf8");
process.stdin.on("data",chunk=>{buffer+=chunk;let i;while((i=buffer.indexOf("\\n"))>=0){
 const r=JSON.parse(buffer.slice(0,i));buffer=buffer.slice(i+1);requests.push(r);let reply;
 if(r.op==="open"){ if(r.document&&r.document.nodes&&r.document.nodes[0].energy==="bright") reply={id:r.id,ok:false,kind:"scene",error:"load scene (document): nodes.key.energy: invalid type: string"}; else {revision++;reply={id:r.id,ok:true,revision,reuse:revision===1?"reinstalled":"updated",changed:revision===1?[]:[["nodes","key","energy"]]};} }
 else if(r.op==="render") reply=r.out[0].endsWith(".jpg")?{id:r.id,ok:false,kind:"usage",error:"unrecognized extension"}
  :{id:r.id,ok:true,revision,reuse:requests.filter(q=>q.op==="render").length===1?"reinstalled":"updated",installs:1,provisional:r.provisional===true,result:{warnings:[{code:"x",requested:{},dropped:"y"}],exposure:[{output:r.out[0],exposure:{frames:1}}],frames:1}};
 else if(r.op==="ask") reply={id:r.id,ok:true,answers:r.questions.map(q=>q.kind)};
 else reply={id:r.id,ok:true};
 process.stdout.write(JSON.stringify(reply)+"\\n");
 if(r.op==="close"){fs.writeFileSync(process.env.VORK_ARGS_CAPTURE,JSON.stringify({argv:process.argv.slice(2),requests}));process.exit(0);}
}});
`;

test('Session speaks the worker protocol and surfaces refusals as errors', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-session-'));
 try {
  const capture=join(directory,'requests.json'),executable=join(directory,'vork');
  writeExecutable(executable,WORKER,{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  const source=`import {Session} from ${JSON.stringify(module)};
   const s=new Session({gpu:'cpu'});
   const out=[];
   out.push(await s.open('shot.json'));
   out.push(await s.render({out:'a.png',at:0.5,profile:'std/draft',set:{samples:4,denoise:false}}));
   out.push(await s.open({version:2,nodes:[{name:'key',type:'light',energy:1400}]},{base:'shots'}));
   out.push(await s.render({out:['b.png','c.mov'],frameStart:2,frameEnd:5,channels:['all']}));
   out.push(await s.render({out:'early.png',provisional:true}));
   out.push(await s.render({out:'final.png'}));
   try { await s.open({version:2,nodes:[{name:'key',type:'light',energy:'bright'}]}); } catch (e) { out.push({error:e.message}); }
   try { await s.render({out:'x.jpg'}); } catch (e) { out.push({error:e.message}); }
   try { s.render({out:'x.png',set:{sample:1}}); } catch (e) { out.push({error:e.message}); }
   try { s.open('shot.json',{root:'x'}); } catch (e) { out.push({error:e.message}); }
   await s.close(); await s.close();
   try { await s.open('again.json'); } catch (e) { out.push({error:e.message}); }
   console.log(JSON.stringify(out));`;
  const output=execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable,VORK_ARGS_CAPTURE:capture},encoding:'utf8'});
  const result=JSON.parse(output.trim().split('\n').at(-1));
  assert.deepEqual(result[0],{revision:1,reuse:'reinstalled',changed:[]});
  assert.deepEqual(result[1],{warnings:[{code:'x',requested:{},dropped:'y'}],exposure:[{output:'a.png',exposure:{frames:1}}],revision:1,reuse:'reinstalled',installs:1,provisional:false});
  assert.deepEqual(result[2],{revision:2,reuse:'updated',changed:[['nodes','key','energy']]});
  assert.equal(result[3].reuse,'updated');
  // The first picture available is labelled; the full render after it is not.
  assert.equal(result[4].provisional,true);
  assert.equal(result[5].provisional,false);
  assert.match(result[6].error,/^scene: load scene .*nodes\.key\.energy/);
  assert.match(result[7].error,/^usage: unrecognized extension/);
  assert.match(result[8].error,/unknown option "sample"; did you mean "samples"\?/);
  assert.match(result[9].error,/unknown option "root"/);
  assert.match(result[10].error,/session is closed/);
  const {argv,requests}=JSON.parse(await readFile(capture,'utf8'));
  assert.deepEqual(argv,['session','--gpu','cpu']);
  assert.deepEqual(requests,[
   {id:1,op:'open',scene:'shot.json'},
   {id:2,op:'render',out:['a.png'],at:0.5,profile:'std/draft',set:{samples:4,denoise:false}},
   {id:3,op:'open',document:{version:2,nodes:[{name:'key',type:'light',energy:1400}]},base:'shots'},
   {id:4,op:'render',out:['b.png','c.mov'],frame_start:2,frame_end:5,channels:['all']},
   {id:5,op:'render',out:['early.png'],provisional:true},
   {id:6,op:'render',out:['final.png']},
   {id:7,op:'open',document:{version:2,nodes:[{name:'key',type:'light',energy:'bright'}]}},
   {id:8,op:'render',out:['x.jpg']},
   {id:9,op:'close'},
  ]);
 } finally {await rm(directory,{recursive:true,force:true});}
});

test('Session keeps its listener count flat over a long request sequence', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-session-long-'));
 try {
  const capture=join(directory,'requests.json'),executable=join(directory,'vork');
  writeExecutable(executable,WORKER,{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  // 100 requests, half refused; after each one the child must carry exactly the
  // constructor's error handler and no exit handler, and Node must not warn.
  const source=`import {Session} from ${JSON.stringify(module)};
   const warnings=[]; process.on('warning',w=>warnings.push(w.name+': '+w.message));
   const s=new Session(); const child=s.child;
   await s.open('shot.json');
   const counts=new Set();
   for (let i=0;i<100;i++) {
     try { await s.render({out:i%2?'x.jpg':'a.png'}); } catch (e) { if (!/unrecognized extension/.test(e.message)) throw e; }
     counts.add(child.listenerCount('error')+'/'+child.listenerCount('exit'));
   }
   await s.close();
   console.log(JSON.stringify({counts:[...counts],warnings,errorListeners:child.listenerCount('error'),exitListeners:child.listenerCount('exit')}));`;
  const output=execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable,VORK_ARGS_CAPTURE:capture},encoding:'utf8'});
  const result=JSON.parse(output.trim().split('\n').at(-1));
  assert.deepEqual(result.counts,['1/0']);
  assert.deepEqual(result.warnings,[]);
  assert.equal(result.errorListeners,1);
  assert.equal(result.exitListeners,0);
  const {requests}=JSON.parse(await readFile(capture,'utf8'));
  assert.equal(requests.length,102);
 } finally {await rm(directory,{recursive:true,force:true});}
});

test('Session reports a worker that dies instead of hanging', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-session-dead-'));
 try {
  const executable=join(directory,'vork');
  writeExecutable(executable,'#!/usr/bin/env node\nprocess.stdin.once("data",()=>process.exit(3));\n',{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  const source=`import {Session} from ${JSON.stringify(module)}; const s=new Session(); try { await s.open('shot.json'); } catch (e) { console.log(e.message); }`;
  const output=execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable},encoding:'utf8'});
  assert.match(output,/vork session ended .*before replying to open/);
 } finally {await rm(directory,{recursive:true,force:true});}
});

test('A queued Session request sends the values its arguments had at the call', async () => {
 // audit-core 12: the queue used to serialise when the request ran, after the caller's later edits.
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-session-snapshot-'));
 try {
  const capture=join(directory,'requests.json'),executable=join(directory,'vork');
  writeExecutable(executable,WORKER,{mode:0o700});
  const module=new URL('../dist/index.js',import.meta.url).href;
  const source=`import {Session} from ${JSON.stringify(module)};
   const s=new Session();
   const scene={version:2,nodes:[{name:'key',type:'light',energy:1}]};
   const first=s.open(scene);
   scene.nodes[0].energy=2;
   const second=s.open(scene);
   scene.nodes[0].energy=3;
   const options={out:['a.png'],set:{samples:4},channels:['color']};
   const render=s.render(options);
   options.out.push('b.png'); options.set.samples=8; options.channels.push('depth');
   const questions=[{kind:'first'}];
   const ask=s.ask(questions);
   questions[0].kind='changed'; questions.push({kind:'added'});
   const emptied=[{kind:'kept'}];
   const second_ask=s.ask(emptied);
   emptied.length=0;
   const answers=await Promise.all([first,second,render,ask,second_ask]);
   let refused; try { s.open({self:1n}); } catch (e) { refused=e instanceof TypeError; }
   const third=s.open('after.json');
   await third; await s.close();
   console.log(JSON.stringify({refused,asked:answers.slice(3)}));`;
  const output=execFileSync(process.execPath,['--input-type=module','-e',source],{env:{...process.env,VORK_BIN:executable,VORK_ARGS_CAPTURE:capture},encoding:'utf8'});
  // Each ask resolves with the answers to the questions it was called with.
  assert.deepEqual(JSON.parse(output.trim().split('\n').at(-1)),{refused:true,asked:[['first'],['kept']]});
  const {requests}=JSON.parse(await readFile(capture,'utf8'));
  assert.deepEqual(requests.map(r=>r.op),['open','open','render','ask','ask','open','close']);
  assert.equal(requests[0].document.nodes[0].energy,1);
  assert.equal(requests[1].document.nodes[0].energy,2);
  assert.deepEqual(requests[2].out,['a.png']);
  assert.deepEqual(requests[2].set,{samples:4});
  assert.deepEqual(requests[2].channels,['color']);
  assert.deepEqual(requests[3].questions,[{kind:'first'}]);
  assert.deepEqual(requests[4].questions,[{kind:'kept'}]);
  // A request that cannot be serialised is refused at the call and uses no id.
  assert.deepEqual(requests.map(r=>r.id),[1,2,3,4,5,6,7]);
 } finally { await rm(directory,{recursive:true,force:true}); }
});
