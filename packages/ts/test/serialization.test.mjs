import {writeExecutable} from './executable-fixture.mjs';
import { readdir } from 'node:fs/promises';
import {existsSync} from 'node:fs';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm, rename, mkdir, writeFile, symlink, realpath, chmod, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import * as sdk from '../dist/index.js';
import {mkdtempSync, readFileSync, rmSync} from 'node:fs';
// Exercise argv through the public render wrapper and a child process.
function renderArguments(scene, options) {
  const dir = mkdtempSync(join(tmpdir(), 'vork-render-argv-'));
  const binary = join(dir, 'vork');
  const capture = join(dir, 'argv.json');
  const previous = process.env.VORK_BIN;
  writeExecutable(binary, '#!/usr/bin/env node\n' +
    `require('node:fs').writeFileSync(${JSON.stringify(capture)}, JSON.stringify(process.argv.slice(2)));`, {mode:0o755});
  process.env.VORK_BIN = binary;
  try {
    // The child only captures arguments; reading absent exposure metadata then fails.
    try { sdk.render(scene, options); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    return JSON.parse(readFileSync(capture, 'utf8'));
  } finally {
    if (previous === undefined) delete process.env.VORK_BIN; else process.env.VORK_BIN = previous;
    rmSync(dir, {recursive:true,force:true});
  }
}

test('argv builders are absent from the public SDK', () => {
  assert.equal('renderArguments' in sdk, false);
});

test('SDK resolves resource paths without mutating its input', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-'));
  const scene = { version: 2, images: { sky: { file: 'studio.hdr' } }, render: { camera: 'hero' }, animation: [{ tracks: [{path: ['render', 'camera'], keys: [[0, 'hero']]}] }], nodes: [
    { type: 'node', children: [{ type: 'asset', file: 'model.glb' }] },
  ] };
  try {
    const path = await sdk.writeScene(scene, join(directory, 'scenes/shot.json'), directory);
    const saved = JSON.parse(await readFile(path, 'utf8'));
    assert.equal(saved.$schema, undefined);
    assert.equal(scene.$schema, undefined);
    const annotated = {...scene, $schema: 'https://voyager.so/vork/schema/scene.schema.json'};
    const annotatedPath = await sdk.writeScene(annotated, join(directory, 'annotated.json'), directory);
    assert.equal(JSON.parse(await readFile(annotatedPath, 'utf8')).$schema, annotated.$schema);
    assert.equal(saved.nodes[0].children[0].file, '../model.glb');
    assert.equal(saved.images.sky.file, '../studio.hdr');
    assert.deepEqual(saved.animation, scene.animation);
    assert.equal(scene.nodes[0].children[0].file, 'model.glb');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('SDK rebases an inline program origin so its local imports still resolve', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-origin-'));
  const absolute = join(directory, 'lib/abs.slang');
  const scene = { version: 2, nodes: [], programs: {
    relative: { source: 'import vork;', source_origin: 'shaders/main.slang' },
    absolute: { source: 'import vork;', source_origin: absolute },
    unanchored: { source: 'import vork;' },
  } };
  try {
    const path = await sdk.writeScene(scene, join(directory, 'out/nested/shot.json'), directory);
    const saved = JSON.parse(await readFile(path, 'utf8'));
    assert.equal(saved.programs.relative.source_origin, '../../shaders/main.slang');
    assert.equal(resolve(dirname(path), saved.programs.absolute.source_origin), resolve(await realpath(directory), 'lib/abs.slang'));
    assert.equal(saved.programs.unanchored.source_origin, undefined);
    assert.equal(scene.programs.relative.source_origin, 'shaders/main.slang');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('SDK externalizes typed large buffers as one little-endian sidecar', async()=>{
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-buffers-'));
 const matrix=[1,0,0,0,0,1,0,0,0,0,1,0,2,3,4,1];
 const scene={version:2,nodes:[],buffers:{
  signed:{type:'i32',data:Array(1024).fill(-1)},
  unsigned:{type:'uvec4',data:Array(256).fill([0xffffffff,0,17,1])},
  matrices:{type:'mat4',data:Array(64).fill(matrix)},
  positions:{type:'vec3',data:Array(342).fill([0.1,-2,3])},
  small:{type:'f32',data:[0.5]},
 }};
 const original=structuredClone(scene);
 try{
  const path=await sdk.writeScene(scene,join(directory,'shot.json'));
  const saved=JSON.parse(await readFile(path,'utf8'));
  const binary=await readFile(resolve(directory, saved.buffers.signed.file));
  assert.deepEqual(saved.buffers.small,scene.buffers.small);
  assert.equal(saved.buffers.signed.offset,0);assert.equal(saved.buffers.signed.count,1024);
  assert.equal(saved.buffers.unsigned.offset,4096);assert.equal(saved.buffers.matrices.offset,8192);
  assert.equal(saved.buffers.positions.offset,12288);assert.equal(binary.length,16392);
  assert.equal(binary.readInt32LE(0),-1);assert.equal(binary.readUInt32LE(4096),0xffffffff);
  assert.equal(binary.readFloatLE(8192+12*4),2);assert.equal(binary.readFloatLE(12288),Math.fround(0.1));
  assert.deepEqual(scene,original);
  const republished = join(directory,'shot.json');
  await sdk.writeScene(saved,republished,directory);
  assert.equal((await readFile(resolve(directory, saved.buffers.signed.file))).length,binary.length,'referenced sidecar survives a rewrite');
  await sdk.writeScene({version:2,nodes:[]},republished);
  await assert.rejects(readFile(resolve(directory, saved.buffers.signed.file)), {code:'ENOENT'});
 }finally{await rm(directory,{recursive:true,force:true});}
});

test('SDK rejects invalid packed scalars instead of truncating or wrapping', async()=>{
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-buffer-errors-'));
 try{
  for(const [type,row] of [['u32',-1],['i32',2147483648],['f32',1e100],['vec3',[0,1]]]){
   const scene={version:2,nodes:[],buffers:{invalid:{type,data:Array(1024).fill(row)}}};
   await assert.rejects(sdk.writeScene(scene,join(directory,'bad.json')),RangeError);
  }
 }finally{await rm(directory,{recursive:true,force:true});}
});


test('SDK scene and sidecar remain portable after moving the scene tree', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-portable-'));
 try {
  const scene={version:2,nodes:[{type:'asset',name:'actor',file:'assets/model.glb'}],
   images:{sky:{file:join(directory,'assets/sky.hdr')}},
   programs:{surface:{file:'programs/surface.slang'}},
   animation:[{clip:{target:'actor',file:'assets/walk.glb'}}],
   buffers:{external:{type:'f32',file:'assets/data.bin'},packed:{type:'f32',data:Array(1024).fill(1)}}};
  const destination=join(directory,'scenes/shot.json');
  await sdk.writeScene(scene,destination,directory);
  const saved=JSON.parse(await readFile(destination,'utf8'));
  assert.equal(saved.nodes[0].file,'../assets/model.glb');
  assert.equal(saved.images.sky.file,'../assets/sky.hdr');
  assert.equal(saved.programs.surface.file,'../programs/surface.slang');
  assert.equal(saved.animation[0].clip.file,'../assets/walk.glb');
  assert.equal(saved.buffers.external.file,'../assets/data.bin');
  assert.match(saved.buffers.packed.file,/^shot\.json\.buffers\.[0-9a-f-]+\.bin$/);
  await rename(dirname(destination),join(directory,'moved'));
  assert.equal((await readFile(join(directory,'moved',saved.buffers.packed.file))).length,4096);
 } finally {await rm(directory,{recursive:true,force:true});}
});


test('SDK relative resources open correctly through a symlinked output directory', async () => {
 const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-symlink-'));
 try {
  await mkdir(join(directory, 'project/assets'), {recursive:true});
  await mkdir(join(directory, 'scratch/render'), {recursive:true});
  await writeFile(join(directory, 'project/assets/model.glb'), 'model');
  await symlink(join(directory, 'scratch/render'), join(directory, 'project/out'), 'dir');
  const path = await sdk.writeScene({version:2,nodes:[{type:'asset',file:'assets/model.glb'}]},
   join(directory, 'project/out/shot.json'), join(directory, 'project'));
  const saved = JSON.parse(await readFile(path, 'utf8'));
  assert.equal(await readFile(join(dirname(path), saved.nodes[0].file), 'utf8'), 'model');
 } finally { await rm(directory, {recursive:true,force:true}); }
});


test('SDK protects referenced sidecars through filesystem aliases', async () => {
 const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-alias-'));
 try {
  const physical = join(directory, 'real'), alias = join(directory, 'link');
  await mkdir(physical);
  await symlink(physical, alias, 'dir');
  const original = Buffer.from([0,0,0,0]);
  const sidecar = join(physical, 'shot.json.buffers.bin');
  await writeFile(sidecar, original);
  for (const packed of [false, true]) {
   await sdk.writeScene({version:2,nodes:[],buffers:{
    external:{type:'f32',file:'shot.json.buffers.bin',count:1},
    ...(packed ? {packed:{type:'f32',data:Array(1024).fill(1)}} : {})}}, join(alias,'shot.json'), alias);
   assert.deepEqual(await readFile(sidecar), original);
  }
  await symlink(join(physical,'shot.json'), join(directory,'scene-link.json'));
  await assert.rejects(sdk.writeScene({version:2,nodes:[]}, join(directory,'scene-link.json')), /symbolic link/);
  assert.deepEqual(await readFile(sidecar), original);
 } finally { await rm(directory, {recursive:true,force:true}); }
});


test('SDK paths survive moving a project written through a symlink', async () => {
 const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-project-link-'));
 try {
  const project = join(directory, 'project'), alias = join(directory, 'link');
  await mkdir(join(project,'assets'), {recursive:true});
  await writeFile(join(project,'assets/model.glb'), 'model');
  await symlink(project, alias, 'dir');
  await sdk.writeScene({version:2,nodes:[{type:'asset',file:'assets/model.glb'}]}, join(alias,'scenes/shot.json'), alias);
  await rename(project, join(directory,'moved'));
  const path = join(directory,'moved/scenes/shot.json');
  const saved = JSON.parse(await readFile(path, 'utf8'));
  assert.equal(saved.nodes[0].file, '../assets/model.glb');
  assert.equal(await readFile(join(dirname(path),saved.nodes[0].file), 'utf8'), 'model');
 } finally { await rm(directory, {recursive:true,force:true}); }
});


test('SDK preserves the containing directory of symlinked resource files', async () => {
 const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-file-link-'));
 try {
  await mkdir(join(directory,'project/assets'), {recursive:true});
  await mkdir(join(directory,'cache'));
  await writeFile(join(directory,'cache/model.gltf'), JSON.stringify({buffers:[{uri:'mesh.bin'}]}));
  await writeFile(join(directory,'project/assets/mesh.bin'), 'intended');
  await writeFile(join(directory,'cache/mesh.bin'), 'other');
  await symlink(join(directory,'cache/model.gltf'), join(directory,'project/assets/model.gltf'));
  const path = await sdk.writeScene({version:2,nodes:[{type:'asset',file:'assets/model.gltf'}]},
   join(directory,'project/scenes/shot.json'), join(directory,'project'));
  const saved = JSON.parse(await readFile(path,'utf8'));
  const model = join(dirname(path), saved.nodes[0].file);
  const document = JSON.parse(await readFile(model,'utf8'));
  assert.equal(await readFile(join(dirname(model),document.buffers[0].uri),'utf8'), 'intended');
 } finally { await rm(directory, {recursive:true,force:true}); }
});

test('SDK render options map to vork render flags', () => {
  assert.deepEqual(renderArguments('shot.json', {out: 'a.png'}), ['render', 'shot.json', '--out', 'a.png']);
  assert.deepEqual(renderArguments('shot.json', {out: 'a.png', channels: ['all']}), ['render', 'shot.json', '--out', 'a.png', '--channels', 'all']);
  assert.deepEqual(renderArguments('shot.json', {out: 'a.png', at: 0, set: {'resolution': [64, 32]}}),
    ['render', 'shot.json', '--out', 'a.png', '--at', '0', '--set', 'resolution=[64,32]']);
  for (const key of ['resolution','samples','previs']) assert.throws(() => renderArguments('shot.json', {out:'a.png', [key]:1}), /unknown option/);
  assert.deepEqual(renderArguments('shot.json', {out: 'a.png', set: {samples: 7}}),
    ['render', 'shot.json', '--out', 'a.png', '--set', 'samples=7']);
  assert.ok(!renderArguments('shot.json', {out: ['a.mp4', 'b']}).includes('--previs'));
  assert.deepEqual(renderArguments('shot.json', {out: 'a.png', profile: 'std/draft', set: {seed: 4}}),
    ['render', 'shot.json', '--out', 'a.png', '--profile', 'std/draft', '--set', 'seed=4']);
  assert.deepEqual(renderArguments('shot.json', {out: 'frames/', frameStart: 3, frameEnd: 7}),
    ['render', 'shot.json', '--out', 'frames/', '--frame-start', '3', '--frame-end', '7']);
  for (const key of ['render.camera','camera','render.background','render.resolution','settings.seed','exposure']) {
    assert.throws(() => renderArguments('shot.json', {out:'a.png', set:{[key]:1}}), /unknown option/);
  }
  assert.throws(() => sdk.render('shot.json', {out: 'a.png', preview: true}));
});

test('SDK writes std shapes to the buffer sidecar', async () => {
 const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-shapes-'));
 try {
  const shape = sdk.roundedBox('ball');
  const path = await sdk.writeScene({version:2, ...shape, nodes:[{type:'mesh', geometry:'ball'}]}, join(directory, 'ball.json'));
  const saved = JSON.parse(await readFile(path, 'utf8'));
  for (const buffer of Object.values(saved.buffers)) assert.match(buffer.file, /^ball\.json\.buffers\.[0-9a-f-]+\.bin$/);
  assert.ok(!JSON.stringify(saved).includes('"data"'));
 } finally { await rm(directory, {recursive:true, force:true}); }
});

test('SDK sidecar never replaces or removes a resource the scene names', async () => {
 const directory = await mkdtemp(join(tmpdir(), 'vork-sdk-reserved-'));
 try {
  const shader = join(directory, 'shot.json.buffers.bin');
  await writeFile(shader, 'import vork;');
  let saved;
  for (const buffers of [{}, {x:{type:'f32', data:Array(1024).fill(1)}}]) {
   const path = await sdk.writeScene({version:2, nodes:[], buffers, programs:{p:{file:'shot.json.buffers.bin'}}}, join(directory, 'shot.json'), directory);
   saved = JSON.parse(await readFile(path, 'utf8'));
   assert.equal(await readFile(shader, 'utf8'), 'import vork;');
  }
  assert.notEqual(saved.buffers.x.file, 'shot.json.buffers.bin');
 } finally { await rm(directory, {recursive:true, force:true}); }
});


test('atomic saves publish complete buffers and preserve private JSON mode', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-sdk-atomic-'));
 const path=join(directory,'shot.json');
 const scene=n=>({version:2,nodes:[{type:'node',name:String(n)}],buffers:{x:{type:'f32',data:Array(1024).fill(n)}}});
 const snapshots=[];
 try {
  await sdk.writeScene(scene(0),path);
  await chmod(path,0o600);
  snapshots.push(JSON.parse(await readFile(path,'utf8')));
  let done=false;
  const writes=Promise.all(Array.from({length:20},(_,i)=>sdk.writeScene(scene(i+1),path))).finally(()=>{done=true;});
  while(!done) snapshots.push(JSON.parse(await readFile(path,'utf8')));
  await writes;
  snapshots.push(JSON.parse(await readFile(path,'utf8')));
  for(const snapshot of snapshots.slice(-1)) {
   const bytes=await readFile(join(directory,snapshot.buffers.x.file));
   assert.equal(bytes.length,4096);
   for(let offset=0;offset<bytes.length;offset+=4) assert.equal(bytes.readFloatLE(offset),Number(snapshot.nodes[0].name));
  }
  assert.equal((await stat(path)).mode&0o777,0o600);
  const before=await readFile(path);
  await assert.rejects(sdk.writeScene({...scene(99),invalid:1n},path));
  assert.deepEqual(await readFile(path),before);
 } finally {await rm(directory,{recursive:true,force:true});}
});

test('SDK rebases a published nested sidecar when writing to another directory',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'vork-inline-rebase-'));
  try {
    const source=join(directory,'a'),destination=join(directory,'b');
    const scene={version:2,nodes:[{type:'node',children:[{type:'mesh',geometry:{attributes:{
      POSITION:{type:'vec3',data:Array.from({length:342},(_,i)=>[i,0,0])}}}}]}]};
    const first=JSON.parse(await readFile(await sdk.writeScene(scene,join(source,'shot.json')),'utf8'));
    const original=first.nodes[0].children[0].geometry.attributes.POSITION;
    const second=JSON.parse(await readFile(await sdk.writeScene(first,join(destination,'shot.json'),source),'utf8'));
    const moved=second.nodes[0].children[0].geometry.attributes.POSITION;
    assert.equal(moved.file,'../a/'+original.file);
    assert.deepEqual(await readFile(join(destination,moved.file)),await readFile(join(source,original.file)));
    assert.equal(first.nodes[0].children[0].geometry.attributes.POSITION.file,original.file);
  } finally {await rm(directory,{recursive:true,force:true});}
});

test('SDK rebases a shared nested file buffer only once',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'vork-shared-rebase-'));
  try {
    const source=join(directory,'input'),destination=join(directory,'output/deep');
    await mkdir(source);await writeFile(join(source,'positions.bin'),new Uint8Array(36));
    const buffer={type:'vec3',file:'positions.bin',count:3};
    const node=()=>({type:'mesh',geometry:{attributes:{POSITION:buffer}}});
    const path=await sdk.writeScene({version:2,nodes:[node(),node()]},join(destination,'shot.json'),source);
    const saved=JSON.parse(await readFile(path,'utf8'));
    for (const node of saved.nodes) {
      assert.equal(node.geometry.attributes.POSITION.file,'../../input/positions.bin');
      assert.equal((await readFile(join(destination,node.geometry.attributes.POSITION.file))).length,36);
    }
    assert.equal(buffer.file,'positions.bin');
  } finally {await rm(directory,{recursive:true,force:true});}
});


test('text font paths survive relocation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-text-fonts-'));
  try {
    await mkdir(join(directory, 'fonts'));
    for (const name of ['a.ttf', 'b.otf']) await writeFile(join(directory, 'fonts', name), name);
    for (const font of ['fonts/a.ttf', ['fonts/a.ttf', 'fonts/b.otf']]) {
      const text = {type:'text',content:'Hello',font};
      const run = {content:'Hello',font};
      const scene = {version:2,composition:{space:'2d'},nodes:[text,{children:[{...text,content:[run]}]}]};
      const original = structuredClone(scene);
      const path = await sdk.writeScene(scene, join(directory, 'out/deep/scene.json'), directory);
      const saved = JSON.parse(await readFile(path, 'utf8'));
      const expected = Array.isArray(font) ? font.map(f => '../../' + f) : '../../' + font;
      const nested = saved.nodes[1].children[0];
      for (const node of [saved.nodes[0], nested, nested.content[0]]) {
        assert.deepEqual(node.font, expected);
        const files = [node.font].flat(), authored = [font].flat();
        for (let i = 0; i < files.length; i++) {
          assert.equal(await realpath(resolve(dirname(path), files[i])), await realpath(resolve(directory, authored[i])));
        }
      }
      assert.deepEqual(scene, original);
    }
  } finally { await rm(directory, {recursive:true,force:true}); }
});

test('inline resource paths survive relocation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-inline-paths-'));
  try {
    await mkdir(join(directory, 'textures'));
    await mkdir(join(directory, 'shaders'));
    await writeFile(join(directory, 'textures/a.png'), 'image');
    await writeFile(join(directory, 'shaders/main.slang'), 'import vork;');
    const positions = Buffer.alloc(36);
    [0,0,0,1,0,0,0,1,0].forEach((value, i) => positions.writeFloatLE(value, i*4));
    await writeFile(join(directory, 'positions.bin'), positions);
    for (const program of [{file:'shaders/main.slang'}, {source:'import vork;',source_origin:'shaders/main.slang'}]) {
      const scene = {version:2,nodes:[{type:'camera',name:'camera'}, {type:'node',children:[{type:'mesh',
        geometry:{attributes:{POSITION:{type:'vec3',file:'positions.bin',count:3}}},
        materials:[{program,inputs:{texture:{image:{file:'textures/a.png'}},file:'opaque_token'}}],
      }]}]};
      const original = structuredClone(scene);
      const path = await sdk.writeScene(scene, join(directory, 'out/deep/scene.json'), directory);
      const saved = JSON.parse(await readFile(path, 'utf8'));
      const mesh = saved.nodes[1].children[0];
      assert.equal(mesh.materials[0].inputs.file, 'opaque_token');
      const fields = [[mesh.geometry.attributes.POSITION.file,'positions.bin'],
        [mesh.materials[0].inputs.texture.image.file,'textures/a.png']];
      for (const field of ['file','source_origin']) if (field in program) fields.push([mesh.materials[0].program[field],'shaders/main.slang']);
      for (const [file,expected] of fields) assert.equal(await realpath(resolve(dirname(path),file)), await realpath(join(directory,expected)));
      assert.deepEqual(scene, original);
    }
  } finally { await rm(directory, {recursive:true,force:true}); }
});


test('resource slots match inline lowering, including shared definitions', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-inline-slots-'));
  const buffer = {type:'f32',file:'data.bin',count:1};
  const image = {file:'texture.png'};
  const material = {program:{file:'shader.slang',bindings:{data:buffer}},
    inputs:{texture:{image},density:{grid:buffer}}};
  const scene = {version:2,nodes:[
    {type:'mesh',geometry:{indices:buffer,widths:buffer,curve_vertex_counts:buffer,
      attributes:{POSITION:buffer},morphs:[{POSITION:buffer,NORMAL:buffer}]},materials:[material],
      instances:{count:buffer,attributes:{x:buffer}},skin:{inverse_bind_matrices:buffer}},
    {type:'light',image,medium:material,material},
    {type:'asset',file:'asset.glb',materials:{slot:material}},
  ],images:{nested:{render:{},composition:{compositor:[material]}}},
  composition:{space:'3d',compositor:[material]},render:{compositor_per_step:[material]},
  animation:[{tracks:[{keys:{times:buffer,values:buffer}}]}]};
  try {
    const path = await sdk.writeScene(scene,join(directory,'out/scene.json'),directory);
    const saved = JSON.parse(await readFile(path,'utf8'));
    const check = value => {
      if (!value || typeof value !== 'object') return;
      for (const [key,child] of Object.entries(value)) {
        if (key === 'file') { assert.ok(child.startsWith('../'),child); assert.ok(!child.startsWith('../../'),child); }
        else check(child);
      }
    };
    check(saved);
  } finally { await rm(directory,{recursive:true,force:true}); }
});

test('relocation preserves expanded text geometry and nested display images', {skip: !existsSync(new URL('../../../crates/vork-std/tests/fonts/LiberationSans-Regular.ttf',import.meta.url)) && 'private font fixture is absent'}, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-display-font-'));
  const font = await readFile(new URL('../../../crates/vork-std/tests/fonts/LiberationSans-Regular.ttf',import.meta.url));
  await mkdir(join(directory,'fonts'));
  await writeFile(join(directory,'fonts/a.ttf'),font);
  const options = {body:'Hello',font:join(directory,'fonts/a.ttf')};
  const text = sdk.text('label',options);
  const display = {program:'blender/filmic',inputs:{lut:{image:{file:'lut.exr'}}}};
  const scene = {version:2,...text,nodes:[{type:'mesh',geometry:{...text.geometry.label}}],
    composition:{space:'3d',view:display},images:{capture:{render:{},composition:{compositor:[{program:'authored_white_balance',
      inputs:{texture:{image:{file:'nested.exr'}}}}]}}}};
  try {
    const path = await sdk.writeScene(scene,join(directory,'out/scene.json'),directory);
    const saved = JSON.parse(await readFile(path,'utf8'));
    assert.ok(saved.geometry.label.attributes.POSITION);
    assert.deepEqual(saved.nodes[0].geometry.attributes,saved.geometry.label.attributes);
    assert.equal(saved.geometry.label.font,undefined);
    assert.equal(saved.composition.view.inputs.lut.image.file,'../lut.exr');
    assert.equal(saved.images.capture.composition.compositor[0].inputs.texture.image.file,'../nested.exr');
    assert.equal(options.font,join(directory,'fonts/a.ttf'));
    assert.deepEqual(await readFile(options.font),font);
  } finally { await rm(directory,{recursive:true,force:true}); }
});

test('numeric program and material tracks preserve the shared scene IR', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'vork-input-tracks-'));
  const animation = [{ tracks: [
    { path: ['materials', 'paint', 'inputs', 'emission_color'], keys: [[0, [1, 0, 0]], [1, [0, 0, 1]]], interpolation: 'LINEAR' },
    { path: ['materials', 'sky', 'inputs', 'sun_elevation'], keys: [[0, 0.2], [1, 0.8]] },
    { path: ['materials', 'fog', 'inputs', 'density'], keys: [[0, 0.1], [1, 0.5]], interpolation: 'STEP' },
  ] }];
  const scene = { version: 2, nodes: [], materials: {
    paint: {}, sky: { program: 'std/sky' }, fog: { program: 'std/fog' },
  }, animation };
  try {
    const saved = JSON.parse(await readFile(await sdk.writeScene(scene, join(directory, 'scene.json')), 'utf8'));
    assert.deepEqual(saved.animation, animation);
    assert.deepEqual(saved.materials, scene.materials);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('frame selection uses flags, never --set keys', () => {
  for (const key of ['frame_start','frame_end'])
    assert.throws(() => renderArguments('shot.json', {out:'a.png', set:{[key]:1}}), /unknown option/);
});

 test('regeneration retires only obsolete owned sidecars and respects umask', async () => {
  const directory=await mkdtemp(join(tmpdir(),'vork-sdk-cleanup-'));
  const path=join(directory,'shot.json');
  const scene=n=>({version:2,nodes:[],buffers:{x:{type:'f32',data:Array(1024).fill(n)}}});
  try {
   const unrelated=join(directory,'shot.json.buffers.00000000-0000-0000-0000-000000000000.bin');
   await writeFile(unrelated,'unrelated');
   await sdk.writeScene({version:2,nodes:[],programs:{keep:{file:unrelated}}},path,directory);
   for(let n=0;n<3;n++) {
    await sdk.writeScene(scene(n),path);
    const saved=JSON.parse(await readFile(path,'utf8'));
    const files=(await readdir(directory)).filter(f=>f.endsWith('.bin'));
    assert.equal(files.length,2);
    assert.equal((await stat(join(directory,saved.buffers.x.file))).mode&0o777,0o666&~process.umask());
   }
   const saved=JSON.parse(await readFile(path,'utf8'));
   await symlink(saved.buffers.x.file,join(directory,'buffer-alias.bin'));
   await sdk.writeScene({version:2,nodes:[],images:{keep:{file:'buffer-alias.bin'}}},path,directory);
   assert.equal((await readFile(join(directory,saved.buffers.x.file))).length,4096);
   assert.equal(await readFile(unrelated,'utf8'),'unrelated');
  } finally {await rm(directory,{recursive:true,force:true});}
 });

test('SDK preserves explicit geometry stage order', async () => {
 const directory=await mkdtemp(join(tmpdir(),'vork-stage-order-'));
 const scene={version:2,nodes:[],geometry_stages:['displace','deform','compute']};
 try {
  const path=await sdk.writeScene(scene,join(directory,'scene.json'));
  assert.deepEqual(JSON.parse(await readFile(path,'utf8')).geometry_stages,scene.geometry_stages);
 } finally {await rm(directory,{recursive:true,force:true});}
});


test('dense fields round trip through check below, at and above 4 KiB', async () => {
  const binary=process.env.VORK_BIN ?? resolve(process.env.CARGO_TARGET_DIR ?? '../../target','release','vork');
  const directory=await mkdtemp(join(tmpdir(),'vork-bb107-'));
  try {
    for(const x of [15,16,17]) {
      const data=Array.from({length:x*8*8},(_,i)=>i%x-(x-1)/2);
      const scene={version:2,buffers:{field:{type:'f32',dims:[x,8,8],data}},
        geometry:{shape:{topology:'implicit',attributes:{field:'field'}}},
        nodes:[{type:'mesh',geometry:'shape'},{type:'camera'}]};
      const path=await sdk.writeScene(scene,join(directory,`scene-${x}.json`));
      const saved=JSON.parse(await readFile(path,'utf8')).buffers.field;
      assert.deepEqual(saved.dims,[x,8,8]);
      if(x<16) assert.deepEqual(saved.data,data);
      else {
        assert.equal(saved.count,data.length);
        assert.equal(saved.offset,0);
        const bytes=await readFile(join(directory,saved.file));
        assert.deepEqual(Array.from({length:data.length},(_,i)=>bytes.readFloatLE(i*4)),data);
      }
      const result=spawnSync(binary,['check',path],{encoding:'utf8'});
      assert.equal(result.status,0,result.stderr ?? String(result.error));
      assert.deepEqual(scene.buffers.field.data,data);
      if(x>=16) {
        scene.buffers.field.dims=[2,2,2];
        await assert.rejects(()=>sdk.writeScene(scene,join(directory,'invalid.json')),/dims product/);
      }
    }
  } finally {await rm(directory,{recursive:true,force:true});}
});
