#!/usr/bin/env python3
"""Author After Rain's original geometry, proxies, lighting and image graph."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from beacon import write as write_beacon,SCALE

HERE=Path(__file__).resolve().parent

def write(media):
    media=media.resolve()
    for file in ['camera.json','plate.mp4']:
        if not (media/file).is_file():raise FileNotFoundError(f'Prepare missing input: {media/file}')
    camera=json.loads((media/'camera.json').read_text())
    base=np.array(camera['base']);size=SCALE
    write_beacon(media/'beacon.json',camera['wall_x']);beacon=json.loads((media/'beacon.json').read_text())
    for name in ['composite.slang','wet.slang']:shutil.copy2(HERE/name,media/name)
    buffers={};geometry={};nodes=[];materials={}
    def mesh(name,p,n,idx,uv=None):
        attrs={}
        for key,data,kind,suffix in [('POSITION',p,'vec3','p'),('NORMAL',n,'vec3','n'),('TEXCOORD_0',uv,'vec2','uv')]:
            if data is not None:buffers[name+'_'+suffix]={'type':kind,'data':data};attrs[key]=name+'_'+suffix
        buffers[name+'_i']={'type':'u32','data':idx};geometry[name]={'attributes':attrs,'indices':name+'_i'}
    def quad(name,p,normal):
        cross=np.cross(np.array(p[1])-p[0],np.array(p[2])-p[0])
        indices=[0,1,2,0,2,3] if np.dot(cross,normal)>0 else [0,2,1,0,3,2]
        mesh(name,p,[normal]*4,indices)
    wallx=camera['wall_x']
    quad('road',[[-5,0,4],[-5,0,-40],[wallx,0,-40],[wallx,0,4]],[0,1,0])
    quad('wall_receiver',[[wallx,0,-12],[wallx,5,-12],[wallx,5,-30],[wallx,0,-30]],[-1,0,0])
    materials['road']={'program':'wet','inputs':{}}
    materials['brick']={'inputs':{'base_color':[.08,.08,.085],'specular_roughness':.4}}
    for name,mat in [('road','road'),('wall_receiver','brick')]:
        nodes.append(dict(type='mesh',name=name,geometry=name,materials=[mat],is_shadow_catcher=True))
    first=camera['poses'][0]
    materials['camera_optics']={'program':'std/camera','inputs':dict(
        lens=camera['intrinsics']['f']*36/1920,sensor_width=36,sensor_height=20.25,
        clip_start=.05,clip_end=100,shift=[-.5/1920,0])}
    nodes.append(dict(type='camera',name='camera',translation=first['position'],rotation=first['rotation'],
        material='camera_optics',
        children=[]))
    for name,color,power in [('cyan',[.015,.65,1],20),('amber',[1,.29,.035],5),('metal',[0,0,0],0),('orbit',[.015,.65,1],7)]:
        m=beacon[name];mesh(name,m['positions'],m['normals'],m['indices'])
        materials[name]={'inputs':{'base_weight':0,'specular_weight':0,'emission_color':color,'emission_luminance':power}}
        if name=='metal':materials[name]={'inputs':{'base_color':[.025,.035,.045],'base_metalness':.75,'specular_roughness':.28}}
        translation=base
        nodes.append(dict(type='mesh',name=name,geometry=name,materials=[name],translation=translation.tolist(),scale=[size]*3))
    # Single-exposure ambient estimate; no measured HDR environment exists.
    nodes.append(dict(type='light',name='street_ambient',shape='environment',color=[.12,.20,.20],energy=.12))
    imgs={'plate':{'file':'plate.mp4'}}
    # No built-in bloom/pyramid: each blur is this example's own graph program.
    common={'plate':{'image':'plate'}}
    def comp(name,mode,**inputs):
        # Every opaque Image input has an explicit binding, including modes
        # that do not consume it. Unused bindings point at the existing plate.
        bindings={key:{'image':'plate'} for key in ['source','plate','catcher','integrated','glow','wide']}
        bindings.update(inputs)
        materials[name]={'program':'composite','inputs':dict(mode=mode,**bindings)}
    comp('save',0)
    comp('catcher_filter',5,catcher={'channel':'shadow_catcher'})
    comp('over',1,source={'step':'cg'},catcher={'step':'reflection'},**common)
    comp('highlight',2,source={'step':'cg'},integrated={'step':'integrated'},**common)
    comp('horizontal',3,direction=[1,0]);comp('vertical',3,direction=[0,1])
    comp('finish',4,integrated={'step':'integrated'},glow={'step':'glow'},wide={'step':'wide'},**common)
    graph=[dict(material='save',name='cg'),
        dict(material='catcher_filter',name='reflection'),dict(material='over',name='integrated'),
        dict(material='highlight',resolution=.5),dict(material='horizontal',resolution=.25),
        dict(material='vertical',name='glow',resolution=.25),
        dict(material='horizontal',resolution=.0625),dict(material='vertical',name='wide',resolution=.0625),'finish']
    # The seed rotates; wall brackets and lettering stay fixed.
    tracks=[dict(path=['nodes','orbit','rotation'],keys=[[0,[0,0,0,1]],[5,[0, .258819,0,.965926]]])]
    for field in ['translation','rotation']:
        key='position' if field=='translation' else 'rotation'
        tracks.append(dict(path=['nodes','camera',field],keys=[[p['time'],p[key]] for p in camera['poses']]))
    clock_keys=[[n/24,(n+.1)/24] for n in range(120)]
    tracks.append(dict(path=['images','plate','time'],interpolation='STEP',keys=clock_keys))
    scene=dict(version=2,composition=dict(space='3d',view={'program':'std/standard'},compositor=graph),render=dict(resolution=[1920,1080],fps=24,duration=5,background='transparent',
        motion_blur={'shutter':0}),
        images=imgs,programs={n:{'file':n+'.slang'} for n in ['composite','wet']},
        materials=materials,buffers=buffers,geometry=geometry,nodes=nodes,animation=[dict(tracks=tracks)])
    for clean in [False,True]:
        scene['materials']['finish']['inputs']['wipe']=0 if clean else 1
        (media/('clean.json' if clean else 'scene.json')).write_text(json.dumps(scene,separators=(',',':'))+'\n')
    return scene

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--media',type=Path,required=True);a=p.parse_args();write(a.media)
