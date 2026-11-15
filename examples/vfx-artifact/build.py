"""Original split-seed artifact and plate-matched proxy scene (all media external)."""
import argparse
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from track import on_plane, project, FOCAL

HERE=Path(__file__).resolve().parent


def build(media):
    media=media.resolve()
    if HERE.parents[1] in media.parents: raise ValueError('media must be outside checkout')
    solve=json.loads((media/'camera-solve.json').read_text())
    chunks=[]; buffers={}; geometry={}; nodes=[]
    def buffer(name,data,kind):
        data=np.asarray(data,dtype='<u4' if kind=='u32' else '<f4')
        offset=sum(len(x) for x in chunks)
        chunks.append(data.tobytes())
        buffers[name]=dict(type=kind,file='geometry.bin',offset=offset,count=len(data))
        return name
    def mesh(name,p,idx,material,**kw):
        p=np.asarray(p,float); idx=np.asarray(idx,int).reshape(-1,3)
        n=np.zeros_like(p)
        fn=np.cross(p[idx[:,1]]-p[idx[:,0]],p[idx[:,2]]-p[idx[:,0]])
        for k in range(3): np.add.at(n,idx[:,k],fn)
        n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-10)
        geometry[name]=dict(attributes=dict(POSITION=buffer(name+'_p',p,'vec3'),NORMAL=buffer(name+'_n',n,'vec3')),
                            indices=buffer(name+'_i',idx.reshape(-1),'u32'))
        nodes.append(dict(type='mesh',name=name,geometry=name,materials=[material],**kw))
    camera=np.array(solve[0]['translation'])
    def on_table(x,y):
        return on_plane([[x,y]])[0]
    base=on_table(730,925)
    # Keep the approved plate lighting and focus fixed as the seed moves inward.
    lighting_base=on_table(740,950)
    seed_scale=.75
    # Six unequal, closed, lenticular shells twist around a displaced core.
    for petal in range(6):
        p=[];tri=[];rings=48;sides=16
        angle=petal*math.tau/6+.11*math.sin(petal*3)
        height=.225*(1+.11*math.sin(petal*2.1))
        for j in range(rings+1):
            t=j/rings; a=angle+.72*t
            radius=.033+.027*math.sin(math.pi*t)+.014*t*t
            width=.025*max(math.sin(math.pi*t),.015)**.7
            thickness=.006*max(math.sin(math.pi*t),.015)**.6
            center=base+np.array([radius*math.cos(a),.012+height*t,radius*math.sin(a)])
            for k in range(sides):
                b=k*math.tau/sides
                p.append(center+np.array([-math.sin(a)*width*math.cos(b)+math.cos(a)*thickness*math.sin(b),0,
                                          math.cos(a)*width*math.cos(b)+math.sin(a)*thickness*math.sin(b)]))
                if j<rings:
                    q=j*sides+k; z=j*sides+(k+1)%sides
                    tri.extend([[q,q+sides,z],[z,q+sides,z+sides]])
        mesh('shell_'+str(petal),p,tri,'ceramic' if petal%3 else 'metal')
    def ellipsoid(name,center,scale,material,**kw):
        p=[];idx=[];rings=32;sides=48
        for j in range(rings+1):
            a=math.pi*j/rings
            for k in range(sides):
                b=math.tau*k/sides
                p.append(np.array(center)+np.array(scale)*[math.sin(a)*math.cos(b),math.cos(a),math.sin(a)*math.sin(b)])
                if j<rings:
                    q=j*sides+k;z=j*sides+(k+1)%sides
                    idx.extend([[q,z,q+sides],[z,z+sides,q+sides]])
        mesh(name,p,idx,material,**kw)
    ellipsoid('core',base+[.003,.12,0],[.023,.087,.024],'core')
    ellipsoid('foot',base+[0,.009,0],[.047,.009,.042],'metal')
    # A luminous oval recessed between the shell tips accents the core.
    ellipsoid('upper_aperture',base+[0,.195,-.012],[.012,.029,.012],'aperture')
    # Fixed world placement: no per-frame artifact compensation.
    nodes=[dict(type='node',name='seed',children=nodes,scale=[seed_scale]*3,
                translation=(base*(1-seed_scale)).tolist())]
    reveal_center=[]
    for row in solve:
        pose=np.r_[Rotation.from_quat(row['rotation']).as_rotvec(),row['translation']]
        uv=project(pose,(base+[0,.12*seed_scale,0])[None])[0]/[1920,1080]
        reveal_center.append([row['time'],uv.tolist()])
    # A finite tabletop proxy receives direct/contact shadows and light spill.
    tabletop=[on_table(x,y) for x,y in [(510,933),(1580,875),(1920,960),(1920,1040),(1305,1015),(510,955)]]
    mesh('table',tabletop,[[0,k+1,k] for k in range(1,len(tabletop)-1)],
         'glass_table',is_shadow_catcher=True)
    # Use the keyboard solve's estimated 15 mm height for the laptop deck.
    # It receives the seed's light separately from the tabletop.
    deck=on_plane([[850,875],[1105,782],[1390,801],[1368,919]],.015)
    mesh('laptop_deck',deck,[[0,2,1],[0,3,2]],'laptop',is_shadow_catcher=True)
    # Animated depth relief: fixed camera-space grid, video silhouette, one
    # morph per plate frame. Both normal and position follow the plate proxy.
    import cv2
    yy,xx=np.mgrid[0:1081:18,0:1921:24]
    uv=np.stack([xx/1920,yy/1080],axis=-1).reshape(-1,2)
    grid_shape=xx.shape
    positions=[];normals=[]
    for row in solve:
        z=cv2.imread(str(media/'calibrated-depth'/f'{row["frame"]:06d}.png'),cv2.IMREAD_UNCHANGED).astype(float)/65535*8
        # Anchor the relative relief to the sleeve resting on the solved desk.
        # Its first-frame centre (740, 840) is about 4 cm above the tabletop.
        # Keep the relative shape; the old camera-space hand clamp put the
        # entire forearm almost a metre in front of its actual desk support.
        anchor=on_plane([[740,840]],.04)[0]
        first_rotation=Rotation.from_quat(solve[0]['rotation']).as_matrix()
        anchor_depth=-((anchor-camera)@first_rotation)[2]
        if not positions:
            depth_offset=anchor_depth-z[420,370]
        zz=cv2.remap(z.astype('float32'),(xx/2).astype('float32'),(yy/2).astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)+depth_offset
        f=row['lens']/36*1920
        local=np.stack([(xx-960)/f*zz,-(yy-540)/f*zz,-zz],axis=-1)
        r=Rotation.from_quat(row['rotation']).as_matrix()
        world=local @ r.T+np.array(row['translation'])
        dy,dx=np.gradient(world,axis=(0,1))
        n=np.cross(dy,dx);n/=np.maximum(np.linalg.norm(n,axis=2)[...,None],1e-10)
        positions.append(world.reshape(-1,3));normals.append(n.reshape(-1,3))
    idx=[];cols=grid_shape[1]
    for y in range(grid_shape[0]-1):
        for x in range(cols-1):
            q=y*cols+x;idx.extend([[q,q+cols,q+1],[q+1,q+cols,q+cols+1]])
    mesh('person_proxy',positions[0],idx,'skin',is_shadow_catcher=True)
    geometry['person_proxy']['attributes']['TEXCOORD_0']=buffer('person_uv',uv,'vec2')
    geometry['person_proxy']['attributes']['NORMAL']=buffer('person_normal',normals[0],'vec3')
    geometry['person_proxy']['morphs']={}
    proxy_tracks=[]
    for frame in range(1,len(solve)):
        name=f'frame_{frame:03d}'
        geometry['person_proxy']['morphs'][name]=dict(
            POSITION=buffer(name+'_position',positions[frame]-positions[0],'vec3'),
            NORMAL=buffer(name+'_normal',normals[frame]-normals[0],'vec3'))
        keys=[[0.,0.]]
        if frame>1:keys.append([(frame-1)/25,0.])
        keys.append([frame/25,1.])
        if frame<len(solve)-1:keys.append([(frame+1)/25,0.])
        proxy_tracks.append(dict(path=['nodes','person_proxy','morph_weights',name],keys=keys))
    materials=dict(
        ceramic=dict(inputs=dict(base_color=[.32,.27,.20],specular_roughness=.27,base_metalness=.35,coat_weight=.6,coat_roughness=.18)),
        metal=dict(inputs=dict(base_color=[.14,.09,.046],base_metalness=.93,specular_roughness=.26)),
        camera_optics=dict(program='std/camera',inputs=dict(lens=FOCAL*36/1920,sensor_width=36,clip_start=.05,clip_end=50,
                          use_dof=True,aperture_fstop=4.,focus_distance=float(np.linalg.norm(lighting_base+[0,.04,0]-camera)))),
        core=dict(inputs=dict(base_color=[.025,.17,.13],emission_color=[.055,1,.66],emission_luminance=18.)),
        aperture=dict(double_sided=False,inputs=dict(base_color=[.025,.17,.13],emission_color=[.055,1,.66],emission_luminance=80.)),
        glass_table=dict(inputs=dict(base_color=[.28,.34,.30],specular_roughness=.24,specular_weight=.5)),
        laptop=dict(inputs=dict(base_color=[.38,.40,.42],specular_roughness=.32,specular_weight=.5)),
        skin=dict(program='skin',inputs=dict(person=dict(image='person'))),
        combine=dict(program='combine',inputs=dict(plate=dict(image='plate'),
            catcher=dict(step='ratio_v'),cg=dict(step='cg'),hand=dict(image='hand'))),
    )
    nodes.extend([
        dict(type='camera',name='camera',translation=camera.tolist(),rotation=solve[0]['rotation'],material='camera_optics'),
        dict(type='light',name='practical',shape='rectangle',translation=[1.4,1.8,-.6],
             target=lighting_base.tolist(),size=.65,size_y=.8,energy=48,color=[1,.54,.25]),
        dict(type='light',name='laptop_fill',shape='rectangle',translation=(lighting_base+[.2,.34,-.25]).tolist(),
             target=(lighting_base+[0,.2,.1]).tolist(),size=.3,size_y=.2,energy=2,color=[.43,.72,1]),
        dict(type='light',name='room',shape='environment',material={'program':'std/environment_image','inputs':{'image':{'image':'room'}}},energy=.45),
    ])
    tracks=proxy_tracks+[dict(path=['nodes','camera','rotation'],keys=[[x['time'],x['rotation']] for x in solve]),
            dict(path=['materials','camera_optics','inputs','lens'],keys=[[x['time'],x['lens']] for x in solve]),
            dict(path=['nodes','camera','translation'],keys=[[x['time'],x['translation']] for x in solve])]
    pulse=[[float(t),18.*(1+.07*math.sin(t*2.4))] for t in np.arange(0,8.01,.2)]
    tracks.append(dict(path=['materials','core','inputs','emission_luminance'],keys=pulse))
    tracks.append(dict(path=['materials','aperture','inputs','emission_luminance'],
                       keys=[[t,v*80/18] for t,v in pulse]))
    scene=dict(version=2,composition=dict(space='3d',view=dict(program='std/standard'),compositor=['combine']),render=dict(resolution=[1920,1080],fps=25,duration=8,
                background='transparent',
                motion_blur=dict(shutter=0.)),
        images=dict(plate=dict(file='plate.mp4'),person=dict(file='person.mkv',color_space='Non-Color'),hand=dict(file='hand.mkv',color_space='Non-Color'),room=dict(file='room.hdr')),
        programs=dict(combine=dict(file='combine.slang'),skin=dict(file='skin.slang')),materials=materials,
        buffers=buffers,geometry=geometry,nodes=nodes,animation=[dict(name='shot',tracks=tracks)])
    # Save beauty before filtering the catcher; all image processing is authored here.
    scene['programs'].update(halo=dict(file='halo.slang'),finish=dict(file='finish.slang'),ratio=dict(file='ratio.slang'),capture=dict(file='capture.slang'))
    materials['capture']=dict(program='capture')
    materials.update(ratio_h=dict(program='ratio',inputs=dict(source=dict(channel='shadow_catcher'),plate=dict(image='plate'),person=dict(image='person'),direction=[1,0])),
                     ratio_v=dict(program='ratio',inputs=dict(source=dict(step='ratio_h'),plate=dict(image='plate'),person=dict(image='person'),direction=[0,1])))
    materials.update(halo=dict(program='halo',inputs=dict(beauty=dict(step='beauty'),plate=dict(image='plate'))),
                     finish=dict(program='finish',inputs=dict(beauty=dict(step='beauty'),halo=dict(step='halo'),
                        plate=dict(image='plate'),artifact_uv=reveal_center[0][1],reveal=1.)))
    tracks.append(dict(path=['materials','finish','inputs','artifact_uv'],keys=reveal_center))
    scene.setdefault('composition', {'space': '3d'})['compositor'] = [dict(name='cg',material='capture'),dict(name='ratio_h',material='ratio_h',resolution=.5),dict(name='ratio_v',material='ratio_v',resolution=.5),
                                   dict(name='beauty',material='combine'),dict(name='halo',material='halo',resolution=.25),
                                   dict(material='finish')]
    (media/'geometry.bin').write_bytes(b''.join(chunks))
    (media/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    for f in HERE.glob('*.slang'): shutil.copy2(f,media/f.name)
    clean=json.loads(json.dumps(scene))
    clean['materials']['finish']['inputs']['reveal']=0.
    (media/'clean.json').write_text(json.dumps(clean,indent=2)+'\n')
    (media/'proxy-layout.json').write_text(json.dumps(dict(artifact=base.tolist(),camera=camera.tolist(),seed_scale=seed_scale,contact_points=[(base+[x,0,z]).tolist() for x,z in [(0,0),(.047*seed_scale,0),(-.047*seed_scale,0),(0,.042*seed_scale),(0,-.042*seed_scale)]]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('media',type=Path)
    a=p.parse_args()
    build(a.media)
