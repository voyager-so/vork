#!/usr/bin/env python3
"""Real geometry, finite emitters and a twelve-second descending flyby.

The program owns only authoring maths. Every camera, BSDF, reflection and
occlusion path is integrated by vork's propagation provider.
"""
import json
import math
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from compositor_graph import bloom
DURATION = 12
FPS = 24


def quaternion(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def camera(t):
    u = t / DURATION
    smooth = u*u*(3-2*u)
    angle = -.43 + .95*smooth
    radius = 78 - 22*math.sin(math.pi*u)**1.4
    height = 6.8 + 4.0*math.sin(math.pi*u) + 2.5*u
    position = [radius*math.sin(angle), height, radius*math.cos(angle)]
    pitch = -math.atan2(height, radius)
    rotation = quaternion([0, math.sin(angle/2), 0, math.cos(angle/2)],
                          [math.sin(pitch/2), 0, 0, math.cos(pitch/2)])
    roll = .1*math.sin(2*math.pi*u)
    rotation = quaternion(rotation, [0, 0, math.sin(roll/2), math.cos(roll/2)])
    return position, rotation, angle, 43-7*math.sin(math.pi*u)


def areal_to_isotropic(r):
    return (r-1+math.sqrt(r*(r-2)))/2


def ship_bounds():
    # Bounds written from the decimated asset's world-space vertices.
    return json.loads((HERE/'assets/corinth/bounds.json').read_text())


class Mesh:
    def __init__(self):
        self.positions, self.normals, self.indices = [], [], []

    def triangle(self, a, b, c):
        x = [b[k]-a[k] for k in range(3)]
        y = [c[k]-a[k] for k in range(3)]
        n = [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]]
        length = math.sqrt(sum(v*v for v in n))
        n = [v/length for v in n]
        start = len(self.positions)
        self.positions.extend([a, b, c])
        self.normals.extend([n]*3)
        self.indices.extend([start, start+1, start+2])

    def write(self, name):
        # One retained native geometry, not one draw or hierarchy per dust grain.
        file = HERE/'assets'/f'{name}.bin'
        data = bytearray()
        buffers = {}
        for channel, values, kind, width in [('position', self.positions, 'vec3', 3),
                                             ('normal', self.normals, 'vec3', 3),
                                             ('index', self.indices, 'u32', 1)]:
            offset = len(data)
            flat = values if width == 1 else [x for row in values for x in row]
            data.extend(struct.pack('<'+('I' if width == 1 else 'f')*len(flat), *flat))
            buffers[name+'_'+channel] = dict(type=kind, file='assets/'+file.name, offset=offset, count=len(values))
        file.write_bytes(data)
        return buffers, {'attributes': {'POSITION': name+'_position', 'NORMAL': name+'_normal'}, 'indices': name+'_index'}


def annulus():
    mesh = Mesh()
    inner, outer = areal_to_isotropic(6), areal_to_isotropic(24)
    rings, sectors = 12, 192
    for j in range(rings):
        r0 = inner*(outer/inner)**(j/rings)
        r1 = inner*(outer/inner)**((j+1)/rings)
        for k in range(sectors):
            t0, t1 = 2*math.pi*k/sectors, 2*math.pi*(k+1)/sectors
            a, b = [r0*math.cos(t0), 0, r0*math.sin(t0)], [r1*math.cos(t0), 0, r1*math.sin(t0)]
            c, d = [r1*math.cos(t1), 0, r1*math.sin(t1)], [r0*math.cos(t1), 0, r0*math.sin(t1)]
            # A finite photosphere, tapered to the inner and outer rims.
            for point in (a,b,c,d):
                radius = math.hypot(point[0],point[2])
                point[1] = 1.2*math.sin(math.pi*(radius-inner)/(outer-inner))
            mesh.triangle(a, c, b)
            mesh.triangle(a, d, c)
            a,b,c,d = [[p[0],-p[1],p[2]] for p in (a,b,c,d)]
            mesh.triangle(a, b, c)
            mesh.triangle(a, c, d)
    return mesh


def debris(seed, count, near, far):
    rng, mesh = random.Random(seed), Mesh()
    corners = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    faces = [(0,2,4),(4,2,1),(1,2,5),(5,2,0),(4,3,0),(1,3,4),(5,3,1),(0,3,5)]
    def unit(p):
        length=math.sqrt(sum(x*x for x in p))
        return [x/length for x in p]
    for j in range(count):
        centre = [rng.uniform(-22, 12), rng.uniform(-4, 7), rng.uniform(near, far)]
        radius = rng.uniform(.012, .045)
        stretch = [rng.uniform(.8, 1.2) for _ in range(3)]
        for face in faces:
            a,b,c = [corners[k] for k in face]
            ab,bc,ca = [unit([x+y for x,y in zip(u,v)]) for u,v in [(a,b),(b,c),(c,a)]]
            for tri in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]:
                mesh.triangle(*[[centre[k]+radius*p[k]*stretch[k] for k in range(3)] for p in tri])
                mesh.normals[-3:] = [unit([p[k]/stretch[k] for k in range(3)]) for p in tri]
    return mesh


def _moon_mesh():
 radius,sectors,rings=3.5,64,32
 positions=[];normals=[];uv=[];indices=[]
 for y in range(rings+1):
  for x in range(sectors+1):
   a=y*math.pi/rings;b=(0 if x==sectors else x)*math.tau/sectors
   n=[0.,1.,0.] if y==0 else [0.,-1.,0.] if y==rings else [math.sin(a)*math.cos(b),math.cos(a),math.sin(a)*math.sin(b)]
   positions.append([v*radius for v in n]);normals.append(n);uv.append([1-x/sectors,y/rings])
 for y in range(rings):
  for x in range(sectors):
   a=y*(sectors+1)+x;b=a+sectors+1
   if y<rings-1:indices.extend([a,b+1,b])
   if y>0:indices.extend([a,a+1,b+1])
 return dict(attributes={k:dict(type=t,data=v) for k,t,v in [('POSITION','vec3',positions),('NORMAL','vec3',normals),('TEXCOORD_0','vec2',uv)]},indices=dict(type='u32',data=indices))

def generate():
    scene = {'version': 2,
             "composition":{"space":"3d","view":{'program': 'std/agx'},'compositor':bloom({'threshold':1.,'strength':.045,'size':.28},[1920,1080])},'render': {'resolution': [1920,1080], 'duration': DURATION, 'fps': FPS,
                        'background': 'environment'},
             'propagation': {'program': 'propagation', 'bounds': [[-1000]*3, [1000]*3]},
             'programs': {name: {'file': name+'.slang'} for name in ['propagation', 'disk', 'environment', 'moon']},
             'materials': {'disk': {'program': 'disk'},
                           'world': {'program': 'environment'}, 'moon': {'program': 'moon'},
                           'rock': {'program': 'std/openpbr', 'inputs': {'base_color': [.5,.32,.15], 'specular_roughness': .32, 'emission_color': [1,.35,.08], 'emission_luminance': .18}}},
             'buffers': {}, 'geometry': {'moon': _moon_mesh()},
             'nodes': [], 'animation': [{'tracks': []}]}
    for name, mesh in [('annulus',annulus()), ('near_dust',debris(418,200,25,53)), ('far_dust',debris(631,350,-20,25))]:
        buffers, geometry = mesh.write(name)
        scene['buffers'].update(buffers)
        scene['geometry'][name] = geometry
        scene['nodes'].append({'type': 'mesh', 'name': name, 'geometry': name, 'materials': ['disk' if name=='annulus' else 'rock']})
    centre, extent = ship_bounds()
    scale = 16/extent
    ship_position = [-20,6,10]
    def unit(v):
        length = math.sqrt(sum(x*x for x in v))
        return [x/length for x in v]
    view = unit([camera(0)[0][k]-ship_position[k] for k in range(3)])
    light = unit([p-ship_position[k] for k,p in enumerate([-12,0,15])])
    normal = unit([a+b for a,b in zip(view,light)])
    tilt = [normal[2],0,-normal[0],1+normal[1]]
    tilt = [x/math.sqrt(sum(v*v for v in tilt)) for x in tilt]
    ship_rotation = quaternion(tilt,[0,math.sin(-.6),0,math.cos(-.6)])
    scene['nodes'].extend([
        {'type': 'light', 'name': 'stars', 'shape': 'environment', 'material': 'world', 'energy': 1},
        {'type': 'mesh', 'name': 'moon', 'geometry': 'moon', 'materials': ['moon']},
        {'type': 'node', 'name': 'corinth', 'translation': ship_position, 'rotation': ship_rotation,
         'children': [{'type': 'asset', 'name': 'cargo_ship', 'file': 'assets/corinth/scene.gltf',
                       'translation': [-v*scale for v in centre], 'scale': [scale]*3}]},
        {'type':'camera','name':'camera','translation':camera(0)[0],'rotation':camera(0)[1],'material':{'program':'std/camera','inputs':{'lens':camera(0)[3],'clip_end':4000}}}])
    scene['materials']['camera-optics'] = scene['nodes'][-1]['material']
    scene['nodes'][-1]['material'] = 'camera-optics'
    tracks = scene['animation'][0]['tracks']
    times = [j/FPS for j in range(DURATION*FPS+1)]
    for field, component in [('translation',0), ('rotation',1)]:
        tracks.append({'path': ['nodes','camera',field], 'keys': [[t,camera(t)[component]] for t in times]})
    tracks.append({'path':['materials','camera-optics','inputs','lens'], 'keys':[[t,camera(t)[3]] for t in times]})
    positions = []
    for t in times:
        eye, _, angle, _ = camera(t)
        moon_y = -90*eye[1]/math.hypot(eye[0],eye[2]) + 20*math.cos(math.pi*t/DURATION)
        offset = 28*math.cos(math.pi*t/DURATION)
        positions.append([t, [-90*math.sin(angle)+offset*math.cos(angle), moon_y, -90*math.cos(angle)-offset*math.sin(angle)]])
    tracks.extend([{'path': ['nodes','moon','translation'], 'keys': positions},
                   {'path': ['nodes','near_dust','translation'], 'keys': [[0,[0,0,0]], [DURATION,[.7,.15,-.5]]]},
                   {'path': ['nodes','far_dust','translation'], 'keys': [[0,[0,0,0]], [DURATION,[-.3,.3,.5]]]}])
    (HERE/'scene.json').write_text(json.dumps(scene, separators=(',',':'))+'\n')


if __name__ == '__main__':
    generate()
