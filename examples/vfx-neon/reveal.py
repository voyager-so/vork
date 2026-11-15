"""Build the inexpensive reveal graph over a lossless clean image sequence."""
import argparse,json,shutil
from pathlib import Path

def write(media,frames,extension='png'):
    media=media.resolve();frames=frames.resolve()
    shutil.copy2(Path(__file__).with_name('reveal.slang'),media/'reveal.slang')
    scene=dict(version=2,composition=dict(space='3d',view={'program':'std/standard'},compositor=[dict(material='read',name='clean'),'wipe']),render=dict(resolution=[1920,1080],fps=24,duration=5),
        nodes=[dict(type='camera',name='camera')],programs={'reveal':{'file':'reveal.slang'}},
        images={'clean':{'file':str(frames/('%06d.'+extension))},'plate':{'file':'plate.mp4'}},
        materials={'read':{'program':'reveal','inputs':{'source':{'image':'clean'},'plate':{'image':'plate'}}},
                   'wipe':{'program':'reveal','inputs':{'source':{'step':'clean'},'plate':{'image':'plate'},'mode':1}}})
    keys=[[0,.1/24]]+[[float((n-.5)/24),float((n+.1)/24)] for n in range(1,121)]
    scene['animation']=[dict(tracks=[dict(path=['images',name,'time'],interpolation='STEP',keys=keys) for name in ['clean','plate']])]
    (media/'reveal.json').write_text(json.dumps(scene,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--media',type=Path,required=True);p.add_argument('--frames',type=Path,required=True);a=p.parse_args();write(a.media,a.frames)
