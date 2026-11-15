#!/usr/bin/env python3
"""Pinned target-family data. No Blender renderer code is imported.
uv run --with opencolorio==2.6.0 --with numpy --with OpenEXR==3.4.0 \
  scripts/color/bake-device-views.py PROTECTED_REFERENCE_DIRECTORY --evidence OUTSIDE_REPO
Reference: Blender tag v5.2.1 colormanagement/config.ocio and its BSD-3-Clause
OCIO configuration data; see device-views-manifest.json and bundled notice.
"""
import json, hashlib, argparse
from pathlib import Path
import numpy as np
import PyOpenColorIO as ocio
import OpenEXR
root=Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference', type=Path)
parser.add_argument('--evidence', type=Path, required=True,
                    help='Directory outside the repository for validation measurements')
args = parser.parse_args()
ref = args.reference
assert not args.evidence.resolve().is_relative_to(root), 'Evidence must stay outside the repository'
args.evidence.mkdir(parents=True, exist_ok=True)
pinned = json.loads((root/'libraries/std/device-views-manifest.json').read_text())
assert ocio.__version__ == '2.6.0', ocio.__version__
# Verify every input, including the OCIO implementation used to derive the conversion.
for name, digest in pinned['source_sha256'].items():
    assert hashlib.sha256((ref/name).read_bytes()).hexdigest() == digest, name
c=ocio.Config.CreateFromFile(str(ref/'config.ocio'))
assert hashlib.sha256((ref/'config.ocio').read_bytes()).hexdigest()=='47a7d83e79c1d21f49ba6c505efe311da723471688c614b5c366e1da7eb8ea3a'
looks=['None']
lo=-12.47393; hi=12.5260688117
m=np.eye(3,dtype=np.float32);c.getProcessor('Linear Rec.709','Linear FilmLight E-Gamut').getDefaultCPUProcessor().applyRGB(m);matrix=m.T.tolist()
rng=np.random.default_rng(13579)
patches=np.r_[np.repeat(np.exp2(np.linspace(-16,15,125))[:,None],3,axis=1),np.exp2(rng.uniform(-12,12,(500,3))),[[0,0,0],[-.1,.3,1],[1,-.1,.2],[100,-1,2]]].astype(np.float32)
fixtures=[]; sizes={}
for target,view,name,cube in [('Display P3','AgX','agx-p3','AgX_Base_P3.cube'),('Rec.2100-PQ','AgX - HDR 1000 nits','agx-pq','AgX_Rec2100-HLG_p3_lim.cube'),('Rec.2100-HLG','AgX - HDR 1000 nits','agx-pq','AgX_Rec2100-HLG_p3_lim.cube')]:
    # Preserve the authoritative cube's original lattice and tetrahedral interpolation.
    # No resampling or baking of an already encoded SDR result.
    text=(ref/'luts'/cube).read_text().splitlines()
    n=int(next(l.split()[1] for l in text if l.startswith('LUT_3D_SIZE')));sizes[target]=n
    rgb=np.array([[float(v) for v in l.split()] for l in text if l and l[0] in '-.0123456789'],dtype=np.float32)
    image=np.ascontiguousarray(rgb.reshape(n,n,n,3).transpose(1,0,2,3).reshape(n,n*n,3))
    OpenEXR.File({'compression':OpenEXR.ZIP_COMPRESSION,'type':OpenEXR.scanlineimage},{'RGB':image}).write(str(root/f'libraries/std/{name}.exr'))
    display=ocio.DisplayViewTransform(src='Linear Rec.709',display=target,view=view);display.setLooksBypass(True)
    expected=patches.copy();c.getProcessor(display).getDefaultCPUProcessor().applyRGB(expected)
    fixture = dict(display=target, samples=[dict(input=a.tolist(),output=b.tolist()) for a,b in zip(patches,expected)])
    # Preserve the existing reference rows, including HLG's absent look field.
    if target != 'Rec.2100-HLG':
        fixture = dict(display=target, look='None', samples=fixture['samples'])
    fixtures.append(fixture)
# Standard reference values are independently produced by OCIO, not the runtime equations.
standard=[]
for target in ['Display P3','Rec.2100-PQ','Rec.2100-HLG']:
 t=ocio.DisplayViewTransform(src='Linear Rec.709',display=target,view='Standard');t.setLooksBypass(True)
 values=patches.copy()
 if target=='Rec.2100-HLG':
  # Device peak clipping precedes the nonlinear HLG OOTF/OETF.
  c.getProcessor('Linear Rec.709','Linear Rec.2020').getDefaultCPUProcessor().applyRGB(values)
  values=np.clip(values,0,10)
  c.getProcessor('Linear Rec.2020','Linear Rec.709').getDefaultCPUProcessor().applyRGB(values)
 c.getProcessor(t).getDefaultCPUProcessor().applyRGB(values)
 standard.append(dict(display=target,samples=[dict(input=a.tolist(),output=b.tolist()) for a,b in zip(patches,values)]))
# The config has a fixed HDR view, not a configurable diffuse-white control.
# Its 100-nit display-reference unit is NOT scene-linear unit white after AgX.
# Define this derived recipe explicitly: neutral scene input 1 has Y=203 cd/m².
# Insert only a scalar scene exposure, before the original AgX shoulder/gamut LUT.
# This preserves the source peak and never scales an already limited picture.
hdr_view = ocio.DisplayViewTransform(src='Linear Rec.709', display='Rec.2100-HLG',
                                    view='AgX - HDR 1000 nits')
hdr_view.setLooksBypass(True)
linear_view = ocio.GroupTransform([
    hdr_view,
    ocio.BuiltinTransform(style='DISPLAY - CIE-XYZ-D65_to_REC.2100-HLG-1000nit',
                          direction=ocio.TRANSFORM_DIR_INVERSE)])
linear_cpu = c.getProcessor(linear_view).getDefaultCPUProcessor()
def diffuse_nits(gain):
    value = np.full((1, 3), gain, dtype=np.float32)
    linear_cpu.applyRGB(value)
    return float(value[0, 1]) * 100.
left, right = 1., 2.
for _ in range(48):
    midpoint = (left + right) / 2.
    if diffuse_nits(midpoint) < 203.:
        left = midpoint
    else:
        right = midpoint
gain = (left + right) / 2.
exposure = ocio.MatrixTransform.Scale([gain, gain, gain, 1.])

# A scene exposure before the view is a shift of the cube's log2 input allocation,
# so the 203 recipe is the pinned HDR cube read with these bounds: the same
# lattice and the same tetrahedral function, and no second cube. Runtime MUST use
# these bounds. Reference processors below apply the exposure through OCIO.
stops = float(np.log2(gain))
recipe_lo, recipe_hi = lo - stops, hi - stops
new_patches = np.concatenate([patches, np.repeat(
    np.array([0.18, 1., 4., 16., 256., 1.e8], dtype=np.float32)[:,None], 3, axis=1)])
fixtures_203 = []
for target in ['Rec.2100-PQ', 'Rec.2100-HLG']:
    view = ocio.DisplayViewTransform(src='Linear Rec.709', display=target,
                                     view='AgX - HDR 1000 nits')
    view.setLooksBypass(True)
    expected = new_patches.copy()
    c.getProcessor(ocio.GroupTransform([exposure, view])).getDefaultCPUProcessor().applyRGB(expected)
    fixtures_203.append(dict(display=target, look='None', peak_cd_m2=1000,
        reference_white_cd_m2=203, samples=[dict(input=a.tolist(),output=b.tolist())
                                          for a,b in zip(new_patches,expected)]))
(root/'crates/vork-render/data/device-reference.json').write_text(json.dumps(
    dict(agx=fixtures,standard=standard,agx_203=fixtures_203),separators=(',',':'))+'\n')
manifest=dict(hdr_conversion='BT.2100 inverse HLG OETF, 1000 nit OOTF gamma 1.2; OCIO inverse surround floor 1e-4^(1/1.2)/3; ST 2084 encoding. OCIO v2.6.0 Displays.cpp and FixedFunctionOpCPU.cpp reference.',source='https://github.com/blender/blender/tree/v5.2.1/release/datafiles/colormanagement',ocio=ocio.__version__,license='BSD-3-Clause; see dist/licenses/ocio-configuration.txt',preset=dict(view='AgX - HDR 1000 nits',peak_cd_m2=1000,reference_white_cd_m2=100,gamut='P3 D65 limited in Rec.2020 PQ'),size=sizes,min_log2=lo,max_log2=hi,looks=looks,rec709_to_egamut=matrix,format='Original target cube samples; OpenEXR float RGB; x=N*b+r, y=g; tetrahedral interpolation in E-Gamut log2 coordinates; P3 cube returns gamma-2.4 P3, HDR cube returns HLG Rec.2020',source_sha256=pinned['source_sha256'],data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'libraries/std').glob('agx-*.exr')})
manifest['recipes'] = [
    dict(data='agx-pq.exr', size=57, peak_cd_m2=1000, reference_white_cd_m2=100,
         white_definition='Unmodified Blender preset; 100 cd/m² per display-reference unit, not per scene unit',
         min_log2=lo, max_log2=hi),
    dict(data='agx-pq.exr', size=sizes['Rec.2100-PQ'], peak_cd_m2=1000, reference_white_cd_m2=203,
         white_definition='Neutral scene-linear Rec.709 input [1,1,1] has display Y=203 cd/m²',
         scene_exposure_gain=gain, scene_exposure_stops=stops,
         min_log2=recipe_lo, max_log2=recipe_hi,
         derivation='Scalar exposure before the pinned AgX HDR view; original shoulder and gamut mapping',
         output='HLG Rec.2020; PQ destinations use the existing 1000-nit HLG-to-PQ conversion')]
manifest['format'] += '; the 203 recipe reads the HDR cube with its own shifted log2 bounds'
for name in ['agx-p3.exr', 'agx-pq.exr']:
    assert manifest['data_sha256'][name] == pinned['data_sha256'][name], name
(root/'libraries/std/device-views-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))

# Compare the actual EXR data, tetrahedrally sampled as in Rust, with independent
# direct OCIO patches. PQ conversion below intentionally mirrors sample(), rather
# than using OCIO to evaluate both sides of the comparison.
def sample_cube(path, inputs, domain_lo, domain_hi):
    with OpenEXR.File(str(path)) as exr:
        pixels = exr.channels()['RGB'].pixels
    size = pixels.shape[0]
    linear = inputs @ np.array(matrix, dtype=np.float32).T
    t = np.clip((np.log2(np.maximum(linear, np.float32(2**domain_lo))) - np.float32(domain_lo)) /
                np.float32(domain_hi-domain_lo) * np.float32(size-1), 0, size-1)
    low = np.minimum(np.floor(t).astype(int), size-2)
    f = t - low.astype(np.float32)
    axes = np.argsort(-f, axis=1, kind='stable')
    sf = np.take_along_axis(f, axes, axis=1)
    weights = np.stack([1-sf[:,0], sf[:,0]-sf[:,1], sf[:,1]-sf[:,2], sf[:,2]],axis=1)
    corner = low.copy()
    output = np.zeros_like(inputs)
    rows = np.arange(len(inputs))
    for i in range(4):
        if i: corner[rows, axes[:,i-1]] += 1
        output += weights[:,i,None] * pixels[corner[:,1], corner[:,2]*size+corner[:,0]]
    return output

def hlg_to_pq(encoded):
    a, b, cc = np.float32(.17883277), np.float32(.28466892), np.float32(.55991073)
    e = np.where(encoded <= .5, encoded*encoded/3, (np.exp((encoded-cc)/a)+b)/12)
    y = np.maximum(np.abs(e @ np.array([.2627,.678,.0593],np.float32)),
                   np.float32(1e-4)**np.float32(1/1.2)/3)
    nit = 1000 * e * y[:,None]**np.float32(.2)
    v = (np.maximum(nit,0)/10000)**np.float32(2610/16384)
    return ((np.float32(3424/4096)+np.float32(2413/128)*v) /
            (1+np.float32(2392/128)*v))**np.float32(2523/32)

measurements = dict(ocio=ocio.__version__, scene_exposure_gain=gain,
    scene_exposure_stops=float(np.log2(gain)), original_unit_white_nits=diffuse_nits(1.),
    derived_unit_white_nits=diffuse_nits(gain), errors=[])
for fixture in fixtures_203:
    actual = sample_cube(root/'libraries/std/agx-pq.exr', new_patches, recipe_lo, recipe_hi)
    if fixture['display'] == 'Rec.2100-PQ':
        actual = hlg_to_pq(actual)
        ceiling = .7518271
    else:
        ceiling = 1.
    expected = np.array([p['output'] for p in fixture['samples']], np.float32)
    error = np.clip(actual,0,ceiling).astype(float)-np.clip(expected,0,ceiling).astype(float)
    measurements['errors'].append(dict(display=fixture['display'],patches=len(new_patches),
        channels=error.size, maximum=float(np.max(np.abs(error))),rms=float(np.sqrt(np.mean(error**2)))))
(args.evidence/'validation.json').write_text(json.dumps(measurements,indent=2)+'\n')
print(json.dumps(measurements,indent=2))
assert all(row['maximum'] < 0.0003 and row['rms'] < 0.0001
           for row in measurements['errors']), '203 recipe exceeds encoded error tolerance'
assert abs(diffuse_nits(gain) - 203.) < .001, 'Diffuse white calibration'
