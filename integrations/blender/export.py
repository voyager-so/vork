# SPDX-License-Identifier: GPL-3.0-or-later
"""blender -b --disable-autoexec source.blend --python export.py -- --out OUTPUT [--animation]."""
import json, math, sys, time, traceback, collections
from pathlib import Path
import bpy
from mathutils import Vector, Matrix
import argparse
parser = argparse.ArgumentParser(description='Export the current Blender scene as vork v2 files')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--animation', action='store_true')
parser.add_argument('--bake-values', type=int, metavar='RESOLUTION', help='CPU bake unsupported view-independent numeric sockets per source object and frame at this texture resolution')
parser.add_argument('--bake-curves', action='store_true', help='Bake RGB curve transfer functions to reported PNG8 lookup textures')
parser.add_argument('--apply-modifiers', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if args.bake_values is not None and (args.bake_values<=0 or not args.apply_modifiers):
    parser.error('--bake-values requires a positive resolution and --apply-modifiers')
repo, out = Path(__file__).resolve().parents[2], args.out.resolve()
animated = args.animation
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_hdr import write_image
from integrations.blender.animation import export_clip, sample_controls, camera_focus
from integrations.blender.pose import rotation as pose_rotation
from integrations.blender.camera import camera_fit
from integrations.blender.light_graph import translate as translate_light
from integrations.blender.visibility import asset_overrides
from integrations.blender.world import translate as translate_world, preprocess_image, UnsupportedWorld
from integrations.blender.geometry import export_gltf, filter_material_overrides
from integrations.blender.display import translate as translate_display
from integrations.blender.material_export import MaterialWriter
from integrations.blender.instances import particle_occurrences
from integrations.blender.render_geometry import direct_render_objects, render_selection, gltf_render_settings
report = dict(stages={}, omissions=[], materials={}, warnings=[])
clip = None
control_bindings = {}
def wall(kind, detail):
    report['omissions'].append(dict(kind=kind, detail=detail))
    print(f'BRIDGE SUBSTITUTION [{kind}]: {detail}', file=sys.stderr, flush=True)
if args.animation and args.bake_values is not None:
    wall('value bake animation snapshot','value textures retain the source frame; changed poses require new exports')
def stage(name, t0): report['stages'][name] = round(time.perf_counter()-t0, 3)
def save(): (out/'export-report.json').write_text(json.dumps(report, indent=1, default=str))
sc = bpy.context.scene; r = sc.render
assert bpy.app.version == (5, 2, 1), bpy.app.version_string
sc.frame_set(sc.frame_start); bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
res = [int(r.resolution_x*r.resolution_percentage/100), int(r.resolution_y*r.resolution_percentage/100)]
# Features the bridge has no path for: record, never reconstruct.
for ob in sc.objects:
    if ob.hide_render: continue
    for p in getattr(ob, 'particle_systems', []):
        s = p.settings
        if s.render_type not in ('OBJECT', 'COLLECTION', 'NONE'):
            wall('particles/hair not exported', f'{ob.name}/{p.name}: {s.type} render_as={s.render_type} count={s.count} children={s.child_type}x{s.rendered_child_count}')
    if ob.type in ('CURVES', 'VOLUME', 'POINTCLOUD'): wall(f'{ob.type} object not exported', ob.name)
    if ob.type == 'CURVE' and (ob.data.bevel_depth == 0 and ob.data.extrude == 0 and not ob.data.bevel_object): pass
    for m in ob.modifiers:
        if m.type == 'FLUID': wall('fluid/smoke domain', f'{ob.name}/{m.name}')
if r.use_motion_blur: wall('motion blur not exported', 'render.use_motion_blur; frame-sampled animation does not export shutter settings')
comp = getattr(sc, 'compositing_node_group', None)
if comp is not None and len(comp.nodes) > 2:
    wall('compositor node tree not translated', sorted({n.bl_idname for n in comp.nodes}))
# Geometry
t0 = time.perf_counter()
try:
    with gltf_render_settings(sc), particle_occurrences(sc, animated=animated) as count, render_selection(sc):
        report['particle_occurrences'] = count
        if animated:
            clip = export_clip(out/'scene.glb', 'model', apply_modifiers=args.apply_modifiers,
                               export_lights=False, export_cameras=False, use_renderable=True, use_selection=True, use_active_scene=True, bake_sources=args.bake_values is not None)
            report['animation_clip'] = clip
        else:
            export_gltf(filepath=str(out/'scene.glb'), export_format='GLB', export_animations=False, export_current_frame=True,
                        export_apply=args.apply_modifiers, export_lights=False, export_cameras=False, use_renderable=True, use_selection=True, use_active_scene=True, bake_sources=args.bake_values is not None)
except Exception as e:
    wall('glTF export failed', traceback.format_exc()[-2000:]); save(); raise
stage('gltf_export', t0); report['glb_bytes'] = (out/'scene.glb').stat().st_size; save()
# Materials actually used by renderable mesh objects
t0 = time.perf_counter()
used = collections.Counter()
for ob in sc.objects:
    if ob.type in ('MESH','CURVE','SURFACE','META','FONT') and not ob.hide_render:
        for slot in ob.material_slots:
            if slot.material: used[slot.material.name] += 1
writer = MaterialWriter(out, best_effort=True, curve_lookups=args.bake_curves)
# Bake the same render-evaluated mesh/UV layout that glTF exported.
with gltf_render_settings(sc):
    bindings = writer.write_asset(out/'scene.glb',bake_resolution=args.bake_values)
report['material_reports'] = writer.reports
report['asset_reports'] = writer.asset_reports
for name in sorted(used):
    if animated:
        material = bpy.data.materials[name]
        pending = [material, material.node_tree]
        seen = set()
        while pending:
            owner = pending.pop()
            if owner is None or owner.as_pointer() in seen:
                continue
            seen.add(owner.as_pointer())
            animation = getattr(owner, 'animation_data', None)
            if animation is not None and (animation.action is not None or animation.drivers or animation.nla_tracks):
                wall('material animation snapshot', f'{name}/{owner.name}: source-frame numeric values retained; material socket tracks are not exported')
            if hasattr(owner, 'nodes'):
                pending.extend(n.node_tree for n in owner.nodes if n.type == 'GROUP')
    reports = [r for r in writer.reports.values() if r['material'] == name]
    report['materials'][name] = 'lowered' if reports else 'not emitted by glTF'
stage('materials', t0)
reasons = collections.Counter(v.split(':')[0] + ': ' + v.partition(':')[2].split(' node ')[-1][:90] if v != 'lowered' else 'lowered' for v in report['materials'].values())
report['material_summary'] = dict(used=len(used), lowered=sum(v=='lowered' for v in report['materials'].values()), reasons=reasons.most_common(40))
# World
images = {}
try:
    world = translate_world(sc.world, bake_directory=out, on_bake=lambda detail: report.setdefault('world_bakes', []).append(detail)); environment = world.environment; camera_background = world.background; materials = dict(world.materials)
    if world.image is not None:
        write_image(out/'world.hdr', world.image); images['world'] = {'file': 'world.hdr', 'color_space': 'Linear Rec.709'}
        preprocess_image(environment, out, float(sc.world.sun_threshold), float(sc.world.sun_angle))
except UnsupportedWorld as e:
    wall('World export refused', f'{e}; use a constant Background or a supported environment image; for fog use supported homogeneous Volume inputs or disconnect Volume; see vork help fog'); save(); raise
if materials or any(r.get('medium') for r in writer.reports.values()):
    surfaces = [r['material'] for r in writer.reports.values() if not r.get('medium')]
    if surfaces:
        from integrations.blender.materials import UnsupportedMaterial
        raise UnsupportedMaterial(
            'Fog cannot render alongside translated custom surface programs: '
            + ', '.join(sorted(set(surfaces)))
            + '; export fog-only carriers and add native std/openpbr surfaces in the '
            'authored scene, or disconnect Volume for a surface-only export; see vork help fog')
try:
    display = translate_display(sc, emit_display=writer.write_display, on_substitution=lambda field,reason,replacement:
        wall('display setting not translated', f'{field}: {reason}; {replacement}'))
except ValueError as e:
    wall('display export refused', f'{e}; use sRGB with Standard, AgX or Filmic and a matching look; disable unsupported display curves'); save(); raise
path_transport = r.engine == 'CYCLES'
render = dict(resolution=res, fps=r.fps/r.fps_base, duration=2.0)
render['background'] = 'transparent' if r.film_transparent else camera_background
# Render settings live in a sibling profile: vork render shot.json --profile shot.profile.json.
level = 2 if path_transport else 1
settings = dict(seed=sc.cycles.seed if path_transport else 0, denoise=bool(sc.cycles.use_denoising) if path_transport else False)
def cv(v): return [v.x, v.z, -v.y]
basis = Matrix(((1,0,0),(0,0,1),(0,-1,0)))
cam_ob = sc.camera.evaluated_get(deps); cd = cam_ob.data
if r.pixel_aspect_x != r.pixel_aspect_y:
    wall('non-square pixel aspect', 'square pixels at the output resolution')
q = (basis @ cam_ob.matrix_world.to_3x3()).to_quaternion()
cam = {'type':'camera','name':'camera','translation':cv(cam_ob.matrix_world.translation), 'rotation':[q.x,q.y,q.z,q.w], 'material':'camera-optics'}
optics = dict(clip_start=cd.clip_start,clip_end=cd.clip_end,lens=cd.lens,sensor_width=cd.sensor_width,sensor_height=cd.sensor_height,
              **camera_fit(cd, *res),shift=[cd.shift_x,cd.shift_y])
if cd.type == 'ORTHO': optics.update(projection='ortho', ortho_scale=cd.ortho_scale)
elif cd.type != 'PERSP': raise ValueError(f'Unsupported camera type {cd.type}')
control_bindings['camera'] = sc.camera
if cd.dof.use_dof and cd.type != 'PERSP':
    wall('orthographic DOF not exported', cam_ob.name + ': vork supports perspective DOF only')
elif cd.dof.use_dof:
    fd, substituted = camera_focus(cam_ob, deps)
    if substituted:
        wall('DOF focus plane at or inside focal length', cam_ob.name + ': omit unsupported thin-lens DOF')
    else:
        optics.update(use_dof=True, focus_distance=fd, aperture_fstop=cd.dof.aperture_fstop, aperture_blades=cd.dof.aperture_blades,
                          aperture_rotation=cd.dof.aperture_rotation, aperture_ratio=cd.dof.aperture_ratio)
materials['camera-optics'] = dict(program='std/camera',inputs=optics)
nodes = [dict(type='asset', name='model', file='scene.glb', materials=bindings, nodes=asset_overrides(out/'scene.glb',
    lambda obj,reason,replacement: wall('object ray visibility', f'{obj}: {reason}; {replacement}'))), cam]
if environment is not None: nodes.append(environment)
lights = collections.Counter()
visible = direct_render_objects(bpy.context.view_layer.layer_collection)
with gltf_render_settings(sc):
    deps = bpy.context.evaluated_depsgraph_get()
    for oi in deps.object_instances:
        obj = oi.object
        if obj.type != 'LIGHT' or obj.hide_render: continue
        if oi.is_instance:
            if oi.parent is None or oi.parent.original not in visible: continue
        elif obj.original not in visible: continue
        data = obj.data; kind = data.type.lower(); mw = oi.matrix_world.copy()
        if kind not in ('sun','area','point','spot'): wall('light type', kind); continue
        qq = pose_rotation(basis.to_4x4() @ mw, area=kind == 'area')
        node = {'type':'light','shape':{'area':'rectangle'}.get(kind, kind),'name':f'light_{sum(lights.values())}_{obj.name}',
                'translation':cv(mw.translation), 'rotation':[qq.x,qq.y,qq.z,qq.w],**translate_light(data, on_substitution=lambda reason,replacement:
                    wall('light graph', f'{obj.name}/{data.name}: {reason}; {replacement}'))}
        if kind == 'sun': node['angle'] = data.angle
        if kind in ('point','spot'): node['radius'] = data.shadow_soft_size
        if kind == 'spot': node.update(spot_size=data.spot_size, spot_blend=data.spot_blend)
        if kind == 'area':
            node.update(size=data.size*mw.to_3x3().col[0].length, size_y=(data.size_y if data.shape in ('RECTANGLE','ELLIPSE') else data.size)*mw.to_3x3().col[1].length,
                        shape='ellipse' if data.shape in ('DISK','ELLIPSE') else 'rectangle')
            if getattr(data, 'spread', math.pi) < math.pi - 1e-4: wall('area light spread not exported', f'{obj.name}: spread={data.spread:.3f}')
        if not data.use_shadow: node['shadowed_by'] = []
        if not data.normalize: wall('unnormalized light approximation', obj.name + ': retain normalized energy; unnormalized surface radiance is not converted')
        lights[kind] += 1; nodes.append(node)
        # Instanced occurrences need distinct world-transform sampling; do not bind them to the original pose.
        if not oi.is_instance:
            control_bindings[node['name']] = obj.original
        elif animated:
            wall('instanced light animation', obj.name + ': current occurrence snapshot')
report['lights'] = dict(lights)
scene_doc = dict(version=2, composition=dict(space="3d", **display), render=render, nodes=nodes, images={**images, **writer.images}, programs=writer.programs,
                 materials={**materials, **writer.materials})
if animated:
    controls = sample_controls(sc, control_bindings)
    for node in nodes:
        if node['name'] in controls['disabled_dof']:
            scene_doc['materials'][node['material']]['inputs']['use_dof'] = False
    scene_doc['animation'] = clip['layers'] + [controls['layer']]
    scene_doc['render']['duration'] = (sc.frame_end-sc.frame_start+1)/(r.fps/r.fps_base)
    report['animation_controls'] = controls
else:
    wall('animation snapshot', 'one evaluated source frame; use --animation for clips and camera/light tracks')
(out/'shot.json').write_text(json.dumps(scene_doc, indent=1))
profile = out/'shot.profile.json'
if settings:
    profile.write_text(json.dumps(dict(version=1, level=level, description='Render settings exported from Blender ' + ('Cycles' if path_transport else 'EEVEE'), settings=settings), indent=2))
else:
    profile.unlink(missing_ok=True)
report['scene'] = dict(resolution=res, profile='std/path' if path_transport else 'std/final', cycles_samples=sc.cycles.samples, frames=[sc.frame_start, sc.frame_end])
report['animated_objects'] = sorted(o.name for o in bpy.data.objects if o.animation_data and o.animation_data.action)[:50]
report['animated_count'] = sum(1 for o in bpy.data.objects if o.animation_data and (o.animation_data.action or o.animation_data.drivers))
save(); print('EXPORT_OK', out)
