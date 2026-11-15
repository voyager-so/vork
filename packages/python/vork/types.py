# Generated from scene.schema.json; do not edit.
from __future__ import annotations
from typing import Any, Literal, Union, TypedDict, Required

Active = Union[bool, list['Clock']]
AimTarget = Union[str, list[float]]
AovType = Literal['value', 'color']
Background = Union[list[float], 'BackgroundMode', str]
BackgroundMode = Literal['environment', 'transparent']
Blend = Literal['replace', 'combine']
Buffer = Union['InlineBuffer', 'FileBuffer', 'DenseBuffer', 'DenseFileBuffer']
BufferType = Literal['f32', 'i32', 'u32', 'vec2', 'vec3', 'vec4', 'uvec4', 'mat4']
Channel = Literal['r', 'g', 'b', 'a', 'rgb']
Clock = Union[float, str]
ColorSpace = Literal['sRGB', 'Linear Rec.709', 'Linear Rec.2020', 'ACEScg', 'Display P3', 'Non-Color']
CompositeEntry = Union['NameOr', 'CompositeNode']
CompositeResolution = Union[list[int], float]
ComputeDispatch = Union[int, list[int]]
ComputeSchedule = Union[Literal['frame', 'install'], Literal['simulation']]
Content = Union[str, list['Run']]
DashArray = Union['NoDash', list[float]]
Direction = Literal['ltr', 'rtl']
Factor = Union[float, list[float]]
FillRule = Literal['nonzero', 'evenodd']
Fit = Literal['contain', 'cover', 'fill', 'none']
FontFiles = Union[str, list[str]]
FontStyle = Literal['normal', 'italic', 'oblique']
FontWeight = Union['FontWeightKeyword', float]
FontWeightKeyword = Literal['normal', 'bold']
Geometry = Union['BufferGeometry', 'StdGeometry']
GeometryStage = Literal['deform', 'compute', 'displace']
InlineDataSchema = Union[list[float], list[list[float]]]
Input = Union[float, list[float], bool, 'GeometryBinding', 'TextureBinding', 'AttributeBinding', 'GridBinding', 'AovBinding', 'ChannelBinding', 'StepBinding', str]
Interpolation = Literal['STEP', 'LINEAR', 'CUBICSPLINE']
KernelChannel = Literal['color', 'depth', 'view_normal', 'object_index', 'motion', 'shadow_catcher']
Keys = Union[list[list[Any]], 'Keys1']
Layer = Union['TrackLayer', 'ClipLayer']
LightShape = Literal['sun', 'point', 'spot', 'rectangle', 'ellipse', 'environment']
LineBreak = Literal['auto', 'loose', 'normal', 'strict', 'anywhere']
LineCap = Literal['butt', 'round', 'square']
LineHeight = Union['LineHeightNormal', float]
LineHeightNormal = Literal['normal']
LineJoin = Literal['miter', 'round', 'bevel']
MagFilter = Literal['nearest', 'linear']
MinFilter = Literal['nearest', 'linear', 'nearest_mipmap_nearest', 'linear_mipmap_nearest', 'nearest_mipmap_linear', 'linear_mipmap_linear']
MotionBlurPosition = Literal['start', 'center', 'end']
NameOr = Union[str, 'Material']
NameOr2 = Union[str, 'Program']
NameOr3 = Union[str, 'Buffer']
NameOr4 = Union[str, 'Image']
NameOr5 = Union[str, 'Geometry']
NoDash = Literal['none']
NoPaint = Literal['none']
Node = Union['TransformNode', 'MeshNode', 'AssetNode', 'CameraNode', 'LightNode', 'TextNode', 'PathNode']
NodeLayer = Union['NodeTrackLayer', 'NodeClipLayer']
Opacity = Literal['coverage', 'transmittance']
OutputDisplay = Literal['sRGB', 'Display P3', 'Rec.2100-PQ', 'Rec.2100-HLG']
Paint = Union['NoPaint', str, 'ImagePaint']
PathSegment = Union[str, int]
Pitch = Literal['follow', 'keep']
Profile = 'ProfileFile'
Program = Union['FileProgram', 'SourceProgram']
Rotation = Union[list[float], 'Rotation1']
Space = Literal['2d', '3d']
SubdivisionType = Literal['catmull_clark', 'simple']
TextAlign = Literal['start', 'end', 'left', 'right', 'center', 'justify']
Time = Union[float, str]
Topology = Union[Literal['triangles', 'points', 'curves'], Literal['implicit']]
WhiteSpace = Literal['normal', 'nowrap', 'pre', 'pre-wrap', 'pre-line', 'break-spaces']
WordBreak = Literal['normal', 'break-all', 'keep-all', 'break-word']
WorkingColorSpace = Literal['Linear Rec.709', 'Linear Rec.2020', 'ACEScg', 'sRGB', 'Display P3']
Wrap = Literal['repeat', 'clamp_to_edge', 'mirrored_repeat']
Aov = TypedDict('Aov', {'name': Required[str], 'type': Required['AovType']}, total=False)
AovBinding = TypedDict('AovBinding', {'aov': Required[str]}, total=False)
AssetNodeOverrides = TypedDict('AssetNodeOverrides', {'light_link_tags': Union[list[str], None], 'pass_index': Union[int, None], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
AttributeBinding = TypedDict('AttributeBinding', {'attribute': Required[str], 'channel': Union['Channel', None], 'factor': Union['Factor', None]}, total=False)
Audio = TypedDict('Audio', {'file': Union[str, None], 'gain': float}, total=False)
InlineBuffer = TypedDict('InlineBuffer', {'count': Union[int, None], 'data': Required[Union['InlineDataSchema']], 'dims': None, 'file': None, 'grid': Union[str, None], 'offset': int, 'type': Required['BufferType']}, total=False)
FileBuffer = TypedDict('FileBuffer', {'count': Union[int, None], 'data': None, 'dims': None, 'file': Required[Union[str]], 'grid': Union[str, None], 'offset': int, 'type': Required['BufferType']}, total=False)
DenseBuffer = TypedDict('DenseBuffer', {'count': None, 'data': Union['InlineDataSchema', None], 'dims': Required[list[int]], 'file': None, 'grid': None, 'offset': int, 'type': Required['BufferType']}, total=False)
DenseFileBuffer = TypedDict('DenseFileBuffer', {'count': Union[int, None], 'data': None, 'dims': Required[list[int]], 'file': Required[Union[str]], 'grid': None, 'offset': int, 'type': Required['BufferType']}, total=False)
ChannelBinding = TypedDict('ChannelBinding', {'channel': Required['KernelChannel']}, total=False)
Clip = TypedDict('Clip', {'file': Union[str, None], 'index': Union[int, None], 'name': Union[str, None], 'target': Required[str]}, total=False)
CompositeNode = TypedDict('CompositeNode', {'material': Required['NameOr'], 'name': Union[str, None], 'resolution': Union['CompositeResolution', None]}, total=False)
Composition = TypedDict('Composition', {'color_space': 'WorkingColorSpace', 'compositor': list['CompositeEntry'], 'preserveAspectRatio': Union[str, None], 'space': Required['Space'], 'view': 'NameOr', 'viewBox': Union[list[float], None]}, total=False)
EdgeCrease = TypedDict('EdgeCrease', {'crease': Required[float], 'vertices': Required[list[int]]}, total=False)
BufferGeometry = TypedDict('BufferGeometry', {'attributes': Required[dict[str, 'NameOr3']], 'count': Union['NameOr3', None], 'crease_edge': list['EdgeCrease'], 'curve_vertex_counts': Union['NameOr3', None], 'face_vertex_counts': Union[list[int], None], 'ids': Union['NameOr3', None], 'indices': Union['NameOr3', None], 'material_ranges': list['MaterialRange'], 'morphs': dict[str, 'Morph'], 'status': Union['GeometryStatus', None], 'subdivision': Union['SubdivisionSurface', None], 'topology': 'Topology', 'widths': Union['NameOr3', None]}, total=False)
StdGeometry = TypedDict('StdGeometry', {'font': Union[str, None], 'std': Required[str]}, total=False)
GeometryBinding = TypedDict('GeometryBinding', {'geometry': Required[str]}, total=False)
GeometryStatus = TypedDict('GeometryStatus', {'buffer': Required['NameOr3'], 'message': Required[str]}, total=False)
GridBinding = TypedDict('GridBinding', {'grid': Required['NameOr3']}, total=False)
Image = TypedDict('Image', {'color_space': Union['ColorSpace', None], 'composition': Union['ImageComposition', None], 'file': Union[str, None], 'mag_filter': 'MagFilter', 'min_filter': 'MinFilter', 'render': Union['ImageRender', None], 'time': Union['Clock', None], 'wrap_s': 'Wrap', 'wrap_t': 'Wrap'}, total=False)
ImageComposition = TypedDict('ImageComposition', {'compositor': list['CompositeEntry']}, total=False)
ImagePaint = TypedDict('ImagePaint', {'fit': Union['Fit', None], 'image': Required[str]}, total=False)
ImageRender = TypedDict('ImageRender', {'background': 'Background', 'camera': Union[str, None], 'resolution': list[int], 'root': Union[str, None]}, total=False)
Instances = TypedDict('Instances', {'attributes': Required[dict[str, 'NameOr3']], 'count': Union['NameOr3', None]}, total=False)
Keys1 = TypedDict('Keys1', {'times': Required['NameOr3'], 'values': Required['NameOr3']}, total=False)
TrackLayer = TypedDict('TrackLayer', {'blend': 'Blend', 'clip': None, 'loop': bool, 'name': Union[str, None], 'offset': float, 'speed': float, 'tracks': Required[Union[list['Track']]], 'weight': float}, total=False)
ClipLayer = TypedDict('ClipLayer', {'blend': 'Blend', 'clip': Required[Union['Clip']], 'loop': bool, 'name': Union[str, None], 'offset': float, 'speed': float, 'tracks': None, 'weight': float}, total=False)
Material = TypedDict('Material', {'double_sided': bool, 'inputs': dict[str, 'Input'], 'opacity': Union['Opacity', None], 'priority': int, 'program': 'NameOr2'}, total=False)
MaterialRange = TypedDict('MaterialRange', {'count': Required[int], 'material_index': Required[int], 'start': Required[int]}, total=False)
Morph = TypedDict('Morph', {'NORMAL': Union['NameOr3', None], 'POSITION': Required['NameOr3']}, total=False)
MotionBlur = TypedDict('MotionBlur', {'position': 'MotionBlurPosition', 'shutter': float}, total=False)
TransformNode = TypedDict('TransformNode', {'active': 'Active', 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'composition': Union['Composition', None], 'filter': Union[str, None], 'is_holdout': bool, 'is_shadow_catcher': bool, 'light_link_tags': list[str], 'mask': Union[str, None], 'name': str, 'opacity': float, 'pass_index': Union[int, None], 'rotation': 'Rotation', 'scale': list[float], 'target': Union['AimTarget', None], 'time': Union['NodeClock', None], 'translation': list[float], 'type': Literal['node'], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
MeshNode = TypedDict('MeshNode', {'active': 'Active', 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'composition': Union['Composition', None], 'filter': Union[str, None], 'geometry': Required['NameOr5'], 'instances': Union['Instances', None], 'is_holdout': bool, 'is_shadow_catcher': bool, 'light_link_tags': list[str], 'mask': Union[str, None], 'materials': list['NameOr'], 'morph_weights': dict[str, float], 'name': str, 'opacity': float, 'pass_index': Union[int, None], 'rotation': 'Rotation', 'scale': list[float], 'skin': Union['Skin', None], 'target': Union['AimTarget', None], 'time': Union['NodeClock', None], 'translation': list[float], 'type': Required[Literal['mesh']], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
AssetNode = TypedDict('AssetNode', {'active': 'Active', 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'composition': Union['Composition', None], 'file': Required[str], 'filter': Union[str, None], 'is_holdout': bool, 'is_shadow_catcher': bool, 'light_link_tags': list[str], 'mask': Union[str, None], 'materials': dict[str, 'NameOr'], 'name': str, 'nodes': dict[str, 'AssetNodeOverrides'], 'opacity': float, 'pass_index': Union[int, None], 'rotation': 'Rotation', 'scale': list[float], 'target': Union['AimTarget', None], 'time': Union['NodeClock', None], 'translation': list[float], 'type': Required[Literal['asset']], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
CameraNode = TypedDict('CameraNode', {'active': 'Active', 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'composition': Union['Composition', None], 'filter': Union[str, None], 'focus_object': Union[str, None], 'is_holdout': bool, 'is_shadow_catcher': bool, 'light_link_tags': list[str], 'mask': Union[str, None], 'material': Union['NameOr', None], 'name': str, 'opacity': float, 'pass_index': Union[int, None], 'rotation': 'Rotation', 'scale': list[float], 'target': Union['AimTarget', None], 'time': Union['NodeClock', None], 'translation': list[float], 'type': Required[Literal['camera']], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
LightNode = TypedDict('LightNode', {'active': 'Active', 'angle': float, 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'color': Union[list[float], None], 'composition': Union['Composition', None], 'energy': Union[float, None], 'filter': Union[str, None], 'illuminates': Union[list[str], None], 'image': Union['NameOr4', None], 'is_holdout': bool, 'is_shadow_catcher': bool, 'light_link_tags': list[str], 'mask': Union[str, None], 'material': Union['NameOr', None], 'medium': Union['NameOr', None], 'name': str, 'opacity': float, 'pass_index': Union[int, None], 'radius': float, 'rotation': 'Rotation', 'scale': list[float], 'shadowed_by': Union[list[str], None], 'shape': Required['LightShape'], 'size': float, 'size_y': Union[float, None], 'spot_blend': float, 'spot_size': float, 'target': Union['AimTarget', None], 'temperature': Union[float, None], 'time': Union['NodeClock', None], 'translation': list[float], 'type': Required[Literal['light']], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
TextNode = TypedDict('TextNode', {'active': 'Active', 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'composition': Union['Composition', None], 'content': Required['Content'], 'direction': 'Direction', 'fill': 'Paint', 'filter': Union[str, None], 'font': Required['FontFiles'], 'font-feature-settings': str, 'font-size': float, 'font-style': 'FontStyle', 'font-variation-settings': str, 'font-weight': 'FontWeight', 'is_holdout': bool, 'is_shadow_catcher': bool, 'letter-spacing': float, 'light_link_tags': list[str], 'line-break': 'LineBreak', 'line-height': 'LineHeight', 'mask': Union[str, None], 'name': str, 'opacity': float, 'pass_index': Union[int, None], 'rotation': 'Rotation', 'scale': list[float], 'stroke': 'Paint', 'stroke-width': float, 'target': Union['AimTarget', None], 'text-align': 'TextAlign', 'time': Union['NodeClock', None], 'translation': list[float], 'type': Required[Literal['text']], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool, 'white-space': 'WhiteSpace', 'width': Union[float, None], 'word-break': 'WordBreak', 'word-spacing': float}, total=False)
PathNode = TypedDict('PathNode', {'active': 'Active', 'animation': list['NodeLayer'], 'audio': Union['Audio', None], 'children': list['Node'], 'clip-path': Union[str, None], 'composition': Union['Composition', None], 'd': str, 'fill': 'Paint', 'fill-opacity': float, 'fill-rule': 'FillRule', 'filter': Union[str, None], 'is_holdout': bool, 'is_shadow_catcher': bool, 'light_link_tags': list[str], 'mask': Union[str, None], 'name': str, 'opacity': float, 'paint-order': str, 'pass_index': Union[int, None], 'pathLength': Union[float, None], 'rotation': 'Rotation', 'scale': list[float], 'stroke': 'Paint', 'stroke-dasharray': 'DashArray', 'stroke-dashoffset': float, 'stroke-linecap': 'LineCap', 'stroke-linejoin': 'LineJoin', 'stroke-miterlimit': float, 'stroke-opacity': float, 'stroke-width': float, 'target': Union['AimTarget', None], 'time': Union['NodeClock', None], 'translation': list[float], 'type': Required[Literal['path']], 'visible': bool, 'visible_camera': bool, 'visible_diffuse': bool, 'visible_glossy': bool, 'visible_shadow': bool, 'visible_transmission': bool}, total=False)
NodeClock = TypedDict('NodeClock', {'duration': Union['Clock', None], 'loop': bool, 'offset': 'Clock', 'pitch': 'Pitch', 'speed': 'Clock', 'start': Union['Clock', None]}, total=False)
NodeTrackLayer = TypedDict('NodeTrackLayer', {'blend': 'Blend', 'clip': None, 'loop': bool, 'name': Union[str, None], 'offset': float, 'speed': float, 'tracks': Required[Union[list['Track']]], 'weight': float}, total=False)
NodeClipLayer = TypedDict('NodeClipLayer', {'blend': 'Blend', 'clip': Required[Union['Clip']], 'loop': bool, 'name': Union[str, None], 'offset': float, 'speed': float, 'tracks': None, 'weight': float}, total=False)
OutputDestination = TypedDict('OutputDestination', {'color_space': 'WorkingColorSpace', 'display': 'OutputDisplay', 'file': Required[str], 'peak_luminance': float, 'reference_white': float}, total=False)
ProfileFile = TypedDict('ProfileFile', {'description': str, 'level': Required[int], 'settings': 'Settings', 'version': Required[int]}, total=False)
FileProgram = TypedDict('FileProgram', {'bindings': dict[str, 'NameOr3'], 'dispatch': Union['ComputeDispatch', None], 'file': Required[Union[str]], 'schedule': Union['ComputeSchedule', None], 'source': None, 'source_origin': None}, total=False)
SourceProgram = TypedDict('SourceProgram', {'bindings': dict[str, 'NameOr3'], 'dispatch': Union['ComputeDispatch', None], 'file': None, 'schedule': Union['ComputeSchedule', None], 'source': Required[Union[str]], 'source_origin': Union[str, None]}, total=False)
Propagation = TypedDict('Propagation', {'bounds': Required[list[list[float]]], 'program': Required[str]}, total=False)
Render = TypedDict('Render', {'aovs': list['Aov'], 'background': 'Background', 'camera': Union[str, None], 'duration': 'Clock', 'fps': 'Time', 'motion_blur': Union['MotionBlur', None], 'resolution': list[int], 'root': Union[str, None]}, total=False)
Rotation1 = TypedDict('Rotation1', {'euler': Required[list[float]]}, total=False)
Run = TypedDict('Run', {'content': Required[str], 'fill': Union['Paint', None], 'font': Union['FontFiles', None], 'font-feature-settings': Union[str, None], 'font-size': Union[float, None], 'font-style': Union['FontStyle', None], 'font-variation-settings': Union[str, None], 'font-weight': Union['FontWeight', None], 'letter-spacing': Union[float, None], 'stroke': Union['Paint', None]}, total=False)
Settings = TypedDict('Settings', {'denoise': bool, 'samples': int, 'seed': int}, total=False)
Simulation = TypedDict('Simulation', {'dt': Required['Clock'], 'init': Required[list[str]], 'seed': Required[int], 'state': Required[list[str]], 'steps': Required[list[str]], 't0': Required['Clock']}, total=False)
Skin = TypedDict('Skin', {'inverse_bind_matrices': Required['NameOr3'], 'joints': Required[list[str]]}, total=False)
StepBinding = TypedDict('StepBinding', {'step': Required[str]}, total=False)
SubdivisionSurface = TypedDict('SubdivisionSurface', {'render_levels': int, 'subdivision_type': 'SubdivisionType', 'use_limit_surface': bool}, total=False)
TextureBinding = TypedDict('TextureBinding', {'image': Required['NameOr4']}, total=False)
Track = TypedDict('Track', {'interpolation': Union['Interpolation', None], 'keys': Required['Keys'], 'path': Required[list['PathSegment']]}, total=False)
Scene = TypedDict('Scene', {'$schema': Union[str, None], 'animation': list['Layer'], 'buffers': dict[str, 'Buffer'], 'composition': Union['Composition', None], 'geometry': dict[str, 'Geometry'], 'geometry_stages': list['GeometryStage'], 'images': dict[str, 'Image'], 'materials': dict[str, 'Material'], 'nodes': Required[list['Node']], 'programs': dict[str, 'Program'], 'propagation': Union['Propagation', None], 'render': 'Render', 'simulation': Union['Simulation', None], 'version': Required[Literal[2]]}, total=False)
