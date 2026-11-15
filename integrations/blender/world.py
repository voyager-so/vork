"""Translate supported Blender World graphs to vork's environment-light contract.

Run inside Blender. Only nodes reachable from the active Cycles World output
participate. Unsupported links raise a named error instead of selecting unrelated
Background or Environment Texture nodes elsewhere in the tree.
"""
from dataclasses import dataclass
import math

from mathutils import Euler, Matrix


class UnsupportedWorld(ValueError):
    pass


class UnsupportedWorldVolume(UnsupportedWorld):
    """A volume cannot be replaced by fallback World illumination."""


def unsupported(node, socket, reason):
    raise UnsupportedWorld(f'World {node.name!r}.{socket}: {reason}')


def constant(node, name):
    socket = node.inputs[name]
    if socket.is_linked:
        unsupported(node, name, 'linked parameters are not translated')
    return socket.default_value


def mapping(socket):
    if not socket.is_linked:
        return Matrix.Identity(3)
    link = socket.links[0]
    node = link.from_node
    if node.mute:
        unsupported(node, link.from_socket.name, 'muted-node bypass is not translated')
    if node.type == 'TEX_COORD' and link.from_socket.name == 'Generated':
        return Matrix.Identity(3)
    if node.type != 'MAPPING':
        unsupported(node, link.from_socket.name, 'expected a World direction or Mapping node')
    location = constant(node, 'Location') if node.inputs.get('Location') else (0., 0., 0.)
    scale = constant(node, 'Scale')
    # A positive uniform scale cancels when a direction is normalized. Other
    # affine maps need directional resampling, not a light rotation.
    if any(v != 0 for v in location):
        unsupported(node, 'Location', 'translated directions require angular resampling')
    if not (scale[0] == scale[1] == scale[2] and scale[0] > 0):
        unsupported(node, 'Scale', 'nonuniform or nonpositive direction scale requires angular resampling')
    rotation = Euler(tuple(constant(node, 'Rotation')), 'XYZ').to_matrix()
    if node.vector_type == 'TEXTURE':
        rotation.transpose()
    return rotation @ mapping(node.inputs['Vector'])


@dataclass
class Radiance:
    image: object
    color: tuple
    rotation: object


def color(socket, camera):
    if not socket.is_linked:
        return Radiance(None, tuple(socket.default_value[:3]), Matrix.Identity(3))
    link = socket.links[0]
    node = link.from_node
    if node.mute:
        unsupported(node, link.from_socket.name, 'muted-node bypass is not translated')
    if node.type == 'RGB':
        return Radiance(None, tuple(link.from_socket.default_value[:3]), Matrix.Identity(3))
    if node.type == 'BLACKBODY':
        from .blackbody import blackbody
        try:
            rgb = blackbody(float(constant(node, 'Temperature')))
        except ValueError as error:
            unsupported(node, 'Temperature', str(error))
        return Radiance(None, tuple(rgb[:3]), Matrix.Identity(3))
    if node.type == 'TEX_ENVIRONMENT':
        if not node.image:
            unsupported(node, 'Color', 'missing environment image')
        if node.interpolation != 'Linear':
            unsupported(node, 'Color', 'only linear environment filtering is translated')
        if node.projection != 'EQUIRECTANGULAR':
            unsupported(node, 'Color', 'only equirectangular environment images are translated')
        if link.from_socket.name != 'Color':
            unsupported(node, link.from_socket.name, 'expected environment Color')
        return Radiance(node.image, (1., 1., 1.), mapping(node.inputs['Vector']))
    if node.type == 'MIX' and node.data_type == 'RGBA' and node.blend_type == 'MIX':
        # Blender's Mix node has several identically named inactive sockets.
        if node.clamp_result:
            unsupported(node, 'Result', 'clamped World colours require graph evaluation')
        factor = node.inputs[0]
        first, second = node.inputs[6], node.inputs[7]
    elif node.type == 'MIX_RGB' and node.blend_type == 'MIX':
        if node.use_clamp:
            unsupported(node, 'Color', 'clamped World colours require graph evaluation')
        factor, first, second = node.inputs[:3]
    else:
        unsupported(node, link.from_socket.name, f'{node.type} colour evaluation is not translated')
    if factor.is_linked:
        factor_link = factor.links[0]
        if factor_link.from_node.type != 'LIGHT_PATH' or factor_link.from_socket.name != 'Is Camera Ray':
            unsupported(node, 'Factor', 'only Is Camera Ray branching is translated')
        weight = float(camera)
    else:
        weight = float(factor.default_value)
    if weight == 0:
        return color(first, camera)
    if weight == 1:
        return color(second, camera)
    unsupported(node, 'Factor', 'mixed World radiance requires graph evaluation')


def background(socket, camera):
    if not socket.is_linked:
        return Radiance(None, (0., 0., 0.), Matrix.Identity(3))
    node = socket.links[0].from_node
    if node.mute:
        unsupported(node, 'Surface', 'muted-node bypass is not translated')
    if node.type == 'MIX_SHADER':
        factor = node.inputs[0]
        if factor.is_linked:
            link = factor.links[0]
            if link.from_node.type != 'LIGHT_PATH' or link.from_socket.name != 'Is Camera Ray':
                unsupported(node, 'Factor', 'only Is Camera Ray branching is translated')
            weight = float(camera)
        else:
            weight = min(1., max(0., float(factor.default_value)))
        if weight == 0: return background(node.inputs[1], camera)
        if weight == 1: return background(node.inputs[2], camera)
        first, second = background(node.inputs[1], camera), background(node.inputs[2], camera)
        if first.image is None and second.image is None:
            return Radiance(None, tuple(a*(1-weight)+b*weight for a,b in zip(first.color, second.color)), Matrix.Identity(3))
        unsupported(node, 'Surface', 'mixed directional World radiance requires resampling')
    if node.type != 'BACKGROUND':
        unsupported(node, 'Surface', f'{node.type} shader evaluation is not translated')
    strength = constant(node, 'Strength')
    result = color(node.inputs['Color'], camera)
    result.color = tuple(v * strength for v in result.color)
    return result


@dataclass
class WorldImport:
    image: object
    environment: dict
    background: object
    materials: dict


def medium(socket):
    """Keep world fog independent of its surface illumination, including bakes."""
    from .volume import empty_volume, homogeneous
    if empty_volume(socket):
        return {}
    try:
        source = homogeneous(socket)
    except ValueError as error:
        raise UnsupportedWorldVolume(f'World {error}') from error
    return {'world_medium': {'program': {'source': source}}}


def translate(world, *, bake_directory=None, on_bake=None):
    """Return WorldImport with image, environment, background and materials.

    The caller writes image pixels as scene-linear HDR bound as `world` and
    merges materials into the scene. A constant camera background overrides only
    uncovered camera pixels while lighting and media keep the active World graph.
    """
    identity = Matrix.Identity(3)
    materials = {}
    if world is None:
        lighting = camera = Radiance(None, (0., 0., 0.), identity)
    elif not world.use_nodes:
        lighting = camera = Radiance(None, tuple(world.color), identity)
    else:
        output = world.node_tree.get_output_node('CYCLES')
        if output is None:
            raise UnsupportedWorld('World: no active Cycles output')
        materials = medium(output.inputs['Volume'])
        try:
            lighting = background(output.inputs['Surface'], False)
            camera = background(output.inputs['Surface'], True)
        except UnsupportedWorld:
            if bake_directory is None: raise
            from .world_bake import bake
            result = bake(world, bake_directory)
            if on_bake is not None:
                on_bake('Cycles directional World resampled at 2048x1024 into float32 scene-linear EXR; no source geometry or display transform')
            return result
    if not all(math.isfinite(v) and v >= 0 for v in lighting.color + camera.color):
        raise UnsupportedWorld('World: radiance must be finite and nonnegative')
    node = dict(type='light', name='environment', shape='environment', color=list(lighting.color), energy=1)
    if materials:
        node['medium'] = 'world_medium'
    if lighting.image is not None:
        # Mapping transforms the lookup direction. A light transform acts in the
        # opposite direction, after converting Blender Z-up to vork Y-up.
        basis = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
        q = (basis @ lighting.rotation.transposed() @ basis.transposed()).to_quaternion()
        node.update(material={'program':'std/environment_image','inputs':{'image':{'image':'world'}}}, rotation=[q.x, q.y, q.z, q.w])
    if camera.image is None:
        camera_background = list(camera.color)
    elif (camera.image == lighting.image and camera.color == lighting.color
          and camera.rotation == lighting.rotation):
        camera_background = 'environment'
    else:
        raise UnsupportedWorld('World: distinct textured camera and lighting backgrounds need separate environments')
    return WorldImport(lighting.image, node, camera_background, materials)


def preprocess_image(environment, directory, threshold=10, angle=None):
    """Author the HDR split through std, before handing declarations to the kernel."""
    import json
    import os
    from pathlib import Path
    import subprocess
    directory = Path(directory).resolve()
    if threshold <= 0:
        return
    result = subprocess.run([os.environ.get('VORK_BIN', 'vork'), 'std', 'environment', '-'],
                            input=json.dumps(dict(file=str(directory/'world.hdr'), output=str(directory/'world-lighting.exr'),
                                                  name=environment['name'], threshold=threshold, color_space='Linear Rec.709')),
                            text=True, capture_output=True, check=True)
    fragment = json.loads(result.stdout)
    material = fragment['materials'][fragment['nodes'][0]['material']]
    for binding in material['inputs'].values():
        image = dict(fragment['images'][binding['image']])
        image['file'] = os.path.relpath(image['file'], directory)
        binding['image'] = image
    environment['material'] = material
    for sun in fragment['nodes'][1:]:
        if angle is not None:
            sun['angle'] = angle
        sun['energy'] *= environment.get('energy', 1)
        sun['color'] = [v*t for v, t in zip(sun['color'], environment.get('color', [1, 1, 1]))]
        environment.setdefault('children', []).append(sun)
