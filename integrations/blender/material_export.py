"""Write material graph programs and image inputs without changing source data."""
from pathlib import Path

import bpy

from .materials import Compiler, UnsupportedMaterial


def native_exr_channels(data):
    """The native image decoder needs non-deep, unprefixed RGB channels."""
    import struct
    try:
        version = struct.unpack_from('<I', data, 4)[0]
        if version & (0x800 | 0x1000):  # Deep or multipart: snapshot Blender's selected pixels.
            return False
        position = 8
        while data[position]:
            end = data.index(b'\0', position)
            name = data[position:end]
            position = data.index(b'\0', end+1)+1  # Attribute type.
            size = struct.unpack_from('<I', data, position)[0]
            position += 4
            value = data[position:position+size]
            if len(value) != size:
                return False
            position += size
            if name == b'channels':
                names = set()
                channel = 0
                while value[channel]:
                    end = value.index(b'\0', channel)
                    names.add(value[channel:end])
                    # The native decoder rejects subsampling on every channel,
                    # including auxiliary channels outside RGB.
                    if struct.unpack_from('<ii', value, end+9) != (1, 1):
                        return False
                    channel = end+17  # Pixel type, linear flag, reserved bytes, XY sampling.
                return {b'R', b'G', b'B'} <= names
    except (IndexError, ValueError, struct.error):
        pass
    return False


class MaterialWriter:
    """Accumulate v2 scene fragments; bind returned IDs on an asset's materials.

    Export geometry with geometry.export_gltf to retain active/named UV aliases.
    Object/Normal coordinates need no UVs. Generated coordinates fail explicitly.
    """

    def __init__(self, directory, *, best_effort=False, curve_lookups=False):
        if curve_lookups and not best_effort:
            raise ValueError('curve lookup baking requires best_effort=True')
        self.best_effort = best_effort
        self.curve_lookups = curve_lookups
        self.curve_images = {}
        self.reports = {}
        self.asset_reports = []
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.images = {}
        self.materials = {}
        self.programs = {}
        self.image_keys = {}

    def write_display(self, source, view_program, inputs):
        key = '__blender_display_curves'
        while key in self.programs or key in self.materials:
            key += '_'
        filename = key + '.slang'
        (self.directory / filename).write_text(source)
        self.programs[key] = dict(file=filename)
        inputs = dict(inputs)
        if view_program in ('std/agx', 'blender/filmic'):
            import shutil
            name = 'agx' if view_program == 'std/agx' else 'filmic'
            filename = f'blender-{name}.exr'
            shutil.copyfile(Path(__file__).resolve().parents[2]/'libraries/blender'/f'{name}.exr', self.directory/filename)
            inputs['lut'] = {'image': {'file': filename, 'color_space': 'Linear Rec.709'}}
        self.materials[key] = dict(program=key, inputs=inputs)
        return key

    def curve_image(self, table):
        """Bake a context-free scalar transfer function, not a surface UV field."""
        import struct, zlib, math
        if len(table) < 2 or any(len(row) != 4 or any(not math.isfinite(x) for x in row) for row in table):
            raise UnsupportedMaterial('curve lookup needs finite RGBA samples')
        identity = tuple(table)
        if identity in self.curve_images:
            return self.curve_images[identity]
        low = [min(row[c] for row in table) for c in range(4)]
        span = [max(row[c] for row in table)-low[c] for c in range(4)]
        if any(x > 3.4028234663852886e38 for x in span):
            raise UnsupportedMaterial('curve lookup range exceeds shader float precision')
        pixels = bytes(max(0,min(255,round((row[c]-low[c])/span[c]*255))) if span[c] else 0
                       for row in table for c in range(4))
        key = f'blender_curve_{len(self.curve_images)}'
        filename = key+'.png'
        def chunk(kind, data):
            return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
        data = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',len(table),1,8,6,0,0,0))+
                chunk(b'IDAT',zlib.compress(b'\0'+pixels))+chunk(b'IEND',b''))
        (self.directory/filename).write_bytes(data)
        self.images[key] = dict(file=filename,color_space='Non-Color',wrap_s='clamp_to_edge',wrap_t='clamp_to_edge',
                                mag_filter='linear',min_filter='linear')
        result = dict(image=key,scale=span,offset=low,samples=len(table),
                      replacement='sampled curve transfer function; range-encoded PNG8 with 8-bit storage quantization')
        self.curve_images[identity] = result
        return result

    def image(self, node):
        image = node.image
        if image.source not in ('FILE', 'GENERATED'):
            raise UnsupportedMaterial(f'image {image.name!r}: {image.source} is not translated')
        color_space = image.colorspace_settings.name
        fallback = None
        if color_space == '' and self.best_effort:
            floating_file = image.file_format in ('OPEN_EXR','OPEN_EXR_MULTILAYER','HDR')
            color_space = 'Linear Rec.709' if floating_file or (image.source == 'GENERATED' and image.is_float) else 'sRGB'
            fallback = f'Blender file-default {color_space}; unavailable source colour-space role'
        if color_space not in ('sRGB', 'Linear Rec.709', 'Non-Color'):
            raise UnsupportedMaterial(f'image {image.name!r}: unsupported colour space {color_space!r}')
        wraps = {'REPEAT': 'repeat', 'EXTEND': 'clamp_to_edge', 'MIRROR': 'mirrored_repeat'}
        if node.extension not in wraps or node.interpolation not in ('Linear', 'Closest'):
            raise UnsupportedMaterial(f'image node {node.name!r}: unsupported extension/interpolation')
        cache = (image.as_pointer(), node.extension, node.interpolation)
        if cache in self.image_keys:
            return self.image_keys[cache]
        key = f'blender_image_{len(self.images)}'
        # Packed images may be float in Blender but still carry an ordinary
        # PNG/JPEG file. Preserve those bytes, precision and transfer function.
        data = bytes(image.packed_file.data) if image.packed_file and not image.is_dirty else None
        if data is None and image.source == 'FILE' and not image.is_dirty:
            path = Path(bpy.path.abspath(image.filepath, library=image.library))
            if not path.is_file():
                raise UnsupportedMaterial(f'image {image.name!r}: missing file {str(path)!r}')
            data = path.read_bytes()
        if data is not None:
            if data.startswith(b'\x89PNG\r\n\x1a\n'):
                suffix = '.png'
            elif data.startswith(b'\xff\xd8\xff'):
                suffix = '.jpg'
            elif data.startswith(b'v/1\x01'):
                if not native_exr_channels(data):
                    # Native decoding needs unprefixed RGB, unlike Blender render layers.
                    return self.float_image(image, key, cache, node, fallback)
                suffix = '.exr'
            elif data.startswith((b'#?RADIANCE', b'#?RGBE')):
                suffix = '.hdr'
            else:
                raise UnsupportedMaterial(f'image {image.name!r}: only PNG/JPEG/EXR/HDR source bytes are supported')
            if suffix in ('.exr', '.hdr'):
                import numpy as np
                if not len(image.pixels):
                    raise UnsupportedMaterial(f'image {image.name!r}: unavailable float pixels')
                pixels = np.empty(len(image.pixels), dtype=np.float32)
                image.pixels.foreach_get(pixels)
                if not np.isfinite(pixels).all():
                    raise UnsupportedMaterial(f'image {image.name!r}: nonfinite pixels')
            filename = key + suffix
            (self.directory / filename).write_bytes(data)
        else:
            if image.is_float:
                return self.float_image(image, key, cache, node, fallback)
            filename = key + '.png'
            copy = image.copy()
            try:
                # Image.copy() reloads file pixels and omits unsaved generated pixels.
                copy.pixels.foreach_set(image.pixels[:])
                copy.update()
                copy.filepath_raw = str(self.directory / filename)
                copy.file_format = 'PNG'
                copy.save()  # save_render would apply the display transform.
            finally:
                bpy.data.images.remove(copy)
        nearest = node.interpolation == 'Closest'
        self.images[key] = dict(file=filename, color_space=color_space,
                                wrap_s=wraps[node.extension], wrap_t=wraps[node.extension],
                                mag_filter='nearest' if nearest else 'linear',
                                min_filter='nearest' if nearest else 'linear_mipmap_linear')
        result = (dict(image=key,scale=[1.,1.,1.],offset=[0.,0.,0.],
                       reason='source colour-space role is unavailable',replacement=fallback) if fallback else key)
        self.image_keys[cache] = result
        return result

    def float_image(self, image, key, cache, node, fallback=None):
        """Snapshot dirty/generated scene-linear float pixels without an 8-bit codec."""
        import numpy as np
        if image.source != 'FILE' and image.colorspace_settings.name == 'sRGB':
            raise UnsupportedMaterial(f'float image {image.name!r}: generated pixels require a linear colour space')
        width, height = image.size
        if width == 0 or height == 0:
            raise UnsupportedMaterial(f'float image {image.name!r}: unavailable pixels')
        if image.channels not in (1,3,4):
            raise UnsupportedMaterial(f'float image {image.name!r}: unsupported channel count {image.channels}')
        raw = np.empty(width*height*image.channels, dtype=np.float32)
        image.pixels.foreach_get(raw)
        raw = raw.reshape(height,width,image.channels)
        pixels = np.ones((height,width,4), dtype=np.float32)
        pixels[:,:,:3] = raw[:,:,:3] if image.channels >= 3 else raw[:,:,:1]
        if image.channels == 4:
            pixels[:,:,3] = raw[:,:,3]
        if not np.isfinite(pixels).all():
            raise UnsupportedMaterial(f'float image {image.name!r}: nonfinite pixels')
        filename = key + '.exr'
        snapshot = bpy.data.images.new('__vork_float_image__', width=width, height=height,
                                       alpha=True, float_buffer=True)
        save_scene = bpy.data.scenes.new('__vork_float_image_save__')
        try:
            save_scene.render.image_settings.file_format = 'OPEN_EXR'
            save_scene.render.image_settings.color_depth = '32'
            save_scene.render.image_settings.color_mode = 'RGBA'
            snapshot.colorspace_settings.name = 'Linear Rec.709'
            snapshot.pixels.foreach_set(pixels.ravel())
            # EXR save_render writes scene-linear pixels; Image.save on generated
            # buffers can retain the buffer's PNG encoder despite file_format.
            snapshot.save_render(str(self.directory / filename), scene=save_scene)
        finally:
            bpy.data.scenes.remove(save_scene)
            bpy.data.images.remove(snapshot)
        wraps = {'REPEAT':'repeat','EXTEND':'clamp_to_edge','MIRROR':'mirrored_repeat'}
        nearest = node.interpolation == 'Closest'
        self.images[key] = dict(file=filename, color_space='Non-Color',
                                wrap_s=wraps[node.extension], wrap_t=wraps[node.extension],
                                mag_filter='nearest' if nearest else 'linear',
                                min_filter='nearest' if nearest else 'linear_mipmap_linear')
        result = (dict(image=key,scale=[1.,1.,1.],offset=[0.,0.,0.],
                       reason='source colour-space role is unavailable',replacement=fallback) if fallback else key)
        self.image_keys[cache] = result
        return result

    def write(self, material, *, bake_object=None, bake_resolution=None, available_attributes=None):
        return self._write(f'blender_material_{len(self.materials)}', material,
                           bake_object=bake_object, bake_resolution=bake_resolution,
                           available_attributes=available_attributes)

    def _write(self, key, material, *, bake_object=None, bake_resolution=None, available_attributes=None):
        engine = bpy.context.scene.render.engine
        target = 'CYCLES' if engine == 'CYCLES' else 'EEVEE'
        output = material.node_tree.get_output_node(target) if material.use_nodes else None
        from .volume import empty_volume
        if output is not None and not empty_volume(output.inputs['Volume']):
            if output.inputs['Surface'].is_linked or output.inputs['Displacement'].is_linked:
                raise UnsupportedMaterial(
                    f'{material.name}: mixed surface/displacement and Volume export not supported; '
                    'put homogeneous fog on a separate closed mesh with only Volume connected, '
                    'or disconnect Volume for a surface-only export; see vork help fog')
            from .volume import homogeneous
            try:
                source = homogeneous(output.inputs['Volume'])
            except ValueError as error:
                raise UnsupportedMaterial(f'{material.name}: {error}') from error
            filename = key + '.slang'
            (self.directory / filename).write_text(source)
            self.programs[key] = dict(file=filename)
            self.materials[key] = dict(program=key)
            self.reports[key] = dict(material=material.name, status='translated',
                                     substitutions=[], bakes=[], medium=True)
            return key
        bakes, baked = [], {}
        if (bake_object is None) != (bake_resolution is None):
            raise ValueError('value baking requires both bake_object and bake_resolution')
        def bake(socket):
            from .bake import bake_value
            pointer = socket.as_pointer()
            if pointer not in baked:
                image = f'blender_image_{len(self.images)}'
                filename = image + '.exr'
                receipt = bake_value(material, socket, bake_object, self.directory / filename, bake_resolution)
                from types import SimpleNamespace
                loaded = bpy.data.images.load(str(self.directory / filename), check_existing=False)
                try:
                    loaded.colorspace_settings.name = 'Linear Rec.709'
                    encoded = self.image(SimpleNamespace(image=loaded, extension='EXTEND', interpolation='Linear', name='value bake'))
                finally:
                    self.image_keys.pop((loaded.as_pointer(), 'EXTEND', 'Linear'), None)
                    bpy.data.images.remove(loaded)
                receipt['encoding'] = encoded if isinstance(encoded, dict) else dict(image=encoded, storage='native float EXR')
                bakes.append(receipt)
                baked[pointer] = encoded
            return baked[pointer]
        compiler = Compiler(material, self.image, engine=bpy.context.scene.render.engine,
                            best_effort=self.best_effort,
                            bake_input=bake if bake_object is not None else None,
                            curve_input=self.curve_image if self.curve_lookups else None,
                            available_attributes=available_attributes)
        source, inputs = compiler.compile()
        self.reports[key] = dict(material=material.name,
                                 status='partial' if compiler.substitutions else 'translated',
                                 substitutions=compiler.substitutions,
                                 object_specific=compiler.object_specific,
                                 bakes=bakes+[dict(**r,frame=float(bpy.context.scene.frame_current_final)) for r in compiler.curve_bakes])
        filename = key + '.slang'
        (self.directory / filename).write_text(source)
        self.programs[key] = dict(file=filename)
        self.materials[key] = dict(program=key, inputs=inputs,
                                   double_sided=not material.use_backface_culling,
                                   opacity='transmittance' if material.surface_render_method == 'BLENDED' else 'coverage')
        return key

    def write_asset(self, filepath, *, bake_resolution=None):
        """Bind material variants to actual exported primitive UV signatures.

        A shared Blender material may occur on meshes with and without UVs. Split
        only the file's material table; geometry bytes and source datablocks stay
        untouched. Index bindings remain unambiguous even for duplicate names.
        """
        import copy
        import json
        import struct
        path = Path(filepath)
        data = path.read_bytes()
        binary = data[:4] == b'glTF'
        if binary:
            size, kind = struct.unpack_from('<II', data, 12)
            if kind != 0x4E4F534A:
                raise ValueError('GLB has no initial JSON chunk')
            document = json.loads(data[20:20+size])
        else:
            document = json.loads(data)
        if bake_resolution is not None and (not isinstance(bake_resolution,int) or bake_resolution<=0):
            raise ValueError('value bake resolution must be a positive integer')
        from .geometry import _BAKE_SOURCE
        from collections import Counter
        from .render_geometry import direct_render_objects
        direct=direct_render_objects(bpy.context.view_layer.layer_collection)
        nodes=document.get('nodes',[])
        occurrences=Counter(node.get('extras',{}).get(_BAKE_SOURCE) for node in nodes if 'mesh' in node)
        contexts=[]
        if bake_resolution is None:
            contexts=[(mesh,None) for mesh in document.get('meshes',[])]
        else:
            # glTF can share one mesh across source objects. Split its JSON
            # primitive table, preserving shared geometry/accessor bytes, before
            # assigning object-specific baked material variants.
            for node in nodes:
                source=node.get('extras',{}).get(_BAKE_SOURCE)
                if 'mesh' not in node:
                    continue
                candidates=[obj for obj in bpy.data.objects if obj.name==source]
                obj=candidates[0] if len(candidates)==1 else None
                if obj is None or obj.type!='MESH' or obj not in direct or occurrences[source]!=1:
                    self.asset_reports.append(dict(file=str(path),object=source,
                        reason='value baking requires one identified direct source mesh occurrence',
                        replacement='retain translated sockets and reported destination defaults'))
                    obj=None
                mesh=copy.deepcopy(document['meshes'][node['mesh']])
                node['mesh']=len(document['meshes']);document['meshes'].append(mesh)
                contexts.append((mesh,obj))
        for node in nodes:
            extras=node.get('extras',{})
            extras.pop(_BAKE_SOURCE,None)
            if not extras:node.pop('extras',None)
        variants, bindings = {}, {}
        materials = document.get('materials', [])
        original_count = len(materials)
        for mesh,bake_object in contexts:
            for primitive in mesh.get('primitives', []):
                index = primitive.get('material')
                if index is None:
                    continue
                if index >= original_count:
                    raise ValueError('primitive references a missing source material')
                source = materials[index]
                material = bpy.data.materials.get(source.get('name', ''))
                if material is None:
                    self.asset_reports.append(dict(file=str(path), material=source.get('name'), reason='source Blender material unavailable', replacement='retain emitted glTF material'))
                    continue  # Retain the emitted glTF material when source is unavailable.
                attributes = frozenset(k for k in primitive.get('attributes', {}) if k.startswith('_VORK_UV_'))
                shared_identity = index, attributes
                own_identity = index, attributes, bake_object
                identity = shared_identity if bake_resolution is None or shared_identity in variants else own_identity
                if identity not in variants:
                    key = self.write(material, available_attributes=attributes,
                                     bake_object=bake_object, bake_resolution=bake_resolution if bake_object is not None else None)
                    if not self.reports[key].get('object_specific', False):
                        identity = shared_identity
                    target = len(materials)
                    materials.append(copy.deepcopy(source))
                    variants[identity] = target
                    bindings[f'#{target}'] = key
                    self.reports[key]['source_material_index'] = index
                    self.reports[key]['attributes'] = sorted(attributes)
                primitive['material'] = variants[identity]
        if bake_resolution is not None:
            # Per-object variants replace the original primitive tables. Keep
            # only referenced meshes; accessor and buffer storage remains shared.
            used = sorted({node['mesh'] for node in nodes if 'mesh' in node})
            remap = {old: new for new, old in enumerate(used)}
            document['meshes'] = [document['meshes'][old] for old in used]
            for node in nodes:
                if 'mesh' in node: node['mesh'] = remap[node['mesh']]
        if binary:
            encoded = json.dumps(document, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 4)
            tail = data[20+size:]
            path.write_bytes(struct.pack('<III', 0x46546C67, 2, 20+len(encoded)+len(tail))
                             + struct.pack('<II', len(encoded), kind) + encoded + tail)
        else:
            path.write_text(json.dumps(document, separators=(',', ':'))+'\n')
        return bindings
