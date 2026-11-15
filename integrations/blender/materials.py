"""Lower supported Blender material sockets to ordinary vork surface programs.

No shader-node interpreter runs in the renderer. Source sockets become typed
functions; images remain material inputs, and coordinates retain their footprint.
"""
import math

from mathutils import Euler, Matrix, Vector

from .geometry import uv_attribute


class UnsupportedMaterial(ValueError):
    pass


class InvalidMaterial(ValueError):
    """Malformed source graphs cannot be repaired by feature substitution."""
    pass


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise UnsupportedMaterial('material contains a nonfinite value')
    return repr(value)


def vector(value):
    return 'float3(' + ','.join(number(v) for v in value[:3]) + ')'


def width(socket):
    return 3 if socket.type in ('VECTOR', 'RGBA') else 1


def convert(expression, source, destination):
    if source == destination:
        return expression
    if source == 1 and destination == 3:
        return f'float3({expression})'
    if source == 3 and destination == 1:
        return f'dot({expression},float3(0.2126,0.7152,0.0722))'
    raise UnsupportedMaterial(f'cannot convert {source} to {destination}')


PRELUDE = '''
Closure blender_empty_shader() {
    Closure c={};c.base_weight=0.;c.specular_weight=0.;c.geometry_opacity=1.;return c;
}

float3 blender_to_hsv(float3 rgb) {
    float hi=max(rgb.x,max(rgb.y,rgb.z)),lo=min(rgb.x,min(rgb.y,rgb.z));
    float delta=hi-lo,h=0.;
    if(delta>0.) {
        if(hi==rgb.x) h=(rgb.y-rgb.z)/delta;
        else if(hi==rgb.y) h=2.+(rgb.z-rgb.x)/delta;
        else h=4.+(rgb.x-rgb.y)/delta;
        h=frac(h/6.);
    }
    return float3(h,hi==0.?0.:delta/hi,hi);
}
float3 blender_from_hsv(float3 hsv) {
    float3 rgb=saturate(abs(frac(hsv.x+float3(0.,2./3.,1./3.))*6.-3.)-1.);
    return hsv.z*lerp(float3(1.),rgb,hsv.y);
}
float3 blender_box_weights(float3 normal,float blend) {
    float3 w=abs(normal);
    if(blend==0.) {
        if(w.x>=w.y && w.x>=w.z) return float3(1,0,0);
        if(w.y>=w.z) return float3(0,1,0);
        return float3(0,0,1);
    }
    w/=max(w.x+w.y+w.z,1.e-20);
    float limit=(1.+blend)*.5;
    if(w.x>limit*(w.x+w.y) && w.x>limit*(w.x+w.z)) return float3(1,0,0);
    if(w.y>limit*(w.y+w.x) && w.y>limit*(w.y+w.z)) return float3(0,1,0);
    if(w.z>limit*(w.z+w.x) && w.z>limit*(w.z+w.y)) return float3(0,0,1);
    if(w.x<(1.-limit)*(w.y+w.z)) {
        float y=(w.y/(w.y+w.z)-(1.-limit))/blend;
        return float3(0,y,1.-y);
    }
    if(w.y<(1.-limit)*(w.x+w.z)) {
        float x=(w.x/(w.x+w.z)-(1.-limit))/blend;
        return float3(x,0,1.-x);
    }
    if(w.z<(1.-limit)*(w.x+w.y)) {
        float x=(w.x/(w.x+w.y)-(1.-limit))/blend;
        return float3(x,1.-x,0);
    }
    return ((2.-limit)*w+limit-1.)/(2.*limit-1.);
}
float2 blender_box_uv(float3 p,float3 n,int axis) {
    if(axis==0) return float2(n.x<0.?1.-p.y:p.y,1.-p.z);
    if(axis==1) return float2(n.y>0.?1.-p.x:p.x,1.-p.z);
    return float2(n.z>0.?1.-p.y:p.y,1.-p.x);
}
float4 blender_image(Image image,float2 uv,float2 x,float2 y,ShadingContext ctx) {
    ctx.duv_dx[0]=x-uv;ctx.duv_dy[0]=y-uv;
    return sample(image,uv,ctx);
}
'''


class Compiler:
    def __init__(self, material, image_input, engine="EEVEE", *, best_effort=False, bake_input=None, curve_input=None, available_attributes=None, medium=False):
        if engine not in ("EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
            raise UnsupportedMaterial(f'unsupported surface engine {engine!r}')
        self.engine = "CYCLES" if engine == "CYCLES" else "EEVEE"
        self.medium = medium
        self.available_attributes = available_attributes
        self.bake_input = bake_input
        self.object_specific = False
        self.curve_input = curve_input
        self.curve_bakes = []
        self.curve_lookup_names = set()
        self.best_effort = best_effort
        self.substitutions = []
        self.material = material
        self.image_input = image_input
        self.frames = []
        self.functions = []
        self.cache = {}
        self.degradations = 0
        self.visiting = set()
        self.lut_counter = 0
        self.images = {}
        self.uv_fields = {}
        self.inputs = {}
        # Numeric program inputs by field name; other fields are images or UV attributes.
        self.field_types = {}
        # The surface's representative base colour and roughness sources, exposed as the
        # program inputs `base_color` and `roughness` (see preview_inputs).
        self.preview_constants = {}
        self.preview_images = {}

    def subsurface_transport_restriction(self):
        output = self.material.node_tree.get_output_node(self.engine)
        if output is None:
            return 'no active surface output'
        def visit_input(socket, frames, active):
            if not socket.is_linked:
                return None  # The emitted empty shader is opaque black.
            return visit(socket.links[0].from_socket, frames, active)
        def visit(socket, frames, active):
            node=socket.node
            identity=(node.as_pointer(),socket.identifier,tuple(n.as_pointer() for n in frames))
            if identity in active:
                return 'cyclic surface transport'
            active=active|{identity}
            if node.mute:
                bypass=next((l.from_socket for l in node.internal_links if l.to_socket==socket),None)
                return visit_input(bypass,frames,active) if bypass is not None else 'muted surface has no bypass'
            if node.type=='REROUTE':
                return visit_input(node.inputs[0],frames,active)
            if node.type=='GROUP':
                if node in frames:
                    return 'cyclic group surface transport'
                output=next((n for n in node.node_tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output),None)
                target=next((s for s in output.inputs if s.identifier==socket.identifier),None) if output else None
                return visit_input(target,frames+[node],active) if target else 'group surface output unavailable'
            if node.type=='GROUP_INPUT':
                target=next((s for s in frames[-1].inputs if s.identifier==socket.identifier),None) if frames else None
                return visit_input(target,frames[:-1],active) if target else 'group surface input unavailable'
            if node.type in ('MIX_SHADER','ADD_SHADER'):
                for source in node.inputs:
                    if source.type=='SHADER':
                        restriction=visit_input(source,frames,active)
                        if restriction:return restriction
                return None
            if node.type=='BSDF_PRINCIPLED':
                for name,expected in [('Alpha',1.),('Transmission Weight',0.),('Thin Wall',False)]:
                    source=node.inputs[name]
                    if source.is_linked or source.default_value!=expected:
                        return f'{node.name}.{name}: subsurface requires proven opaque, nontransmitting thick surfaces'
                return None
            if node.type in ('BSDF_DIFFUSE','BSDF_GLOSSY','EMISSION'):
                return None
            return f'{node.name}: {node.type} is not proven compatible with opaque subsurface transport'
        return visit_input(output.inputs['Surface'],[],set())

    def fail(self, node, socket, message):
        raise UnsupportedMaterial(f'material {self.material.name!r}: {node.name!r}.{socket}: {message}')

    def input(self, socket, desired=None):
        desired = desired or width(socket)
        if socket.is_linked and not getattr(socket, 'is_unavailable', False):
            link = socket.links[0]
            try:
                name, source = self.output(link.from_socket)
            except UnsupportedMaterial as error:
                if not self.best_effort or desired in ('Closure', 'Blend'):
                    raise
                self.object_specific = True
                reason = str(error)
                if self.bake_input is not None:
                    try:
                        expression = self.baked_value(link.from_socket, desired)
                    except (UnsupportedMaterial, RuntimeError) as bake_error:
                        reason += f'; value bake unavailable: {bake_error}'
                    else:
                        self.report(socket.node, socket.name, reason, 'object- and frame-specific CPU value texture bake')
                        return expression
                if socket.name in ('Normal', 'Coat Normal', 'Tangent') and socket.type == 'VECTOR':
                    field = 'TANGENT' if socket.name == 'Tangent' else 'NORMAL'
                    self.report(socket.node, socket.name, reason, f'use unperturbed shading {field.lower()}')
                    return convert(f'float3(ctx.{field}.x,-ctx.{field}.z,ctx.{field}.y)', 3, desired)
                self.report(socket.node, socket.name, reason, 'use destination socket default')
                return self.default(socket, desired)
            expression = f'{name}(ctx,inputs,off)'
            if source == 3 and desired == 1 and link.from_socket.type == 'VECTOR':
                return f'dot({expression},float3(1./3.))'
            return convert(expression, source, desired)
        if desired in ('Closure', 'Blend'):
            self.fail(socket.node, socket.name, 'unconnected shader socket')
        preview = self.preview_input(socket, desired)
        return preview if preview is not None else self.default(socket, desired)

    def frame_key(self, frames, socket):
        return tuple(frame.as_pointer() for frame in frames), socket.as_pointer()

    def preview_inputs(self, output):
        """Find the surface's representative BSDF and where its base colour and roughness come from.

        The representative is the heaviest Principled or Diffuse BSDF in the surface's
        shader mixture, else its heaviest Glossy BSDF. Its colour source is the constant, or
        the image read through the active UV (Mapping placement is not carried), that feeds
        Base Color, followed back through colour adjustments and the heavier side of colour
        mixes. Those values become the
        program inputs `base_color` and `roughness`, which the program reads like any input,
        so it renders unchanged. A blockout preview that cannot run material programs reads
        them as data, without the adjustments. Emission, the light a blockout keeps, becomes
        `emission_color` and `emission_luminance` from the heaviest Emission shader or emitting
        Principled BSDF, the same way.
        """
        def linked(socket):
            if not socket.is_linked or socket.links[0].is_muted:
                return None
            link = socket.links[0]
            return None if link.from_node.mute else link
        def closure(socket, frames, pick, depth=0):
            """The best closure `pick` ranks in this shader mixture: (rank, node, frames, ...)."""
            link = linked(socket)
            if link is None or depth > 64:
                return None
            node = link.from_node
            if node.type == 'REROUTE':
                return closure(node.inputs[0], frames, pick, depth+1)
            if node.type == 'GROUP' and node.node_tree is not None:
                group = next((n for n in node.node_tree.nodes if n.type == 'GROUP_OUTPUT' and n.is_active_output), None)
                target = next((s for s in group.inputs if s.identifier == link.from_socket.identifier), None) if group else None
                return closure(target, frames+[node], pick, depth+1) if target is not None else None
            if node.type == 'GROUP_INPUT' and frames:
                target = next((s for s in frames[-1].inputs if s.identifier == link.from_socket.identifier), None)
                return closure(target, frames[:-1], pick, depth+1) if target is not None else None
            if node.type in ('MIX_SHADER', 'ADD_SHADER'):
                shaders = [s for s in node.inputs if s.type == 'SHADER']
                if node.type == 'MIX_SHADER' and not node.inputs[0].is_linked:
                    factor = node.inputs[0].default_value
                    # A closure with no weight is not part of the surface; the heavier one represents it.
                    shaders = [shaders[0]] if factor <= 0 else [shaders[1]] if factor >= 1 else shaders[::-1] if factor > .5 else shaders
                found = [closure(s, frames, pick, depth+1) for s in shaders]
                return min((f for f in found if f), key=lambda f: f[0], default=None)
            found = pick(node)
            return (found[0], node, frames, *found[1:]) if found else None
        def surface(node):
            return {'BSDF_PRINCIPLED': (0, 'Base Color', 'Roughness'), 'BSDF_DIFFUSE': (0, 'Color', None),
                    'BSDF_GLOSSY': (1, 'Color', 'Roughness')}.get(node.type)
        def emitter(node):
            if node.type == 'EMISSION':
                return 0, 'Color', 'Strength'
            strength = node.inputs.get('Emission Strength') if node.type == 'BSDF_PRINCIPLED' else None
            if strength is not None and (strength.is_linked or strength.default_value != 0):
                return 0, 'Emission Color', 'Emission Strength'
            return None
        def active_uv(socket, depth=0):
            """Whether an image vector is the active UV, ignoring Mapping placement."""
            link = linked(socket) if socket.is_linked else None
            if not socket.is_linked:
                return True
            if link is None or depth > 64:
                return False
            node = link.from_node
            if node.type == 'REROUTE':
                return active_uv(node.inputs[0], depth+1)
            if node.type == 'MAPPING' and node.vector_type in ('POINT', 'TEXTURE'):
                return active_uv(node.inputs['Vector'], depth+1)
            if node.type == 'TEX_COORD':
                return link.from_socket.name == 'UV' and not node.from_instancer
            return node.type == 'UVMAP' and not node.uv_map and not node.from_instancer
        def source(socket, frames, depth=0):
            """('constant', key of the socket holding the value, that socket), ('image', node), or
            ('texture',) for an image this cannot carry (its colour is the texture's, so no constant is)."""
            if depth > 64:
                return None
            if not socket.is_linked:
                return 'constant', self.frame_key(frames, socket), socket
            link = linked(socket)
            if link is None:
                return None
            node, out = link.from_node, link.from_socket
            if node.type == 'REROUTE':
                return source(node.inputs[0], frames, depth+1)
            if node.type in ('RGB', 'VALUE'):
                return 'constant', self.frame_key(frames, out), out
            if node.type == 'GROUP' and node.node_tree is not None:
                group = next((n for n in node.node_tree.nodes if n.type == 'GROUP_OUTPUT' and n.is_active_output), None)
                target = next((s for s in group.inputs if s.identifier == out.identifier), None) if group else None
                return source(target, frames+[node], depth+1) if target is not None else None
            if node.type == 'GROUP_INPUT' and frames:
                target = next((s for s in frames[-1].inputs if s.identifier == out.identifier), None)
                return source(target, frames[:-1], depth+1) if target is not None else None
            if node.type == 'TEX_IMAGE':
                if (out.name == 'Color' and node.image is not None and node.projection == 'FLAT'
                        and active_uv(node.inputs['Vector']) and not node.outputs['Alpha'].is_linked):
                    return 'image', node
                return 'texture',
            if width(socket) != 3:
                return None
            # Colour adjustments and mixes: the colour they start from.
            if node.type in ('CURVE_RGB', 'HUE_SAT', 'BRIGHTCONTRAST', 'GAMMA'):
                return source(node.inputs['Color'], frames, depth+1)
            if node.type in ('MIX', 'MIX_RGB') and (node.type == 'MIX_RGB' or node.data_type == 'RGBA'):
                factor, first, second = (node.inputs[0], node.inputs[6], node.inputs[7]) if node.type == 'MIX' else node.inputs[:3]
                if not factor.is_linked:
                    heavier = [second, first] if factor.default_value > .5 else [first, second]
                    return next((found for found in (source(s, frames, depth+1) for s in heavier) if found), None)
                # A varying mix: a texture on either side outweighs a constant.
                found = [f for f in (source(s, frames, depth+1) for s in (first, second)) if f]
                return next((f for kind in ('image', 'texture', 'constant') for f in found if f[0] == kind), None)
            return None
        sockets = []
        found = closure(output.inputs['Surface'], [], surface)
        if found is not None:
            _, node, frames, color, roughness = found
            sockets += [('base_color', node, frames, color, (1, 3)), ('roughness', node, frames, roughness, (1,))]
        # Emission is the light a blockout keeps: its colour and strength (Blender's Strength is
        # the closure's emission_luminance).
        found = closure(output.inputs['Surface'], [], emitter)
        if found is not None:
            _, node, frames, color, strength = found
            sockets += [('emission_color', node, frames, color, (1, 3)), ('emission_luminance', node, frames, strength, (1,))]
        for field, node, frames, name, widths in sockets:
            if name is None:
                continue
            value = source(node.inputs[name], frames)
            if value is None:
                continue
            if value[0] == 'texture':
                continue
            if value[0] == 'image':
                if field in ('base_color', 'emission_color'):
                    self.preview_images.setdefault(value[1].as_pointer(), []).append(field)
                continue
            _, key, socket = value
            if width(socket) in widths:
                self.preview_constants.setdefault(key, []).append(field)

    def preview_input(self, socket, desired):
        """The program input expression for a representative constant, registering the field."""
        fields = self.preview_constants.get(self.frame_key(self.frames, socket))
        source = width(socket)
        if not fields or desired != source:
            return None
        value = socket.default_value
        value = [float(v) for v in value[:3]] if source == 3 else float(value)
        if not all(math.isfinite(v) for v in (value if source == 3 else [value])):
            return None
        # One socket may feed both fields; each holds the value and the program reads the first.
        for field in fields:
            if self.inputs.setdefault(field, value) != value:
                return None
            self.field_types[field] = 'float3' if source == 3 else 'float'
        return f'inputs.{fields[0]}'

    def baked_value(self, socket, desired):
        if self.available_attributes is not None and uv_attribute() not in self.available_attributes:
            raise UnsupportedMaterial('value baking requires an exported active render UV')
        image = self.bake_input(socket)
        encoding = image if isinstance(image, dict) else None
        if encoding is not None:
            image = encoding['image']
        field = f'baked_image_{len(self.inputs)}'
        self.inputs[field] = {'image': image}
        self.uv(None, socket.node)
        uv = self.uv_fields[uv_attribute()]
        name = f'baked_value_{len(self.functions)}'
        decode = f'*{vector(encoding["scale"])}+{vector(encoding["offset"])}' if encoding else ''
        self.functions.append(f'float3 {name}(ShadingContext ctx,Inputs inputs,float2 off) {{'
                              f'let uv=read_attribute(inputs.{uv},ctx);'
                              'float2 p=uv.value+uv.dx*off.x+uv.dy*off.y;'
                              f'return blender_image(inputs.{field},p,p+uv.dx,p+uv.dy,ctx).rgb{decode};}}')
        expression = f'{name}(ctx,inputs,off)'
        if socket.type == 'VECTOR' and desired == 1:
            return f'dot({expression},float3(1./3.))'
        return convert(expression, 3, desired)

    def report(self, node, socket, reason, replacement):
        if replacement != 'object- and frame-specific CPU value texture bake':
            self.degradations += 1
        row = dict(material=self.material.name, node=node.name, socket=socket,
                   reason=reason, replacement=replacement)
        if row not in self.substitutions:
            self.substitutions.append(row)

    def default(self, socket, desired):
        if socket.node.type not in ('GROUP', 'GROUP_INPUT', 'GROUP_OUTPUT') and socket.type == 'VECTOR' and socket.name in ('Normal', 'Coat Normal', 'Tangent'):
            if self.medium:
                self.fail(socket.node, socket.name, 'implicit surface frame is unavailable in medium programs')
            field = 'TANGENT' if socket.name == 'Tangent' else 'NORMAL'
            return convert(f'float3(ctx.{field}.x,-ctx.{field}.z,ctx.{field}.y)', 3, desired)
        value = socket.default_value
        source = width(socket)
        expression = vector(value) if source == 3 else number(value)
        if source == 3 and desired == 1 and socket.type == 'VECTOR':
            return f'dot({expression},float3(1./3.))'
        return convert(expression, source, desired)

    def output(self, socket):
        key = (tuple(frame.as_pointer() for frame in self.frames), socket.as_pointer())
        if key in self.cache:
            name, result, degraded = self.cache[key]
            self.degradations += int(degraded)
            return name, result
        node = socket.node
        if key in self.visiting:
            raise InvalidMaterial(f'material {self.material.name!r}: {node.name!r}.{socket.name}: cyclic graph')
        self.visiting.add(key)
        defaults_before = self.degradations
        name = f'node_{len(self.cache)}_{len(self.visiting)}'
        try:
            try:
                body, result = self.lower(node, socket)
            except UnsupportedMaterial as error:
                if not self.best_effort or socket.type != 'SHADER':
                    raise
                self.report(node, socket.name, str(error),
                            'diffuse surface using Color or Base Color; viewport colour otherwise')
                color = node.inputs.get('Color') or node.inputs.get('Base Color')
                base = self.input(color, 3) if color is not None else vector(self.material.diffuse_color)
                body = f'Closure c={{}};c.base_color={base};c.specular_weight=0.;return c;'
                result = 'Closure'
        finally:
            self.visiting.remove(key)
        self.cache[key] = (name, result, self.degradations != defaults_before)
        ty = 'float3' if result == 3 else 'float' if result == 1 else result
        self.functions.append(f'{ty} {name}(ShadingContext ctx,Inputs inputs,float2 off) {{\n{body}\n}}')
        return name, result

    def uv(self, name, node):
        if getattr(node, 'from_instancer', False):
            self.fail(node, 'UV', 'instancer UV coordinates are not translated')
        attribute = uv_attribute(name)
        if self.available_attributes is not None and attribute not in self.available_attributes:
            if not self.best_effort:
                self.fail(node, name or 'UV', f'missing exported source UV {attribute}')
            self.report(node, name or 'UV', f'missing exported source UV {attribute}', 'zero UV for this primitive variant')
            return 'return float3(0.);', 3
        if attribute not in self.uv_fields:
            field = 'uv_' + str(len(self.uv_fields))
            self.uv_fields[attribute] = field
            self.inputs[field] = dict(attribute=attribute)
        field = self.uv_fields[attribute]
        return (f'let uv=read_attribute(inputs.{field},ctx);'
                'float2 p=uv.value+uv.dx*off.x+uv.dy*off.y;'
                'return float3(p.x,1.-p.y,0.);', 3)

    def lower(self, node, socket):
        get = lambda name, desired=None: self.input(node.inputs[name], desired)
        if node.mute:
            bypass = next((link.from_socket for link in node.internal_links if link.to_socket == socket), None)
            if bypass is None:
                self.fail(node, socket.name, 'muted output has no bypass')
            if socket.type == 'SHADER':
                if not bypass.is_linked:
                    self.fail(node, socket.name, 'unconnected muted shader bypass')
                name, result = self.output(bypass.links[0].from_socket)
                return f'return {name}(ctx,inputs,off);', result
            return f'return {self.input(bypass, width(socket))};', width(socket)
        if self.medium and node.type in ('TEX_COORD','UVMAP','NEW_GEOMETRY','LAYER_WEIGHT','FRESNEL','NORMAL_MAP','BUMP','TEX_IMAGE'):
            self.fail(node, socket.name, 'surface context/image inputs are unavailable in numeric medium programs')
        if node.type == 'LIGHT_PATH' and self.best_effort:
            if socket.name in ('Is Camera Ray','Is Shadow Ray','Is Diffuse Ray','Is Glossy Ray','Is Singular Ray','Is Reflection Ray','Is Transmission Ray','Ray Depth','Diffuse Depth','Glossy Depth','Transparent Depth','Transmission Depth'):
                self.report(node, socket.name, 'ray-kind context is not exposed to material programs', 'primary camera-ray value; secondary/shadow contexts approximated')
                return 'return '+('1.' if socket.name == 'Is Camera Ray' else '0.')+';', 1
        if node.type == 'GROUP':
            output = next((n for n in node.node_tree.nodes if n.type == 'GROUP_OUTPUT' and n.is_active_output), None)
            if output is None:
                self.fail(node, socket.name, 'missing active group output')
            target = next((s for s in output.inputs if s.identifier == socket.identifier), None)
            if target is None:
                self.fail(node, socket.name, 'missing group interface socket')
            self.frames.append(node)
            try:
                if target.type == 'SHADER':
                    if not target.is_linked:
                        self.fail(node, socket.name, 'unconnected group shader output')
                    name, result = self.output(target.links[0].from_socket)
                    return f'return {name}(ctx,inputs,off);', result
                return f'return {self.input(target, width(socket))};', width(socket)
            finally:
                self.frames.pop()
        if node.type == 'GROUP_INPUT':
            if not self.frames:
                self.fail(node, socket.name, 'group input outside an instance')
            instance = self.frames.pop()
            try:
                target = next((s for s in instance.inputs if s.identifier == socket.identifier), None)
                if target is None:
                    self.fail(node, socket.name, 'missing group input socket')
                if target.type == 'SHADER':
                    if not target.is_linked:
                        self.fail(node, socket.name, 'unconnected group shader input')
                    name, result = self.output(target.links[0].from_socket)
                    return f'return {name}(ctx,inputs,off);', result
                return f'return {self.input(target, width(socket))};', width(socket)
            finally:
                self.frames.append(instance)
        if node.type in ('RGB', 'VALUE'):
            preview = self.preview_input(socket, width(socket))
            value = preview or (vector(socket.default_value) if width(socket) == 3 else number(socket.default_value))
            return f'return {value};', width(socket)
        if node.type == 'REROUTE':
            if not node.inputs[0].is_linked:
                self.fail(node, socket.name, 'unconnected reroute')
            name, result = self.output(node.inputs[0].links[0].from_socket)
            return f'return {name}(ctx,inputs,off);', result
        if node.type == 'TEX_COORD':
            if socket.name == 'UV':
                return self.uv(None, node)
            if node.object is not None:
                self.fail(node, socket.name, 'external coordinate objects are not translated')
            if socket.name == 'Generated':
                if not self.best_effort:
                    self.fail(node, socket.name, 'source ORCO coordinates are not retained by glTF')
                self.report(node, socket.name, 'source ORCO coordinates are not retained by glTF', 'normalized imported rest bounds; Generated coordinates approximated')
                return 'float3 p=texture_coordinate(ctx,off).Generated;return float3(p.x,1.-p.z,p.y);', 3
            if socket.name in ('Object', 'Normal'):
                return f'float3 p=texture_coordinate(ctx,off).{socket.name};return float3(p.x,-p.z,p.y);', 3
            self.fail(node, socket.name, 'coordinate output is not translated')
        if node.type == 'UVMAP':
            return self.uv(node.uv_map or None, node)
        if node.type == 'MAPPING':
            for input in node.inputs:
                if input.name != 'Vector' and input.is_linked:
                    self.fail(node, input.name, 'linked mapping parameter is not translated')
            rotation = Euler(tuple(node.inputs['Rotation'].default_value), 'XYZ').to_matrix()
            scale = Vector(node.inputs['Scale'].default_value)
            location = Vector(node.inputs['Location'].default_value) if node.inputs.get('Location') else Vector((0, 0, 0))
            matrix = rotation @ Matrix.Diagonal(scale)
            if node.vector_type in ('TEXTURE', 'NORMAL'):
                if any(v == 0 for v in scale):
                    self.fail(node, 'Scale', 'singular inverse mapping')
                matrix.invert()
                if node.vector_type == 'NORMAL':
                    matrix.transpose()
                else:
                    location = -(matrix @ location)
            if node.vector_type != 'POINT' and node.vector_type != 'TEXTURE':
                location = Vector((0, 0, 0))
            components = [f'dot(p,{vector(row)})+{number(offset)}' for row, offset in zip(matrix, location)]
            expression = 'float3(' + ','.join(components) + ')'
            if node.vector_type == 'NORMAL':
                expression = f'normalize({expression})'
            return f'float3 p={get("Vector",3)};return {expression};', 3
        if node.type == 'TEX_IMAGE':
            if node.image is None:
                self.fail(node, socket.name, 'missing image')
            alpha = socket.name == 'Alpha'
            if node.projection not in ('FLAT', 'BOX'):
                self.fail(node, socket.name, f'{node.projection} projection is not translated')
            if not alpha and node.outputs['Alpha'].is_linked:
                self.fail(node, socket.name, 'image colour with linked Alpha requires qualified premultiplication semantics')
            key = node.as_pointer()
            if key not in self.images:
                # A representative image is named for its fields; the program reads the first.
                field, *aliases = self.preview_images.get(key, [f'image_{len(self.images)}'])
                image_name = self.image_input(node)
                encoding = image_name if isinstance(image_name, dict) else None
                self.images[key] = (field, encoding)
                for name in [field, *aliases]:
                    self.inputs[name] = {'image': encoding['image'] if encoding else image_name}
            field, encoding = self.images[key]
            if encoding:
                if not self.best_effort:
                    self.fail(node, socket.name, 'image export requires a reported encoding substitution')
                self.report(node, socket.name, encoding.get('reason','source image exported with range encoding'), encoding['replacement'])
            if node.inputs['Vector'].is_linked:
                coordinate = get('Vector', 3)
            else:
                # Image Texture's implicit vector uses the active render UV.
                body, _ = self.uv(None, node)
                name = f'image_uv_{len(self.functions)}'
                self.functions.append(f'float3 {name}(ShadingContext ctx,Inputs inputs,float2 off) {{{body}}}')
                coordinate = f'{name}(ctx,inputs,off)'
            # Reevaluate the complete coordinate expression at the footprint neighbours.
            body = f'float3 p={coordinate};off.x+=1.;float3 px={coordinate};off.x-=1.;off.y+=1.;float3 py={coordinate};off.y-=1.;\n'
            if node.projection == 'FLAT':
                body += f'float4 value=blender_image(inputs.{field},float2(p.x,1.-p.y),float2(px.x,1.-px.y),float2(py.x,1.-py.y),ctx);'
            else:
                body += f'''float3 normal=texture_coordinate(ctx,off).Normal;
float3 n=float3(normal.x,-normal.z,normal.y);
float3 weights=blender_box_weights(n,{number(node.projection_blend)});
float4 value=float4(0.);
for(int axis=0;axis<3;axis++) {{
    if(weights[axis]>0.) value+=weights[axis]*blender_image(inputs.{field},blender_box_uv(p,n,axis),blender_box_uv(px,n,axis),blender_box_uv(py,n,axis),ctx);
}}'''
            if encoding:
                body += f'value.rgb=value.rgb*{vector(encoding["scale"])}+{vector(encoding["offset"])};'
            channel, w = ('a', 1) if alpha else ('rgb', 3)
            return body + f'return value.{channel};', w
        if node.type in ('MIX', 'MIX_RGB'):
            if node.type == 'MIX':
                if node.data_type in ('FLOAT', 'VECTOR'):
                    w = 1 if node.data_type == 'FLOAT' else 3
                    first, second = (2,3) if w == 1 else (4,5)
                    factor = self.input(node.inputs[1 if w == 3 and node.factor_mode == 'NON_UNIFORM' else 0], w if w == 3 and node.factor_mode == 'NON_UNIFORM' else 1)
                    if node.clamp_factor:
                        factor = f'saturate({factor})'
                    return f'return lerp({self.input(node.inputs[first],w)},{self.input(node.inputs[second],w)},{factor});',w
                if node.data_type != 'RGBA':
                    self.fail(node, socket.name, 'unsupported Mix data type')
                factor, first, second = node.inputs[0], node.inputs[6], node.inputs[7]
                clamp_factor, clamp_result = node.clamp_factor, node.clamp_result
            else:
                factor, first, second = node.inputs[:3]
                clamp_factor, clamp_result = True, node.use_clamp
            modes = {'MIX':'Mix','MULTIPLY':'Multiply','ADD':'Add','SUBTRACT':'Subtract','SCREEN':'Screen','OVERLAY':'Overlay','DIVIDE':'Divide','DIFFERENCE':'Difference','DARKEN':'Darken','LIGHTEN':'Lighten','SOFT_LIGHT':'Soft_Light','LINEAR_LIGHT':'Linear_Light','HUE':'Hue','SATURATION':'Saturation','COLOR':'Color','VALUE':'Value'}
            if node.blend_type not in modes:
                self.fail(node, socket.name, f'{node.blend_type} blend is not translated')
            return f'return mix({self.input(first,3)},{self.input(second,3)},{self.input(factor,1)},BlendMode::{modes[node.blend_type]},{str(clamp_factor).lower()},{str(clamp_result).lower()});', 3
        if node.type == 'INVERT':
            return f'return lerp({get("Color",3)},1.-{get("Color",3)},{get("Fac",1)});', 3
        if node.type == 'VALTORGB':
            ramp=node.color_ramp
            interpolation={'CONSTANT':'Constant','LINEAR':'Linear','EASE':'Ease','CARDINAL':'Cardinal','B_SPLINE':'B_Spline'}[ramp.interpolation]
            hue={'NEAR':'Near','FAR':'Far','CW':'CW','CCW':'CCW'}[ramp.hue_interpolation]
            stops=','.join('{'+number(e.position)+',float4('+','.join(number(v) for v in e.color)+')}' for e in ramp.elements)
            channel='a' if socket.name=='Alpha' else 'rgb'
            return f'RampStop stops[{len(ramp.elements)}]={{{stops}}};return color_ramp({get("Fac",1)},stops,RampInterpolation::{interpolation},RampColorMode::{ramp.color_mode},HueInterpolation::{hue}).{channel};',1 if channel=='a' else 3
        if node.type == 'LAYER_WEIGHT':
            if socket.name == 'Fresnel':
                normal=get('Normal',3) if node.inputs['Normal'].is_linked else 'float3(ctx.NORMAL.x,-ctx.NORMAL.z,ctx.NORMAL.y)'
                return f'float3 n={normal};return Layer_Weight_Fresnel({get("Blend",1)},float3(n.x,n.z,-n.y),ctx);',1
            if socket.name!='Facing':
                self.fail(node,socket.name,'unknown Layer Weight output')
            normal=get('Normal',3) if node.inputs['Normal'].is_linked else 'float3(surface_normal.x,-surface_normal.z,surface_normal.y)'
            return f'''float3 surface_normal=ctx.NORMAL+ctx.unperturbed_normal_dx*off.x+ctx.unperturbed_normal_dy*off.y;
float3 incoming=ctx.incoming+ctx.incoming_dx*off.x+ctx.incoming_dy*off.y;
float3 n={normal};float3 v=float3(incoming.x,-incoming.z,incoming.y);
float facing=abs(dot(normalize(n),normalize(v)));float blend={get('Blend',1)};
if(blend>=1.) return facing<1.?1.:0.;
float exponent=blend<.5?2.*blend:.5/(1.-blend);
return 1.-pow(facing,exponent);''',1
        if node.type == 'BSDF_DIFFUSE':
            roughness = node.inputs['Roughness']
            if self.engine == 'CYCLES' and (roughness.is_linked or roughness.default_value != 0):
                if not self.best_effort:
                    self.fail(node, 'Roughness', 'Cycles rough diffuse response is not translated')
                self.report(node, 'Roughness', 'Cycles rough diffuse response is not translated', 'EEVEE Lambert response')
            normal = get('Normal',3) if node.inputs['Normal'].is_linked else 'float3(ctx.NORMAL.x,-ctx.NORMAL.z,ctx.NORMAL.y)'
            return f'float3 n={normal};return Diffuse_BSDF({get("Color",3)},{get("Roughness",1)},float3(n.x,n.z,-n.y));', 'Closure'
        if node.type == 'BSDF_TRANSPARENT':
            color = node.inputs['Color']
            if color.is_linked or tuple(color.default_value[:3]) != (1.,1.,1.):
                if not self.best_effort:
                    self.fail(node, 'Color', 'coloured transparent transport is not translated')
                self.report(node, 'Color', 'coloured transparent transport is not translated', 'neutral transparent coverage')
            return f'return Transparent_BSDF({get("Color",3)});', 'Closure'
        if node.type == 'MATH':
            operations = {'ADD','SUBTRACT','MULTIPLY','DIVIDE','MULTIPLY_ADD','MINIMUM','MAXIMUM','LESS_THAN','GREATER_THAN','ABSOLUTE','FLOOR','CEIL','POWER','FRACT','TRUNC','SINE','COSINE','TANGENT','ARCTANGENT','SQRT','SIGN','RADIANS','DEGREES','EXPONENT','ROUND','MODULO','FLOORED_MODULO'}
            if node.operation not in operations:
                self.fail(node, socket.name, f'{node.operation} Math is not translated')
            args = [self.input(s,1) for s in node.inputs[:3]]
            return f'return Math(MathOperation::{node.operation},{",".join(args)},{str(node.use_clamp).lower()});', 1
        if node.type == 'BLACKBODY':
            return f'return blackbody({get("Temperature",1)});',3
        if node.type == 'NORMAL_MAP':
            if self.available_attributes is not None and uv_attribute() not in self.available_attributes:
                self.fail(node, socket.name, 'tangent Normal Map needs an exported active render UV')
            if node.space != 'TANGENT' or node.uv_map:
                self.fail(node, socket.name, 'only active render UV tangent Normal Map is translated')
            return f'float3 n=Normal_Map({get("Strength",1)},{get("Color",3)},ctx);return float3(n.x,-n.z,n.y);',3
        if node.type == 'FRESNEL':
            normal = get('Normal',3) if node.inputs['Normal'].is_linked else 'float3(ctx.NORMAL.x,-ctx.NORMAL.z,ctx.NORMAL.y)'
            return f'float3 n={normal};return Fresnel({get("IOR",1)},float3(n.x,n.z,-n.y),ctx);', 1
        if node.type == 'NEW_GEOMETRY':
            if socket.name == 'Backfacing':
                return 'return ctx.front_facing?0.:1.;',1
            fields = {'Position':'POSITION','Normal':'NORMAL','True Normal':'true_normal','Tangent':'TANGENT','Incoming':'incoming'}
            if socket.name in fields:
                field = fields[socket.name]
                derivatives = {'POSITION':('position_dx','position_dy'), 'NORMAL':('unperturbed_normal_dx','unperturbed_normal_dy'), 'incoming':('incoming_dx','incoming_dy')}
                expression = f'ctx.{field}'
                if field in derivatives:
                    x,y = derivatives[field]
                    expression += f'+ctx.{x}*off.x+ctx.{y}*off.y'
                return f'float3 p={expression};return float3(p.x,-p.z,p.y);', 3
            self.fail(node, socket.name, 'geometry output is not translated')
        if node.type == 'BSDF_PRINCIPLED':
            # A Principled metal ignores its dielectric Specular IOR Level/Tint.
            # OpenPBR's specular weight also scales metal, so fractional metallic
            # graphs require an independently shaded dielectric/metal mixture.
            pairs = [('Base Color','base_color',3),('Metallic','base_metalness',1),('Roughness','specular_roughness',1),('IOR','specular_ior',1),('Alpha','geometry_opacity',1),('Specular Tint','specular_color',3),('Transmission Weight','transmission_weight',1),('Coat Weight','coat_weight',1),('Coat Roughness','coat_roughness',1),('Coat IOR','coat_ior',1),('Coat Tint','coat_color',3),('Sheen Weight','fuzz_weight',1),('Sheen Roughness','fuzz_roughness',1),('Sheen Tint','fuzz_color',3),('Emission Color','emission_color',3),('Emission Strength','emission_luminance',1)]
            if node.distribution not in ('GGX', 'MULTI_GGX'):
                self.fail(node, 'distribution', f'{node.distribution} Principled is not translated')
            metallic=node.inputs['Metallic']
            body='Closure c={};\n'+'\n'.join(f'c.{target}={get(source,w)};' for source,target,w in pairs)
            body+=f'\nc.specular_weight=2.*{get("Specular IOR Level",1)};c.geometry_thin_walled=({get("Thin Wall",1)}!=0.);'
            for source,target in [('Normal','geometry_normal'),('Coat Normal','geometry_coat_normal'),('Tangent','geometry_tangent')]:
                if node.inputs[source].is_linked:
                    body+=f'{{float3 n={get(source,3)};c.{target}=float3(n.x,n.z,-n.y);}}'
            if self.engine == 'CYCLES' and node.distribution == 'GGX' and (metallic.is_linked or metallic.default_value != 0.):
                reason = 'Cycles single-scatter GGX metal reflection is not represented'
                if not self.best_effort:
                    self.fail(node, 'distribution', reason)
                self.report(node, 'distribution', reason, 'multiscatter-compensated GGX metal reflection')
            body+='c.transmission_color=sqrt(max(c.base_color,float3(0.)));'
            for name in ['Anisotropic','Anisotropic Rotation','Diffuse Roughness','Thin Film Thickness']:
                source=node.inputs[name]
                if source.is_linked or source.default_value!=0:
                    if not self.best_effort:
                        self.fail(node,name,'this response is not translated yet')
                    self.report(node, name, 'this response is not translated yet', 'omit response; preserve other Principled sockets')
            subsurface=node.inputs['Subsurface Weight']
            if subsurface.is_linked or subsurface.default_value!=0.:
                restriction=self.subsurface_transport_restriction()
                if not self.best_effort:
                    self.fail(node,'Subsurface Weight',restriction or 'native Burley diffusion approximates Blender subsurface transport')
                if restriction:
                    self.report(node,'Subsurface Weight',restriction,'omit subsurface; preserve other Principled sockets')
                else:
                    defaults_before = self.degradations
                    subsurface_body=f'c.subsurface_weight={get("Subsurface Weight",1)};c.subsurface_color=c.base_color;'
                    subsurface_body+=f'c.subsurface_radius={get("Subsurface Scale",1)};c.subsurface_radius_scale={get("Subsurface Radius",3)};'
                    if self.engine=='CYCLES':
                        subsurface_body+=f'c.subsurface_scatter_anisotropy={get("Subsurface Anisotropy",1)};'
                    if self.degradations != defaults_before:
                        self.report(node,'Subsurface Weight','subsurface numeric inputs required substitutions',
                                    'omit subsurface; preserve other Principled sockets')
                    else:
                        body+=subsurface_body
                        self.report(node,'Subsurface Weight',
                                f'Blender {self.engine}/{node.subsurface_method} subsurface transport is approximated',
                                'opaque native Burley diffusion; retain weight, base colour and world-space RGB radii')
            if not metallic.is_linked and metallic.default_value in (0.,1.):
                if metallic.default_value==1.:
                    body+='c.specular_weight=1.;c.specular_color=float3(1.);'
                return body+'return c;','Closure'
            body+='float metallic=c.base_metalness;Closure metal=c;metal.base_metalness=1.;metal.specular_weight=1.;metal.specular_color=float3(1.);c.base_metalness=0.;'
            body+='return blend(c,metal,metallic);'
            return body,'Blend'
        if node.type == 'BSDF_GLASS':
            if node.distribution not in ('GGX', 'MULTI_GGX', 'BECKMANN'):
                self.fail(node, 'distribution', f'{node.distribution} Glass is not translated')
            if self.engine == 'CYCLES' and node.distribution != 'GGX':
                model = {'BECKMANN':'Beckmann','MULTI_GGX':'Multiscatter GGX'}[node.distribution]
                if not self.best_effort:
                    self.fail(node, 'distribution', f'{model} Glass is not represented')
                self.report(node, 'distribution', f'{model} Glass is not represented', 'single-scatter GGX reflection/refraction')
            normal = 'ctx.NORMAL'
            body = ''
            if node.inputs['Normal'].is_linked:
                body = f'float3 n={get("Normal",3)};'
                normal = 'float3(n.x,n.z,-n.y)'
            body += f'Closure c=Glass_BSDF({get("Color",3)},{get("Roughness",1)},{get("IOR",1)},{normal},{get("Thin Film Thickness",1)},{get("Thin Film IOR",1)});'
            film = node.inputs['Thin Film Thickness']
            if not film.is_linked and film.default_value == 0.:
                # Preserve the authored exact off state in the lowered closure.
                # General float predicate proofs must allow GPU contraction/FTZ.
                body += 'c.thin_film_weight=0.;'
            return body+'return c;', 'Closure'
        if node.type == 'BSDF_GLOSSY':
            for name in ('Anisotropy', 'Rotation'):
                value = node.inputs.get(name)
                if value and (value.is_linked or value.default_value != 0):
                    self.fail(node, name, 'anisotropic Glossy is not translated yet')
            if node.distribution not in ('GGX', 'MULTI_GGX', 'BECKMANN', 'ASHIKHMIN_SHIRLEY'):
                self.fail(node, 'distribution', f'{node.distribution} Glossy is not translated')
            # OpenPBR has no constant-colour reflection: a dielectric whose IOR gives the colour's
            # brightest channel at normal incidence (specular_weight carries the rest of an HDR
            # colour), specular_color its hue; reflectance rises toward that hue at grazing angles.
            # Other distributions become GGX of the same roughness (EEVEE already renders GGX).
            approximation = 'constant-colour Glossy reflection'
            if self.engine == 'CYCLES' and node.distribution != 'GGX':
                approximation = f'{node.distribution} {approximation}'
            if not self.best_effort:
                self.fail(node, 'BSDF', f'{approximation} is not represented')
            self.report(node, 'BSDF', f'{approximation} is not represented',
                        'GGX reflection with OpenPBR dielectric Fresnel matching the colour at normal incidence')
            body = (f'Closure c={{}};c.base_weight=0.;float3 k=max({get("Color",3)},float3(0.));float p=max(k.x,max(k.y,k.z));'
                    'float f0=min(p,.9999);float r=sqrt(f0);c.specular_ior=(1.+r)/(1.-r);'
                    f'c.specular_weight=p>f0?p/f0:1.;c.specular_color=p>0.?k/p:float3(0.);c.specular_roughness={get("Roughness",1)};')
            if node.inputs['Normal'].is_linked:
                body += f'float3 n={get("Normal",3)};c.geometry_normal=float3(n.x,n.z,-n.y);'
            return body+'return c;', 'Closure'
        if node.type == 'CURVE_RGB':
            curve=self.curve(node)
            args=',inputs,ctx' if curve in self.curve_lookup_names else ''
            black=node.mapping.black_level
            white=node.mapping.white_level
            if any(a==b for a,b in zip(black,white)):
                self.fail(node,'Color','singular curve black/white range')
            body=f'float3 original={get("Color",3)};float3 p=(original-{vector(black)})/({vector(white)}-{vector(black)});'
            body+=f'p=float3({curve}_w(p.x{args}),{curve}_w(p.y{args}),{curve}_w(p.z{args}));'
            body+=f'float3 result=float3({curve}_x(p.x{args}),{curve}_y(p.y{args}),{curve}_z(p.z{args}));return lerp(original,result,{get("Fac",1)});'
            return body,3
        if node.type == 'BUMP':
            height=self.input(node.inputs['Height'],1)
            name=f'Bump_{self.lut_counter}'
            self.lut_counter+=1
            self.functions.append(f'''struct {name}:BumpHeight {{
ShadingContext context;Inputs parameters;float2 origin;
float evaluate(float2 delta) {{let ctx=context;let inputs=parameters;let off=origin+delta;return {height};}}
}};''')
            normal='ctx.NORMAL'
            if node.inputs['Normal'].is_linked:
                normal=f'float3(({get("Normal",3)}).x,({get("Normal",3)}).z,-({get("Normal",3)}).y)'
            filter_width=get('Filter Width',1) if node.inputs.get('Filter Width') else '.1'
            return f'''{name} height={{}};height.context=ctx;height.parameters=inputs;height.origin=off;
float3 n=bump(height,{normal},ctx.position_dx,ctx.position_dy,{get('Strength',1)},{get('Distance',1)},{filter_width},{str(node.invert).lower()});
return float3(n.x,-n.z,n.y);''',3
        if node.type == 'HUE_SAT':
            return f'''float3 original={get('Color',3)};
float3 hsv=blender_to_hsv(original);hsv.x=frac(hsv.x+{get('Hue',1)}-.5);hsv.y*={get('Saturation',1)};hsv.z*={get('Value',1)};
return lerp(original,max(float3(0.),blender_from_hsv(hsv)),{get('Fac',1)});''',3
        if node.type == 'EMISSION':
            return f'Closure c={{}};c.base_weight=0.;c.specular_weight=0.;c.emission_color={get("Color",3)};c.emission_luminance={get("Strength",1)};return c;', 'Closure'
        if node.type in ('ADD_SHADER', 'MIX_SHADER'):
            sockets = [s for s in node.inputs if s.type == 'SHADER']
            first, first_type = self.output(sockets[0].links[0].from_socket) if sockets[0].is_linked else (None, None)
            second, second_type = self.output(sockets[1].links[0].from_socket) if sockets[1].is_linked else (None, None)
            if first is None or second is None:
                if not self.best_effort:
                    self.fail(node, socket.name, 'unconnected shader mixture')
                self.report(node, socket.name, 'unconnected shader mixture input', 'black absorbing empty closure for the missing input; retain connected sibling')
            a = f'{first}(ctx,inputs,off)' if first else 'blender_empty_shader()'
            b = f'{second}(ctx,inputs,off)' if second else 'blender_empty_shader()'
            expression = f'add({a},{b})' if node.type == 'ADD_SHADER' else f'blend({a},{b},{get("Fac",1)})'
            return f'return {expression};', 'Blend'
        self.fail(node, socket.name, f'{node.type} is not translated')

    def curve(self, node):
        """Resample Blender's evaluated curves, probing interpolation error.

        Include the source knots and their interval quarters in the probes so a
        narrow authored feature is not accepted just because a coarse grid misses it.
        """
        mapping=node.mapping
        mapping.update()
        knots=sorted({0.,1.,*(float(p.location.x) for curve in mapping.curves for p in curve.points)})
        lo,hi=knots[0],knots[-1]
        fixed=knots+[a+(b-a)*f for a,b in zip(knots,knots[1:]) for f in (.25,.5,.75)]
        cache={}
        def evaluate(x):
            if x not in cache:
                cache[x]=tuple(float(mapping.evaluate(curve,x)) for curve in mapping.curves)
            return cache[x]
        intervals=8
        tolerance=1./65536.  # Below a 16-bit unit-range channel code value.
        while True:
            table=[evaluate(lo+(hi-lo)*i/intervals) for i in range(intervals+1)]
            probes=fixed+[lo+(hi-lo)*(i+f)/intervals for i in range(intervals) for f in (.25,.5,.75)]
            error=0.
            for x in probes:
                position=(x-lo)/(hi-lo)*intervals
                index=max(0,min(intervals-1,int(position)))
                fraction=position-index
                error=max(error,*(abs(a*(1.-fraction)+b*fraction-c) for a,b,c in zip(table[index],table[index+1],evaluate(x))))
            if error<=tolerance:
                break
            if intervals>=4096:
                self.fail(node,'Color','curve cannot meet the export interpolation error bound')
            intervals*=2
        name=f'curve_{self.lut_counter}'
        self.lut_counter+=1
        left=tuple((a-b)/(hi-lo) for a,b in zip(table[0],evaluate(lo-(hi-lo))))
        right=tuple((a-b)/(hi-lo) for a,b in zip(evaluate(hi+(hi-lo)),table[-1]))
        if self.curve_input is not None:
            if not self.best_effort:
                self.fail(node,'Color','curve lookup baking requires best effort')
            image=self.curve_input(table)
            field=name+'_image'
            self.inputs[field]={'image':image['image']}
            self.curve_lookup_names.add(name)
            self.report(node,'Color','explicit curve value-field bake',image['replacement'])
            self.curve_bakes.append(dict(node=node.name,kind='curve transfer function',domain=[lo,hi],**image))
            vec4=lambda v:'float4('+','.join(map(number,v))+')'
            self.functions.append(f'''float4 {name}_lookup(float x,Inputs inputs,ShadingContext ctx) {{
float2 uv=float2(((x-({number(lo)}))/{number(hi-lo)}*{intervals}.+0.5)/{intervals+1}.,0.5);
return blender_image(inputs.{field},uv,uv,uv,ctx)*{vec4(image['scale'])}+{vec4(image['offset'])};
}}''')
            for channel,suffix in enumerate('xyzw'):
                self.functions.append(f'''float {name}_{suffix}(float x,Inputs inputs,ShadingContext ctx) {{
if(x<{number(lo)}) return {number(table[0][channel])}+(x-({number(lo)}))*{number(left[channel])};
if(x>{number(hi)}) return {number(table[-1][channel])}+(x-({number(hi)}))*{number(right[channel])};
return {name}_lookup(x,inputs,ctx).{suffix};
}}''')
            return name
        # Exact segment data uses the existing immutable material-word buffer.
        # All channels share this calling convention, including constant/linear
        # channels whose direct expressions can still fold at compile time.
        self.curve_lookup_names.add(name)
        # Keep the validated table/grid, but project before emitting control flow.
        # A caller needs one channel at each x; a float4 return forces drivers to
        # compile/carry all four channels through every nested branch and callback.
        for channel, suffix in enumerate('xyzw'):
            value=lambda v:number(v[channel])
            segments = []
            def segment(first, last):
                # The existing probe error and this deviation add to at most
                # the established curve export bound. Piecewise-linear error
                # attains its maximum at a retained grid vertex, so checking
                # all vertices also bounds compaction between those vertices.
                budget = max(0., tolerance-error)
                deviation = max(abs(table[i][channel] -
                                    (table[first][channel] + (table[last][channel]-table[first][channel])*(i-first)/(last-first)))
                                for i in range(first,last+1))
                if last-first == 1 or deviation <= budget:
                    a=lo+(hi-lo)*first/intervals
                    inverse=number(intervals/((hi-lo)*(last-first)))
                    # Parse the same emitted literals, rather than rounding the
                    # original Blender doubles along a different path.
                    segments.append([float(v) for v in (number(a), inverse, value(table[first]), value(table[last]))])
                    return f'return lerp({value(table[first])},{value(table[last])},(x-({number(a)}))*{inverse});'
                middle=(first+last)//2
                boundary=lo+(hi-lo)*middle/intervals
                return f'if(x<{number(boundary)}) {{{segment(first,middle)}}} else {{{segment(middle,last)}}}'
            interior = segment(0,intervals)
            if len(segments) >= 3:
                field = f'{name}_{suffix}_segments'
                self.field_types[field] = 'FloatGrid'
                self.inputs[field] = {'grid': {'type': 'f32', 'dims': [4, len(segments), 1],
                                              'data': [v for row in segments for v in row]}}
                interior = f'return curve_segments(inputs.{field},x);'
            self.functions.append(f"""float {name}_{suffix}(float x,Inputs inputs,ShadingContext ctx) {{
    if(x<{number(lo)}) return {value(table[0])}+(x-({number(lo)}))*{number(left[channel])};
    if(x>{number(hi)}) return {value(table[-1])}+(x-({number(hi)}))*{number(right[channel])};
    {interior}
}}""")
        return name

    def validate_graph(self, root):
        """Check source dependency cycles before unsupported-feature substitution."""
        done, active = set(), set()
        def visit(node):
            key = node.as_pointer()
            if key in active:
                raise InvalidMaterial(f'material {self.material.name!r}: {node.name!r}: cyclic graph')
            if key in done:
                return
            active.add(key)
            if node.type == 'GROUP' and node.node_tree is not None:
                for output in node.node_tree.nodes:
                    if output.type == 'GROUP_OUTPUT' and output.is_active_output:
                        visit(output)
            for socket in node.inputs:
                if not getattr(socket, 'is_unavailable', False):
                    for link in socket.links:
                        visit(link.from_node)
            active.remove(key)
            done.add(key)
        visit(root)

    def compile(self):
        if not self.material.use_nodes:
            if not self.best_effort:
                raise UnsupportedMaterial(f'material {self.material.name!r}: nodes are required')
            self.report(self.material, 'Surface', 'material has no node graph', 'viewport diffuse surface')
            self.functions.append('Closure viewport(ShadingContext ctx,Inputs inputs,float2 off) {'
                                  f'Closure c={{}};c.base_color={vector(self.material.diffuse_color)};'
                                  'c.specular_weight=0.;return c;}')
            return self.finish('viewport', 'Closure')
        output = self.material.node_tree.get_output_node(self.engine)
        if output is not None:
            self.validate_graph(output)
        if output is None or not output.inputs['Surface'].is_linked:
            if not self.best_effort:
                raise UnsupportedMaterial(f'material {self.material.name!r}: no active surface output')
            self.report(output or self.material, 'Surface', 'no active surface output',
                        'empty transparent surface; no volume transport exported')
            self.functions.append('Closure empty(ShadingContext ctx,Inputs inputs,float2 off) {'
                                  'Closure c={};c.base_weight=0.;c.specular_weight=0.;'
                                  'c.geometry_opacity=0.;return c;}')
            return self.finish('empty', 'Closure')
        from .volume import empty_volume
        for channel in ('Volume', 'Displacement'):
            if channel == 'Volume' and empty_volume(output.inputs[channel]):
                continue
            if output.inputs[channel].is_linked:
                if not self.best_effort:
                    self.fail(output, channel, 'only surface graphs are translated')
                self.report(output, channel, 'only surface graphs are translated',
                            f'omit {channel.lower()}; preserve surface')
        self.preview_inputs(output)
        root, result = self.output(output.inputs['Surface'].links[0].from_socket)
        return self.finish(root, result)

    def finish(self, root, result):
        source = self.module_source()
        source += f'\n{result} surface(ShadingContext ctx,Inputs inputs) {{return {root}(ctx,inputs,float2(0.));}}\n'
        return source, self.inputs

    def module_source(self):
        fields = ''.join(f'{self.field_types.get(field) or ("Attribute<float2>" if field.startswith("uv_") else "Image")} {field};'
                         for field in self.inputs)
        source = 'import vork;import std.color;import blender.vector;import blender.texture_coordinate;import blender.color;import blender.nodes;\n'
        source += f'struct Inputs {{{fields}}};\n' + PRELUDE + '\n'.join(self.functions)
        return source
