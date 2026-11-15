"""Lower numeric Blender volume closures to authored homogeneous coefficients."""
import math


def homogeneous(socket):
    """Return a medium-only program; never bake or substitute a density field."""
    node = socket.links[0].from_node

    def refuse(field, reason):
        raise ValueError(
            f'{node.name!r}.{field}: Volume export not supported: {reason}; '
            'use a single volume node with unlinked constant numeric inputs, '
            'clear volume attribute names and disable blackbody emission; '
            'use a closed mesh for finite fog, or disconnect Volume for a '
            'surface-only export; see vork help fog')

    if node.mute:
        refuse('Volume', 'muted-node bypass is not translated')
    if node.type not in ('PRINCIPLED_VOLUME', 'VOLUME_SCATTER', 'VOLUME_ABSORPTION', 'EMISSION'):
        refuse('Volume', f'{node.type} is not a homogeneous volume closure')
    if node.type == 'VOLUME_SCATTER' and node.phase != 'HENYEY_GREENSTEIN':
        refuse('Phase', 'only Henyey-Greenstein scattering is represented')

    def value(name, width=1):
        inp = node.inputs[name]
        if inp.is_linked:
            refuse(name, 'linked inputs, textures and varying density are not translated')
        result = list(inp.default_value[:width]) if width > 1 else [float(inp.default_value)]
        if any(not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in result):
            refuse(name, 'coefficients must be finite shader floats')
        if name != 'Anisotropy' and any(v < 0 for v in result):
            refuse(name, 'coefficients must be nonnegative')
        return result if width > 1 else result[0]

    scattering = absorption = emission = [0., 0., 0.]
    anisotropy = 0.
    if node.type == 'PRINCIPLED_VOLUME':
        for name in ('Density Attribute', 'Color Attribute', 'Temperature Attribute'):
            inp = node.inputs[name]
            if inp.is_linked or inp.default_value:
                refuse(name, 'attribute-driven coefficients are not homogeneous')
        if value('Blackbody Intensity') != 0:
            refuse('Blackbody Intensity', 'use constant Emission Color and Emission Strength instead')
        color = value('Color', 3)
        density = value('Density')
        absorption_color = value('Absorption Color', 3)
        scattering = [density * c for c in color]
        absorption = [density * max(1-c, 0) * max(1-math.sqrt(a), 0)
                      for c, a in zip(color, absorption_color)]
        emission = [value('Emission Strength') * c for c in value('Emission Color', 3)]
        anisotropy = value('Anisotropy')
    elif node.type == 'EMISSION':
        emission = [value('Strength') * c for c in value('Color', 3)]
    elif node.type == 'VOLUME_SCATTER':
        scattering = [value('Density') * c for c in value('Color', 3)]
        anisotropy = value('Anisotropy')
    else:
        absorption = [value('Density') * max(1-c, 0) for c in value('Color', 3)]
    if not -1 < anisotropy < 1:
        refuse('Anisotropy', 'set anisotropy strictly between -1 and 1')
    if any(not math.isfinite(v) or v > 3.4028234663852886e38
           for v in scattering + absorption + emission):
        refuse('Volume', 'reduce density or emission strength to keep coefficients finite')

    def rgb(values):
        return 'float3(' + ', '.join(repr(float(v)) for v in values) + ')'

    return ('import vork;\nstruct Inputs {};\n'
            'MediumCoefficients medium(MediumContext ctx, Inputs inputs) {\n'
            '    MediumCoefficients c;\n'
            f'    c.scattering = {rgb(scattering)};\n'
            f'    c.absorption = {rgb(absorption)};\n'
            f'    c.emission = {rgb(emission)};\n'
            f'    c.anisotropy = {anisotropy!r};\n'
            '    return c;\n}\n')


def empty_volume(socket, pending=None):
    """Prove a connected shader has no volume closure under Cycles semantics.

    Cycles svm_node_closure_bsdf skips every BSDF/BSSRDF outside surface
    evaluation. Unknown closures remain unsupported, including node groups.
    """
    if not socket.is_linked:
        return True
    link = socket.links[0]
    node = link.from_node
    pending = set() if pending is None else pending
    key = node.as_pointer(), link.from_socket.identifier
    if key in pending:
        return False
    pending.add(key)
    try:
        if node.mute:
            return any(connection.to_socket == link.from_socket
                       and empty_volume(connection.from_socket, pending)
                       for connection in node.internal_links)
        if node.type == 'REROUTE':
            return empty_volume(node.inputs[0], pending)
        if node.type in ('MIX_SHADER', 'ADD_SHADER'):
            return all(empty_volume(inp, pending) for inp in node.inputs if inp.type == 'SHADER')
        return node.type.startswith('BSDF_') or node.type == 'SUBSURFACE_SCATTERING'
    finally:
        pending.remove(key)
