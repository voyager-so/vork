"""Transfer Blender display settings without silently changing the authored look."""
from .view_curves import display_source

def translate(scene, *, on_substitution=None, emit_display=None):
    view = scene.view_settings
    if scene.display_settings.display_device != 'sRGB':
        raise ValueError('Unsupported display device: ' + scene.display_settings.display_device)
    if view.view_transform not in ('Standard', 'AgX', 'Filmic'):
        raise ValueError('Unsupported display transform: ' + view.view_transform)
    filmic = ('Very High Contrast', 'High Contrast', 'Medium High Contrast',
              'Medium Contrast', 'Medium Low Contrast', 'Low Contrast', 'Very Low Contrast')
    agx = {'AgX - ' + name for name in ('Punchy', 'Greyscale', 'Very High Contrast',
           'High Contrast', 'Medium High Contrast', 'Base Contrast', 'Medium Low Contrast',
           'Low Contrast', 'Very Low Contrast')}
    valid = view.look == 'None' or (view.view_transform == 'Filmic' and view.look in filmic) or (
        view.view_transform == 'AgX' and view.look in agx)
    if not valid:
        raise ValueError('Unsupported display look: ' + view.view_transform + '/' + view.look)
    inputs = dict(exposure=view.exposure, gamma=view.gamma)
    if view.view_transform != 'Standard':
        inputs['look'] = view.look.removeprefix('AgX - ').lower().replace(' ', '_')
    program = {'AgX': 'std/agx', 'Standard': 'std/standard', 'Filmic': 'blender/filmic'}[view.view_transform]
    result = {'view': {'program': program, 'inputs': inputs}}

    if view.use_curve_mapping or view.use_white_balance:
        if emit_display is None:
            raise ValueError('display curve mapping/white balance requires a display writer')
        curves = view.curve_mapping if view.use_curve_mapping else None
        balance = (view.white_balance_temperature, view.white_balance_tint) if view.use_white_balance else None
        try:
            source = display_source(program, curves=curves, white_balance=balance)
        except ValueError as error:
            if on_substitution is None: raise
            on_substitution('use_curve_mapping', str(error), 'omit unsupported display curves; retain authored view and white balance')
            if balance is None: return result
            source = display_source(program, white_balance=balance)
        if curves is not None and getattr(scene.render, 'film_transparent', False):
            reason = 'vork display programs receive unassociated RGB without alpha; Blender curves act on premultiplied RGB'
            replacement = 'apply curves to unassociated display RGB; preserve alpha and all scene-linear outputs'
            if on_substitution is None: raise ValueError(reason)
            on_substitution('transparent display curves', reason, replacement)
        result['view'] = emit_display(source, program, inputs)
    return result
