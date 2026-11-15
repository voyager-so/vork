# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit Blender display curves inside display conversion, outside linear outputs."""
import math
import struct

# Blender 5.2.1 CurveMap lookup tables have 256 intervals. These are evaluated
# lookup samples, not authoring defaults or Bezier control-point approximations.
# RGB tables compose their channel response followed by the combined response.
# https://projects.blender.org/blender/blender/src/branch/main/source/blender/blenkernel/intern/colortools.cc
_INTERVALS = 256


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError as error:
        raise ValueError('display curve mapping exceeds float32 range') from error


def _sample_curves(mapping):
    if mapping.tone != 'STANDARD':
        raise ValueError('Film-like display curves couple RGB and need a different transfer representation')
    mapping.initialize()
    mapping.update()
    if len(mapping.curves) != 4:
        raise ValueError('display curve mapping requires RGBA source curves')
    black, white = list(mapping.black_level), list(mapping.white_level)
    if any(not math.isfinite(v) for v in black+white):
        raise ValueError('display curve mapping has an invalid black/white range')
    # Preserve Blender's evaluated normalization, including equal/reversed levels.
    multiplier=[f32(1. / max(f32(b-a),f32(1e-5))) for a,b in zip(black,white)]
    domains = [(min(mapping.clip_min_x, *(p.location.x for p in c.points)),
                max(mapping.clip_max_x, *(p.location.x for p in c.points))) for c in mapping.curves]
    for lo,hi in domains:
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError('display curve mapping has an invalid input domain')
    def evaluate(channel, x):
        return float(mapping.evaluate(mapping.curves[channel], f32(x)))
    def slopes(channel):
        lo,hi=domains[channel];span=hi-lo
        left, right = evaluate(channel,lo-span), evaluate(channel,hi+span)
        # Blender's vertical-tangent fallback is a constant offset, not a
        # line through the endpoint. Its premultiplied channel/combined
        # endpoint vectors are not exposed by RNA; do not guess that case.
        if ((left != evaluate(channel,lo) and left == evaluate(channel,lo-2*span)) or
                (right != evaluate(channel,hi) and right == evaluate(channel,hi+2*span))):
            raise ValueError('vertical display-curve endpoint extrapolation cannot be recovered from evaluated RNA tables')
        return ((evaluate(channel,lo)-left)/span,
                (right-evaluate(channel,hi))/span)
    combined_slopes=slopes(3)
    curves=[]
    for channel in range(3):
        lo,hi=domains[channel]
        table=[]
        for i in range(_INTERVALS+1):
            x=f32(lo+f32((hi-lo)/_INTERVALS)*i)
            y=evaluate(3,evaluate(channel,x))
            table.append([x,y])
        extrapolation=[f32(a*b) for a,b in zip(slopes(channel),combined_slopes)]
        if any(not math.isfinite(v) for point in table for v in point) or any(not math.isfinite(v) for v in extrapolation):
            raise ValueError('display curve mapping has nonfinite evaluated values')
        if any(a[0] >= b[0] for a,b in zip(table,table[1:])):
            raise ValueError('display curve input domain cannot be sampled in float32')
        curves.append((table[0][0],table[-1][0],[p[1] for p in table],extrapolation))
    return black, multiplier, curves


def curve_source(mapping):
    """Emit evaluated RGB transfer functions for a display wrapper."""
    black_level, white_multiplier, curves = _sample_curves(mapping)
    def literal(value):
        return repr(f32(value))
    functions = []
    for channel, (lo, hi, values, extrapolation) in enumerate(curves):
        table = ','.join(literal(v) for v in values)
        left, right = extrapolation
        functions.append(f"""
float curve_{channel}(float x) {{
    const float values[{_INTERVALS+1}] = {{{table}}};
    if (x < {literal(lo)}) return values[0] + (x - {literal(lo)}) * {literal(left)};
    if (x > {literal(hi)}) return values[{_INTERVALS}] + (x - {literal(hi)}) * {literal(right)};
    float t = clamp((x - {literal(lo)}) / ({literal(hi)} - {literal(lo)}) * {_INTERVALS}., 0., {_INTERVALS}.);
    uint i = min(uint(t), {_INTERVALS-1}u);
    return lerp(values[i], values[i+1u], t-float(i));
}}
""")
    black = ','.join(literal(v) for v in black_level)
    white = ','.join(literal(v) for v in white_multiplier)
    return ''.join(functions) + f"""
float3 blender_curves(float3 color) {{
    float3 c = (color - float3({black})) * float3({white});
    return float3(curve_0(c.r), curve_1(c.g), curve_2(c.b));
}}
"""


def display_source(program, *, curves=None, white_balance=None):
    """Wrap the bundled view code and retain its reflected input types."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'libraries'
    source = (root / (program + '.slang')).read_text()
    helpers = ''
    operations = ''
    if curves is not None:
        helpers += curve_source(curves)
        operations += '    ctx.color = blender_curves(ctx.color);\n'
    if white_balance is not None:
        wb = (root / 'blender/white_balance.slang').read_text()
        helpers += wb.replace('WhiteBalanceInputs', 'BlenderWhiteBalanceInputs')
        temperature, tint = white_balance
        operations += f'    BlenderWhiteBalanceInputs wb; wb.temperature = {float(temperature)!r}; wb.tint = {float(tint)!r};\n    ctx.color = white_balance(ctx.color, wb);\n'
    entry = 'public ViewResult view(ViewContext ctx, Inputs inputs) {'
    if source.count(entry) != 1:
        raise ValueError(f'{program}: bundled display entry changed')
    return helpers + source.replace(entry, entry + '\n' + operations)
