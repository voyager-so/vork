"""Author explicit bloom/glare graphs; no renderer-side graph expansion."""
import copy
import math
from pathlib import Path

BLOOM_SOURCE=(Path(__file__).parent/"compositor-effects/bloom.slang").read_text()

COPY = {"program": {"source": "import vork; struct Inputs {}; float4 composite(CompositeContext c, Inputs i) { return compositor_sample(c.uv); }"}}


def bloom(inputs, resolution, prefix="bloom"):
    inputs = dict(inputs)
    quality = inputs.pop("quality", "medium")
    divisor = 2 ** {"high": 0, "medium": 1, "low": 2}[quality]
    radius = inputs.pop("size", .5)
    size = [math.ceil(x / divisor) for x in resolution]
    levels = max(1, int(math.log2(max(1, min(size) * radius))))
    graph = [{"name": prefix + ".original", "material": copy.deepcopy(COPY)}]
    def node(stage, scale, extra, name=None):
        item = {"material": {"program": {"source": BLOOM_SOURCE}, "inputs": {**inputs, "stage": stage, "source": {"step": graph[-1]["name"]}, **extra}}, "resolution": scale}
        item["name"] = name if name is not None else f"{prefix}.stage{len(graph)}"
        graph.append(item)
    # Absolute dimensions retain ceil-divided highlight size and floor mip extents,
    # including odd output dimensions. Regenerate when changing output resolution.
    node("prepare", size, {"divisor": divisor, "original": {"step": prefix + ".original"}}, prefix + ".0")
    for level in range(1, levels):
        node("downsample", [max(1, x >> level) for x in size], {"karis": level == 1}, f"{prefix}.{level}")
    for level in reversed(range(levels - 1)):
        node("upsample", [max(1, x >> level) for x in size], {"fine": {"step": f"{prefix}.{level}"}})
    node("combine", 1, {"original": {"step": prefix + ".original"}, "levels": levels})
    return graph


def glare(material, prefix="glare"):
    return [
        {"name": prefix + ".original", "material": copy.deepcopy(COPY)},
        material,
        {"material": {"program": {"source": "import vork; struct Inputs { Image original; }; float4 composite(CompositeContext c, Inputs i) { return compositor_sample(i.original,c.uv)+compositor_sample(c.uv); }"},
                      "inputs": {"original": {"step": prefix + ".original"}}}},
    ]
