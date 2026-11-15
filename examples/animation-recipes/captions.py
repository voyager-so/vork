"""Plain, non-overlapping SRT cues to a STEP content track (standard library only)."""
import argparse
import json
from pathlib import Path
import re

FONT = Path(__file__).resolve().parents[2] / "crates/vork-std/tests/fonts/LiberationSans-Regular.ttf"
STAMP = r"([0-9]{2,}):([0-5][0-9]):([0-5][0-9]),([0-9]{3})"
TIMING = re.compile(STAMP + r" --> " + STAMP)


def parse_srt(text):
    """Require numbered blocks, millisecond timestamps, text and ordered disjoint cues."""
    cues = []
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("SRT contains no cues")
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.split("\n")
        if len(lines) < 3 or not re.fullmatch(r"[0-9]+", lines[0]):
            raise ValueError("expected cue number, timestamp line and plain text")
        timing = TIMING.fullmatch(lines[1])
        if timing is None:
            raise ValueError(f"invalid SRT timing: {lines[1]!r}")
        h, m, s, ms, eh, em, es, ems = map(int, timing.groups())
        start = ((h * 60 + m) * 60 + s) * 1000 + ms
        end = ((eh * 60 + em) * 60 + es) * 1000 + ems
        if end <= start or (cues and start < cues[-1][1]):
            raise ValueError("cues must have positive duration and be ordered without overlaps")
        cues.append((start, end, "\n".join(lines[2:])))
    return cues


def scene(cues):
    # A shared boundary belongs to the incoming cue. The map removes duplicate keys.
    content = {0: ""}
    for start, end, text in cues:
        content[start] = text
        content[end] = ""
    return {"version": 2, "composition": {"space": "2d", "viewBox": [0, 0, 640, 360]},
            "render": {"resolution": [640, 360], "fps": 30,
                       "duration": f"{cues[-1][1] + 500}/1000", "background": "#142136"},
            "nodes": [{"name": "captions", "type": "text", "font": str(FONT),
                       "content": "", "font-size": 32, "white-space": "pre-line",
                       "width": 580, "text-align": "center", "fill": "white",
                       "translation": [30, 230, 0]}],
            "animation": [{"tracks": [{"path": ["nodes", "captions", "content"],
                            "interpolation": "STEP",
                            "keys": [[f"{ms}/1000", value] for ms, value in sorted(content.items())]}]}]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("srt", type=Path)
    args = parser.parse_args()
    try:
        result = scene(parse_srt(args.srt.read_text(encoding="utf-8-sig")))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
