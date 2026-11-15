CAPTIONS
Convert transcript timing in authoring code; text uses the same tracks as other
canvas content. Use STEP keys for node.content. At each cue's start, key its text;
at its end, key an empty string. At an adjacent boundary, write the incoming
text once at that time. Keys must increase; do not emit two keys at a shared
boundary. Include an empty key at zero when the first cue starts later.

SRT timestamps are integer milliseconds: serialize ms as "ms/1000" seconds.
This preserves the cue's instant independently of render FPS. A cue between
frames appears on the first sampled frame at or after its start and disappears
on the first at or after its end. Set white-space to pre-line to keep explicit
line breaks; width controls wrapping. Fonts are explicit .ttf/.otf files.

Example: examples/animation-recipes/captions.py reads plain UTF-8 SRT, preserves
multiline cues, clears gaps, and emits one text node and a STEP content track.
It accepts BOM and CRLF; malformed timing, nonpositive durations, unordered or
overlapping cues refuse. SRT markup stays literal. For overlapping speakers,
use separate text nodes/tracks and give each a separate position. This example
does not transcribe audio, infer word timing or paginate cues.
Generation and render commands: examples/animation-recipes/README.md.
Text controls: vork help text. Windows and local clocks: vork help sequences.
