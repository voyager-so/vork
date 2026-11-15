SEQUENCES
Author sequences with node.active windows and node-owned animation.
At FPS p/q, frame f starts at f*q/p seconds. Keep integer frame counts in the
generator and serialize times as rational strings. A frame window [a,b) has
b-a frames: set node.active to [a*q/p,b*q/p] in seconds, including the start and
excluding the end. A window starts that node's local clock at zero. Node animation paths start at the node; scene animation
paths start at the scene. Explicit node.time can change the local clock.

For clips of A and B frames with O frames of overlap, place the first at [0,A)
and the next at [A-O,A-O+B). Total length is A+B-O frames. With more clips,
advance each start by its preceding clip's length minus the overlap. Require
0 <= O <= min(A,B). At O=0, omit the fade: the incoming node starts fully opaque.
Windows control existence, including sound; opacity only
changes the picture. Order siblings from back to front.

For a dissolve, keep the outgoing opaque picture underneath and animate the
incoming node's opacity from 0 at local frame 0 to 1 at local frame O. The
outgoing ends as the incoming reaches 1. Fading both pictures over a background
also exposes that background. For a slide, key the incoming translation instead.

Example: examples/animation-recipes/sequences.py uses two 60-frame cards, a
12-frame dissolve and 30000/1001 FPS: starts 0 and 48, ends 60 and 108. Frames
48..59 overlap; frame 60 is entirely the second card. Generation and render
commands: examples/animation-recipes/README.md.
Clock fields: vork help field node.active, vork help field node.time.
Interpolation: vork help animation. Sampled motion: vork help springs.
