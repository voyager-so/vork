IMAGES
image.color_space overrides embedded colour tags; untagged HDR/EXR uses Linear Rec.709, other files sRGB.
Use Non-Color for data. Tagged PQ/HLG inputs and footage tagged Rec.2020 or P3 refuse until decoded; the line names the color_space to set.
Nested camera textures inherit shot profile and timing. A canvas picture uses image.render.root and image.render.resolution;
bind that image to a mesh material. Resolution is fixed, independent of camera distance.
Worlds and canvases may nest.
A canvas inside a canvas cannot set viewBox or preserveAspectRatio.
Missing sampled image files, including environment images, use one magenta texel
and emit a warning naming the file to restore. Corrupt images, IES profiles and
colour-transform library data remain errors. Movie image inputs decode video only;
attach audio: {file: ...} to a node to include the soundtrack.
16-bit integer and floating-point image inputs avoid block compression but use half-float texture precision; exact source integer codes are not preserved.

<!-- all -->
ANIMATED TEXTURE REFERENCE
image.file accepts .mov/.mp4/.mkv and %d/%04d frame patterns. Video requires
user-installed ffmpeg and ffprobe. image.time is seconds into the clip; omitted
follows its node's local clock, or scene time without one. Canvas path fills and
strokes read movies and frame sequences through the same image binding. The ordinary animation path ["images",name,"time"] remaps it.
Sequence time zero is the first on-disk frame. EXR sequences use framesPerSecond
from the first file when present; it must be finite and positive. Without that
metadata, sequences use render.fps. Interior
gaps are errors naming missing frames. Video uses displayed presentation timestamps
relative to the first displayed frame, honours edit lists and rotation, and holds
the last frame at or before time. All footage holds end frames outside its range.

Surface and environment footage follows the transport sample clock. Final
compositor images follow the nominal output-frame clock after exposure integration.

image.render captures use each parent transport sample's pose and time, with a
closed capture shutter. Captures retain scene dependencies even when the final
compositor replaces camera beauty. File/video compositor samples are premultiplied
before filtering; return premultiplied RGBA from custom composites.
