COMPOSITOR
composition.compositor: ordered material-name|{material,name?,resolution?} entries; last image is output; material may be inline.
One spelling: a world's composition, at the root or on a node inside a canvas, runs its graph before its own view; a rendered image runs images.<name>.composition.compositor on its capture.
A 2D canvas, at the root or one a world reads through its picture, runs its graph on its own picture before its view.
On a canvas the images are premultiplied colour in the canvas's working encoding (composition.color_space): sRGB code values by default, not scene-linear light, so decode before linear-light maths; linear light on a linear canvas. {image} files arrive in that encoding too.
On a canvas: channel color only (no depth, view_normal, object_index, motion, shadow_catcher, AOV or rendered image); camera fields are identity or zero and ctx.samples is 1.
A canvas inside a canvas paints inline in its parent and takes no compositor.
resolution: [width,height] or positive output scale (floor, minimum 1); omitted = output; final image resampled to output.
Image bindings: {step:earlier-name}, {image:name}, {channel:color|depth|view_normal|object_index|motion|shadow_catcher}, {aov:name}; names are local to each graph.
Depth, view_normal and object_index describe the first visible hit, not hidden rear edges;
view_normal includes authored smoothing. The compositor has no temporal history.
Channel bindings return raw values, before file encoding: depth in .x is view-axis
metres (background 1e10); view_normal in .xyz is a signed unit vector in camera
space (background zero); object_index in .x is the numeric object ID (background
zero); motion in .xy is pixel displacement, +Y down, current minus previous.
<!-- all -->
COMPOSITOR PROGRAM REFERENCE
float4 composite(CompositeContext,Inputs): premultiplied RGBA in the composition's working encoding (a world: scene-linear; a canvas: sRGB code values unless composition.color_space names another); compositor_sample(uv) reads previous result (first reads accumulated beauty, or the canvas's picture).
File/video image samples become premultiplied RGBA before filtering.
compositor_sample(image,uv): clamped bilinear; compositor_load(image,int2): clamped pixel; compositor_size(image): pixels; compositor_size() reads previous extent. UV origin top left; ctx.size is working extent.
Runs after shutter accumulation/denoising (a canvas: after it is drawn), before the view. A result that is not finite refuses the frame.
