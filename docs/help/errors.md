ERRORS: exit codes: 0 success; 2 usage; 3 invalid scene or program;
4 device, GPU memory, encoder or setup, and any other error not known to be the scene's
(5 when that error is filesystem IO).
A stdout consumer closing its pipe early exits successfully. Other stdout write failures
are filesystem IO errors.
A frame the renderer refuses (non-finite radiance, say) stops the render, a video or a
sequence alike: frames already written stay, without meta.json. A 2D canvas writing a
PNG or EXR sequence skips a frame whose view or compositor result is not finite and
renders the rest of the range, then exits non-zero naming the refused frames, without
meta.json.

If setup cannot open a GPU on Linux, it checks Vulkan ICD driver files and names
libraries that cannot load or expose Vulkan entry points. A GPU visible in
nvidia-smi can still have a broken ICD. Use VK_LOADER_DEBUG=error,warn vork setup
for loader details; see docs/REMOTE-GPU.md for headless NVIDIA driver discovery.
