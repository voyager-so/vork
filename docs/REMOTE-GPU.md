# Run on your own NVIDIA or AMD GPU

Rendering requires Apple Silicon or an NVIDIA or AMD GPU with Vulkan ray queries.
Authoring scenes and `vork help` do not need a GPU. Docker requires a suitable
GPU on its host; it does not provide one.

The Docker runtime below targets your own Linux x86-64 NVIDIA or AMD machine. Use a
matching Linux release archive and SHA256 from its `SHA256SUMS`. There is no
official registry image to pull yet. See [BUILD.md](BUILD.md) for source builds.
Put scene.json and all its assets in the job directory, keeping relative paths.

## Docker on an NVIDIA host

Use Linux x86-64 with a supported NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host supplies its NVIDIA libraries; do not install a mismatched driver in
the image. Docker Desktop on a Mac does not expose Metal to a Linux container.
WSL2 GPU support is a separate host configuration and has not been validated here.

```sh
python3 tools/gpu/prepare.py --bundle /path/to/vork-0.2.0-linux-x64.tar.gz \
  --sha256 RELEASE_SHA256 --context /tmp/vork-image-context
docker build --platform linux/amd64 -t vork-gpu:local /tmp/vork-image-context
docker run --rm --gpus all vork-gpu:local vork setup
docker run --rm --gpus all \
  -v /absolute/job:/job:ro -v /absolute/results:/results \
  vork-gpu:local vork render scene.json --out /results/video.mp4 --profile std/draft
```

Create the results directory first. The image runs as root by default; use
`--user "$(id -u):$(id -g)" -e VORK_CACHE_DIR=/tmp/vork-cache` on Linux
for your own output ownership. It needs a
writable `/tmp` and cache directory. Mount a writable directory and set
`VORK_CACHE_DIR` to reuse caches across containers; never share a writable cache
between mutually untrusted users.

## Vulkan driver discovery

If `nvidia-smi` sees the GPU but Vulkan finds only llvmpipe, inspect the NVIDIA
ICD manifest in `/etc/vulkan/icd.d` or `/usr/share/vulkan/icd.d` (or the files
selected by `VK_DRIVER_FILES` / `VK_ICD_FILENAMES`). Its `ICD.library_path` must
load and expose Vulkan entry points. `VK_LOADER_DEBUG=error,warn vork setup`
prints loader diagnostics. GPU visibility alone does not validate this library.

The packaged engine discovers mounted NVIDIA EGL/GLX entry points itself when
the Vulkan loader supports direct driver loading; normal ICD discovery remains
available. The image leaves host ICDs unchanged and preserves explicit operator
`VK_DRIVER_FILES`/`VK_ICD_FILENAMES` selections. Slang and OIDN resolve beside the
engine. The entrypoint supplies executable and FFmpeg library paths for SSH
sessions, without creating manifests or supplying runtime-path overrides.
NVIDIA graphics, compute, utility and video capabilities are
requested; native NVENC handles MP4 encoding. FFmpeg is built with GPL/nonfree
features disabled and shared LGPL libraries, with matching source and notices
retained in `/opt/ffmpeg/notices`. There is no libx264 fallback in this image;
a working NVIDIA video encoder is required for H.264 video. This version's
`vork setup` can warn that libx264 is missing alongside its muxing message;
libx264 is not needed for native NVENC muxing. Setup exits successfully without libx264. Slang, OIDN, portable
shaders, the Preview License and all release notices accompany the executable.
The GPL Blender bridge is excluded. Ubuntu OS utilities retain their own
licenses, including GPL; the FFmpeg license check applies to its configured
binaries.

## Docker on an AMD host

Use Linux x86-64 with Mesa RADV on the host, and pass `--device /dev/dri
--device /dev/kfd`. The container needs Mesa's Vulkan driver, and the ROCm 7.x
runtime for GPU denoising; without it vork uses the CPU denoiser. Tested
bare-metal on an RX 7900 XTX; the container path is untested.
