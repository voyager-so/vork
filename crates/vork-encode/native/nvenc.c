/* One CUDA/NVENC session with input/bitstream entries sized by the graph ring. Only compressed bytes leave the GPU.
 * ABI declarations are upstream MIT headers; see nvcodec/PROVENANCE. */
#include "constants.h"
#include "nvcodec/nvEncodeAPI.h"
#include "nvcodec/dynlink_cuda.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

typedef struct {
    void *cuda, *library, *encoder;
    CUcontext context;
    CUexternalMemory memory;
    CUexternalSemaphore semaphore;
    CUstream stream;
    tcuWaitExternalSemaphoresAsync *wait_semaphore;
    tcuDestroyExternalSemaphore *destroy_semaphore;
    tcuStreamSynchronize *sync_stream;
    tcuStreamDestroy_v2 *destroy_stream;
    CUdeviceptr pixels;
    NV_ENCODE_API_FUNCTION_LIST api;
    NV_ENC_REGISTERED_PTR (*registered)[2];
    NV_ENC_OUTPUT_PTR *output;
    NV_ENC_INPUT_PTR *mapped;
    unsigned width, height, pitch;
    int *locked, *submitted;
    unsigned capacity;
    double submit_ms, drain_ms;
    tcuCtxPushCurrent_v2 *push;
    tcuCtxPopCurrent_v2 *pop;
    tcuCtxDestroy_v2 *destroy;
    tcuMemFree_v2 *free_pixels;
    tcuDestroyExternalMemory *free_memory;
} NvSession;

void vork_nv_drop(NvSession *s) {
    if (!s) return;
    if (s->context && s->push(s->context) == CUDA_SUCCESS) {
        if (s->stream) s->sync_stream(s->stream);
        for (unsigned slot = 0; slot < s->capacity; ++slot) {
            if (s->submitted[slot] && !s->locked[slot]) {
                NV_ENC_LOCK_BITSTREAM lock = {0};
                lock.version = NV_ENC_LOCK_BITSTREAM_VER;
                lock.outputBitstream = s->output[slot];
                if (s->api.nvEncLockBitstream(s->encoder, &lock) == NV_ENC_SUCCESS)
                    s->locked[slot] = 1;
            }
            if (s->locked[slot]) s->api.nvEncUnlockBitstream(s->encoder, s->output[slot]);
            if (s->mapped[slot]) s->api.nvEncUnmapInputResource(s->encoder, s->mapped[slot]);
            if (s->output[slot]) s->api.nvEncDestroyBitstreamBuffer(s->encoder, s->output[slot]);
            for (unsigned format = 0; format < 2; ++format)
                if (s->registered[slot][format]) s->api.nvEncUnregisterResource(s->encoder, s->registered[slot][format]);
        }
        if (s->encoder) s->api.nvEncDestroyEncoder(s->encoder);
        if (s->stream) s->destroy_stream(s->stream);
        if (s->semaphore) s->destroy_semaphore(s->semaphore);
        if (s->pixels) s->free_pixels(s->pixels);
        if (s->memory) s->free_memory(s->memory);
        CUcontext previous;
        s->pop(&previous);
        s->destroy(s->context);
    }
    if (s->library) dlclose(s->library);
    if (s->cuda) dlclose(s->cuda);
    free(s->registered); free(s->output); free(s->mapped);
    free(s->locked); free(s->submitted);
    free(s);
}

/* Driver initialization needs no Vulkan device. Keep the libraries loaded so
 * prepare can overlap device startup without unloading/reinitializing CUDA. */
NvSession *vork_nv_prepare(int *error) {
    NvSession *s = calloc(1, sizeof(*s));
    *error = -1;
    if (!s) return NULL;
    s->cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    s->library = dlopen("libnvidia-encode.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!s->cuda || !s->library) goto fail;
    tcuInit *init = (tcuInit *)dlsym(s->cuda, "cuInit");
    if (!init) goto fail;
    *error = init(0);
    if (*error) goto fail;
    return s;
fail:
    vork_nv_drop(s);
    return NULL;
}

/* Consumes preparation and fd on every return. Match the actual Vulkan UUID. */
NvSession *vork_nv_open(NvSession *s, int fd, uint64_t allocation_size, const unsigned char *uuid,
                         unsigned width, unsigned height, unsigned pitch, unsigned fps_num, unsigned fps_den, unsigned bitrate, unsigned gop_frames,
                         unsigned primaries, unsigned capacity, int *error) {
    *error = -1;
    if (!s) { close(fd); return NULL; }
    s->capacity = capacity;
    s->registered = calloc(capacity, sizeof(*s->registered));
    s->output = calloc(capacity, sizeof(*s->output));
    s->mapped = calloc(capacity, sizeof(*s->mapped));
    s->locked = calloc(capacity, sizeof(*s->locked));
    s->submitted = calloc(capacity, sizeof(*s->submitted));
    if (!capacity || !s->registered || !s->output || !s->mapped || !s->locked || !s->submitted) goto fail;
    s->width = width; s->height = height; s->pitch = pitch;
#define SYMBOL(type, name, symbol) type *name = (type *)dlsym(s->cuda, symbol); if (!name) goto fail
    SYMBOL(tcuDeviceGetCount, count, "cuDeviceGetCount");
    SYMBOL(tcuDeviceGet, device, "cuDeviceGet");
    SYMBOL(tcuCtxCreate_v2, create, "cuCtxCreate_v2");
    SYMBOL(tcuImportExternalMemory, import_memory, "cuImportExternalMemory");
    SYMBOL(tcuExternalMemoryGetMappedBuffer, map_memory, "cuExternalMemoryGetMappedBuffer");
#define MEMBER(field, symbol) *(void **)(&s->field) = dlsym(s->cuda, symbol); if (!s->field) goto fail
    MEMBER(push, "cuCtxPushCurrent_v2"); MEMBER(pop, "cuCtxPopCurrent_v2");
    MEMBER(destroy, "cuCtxDestroy_v2"); MEMBER(free_pixels, "cuMemFree_v2");
    MEMBER(free_memory, "cuDestroyExternalMemory");
    typedef CUresult (CUDAAPI GetUuid)(CUuuid *, CUdevice);
    GetUuid *get_uuid = (GetUuid *)dlsym(s->cuda, "cuDeviceGetUuid");
    if (!get_uuid) goto fail;
    int n = 0;
    *error = count(&n);
    if (*error) goto fail;
    CUdevice selected = -1;
    for (int i = 0; i < n; ++i) {
        CUdevice candidate;
        CUuuid candidate_uuid;
        *error = device(&candidate, i);
        if (*error) goto fail;
        *error = get_uuid(&candidate_uuid, candidate);
        if (*error) goto fail;
        if (!memcmp(uuid, candidate_uuid.bytes, 16)) { selected = candidate; break; }
    }
    *error = -2;
    if (selected < 0) goto fail;
    *error = create(&s->context, 0, selected);
    if (*error) goto fail;
    /* cuCtxCreate makes the context current; balance it even on construction failure. */
    CUcontext previous;
    s->pop(&previous);
    *error = s->push(s->context);
    if (*error) goto fail;
#define CHECK(expr) do { *error = (expr); if (*error) { fprintf(stderr, "NVENC %s: %d\n", #expr, *error); goto current_fail; } } while (0)
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC external = {0};
    external.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    external.handle.fd = fd;
    external.size = allocation_size;
    external.flags = 1; /* dedicated Vulkan allocation */
    CHECK(import_memory(&s->memory, &external));
    fd = -1; /* CUDA owns the descriptor after successful import. */
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapping = {0};
    mapping.size = (uint64_t)pitch * height * s->capacity;
    CHECK(map_memory(&s->pixels, s->memory, &mapping));
    typedef NVENCSTATUS (NVENCAPI CreateApi)(NV_ENCODE_API_FUNCTION_LIST *);
    CreateApi *create_api = (CreateApi *)dlsym(s->library, "NvEncodeAPICreateInstance");
    *error = -3;
    if (!create_api) goto current_fail;
    s->api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    CHECK(create_api(&s->api));
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open = {0};
    open.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    open.apiVersion = NVENCAPI_VERSION;
    open.device = s->context;
    open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    CHECK(s->api.nvEncOpenEncodeSessionEx(&open, &s->encoder));
    NV_ENC_PRESET_CONFIG preset = {0};
    preset.version = NV_ENC_PRESET_CONFIG_VER;
    preset.presetCfg.version = NV_ENC_CONFIG_VER;
    CHECK(s->api.nvEncGetEncodePresetConfigEx(s->encoder, NV_ENC_CODEC_H264_GUID,
        NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &preset));
    NV_ENC_CONFIG config = preset.presetCfg;
    config.gopLength = gop_frames;
    config.frameIntervalP = 1;
    config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    config.rcParams.averageBitRate = bitrate;
    config.rcParams.enableLookahead = 0;
    config.rcParams.zeroReorderDelay = 1;
    config.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    /* NVENC's RGB input conversion produces limited-range BT.709 YUV. Without
     * VUI tags decoders can assume BT.601 and change the rendered colours.
     * The primaries are the frames' display's, as an H.273 code. */
    NV_ENC_CONFIG_H264_VUI_PARAMETERS *vui = &config.encodeCodecConfig.h264Config.h264VUIParameters;
    vui->videoSignalTypePresentFlag = 1;
    vui->videoFormat = 5;
    vui->videoFullRangeFlag = 0;
    vui->colourDescriptionPresentFlag = 1;
    vui->colourPrimaries = (NV_ENC_VUI_COLOR_PRIMARIES)primaries;
    vui->transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
    vui->colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT709;
    NV_ENC_INITIALIZE_PARAMS initialize = {0};
    initialize.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initialize.encodeGUID = NV_ENC_CODEC_H264_GUID;
    initialize.presetGUID = NV_ENC_PRESET_P1_GUID;
    initialize.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    initialize.encodeWidth = initialize.darWidth = width;
    initialize.encodeHeight = initialize.darHeight = height;
    initialize.frameRateNum = fps_num;
    initialize.frameRateDen = fps_den;
    initialize.enablePTD = 1;
    initialize.encodeConfig = &config;
    CHECK(s->api.nvEncInitializeEncoder(s->encoder, &initialize));
    for (unsigned slot = 0; slot < s->capacity; ++slot) {
        NV_ENC_CREATE_BITSTREAM_BUFFER output = {0};
        output.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        CHECK(s->api.nvEncCreateBitstreamBuffer(s->encoder, &output));
        s->output[slot] = output.bitstreamBuffer;
    }
    s->pop(&previous);
    return s;
current_fail:
    s->pop(&previous);
fail:
    if (fd >= 0) close(fd);
    vork_nv_drop(s);
    return NULL;
}

/* Caller drains a slot before reusing its encoder-owned pixels. */
/* Consume an exported Vulkan timeline semaphore on the NVENC input stream. */
int vork_nv_sync(NvSession *s,int fd) {
    int status=s->push(s->context);
    if(status){close(fd);return status;}
    tcuImportExternalSemaphore *import=(tcuImportExternalSemaphore *)dlsym(s->cuda,"cuImportExternalSemaphore");
    tcuStreamCreate *create=(tcuStreamCreate *)dlsym(s->cuda,"cuStreamCreate");
    *(void **)(&s->wait_semaphore)=dlsym(s->cuda,"cuWaitExternalSemaphoresAsync");
    *(void **)(&s->destroy_semaphore)=dlsym(s->cuda,"cuDestroyExternalSemaphore");
    *(void **)(&s->sync_stream)=dlsym(s->cuda,"cuStreamSynchronize");
    *(void **)(&s->destroy_stream)=dlsym(s->cuda,"cuStreamDestroy_v2");
    if(!import || !create || !s->wait_semaphore || !s->destroy_semaphore || !s->sync_stream || !s->destroy_stream) {status=-4;goto done;}
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC descriptor={0};
    descriptor.type=CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
    descriptor.handle.fd=fd;
    status=import(&s->semaphore,&descriptor);
    if(status)goto done;
    fd=-1;
    status=create(&s->stream,CU_STREAM_NON_BLOCKING);
    /* Despite the void* typedef, NVENC expects addresses of CUstream handles.
     * CUDA's own stream APIs below take the handles themselves. */
    if(!status)status=s->api.nvEncSetIOCudaStreams(s->encoder,&s->stream,&s->stream);
 done:;
    if(fd>=0)close(fd);
    CUcontext previous;s->pop(&previous);
    return status;
}
int vork_nv_push(NvSession *s, uint64_t index, int bgra, uint64_t value) {
    unsigned slot = index % s->capacity;
    int status = s->push(s->context);
    if (status) return status;
    if(s->semaphore) {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait={0};
        wait.params.fence.value=value;
        status=s->wait_semaphore(&s->semaphore,&wait,1,s->stream);
        if(status)goto done;
    }
    NV_ENC_BUFFER_FORMAT format = bgra ? NV_ENC_BUFFER_FORMAT_ARGB : NV_ENC_BUFFER_FORMAT_ABGR;
    /* A stream may mix RGBA/BGRA textures. Re-register only on a format change,
     * after the previous frame in this slot has been drained. */
    if (s->registered[slot][!bgra]) {
        status = s->api.nvEncUnregisterResource(s->encoder, s->registered[slot][!bgra]);
        if (status) goto done;
        s->registered[slot][!bgra] = NULL;
    }
    if (!s->registered[slot][bgra]) {
        NV_ENC_REGISTER_RESOURCE resource = {0};
        resource.version = NV_ENC_REGISTER_RESOURCE_VER;
        resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        resource.resourceToRegister = (void *)(uintptr_t)(s->pixels + (uint64_t)slot * s->pitch * s->height);
        resource.width = s->width; resource.height = s->height; resource.pitch = s->pitch;
        resource.bufferFormat = format;
        resource.bufferUsage = NV_ENC_INPUT_IMAGE;
        status = s->api.nvEncRegisterResource(s->encoder, &resource);
        if (status) goto done;
        s->registered[slot][bgra] = resource.registeredResource;
    }
    NV_ENC_MAP_INPUT_RESOURCE input = {0};
    input.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    input.registeredResource = s->registered[slot][bgra];
    status = s->api.nvEncMapInputResource(s->encoder, &input);
    if (status) goto done;
    s->mapped[slot] = input.mappedResource;
    NV_ENC_PIC_PARAMS picture = {0};
    picture.version = NV_ENC_PIC_PARAMS_VER;
    picture.inputBuffer = input.mappedResource;
    picture.bufferFmt = format;
    picture.inputWidth = s->width; picture.inputHeight = s->height; picture.inputPitch = s->pitch;
    picture.outputBitstream = s->output[slot];
    picture.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picture.inputTimeStamp = index;
    picture.inputDuration = 1;
    double started = milliseconds();
    status = s->api.nvEncEncodePicture(s->encoder, &picture);
    s->submit_ms += milliseconds() - started;
    if (!status) s->submitted[slot] = 1;
done:;
    CUcontext previous;
    s->pop(&previous);
    return status;
}
/* Blocking retrieval only when the bounded ring is full or at finish. */
int vork_nv_read(NvSession *s, unsigned slot, const uint8_t **data, unsigned *size) {
    int status = s->push(s->context);
    if (status) return status;
    NV_ENC_LOCK_BITSTREAM lock = {0};
    lock.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = s->output[slot];
    double started = milliseconds();
    status = s->api.nvEncLockBitstream(s->encoder, &lock);
    s->drain_ms += milliseconds() - started;
    if (!status) { s->locked[slot] = 1; *data = lock.bitstreamBufferPtr; *size = lock.bitstreamSizeInBytes; }
    CUcontext previous;
    s->pop(&previous);
    return status;
}
int vork_nv_unlock(NvSession *s, unsigned slot) {
    int status = s->push(s->context);
    if (status) return status;
    status = s->api.nvEncUnlockBitstream(s->encoder, s->output[slot]);
    if (!status) { s->locked[slot] = 0; s->submitted[slot] = 0; }
    int unmap_status = s->api.nvEncUnmapInputResource(s->encoder, s->mapped[slot]);
    if (!unmap_status) s->mapped[slot] = NULL;
    if (!status) status = unmap_status;
    CUcontext previous;
    s->pop(&previous);
    return status;
}
int vork_nv_finish(NvSession *s, int diagnostics) {
    int status = s->push(s->context);
    if (status) return status;
    if (diagnostics) fprintf(stderr, "nvenc stages: submit_ms=%.3f drain_ms=%.3f\n", s->submit_ms, s->drain_ms);
    NV_ENC_PIC_PARAMS eos = {0};
    eos.version = NV_ENC_PIC_PARAMS_VER;
    eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    status = s->api.nvEncEncodePicture(s->encoder, &eos);
    CUcontext previous;
    s->pop(&previous);
    return status;
}
