// Independent Metal -> IOSurface -> one VideoToolbox session -> AVAssetWriter mux.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include "constants.h"
#include <unistd.h>
#include <stdatomic.h>
#include <time.h>
#include <stdio.h>

typedef struct { CVPixelBufferRef pixel; CVMetalTextureRef texture; dispatch_group_t consumed; uint64_t index; } Surface;
typedef struct {
    VTCompressionSessionRef session;
    CVPixelBufferPoolRef pool;
    CVMetalTextureCacheRef cache;
    void *writer, *input, *queue, *pipeline, *ready, *surfaces;
    uint64_t next;
    unsigned width,height,fps_num,fps_den;
    atomic_int error;
    dispatch_group_t pending;
    dispatch_queue_t handoff,writer_queue;
    dispatch_group_t writing;
    bool started,finished,prores;
    CFStringRef primaries;
} Encoder;
// BGRA input is full-range RGB; VT converts ProRes to video-range BT.709 Y'CbCr.
static void tag_surface(Encoder *e,CVPixelBufferRef pixel) {
    if(e->prores)CVBufferSetAttachment(pixel,kCVImageBufferAlphaChannelModeKey,kCVImageBufferAlphaChannelMode_StraightAlpha,kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(pixel,kCVImageBufferColorPrimariesKey,e->primaries,kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(pixel,kCVImageBufferTransferFunctionKey,kCVImageBufferTransferFunction_ITU_R_709_2,kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(pixel,kCVImageBufferYCbCrMatrixKey,kCVImageBufferYCbCrMatrix_ITU_R_709_2,kCVAttachmentMode_ShouldPropagate);
}
static void append_sample(Encoder *e,CMSampleBufferRef sb,uint64_t index) {

    AVAssetWriter *writer=(__bridge AVAssetWriter *)e->writer;
    AVAssetWriterInput *input=(__bridge AVAssetWriterInput *)e->input;
    if (!e->started) {
        input=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:nil sourceFormatHint:CMSampleBufferGetFormatDescription(sb)];
        input.mediaTimeScale=(CMTimeScale)e->fps_num;
        if (![writer canAddInput:input]) {e->error=-2;return;}
        e->input=(void *)CFBridgingRetain(input);
        [writer addInput:input];
        if (![writer startWriting]) {e->error=-3;return;}
        [writer startSessionAtSourceTime:kCMTimeZero];e->started=true;
    }
    while (!input.readyForMoreMediaData) {
        if (writer.status==AVAssetWriterStatusFailed || writer.status==AVAssetWriterStatusCancelled) {e->error=-4;return;}
        usleep(VORK_WRITER_POLL_US);
    }
    if (![input appendSampleBuffer:sb]) {NSLog(@"append: %@",writer.error);e->error=-5;}

}
static void sample(void *ctx, void *ref, OSStatus status, VTEncodeInfoFlags flags, CMSampleBufferRef sb) {
    Encoder *e=ctx;
    (void)flags;
    Surface *surface=ref;
    uint64_t index=surface->index;

    if(status || !sb) {
        e->error=status ? status : -1;
        dispatch_group_leave(surface->consumed);
        return;
    }
    // VT owns sb only during this callback. The bounded slot credit remains
    // held until the writer consumes it, so a slow disk cannot grow this queue.
    CFRetain(sb);
    dispatch_group_async(e->writing,e->writer_queue,^{
        @autoreleasepool {
            if(!atomic_load(&e->error))append_sample(e,sb,index);
            CFRelease(sb);
            dispatch_group_leave(surface->consumed);
        }
    });
}

void enc_drop(void *ctx) {
    if (!ctx) return;
    Encoder *e=ctx;
    if (e->pending) dispatch_group_wait(e->pending,DISPATCH_TIME_FOREVER);
    if (e->session) {VTCompressionSessionCompleteFrames(e->session,kCMTimeInvalid);}
    if (e->writing) dispatch_group_wait(e->writing,DISPATCH_TIME_FOREVER);
    if (e->session) {VTCompressionSessionInvalidate(e->session);CFRelease(e->session);}
    if(e->surfaces) {
        NSDictionary *surfaces=CFBridgingRelease(e->surfaces);
        for(NSValue *value in surfaces.allValues) {
            Surface *surface=value.pointerValue;
            CFRelease(surface->texture);CVPixelBufferRelease(surface->pixel);surface->consumed=nil;free(surface);
        }
    }
    if (e->pool) CFRelease(e->pool);
    if (e->cache) CFRelease(e->cache);
    if(e->writer) {AVAssetWriter *w=CFBridgingRelease(e->writer); if(!e->finished)[w cancelWriting];}
    if(e->input) CFBridgingRelease(e->input);
    if(e->queue) CFBridgingRelease(e->queue);
    if(e->pipeline) CFBridgingRelease(e->pipeline);
    if(e->ready) CFBridgingRelease(e->ready);
    e->pending=nil;e->handoff=nil;e->writing=nil;e->writer_queue=nil;
    free(e);
}
// `primaries` is the H.273 code of the frames' display: 12 is P3 at D65, anything else BT.709.
void *enc_prepare(unsigned w,unsigned h,unsigned fps_num,unsigned fps_den,unsigned bitrate,unsigned gop_frames,bool prores,unsigned primaries) {
    @autoreleasepool {
    Encoder *e=calloc(1,sizeof(*e));e->width=w;e->height=h;e->fps_num=fps_num;e->fps_den=fps_den;e->prores=prores;
    e->primaries=primaries==12 ? kCVImageBufferColorPrimaries_P3_D65 : kCVImageBufferColorPrimaries_ITU_R_709_2;
    e->ready=(void *)CFBridgingRetain([NSMutableDictionary new]);
    e->surfaces=(void *)CFBridgingRetain([NSMutableDictionary new]);
    e->pending=dispatch_group_create();
    e->writing=dispatch_group_create();
    e->writer_queue=dispatch_queue_create("vork.encode.writer",DISPATCH_QUEUE_SERIAL);
    e->handoff=dispatch_queue_create("vork.encode.handoff",DISPATCH_QUEUE_SERIAL);
    NSDictionary *attrs=@{(id)kCVPixelBufferWidthKey:@(w),(id)kCVPixelBufferHeightKey:@(h),
        (id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA),
        (id)kCVPixelBufferIOSurfacePropertiesKey:@{},(id)kCVPixelBufferMetalCompatibilityKey:@YES};
    OSStatus s=CVPixelBufferPoolCreate(NULL,NULL,(__bridge CFDictionaryRef)attrs,&e->pool);
    NSDictionary *spec=prores
        ? @{(id)kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder:@YES}
        : @{(id)kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder:@YES};
    if(!s)s=VTCompressionSessionCreate(NULL,w,h,prores ? kCMVideoCodecType_AppleProRes4444 : kCMVideoCodecType_H264,(__bridge CFDictionaryRef)spec,(__bridge CFDictionaryRef)attrs,NULL,sample,e,&e->session);
    if(!s && !prores)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_AllowFrameReordering,kCFBooleanFalse);
    if(!s)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_ExpectedFrameRate,(__bridge CFNumberRef)@((double)fps_num/fps_den));
    if(!s && prores)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_PreserveAlphaChannel,kCFBooleanTrue);
    if(!s && !prores)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_AverageBitRate,(__bridge CFNumberRef)@(bitrate));
    if(!s && !prores)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_MaxKeyFrameInterval,(__bridge CFNumberRef)@(gop_frames));
    if(!s)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_ColorPrimaries,e->primaries);
    if(!s)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_TransferFunction,kCVImageBufferTransferFunction_ITU_R_709_2);
    if(!s)s=VTSessionSetProperty(e->session,kVTCompressionPropertyKey_YCbCrMatrix,kCVImageBufferYCbCrMatrix_ITU_R_709_2);
    if(!s)s=VTCompressionSessionPrepareToEncodeFrames(e->session);
    if(s) {NSLog(@"VT initialization: %d",s);enc_drop(e);return NULL;}
    return e;
    }
}
// File creation is deliberately deferred until the renderer validates its device.
int enc_attach(void *ctx,const char *path) {
    @autoreleasepool {
    Encoder *e=ctx;
    NSError *error=nil;
    NSURL *url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
    AVAssetWriter *writer=[[AVAssetWriter alloc] initWithURL:url fileType:(e->prores ? AVFileTypeQuickTimeMovie : AVFileTypeMPEG4) error:&error];
    if(!writer)return -1; writer.movieTimeScale=(CMTimeScale)e->fps_num; e->writer=(void *)CFBridgingRetain(writer);
    return 0;
    }
}
// Surfaces are installed once in graph ring entries; the session retains their CV handles.
void *enc_acquire(void *ctx,void *rawDevice) {
    @autoreleasepool {
    Encoder *e=ctx;if(e->error)return NULL;
    id<MTLDevice> device=(__bridge id<MTLDevice>)rawDevice;
    if(!e->cache && CVMetalTextureCacheCreate(NULL,NULL,device,NULL,&e->cache)) {e->error=-11;return NULL;}
    if(!e->pipeline) {
        // Share wgpu's queue so conversion observes rendering and precedes slot reuse.
        NSString *shader=@"#include <metal_stdlib>\nusing namespace metal; kernel void convert(texture2d<float,access::read> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]], uint2 p [[thread_position_in_grid]]) {if(p.x<dst.get_width() && p.y<dst.get_height())dst.write(src.read(p),p);}";
        NSError *err=nil;
        id<MTLLibrary> lib=[device newLibraryWithSource:shader options:nil error:&err];
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"convert"] error:&err];
        if(!pipeline){e->error=-10;return NULL;}
        e->pipeline=(void *)CFBridgingRetain(pipeline);
        if(!e->cache && CVMetalTextureCacheCreate(NULL,NULL,device,NULL,&e->cache)){e->error=-11;return NULL;}
    }
    Surface *surface=calloc(1,sizeof(*surface));
    if(!surface){e->error=-13;return NULL;}
    surface->consumed=dispatch_group_create();
    OSStatus s=CVPixelBufferPoolCreatePixelBuffer(NULL,e->pool,&surface->pixel);
    if(!s) {
        tag_surface(e,surface->pixel);
        NSDictionary *attrs=@{(id)kCVMetalTextureUsage:@(MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite)};
        s=CVMetalTextureCacheCreateTextureFromImage(NULL,e->cache,surface->pixel,(__bridge CFDictionaryRef)attrs,MTLPixelFormatBGRA8Unorm,e->width,e->height,0,&surface->texture);
    }
    if(s) {
        if(surface->texture)CFRelease(surface->texture);
        if(surface->pixel)CVPixelBufferRelease(surface->pixel);
        surface->consumed=nil;free(surface);e->error=s;return NULL;
    }
    id<MTLTexture> texture=CVMetalTextureGetTexture(surface->texture);
    NSMutableDictionary *surfaces=(__bridge NSMutableDictionary *)e->surfaces;
    surfaces[[NSValue valueWithPointer:(__bridge void *)texture]]=[NSValue valueWithPointer:surface];
    return (__bridge void *)texture;
    }
}
int enc_wait(void *ctx,void *raw) {
    Encoder *e=ctx;
    NSDictionary *surfaces=(__bridge NSDictionary *)e->surfaces;
    Surface *surface=[surfaces[[NSValue valueWithPointer:raw]] pointerValue];
    if(!surface)return -14;
    dispatch_group_wait(surface->consumed,DISPATCH_TIME_FOREVER);
    return atomic_load(&e->error);
}
int enc_push(void *ctx,void *raw,void *destination,void *rawQueue,uint64_t index) {
    @autoreleasepool {
    Encoder *e=ctx;if(e->error)return e->error;
    id<MTLTexture> source=(__bridge id<MTLTexture>)raw;
    NSMutableDictionary *surfaces=(__bridge NSMutableDictionary *)e->surfaces;
    NSValue *key=[NSValue valueWithPointer:destination];
    Surface *surface=[surfaces[key] pointerValue];
    if(!surface)return e->error=-14;
    bool direct=raw==destination;
    surface->index=index;
    if(!e->queue)e->queue=(void *)CFBridgingRetain((__bridge id<MTLCommandQueue>)rawQueue);
    CVPixelBufferRef pixel=surface->pixel;
    CVMetalTextureRef cvtex=surface->texture;
    OSStatus s=0;
    dispatch_group_enter(surface->consumed);
    if(!s) {
        id<MTLCommandBuffer> command=[(__bridge id<MTLCommandQueue>)e->queue commandBuffer];
        if(!direct) {
        id<MTLComputeCommandEncoder> compute=[command computeCommandEncoder];
        [compute setComputePipelineState:(__bridge id<MTLComputePipelineState>)e->pipeline];
        [compute setTexture:source atIndex:0];[compute setTexture:CVMetalTextureGetTexture(cvtex) atIndex:1];
        [compute dispatchThreads:MTLSizeMake(e->width,e->height,1) threadsPerThreadgroup:MTLSizeMake(VORK_CONVERSION_TILE,VORK_CONVERSION_TILE,1)];
        [compute endEncoding];
        }
        dispatch_group_enter(e->pending);
        [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {

            // Reorder completion notifications on a serial queue before feeding one VT session.
            // Codec work stays off the Metal callback and the rendering thread.
            dispatch_async(e->handoff, ^{
                @autoreleasepool {
                NSMutableDictionary *ready=(__bridge NSMutableDictionary *)e->ready;
                ready[@(index)]=^{
                OSStatus result=completed.status==MTLCommandBufferStatusError ? -12 : atomic_load(&e->error);

                if(!result)result=VTCompressionSessionEncodeFrame(e->session,pixel,CMTimeMake(index*e->fps_den,e->fps_num),CMTimeMake(e->fps_den,e->fps_num),NULL,surface,NULL);
                if(result){e->error=result;dispatch_group_leave(surface->consumed);}
                };
                dispatch_group_t pending=e->pending;
                unsigned drained=0;
                dispatch_block_t feed;
                while((feed=ready[@(e->next)])) {
                    [ready removeObjectForKey:@(e->next)];
                    e->next++;feed();drained++;
                }
                // Last leaves may release finish/drop: never touch e after this point.
                while(drained--)dispatch_group_leave(pending);
                }
            });
        }];
        [command commit];
        // The installed surface owns the CV objects; its completion releases the ring entry.
        return atomic_load(&e->error);
    }
    dispatch_group_leave(surface->consumed);
    return e->error=s;
    }
}
int enc_finish(void *ctx) {
    Encoder *e=ctx;
    dispatch_group_wait(e->pending,DISPATCH_TIME_FOREVER);
    OSStatus s=VTCompressionSessionCompleteFrames(e->session,kCMTimeInvalid);
    dispatch_group_wait(e->writing,DISPATCH_TIME_FOREVER);
    if(s || e->error)return s?s:e->error;
    if(!e->started)return -20;
    AVAssetWriter *writer=(__bridge AVAssetWriter *)e->writer;
    [(__bridge AVAssetWriterInput *)e->input markAsFinished];
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    [writer finishWritingWithCompletionHandler:^{dispatch_semaphore_signal(done);}];
    dispatch_semaphore_wait(done,DISPATCH_TIME_FOREVER);
    if(writer.status!=AVAssetWriterStatusCompleted){NSLog(@"finish: %@",writer.error);return -21;}
    e->finished=true;return 0;
}
