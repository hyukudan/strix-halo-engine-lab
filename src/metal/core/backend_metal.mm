#include "backend.hpp"
#include "strix_metal_rms_shader.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <utility>

namespace strix::metal {
namespace {

using core::gpu_port::Capabilities;
using core::gpu_port::DeviceObservation;
using core::gpu_port::ReferenceBackend;
using core::gpu_port::RmsNormParameters;
using core::gpu_port::refuse;

std::string utf8(NSString* value) {
    const char* text = value.UTF8String;
    return text ? text : "";
}

[[noreturn]] void fail(std::string_view action, NSError* error) {
    const auto description = utf8(error.localizedDescription);
    throw std::runtime_error("Metal: " + std::string(action) +
                             (description.empty() ? "" : " (" + description + ")"));
}

class Backend final : public ReferenceBackend {
    struct Buffer final : MappedBuffer {
        id<MTLBuffer> handle = nil;
    };
public:
    Backend(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline)
        : device_(device), command_queue_(queue), pipeline_(pipeline) { initialize_queue(); }

    std::string_view backend_name() const noexcept override { return "metal"; }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLComputePipelineState> pipeline_;

    std::shared_ptr<session::detail::BufferState> allocate_buffer(std::size_t bytes, std::size_t alignment) override {
        @autoreleasepool {
            auto buffer = std::make_shared<Buffer>();
            buffer->handle = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buffer->handle) throw std::runtime_error("Metal: shared buffer allocation failed");
            buffer->mapped = static_cast<std::byte*>(buffer->handle.contents);
            if (!buffer->mapped || reinterpret_cast<std::uintptr_t>(buffer->mapped) % alignment != 0)
                refuse("Metal", "mapped buffer alignment unavailable");
            return buffer;
        }
    }

    void check_rms_norm(const RmsNormParameters&, std::size_t bytes) const override {
        if (bytes > device_.maxBufferLength) refuse("Metal", "RMSNorm exceeds device buffer limits");
    }

    void launch_rms_norm(const RmsNormParameters& params, const BoundBuffer& input,
                         const BoundBuffer& scale, const BoundBuffer& output) override {
        @autoreleasepool {
            id<MTLCommandBuffer> command = [command_queue_ commandBuffer];
            if (!command) fail("create RMSNorm command", nil);
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) fail("create RMSNorm command encoder", command.error);
            [encoder setComputePipelineState:pipeline_];
            const BoundBuffer* bindings[] = {&input, &scale, &output};
            for (NSUInteger index = 0; index < 3; ++index) {
                const auto& view = *bindings[index];
                const auto& buffer = static_cast<const Buffer&>(*view.state);
                [encoder setBuffer:buffer.handle offset:view.offset atIndex:index];
            }
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(params.rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) fail("complete RMSNorm command", command.error);
        }
    }
};

} // namespace

std::unique_ptr<core::DenseReference> open_native(Capabilities* caps) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        std::vector<DeviceObservation> observations;
        observations.reserve(devices.count);
        for (id<MTLDevice> device in devices)
            observations.push_back({utf8(device.name), device.maxThreadsPerThreadgroup.width != 0, 0, {}});
        auto selected = core::gpu_port::select_compute_device("Metal", observations);
        id<MTLDevice> device = devices[selected.ordinal];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) refuse("Metal", "compute command queue unavailable");
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:rms_norm_msl];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) fail("compile RMSNorm library", error);
        id<MTLFunction> function = [library newFunctionWithName:@"rms_norm"];
        if (!function) fail("find RMSNorm function", nil);
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) fail("create RMSNorm pipeline", error);

        // This is the width of this pipeline on the selected device.
        selected.subgroup_width = core::gpu_port::subgroup_width(pipeline.threadExecutionWidth);
        auto backend = std::make_unique<Backend>(device, queue, pipeline);
        if (caps) *caps = std::move(selected);
        return backend;
    }
}

} // namespace strix::metal
