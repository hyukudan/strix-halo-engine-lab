#include "strix/dist/staged_ready_kernels.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix::dist {
namespace {
__global__ __launch_bounds__(256) void pattern_kernel(ProducerStage view, std::size_t bytes, unsigned value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const auto groups = bytes / 16U;
    auto* output = static_cast<uint4*>(view.data);
    const uint4 v{value, value, value, value};
    for (std::size_t i = index; i < groups; i += stride) output[i] = v;
    for (std::size_t i = groups * 16U + index; i < bytes; i += stride)
        static_cast<unsigned char*>(view.data)[i] = static_cast<unsigned char>(value >> ((i & 3U) * 8U));
    publish_staged_block(view.flag, view.token, gridDim.x);
}

template<bool ReadsOnly>
__global__ __launch_bounds__(256) void consume_kernel(ConsumerStage view, unsigned char* output, std::size_t bytes) {
    staged_acquire(view.flag, view.token);
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const auto groups = bytes / 16U;
    if ((reinterpret_cast<unsigned long long>(output) & 15ULL) == 0ULL) {
        for (std::size_t i = index; i < groups; i += stride)
            reinterpret_cast<uint4*>(output)[i] = static_cast<const uint4*>(view.data)[i];
        for (std::size_t i = groups * 16U + index; i < bytes; i += stride)
            output[i] = static_cast<const unsigned char*>(view.data)[i];
    } else {
        for (std::size_t i = index; i < bytes; i += stride) output[i] = static_cast<const unsigned char*>(view.data)[i];
    }
    if constexpr (ReadsOnly) publish_staged_reads_block(view.flag, view.token, gridDim.x);
    else publish_staged_block(view.flag, view.token, gridDim.x);
}

unsigned blocks(std::size_t bytes) {
    return static_cast<unsigned>(std::clamp<std::size_t>((bytes + 4095U) / 4096U, 1U, 32U));
}
void check_launch() {
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("staged readiness launch: ") + hipGetErrorString(status));
}
}

void enqueue_staged_pattern(ProducerStage view, std::size_t bytes, std::uint32_t pattern, hipStream_t stream) {
    if (!view.data || !view.flag || !view.token || !bytes) throw std::invalid_argument("staged producer view");
    hipLaunchKernelGGL(pattern_kernel, dim3(blocks(bytes)), dim3(256), 0, stream, view, bytes, pattern);
    check_launch();
}
void enqueue_staged_consume(ConsumerStage view, void* output, std::size_t bytes, hipStream_t stream) {
    if (!view.data || !view.flag || !view.token || !output || !bytes) throw std::invalid_argument("staged consumer view");
    hipLaunchKernelGGL(consume_kernel<false>, dim3(blocks(bytes)), dim3(256), 0, stream, view, static_cast<unsigned char*>(output), bytes);
    check_launch();
}
void enqueue_staged_consume_reads(ConsumerStage view, void* output, std::size_t bytes, hipStream_t stream, unsigned grid_blocks) {
    if (!view.data || !view.flag || !view.token || !output || !bytes || grid_blocks > 4096U)
        throw std::invalid_argument("staged read-only consumer view/grid");
    hipLaunchKernelGGL(consume_kernel<true>, dim3(grid_blocks ? grid_blocks : blocks(bytes)), dim3(256), 0, stream,
                       view, static_cast<unsigned char*>(output), bytes);
    check_launch();
}
}
