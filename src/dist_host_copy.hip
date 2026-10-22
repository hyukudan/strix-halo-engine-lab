#include "strix/dist/host_copy.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::dist {
namespace {
template <bool Vector>
__global__ __launch_bounds__(256) void host_copy_kernel(unsigned char* __restrict__ dst,
                                                       const unsigned char* __restrict__ src, std::size_t bytes) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t tail = 0;
    if constexpr (Vector) {
        const auto groups = bytes / sizeof(uint4);
        for (std::size_t i = index; i < groups; i += stride)
            reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
        tail = groups * sizeof(uint4);
    }
    for (std::size_t i = tail + index; i < bytes; i += stride) dst[i] = src[i];
    __threadfence_system();
}
}

void enqueue_host_d2h_copy(void* dst, const void* src, std::size_t bytes, hipStream_t stream) {
    if (!bytes) return;
    const unsigned blocks = static_cast<unsigned>(std::clamp<std::size_t>((bytes + 4095U) / 4096U, 1U, 32U));
    const bool vector = ((reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst)) & 15U) == 0U;
    if (vector)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(host_copy_kernel<true>), dim3(blocks), dim3(256), 0, stream,
                           static_cast<unsigned char*>(dst), static_cast<const unsigned char*>(src), bytes);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(host_copy_kernel<false>), dim3(blocks), dim3(256), 0, stream,
                           static_cast<unsigned char*>(dst), static_cast<const unsigned char*>(src), bytes);
    const auto status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string("host D2H copy kernel: ") + hipGetErrorString(status));
}
}
