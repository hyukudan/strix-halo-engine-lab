// CUDA body of the shared embed-rows splice (include/strix/session/embed_splice_device.hpp). Same contract and
// arithmetic as the HIP body (src/session/embed_splice.hip): an exact 16-bit -> FP32 widening per element.
#include "strix/session/embed_splice_device.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace strix::session {
namespace {

// One block per destination row; 256 threads stride the width.
__global__ __launch_bounds__(256) void embed_splice_kernel(float* __restrict__ dst, unsigned copies,
                                                           const std::uint16_t* __restrict__ src, unsigned width,
                                                           unsigned f16) {
    const std::uint16_t* s = src + static_cast<std::size_t>(blockIdx.x / copies) * width;
    float* d = dst + static_cast<std::size_t>(blockIdx.x) * width;
    if (f16 != 0U) {
        for (unsigned c = threadIdx.x; c < width; c += 256U) d[c] = __half2float(__ushort_as_half(s[c]));
    } else {
        for (unsigned c = threadIdx.x; c < width; c += 256U)
            d[c] = __uint_as_float(static_cast<unsigned>(s[c]) << 16U);
    }
}

} // namespace

void enqueue_embed_splice(float* dst, std::uint32_t copies, const std::uint16_t* src, std::uint32_t width,
                          EmbedRowFormat format, std::uint32_t dst_row, std::uint32_t rows, void* stream) {
    if (rows == 0U) return;
    if (dst == nullptr || src == nullptr || copies == 0U || width == 0U)
        throw std::invalid_argument("embed splice: launch arguments");
    embed_splice_kernel<<<rows * copies, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        dst + static_cast<std::size_t>(dst_row) * copies * width, copies, src, width,
        format == EmbedRowFormat::f16 ? 1U : 0U);
    if (const auto e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("embed splice launch: ") + cudaGetErrorString(e));
}

} // namespace strix::session
