#include "grouped_w4a4.hpp"
#include "strix/cuda/core/nvfp4_tile_cuda.hpp"

namespace strix::cuda::core::nvfp4::tile {

bool supported() { return false; }

cudaError_t repack_enqueue(const std::uint8_t*, std::uint32_t, std::uint32_t, std::uint8_t*, cudaStream_t) {
    return cudaErrorNotSupported;
}

cudaError_t quantize_enqueue(const float*, std::uint32_t, std::uint32_t, std::uint8_t*, float*, cudaStream_t) {
    return cudaErrorNotSupported;
}

cudaError_t quantize_enqueue(const std::uint16_t*, std::uint32_t, std::uint32_t, std::uint8_t*, float*, cudaStream_t) {
    return cudaErrorNotSupported;
}

cudaError_t norm_quantize_enqueue(const float*, const float*, float, std::uint32_t, std::uint32_t, std::uint8_t*,
                                  float*, cudaStream_t) {
    return cudaErrorNotSupported;
}

cudaError_t gated_norm_quantize_enqueue(const float*, const float*, const float*, float, std::uint32_t, std::uint32_t,
                                        std::uint8_t*, float*, cudaStream_t) {
    return cudaErrorNotSupported;
}

cudaError_t gemm_enqueue(const GemmArgs&, cudaStream_t) { return cudaErrorNotSupported; }

}  // namespace strix::cuda::core::nvfp4::tile

namespace strix::cuda::core::grouped_w4a4 {
namespace contract = ::strix::moe_core::grouped_w4a4;

cudaError_t quantize(const contract::QuantizeArgs<float>&, cudaStream_t, contract::ScaleRule) {
    return cudaErrorNotSupported;
}

cudaError_t quantize(const contract::QuantizeArgs<std::uint16_t>&, cudaStream_t, contract::ScaleRule) {
    return cudaErrorNotSupported;
}

cudaError_t launch(const contract::Args&, cudaStream_t) { return cudaErrorNotSupported; }

}  // namespace strix::cuda::core::grouped_w4a4
