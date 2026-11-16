#pragma once
// CUDA enqueue side of the nvfp4-a8-v1 converter (contract and host oracles: strix/cuda/core/nvfp4_a8.hpp).
#include <cuda_runtime_api.h>

#include "strix/cuda/core/nvfp4_a8.hpp"

namespace strix::cuda::core::nvfp4_a8 {
// max |fp16(d) * q| over a GGUF Q8_0 [n][k] tensor into *amax (device float, caller zeroes it first).
cudaError_t amax_q8_0_enqueue(const void* q8, std::uint32_t n, std::uint32_t k, float* amax, cudaStream_t stream);
// Q8_0 [n][k] -> image codes/scales (device, preallocated: image.code_bytes() / scale_bytes()) at tensor_scale ts.
// Bit-equal to reference_convert_q8_0 for the same ts.
cudaError_t quantize_q8_0_enqueue(const void* q8, std::uint32_t n, std::uint32_t k, float ts, std::uint8_t* codes,
                                  std::uint8_t* scales, cudaStream_t stream);
// Both steps: amax (one stream sync to read it), tensor scale rule, quantize. image.codes/scales preallocated,
// image.n/k/tensor_scale are written. scratch: one device float.
cudaError_t convert_q8_0(const void* q8, std::uint32_t n, std::uint32_t k, Image& image, float* scratch,
                         cudaStream_t stream);
}  // namespace strix::cuda::core::nvfp4_a8
