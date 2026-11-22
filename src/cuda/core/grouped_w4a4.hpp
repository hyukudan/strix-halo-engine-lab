#pragma once
// CUDA launchers of the grouped W4A4 expert projection (contract: moe_core/grouped_w4a4.hpp; body: arch/sm120/grouped_w4a4.cuh,
// compiled for sm_120a in strix_cuda_core_fp4). A device without the block-scaled FP4 MMA gets cudaErrorInvalidDeviceFunction.
#include <cuda_runtime.h>

#include <cstdint>

#include "moe_core/grouped_w4a4.hpp"

namespace strix::cuda::core::grouped_w4a4 {

// cudaErrorInvalidValue when the arguments are not admitted (moe_core::grouped_w4a4::fits / fits_image).
cudaError_t quantize(const moe_core::grouped_w4a4::QuantizeArgs<float>& args, cudaStream_t stream);
cudaError_t quantize(const moe_core::grouped_w4a4::QuantizeArgs<std::uint16_t>& args, cudaStream_t stream);
// NVFP4 weights only; the caller admits each tile with moe_core::grouped_w4a4::fits_tile.
cudaError_t launch(const moe_core::grouped_w4a4::Args& args, cudaStream_t stream);

}  // namespace strix::cuda::core::grouped_w4a4
