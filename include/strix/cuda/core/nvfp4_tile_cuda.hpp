#pragma once
// CUDA enqueue entry points of the NVFP4 tile GEMM (contract and layouts: strix/cuda/core/nvfp4_tile.hpp). sm_120 only: every call
// returns cudaErrorNotSupported on another device. No allocation; the caller owns every buffer.
#include <cuda_runtime.h>

#include "strix/cuda/core/nvfp4_tile.hpp"

namespace strix::cuda::core::nvfp4::tile {
[[nodiscard]] bool supported();
// W tile of an out x in weight from the NVFP4 decode record layout (strix/cuda_core/nvfp4.hpp); w_tile_bytes(out, in) bytes.
cudaError_t repack_enqueue(const std::uint8_t* records, std::uint32_t out, std::uint32_t in, std::uint8_t* tile, cudaStream_t stream);
// Quantises rows x in FP32 (row stride `in`) into the X tile layout + rs[rows_pad]; rows rows..rows_pad-1 become zero.
cudaError_t quantize_enqueue(const float* x, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile, float* rs, cudaStream_t stream);
// The same quantiser over BF16 rows (bit patterns, 16-byte aligned): bit for bit quantize_enqueue of their exact FP32 widening.
cudaError_t quantize_enqueue(const std::uint16_t* x_bf16, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile, float* rs,
                             cudaStream_t stream);
// rmsnorm(x, gain, eps) of rows x in FP32 rows (stride `in`) quantised as quantize_enqueue would quantise the f32 rmsnorm output:
// bit-identical X tile + rs, with no normalised row written to memory. Requires norm_quantize_fits(in); gain may be null (no gain).
cudaError_t norm_quantize_enqueue(const float* x, const float* gain, float eps, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile,
                                  float* rs, cudaStream_t stream);
// The GDN value-head gated RMSNorm (cuda_core gdn_chunk_norm: per kGatedNormHead-wide head h of row r, y = x * inv_h * gain[pos] *
// silu(z)) quantised as quantize_enqueue would quantise its f32 output: bit-identical X tile + rs. x and z: rows x in (stride in), gain:
// kGatedNormHead values. Requires norm_quantize_fits(in) (in % kK == 0 implies whole heads).
cudaError_t gated_norm_quantize_enqueue(const float* x, const float* z, const float* gain, float eps, std::uint32_t rows,
                                        std::uint32_t in, std::uint8_t* x_tile, float* rs, cudaStream_t stream);
cudaError_t gemm_enqueue(const GemmArgs& args, cudaStream_t stream);
}  // namespace strix::cuda::core::nvfp4::tile
