#pragma once
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// y[r][o] = sum_k W[o][k] * x[r][k] (+ residual[r][o]) for r < rows (1..16). x is FP32 [rows][in], y FP32 [rows][out].
// FIXED ORDER (identical for every row count and launch shape): one warp per output o. bf16/f32: lane l owns chunks
// c = l + 32 i of 8 consecutive elements (i ascending) and adds the rounded products in ascending element order into one
// FP32 accumulator; q8_0: lane l owns blocks b = l + 32 i; per block s = sequential sum_j (float(q_j) * x_j), then
// acc = acc + fl(d * s). The 32 lane accumulators are combined by the xor butterfly (offsets 16,8,4,2,1).
// The optional residual is added after the dot is complete, so y = residual + W x is bitwise (W x) then add. y may alias
// residual. Requires in % 8 == 0 (bf16/f32) or in % 32 == 0 (q8_0).
cudaError_t dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                       cudaStream_t stream);

// y[r][o] = silu(gate_dot) * up_dot with silu(g) = g / (1 + expf(-g)); bitwise equal to two dense_rows calls followed by the
// same epilogue. gate and up must share format/in/out.
cudaError_t dense_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t stream);

// out[r][i] = dequant(row ids[r]) of a [vocab][in] table (embedding gather), FP32.
cudaError_t embedding_rows(const DenseWeight& table, const std::uint32_t* ids, float* out, std::uint32_t rows,
                           cudaStream_t stream);

// out[i] = argmax of logits[i] ... lowest index wins ties. One block; v0 helper for greedy.
cudaError_t argmax_rows(const float* logits, std::uint32_t vocab, std::uint32_t rows, std::uint32_t* out,
                        cudaStream_t stream);

}  // namespace strix::cuda_core
