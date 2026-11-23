#pragma once
// Shared CUDA core: FP32 row-major kernels of the vision tower (strix::vision::CudaEncoder binds them through
// src/vision/encoder_plan.hpp). Launch glue only; the step order and buffer sizes live in the plan.
// Arithmetic: FP32 operands and FP32 accumulation in a fixed k order (no atomics, deterministic), explicit fmaf.
#include <cuda_runtime.h>

#include <cstddef>

namespace strix::cuda_core::vision {

enum class Norm : unsigned { layernorm = 0, rmsnorm = 1 };
enum class Act : unsigned { gelu = 0, gelu_tanh = 1, swiglu = 2 };

// y[m][ldy] (+)= x[m][k] * w[n][k]^T + bias[n]; `accumulate` adds into y (the residual form). bias may be null.
cudaError_t gemm_nt(const float* x, const float* w, const float* bias, float* y, unsigned m, unsigned n, unsigned k, unsigned ldy,
                    bool accumulate, cudaStream_t s);
// y = norm(x) rows of `width` floats (x == y allowed). w required; b null for RMSNorm.
cudaError_t norm_rows(const float* x, float* y, unsigned rows, unsigned width, const float* w, const float* b, float eps, Norm kind,
                      cudaStream_t s);
cudaError_t add_rows(float* x, const float* add, std::size_t count, cudaStream_t s);
// Rotary rotation of q and k inside qkv [rows][3 * heads * d]; cs / sn are [rows][d / 2]. adjacent: pairs (2i, 2i + 1).
cudaError_t rope_qkv(float* qkv, const float* cs, const float* sn, unsigned rows, unsigned heads, unsigned d, bool adjacent, cudaStream_t s);
// Bidirectional attention of one tile: rows [row0, row0 + n) of qkv [.][3 * heads * d] -> ao [.][heads * d]. d <= 128.
cudaError_t attention_tile(const float* qkv, float* ao, unsigned row0, unsigned n, unsigned heads, unsigned d, cudaStream_t s);
// act[r][j] from mid[r][m_out]: gelu / gelu_tanh act on mid[r][j]; swiglu reads gate and up halves (limit 0 = no clamp).
cudaError_t activate_rows(const float* mid, float* act, unsigned rows, unsigned width, Act kind, float swiglu_limit, bool gate_first,
                          cudaStream_t s);

}  // namespace strix::cuda_core::vision
