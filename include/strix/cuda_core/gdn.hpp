#pragma once
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// Gated DeltaNet decode (one layer). Value heads map to key heads TILED: v head t uses key head t % key_heads.
struct GdnDims {
    std::uint32_t key_heads = 16, key_dim = 128, value_heads = 48, value_dim = 128, conv_kernel = 4;  // conv 4 only
    [[nodiscard]] __host__ __device__ std::uint32_t qk_width() const noexcept { return key_heads * key_dim; }
    [[nodiscard]] __host__ __device__ std::uint32_t v_width() const noexcept { return value_heads * value_dim; }
    [[nodiscard]] __host__ __device__ std::uint32_t conv_channels() const noexcept { return 2 * qk_width() + v_width(); }
};

struct GdnLayer {
    const float* conv_w = nullptr;     // [channels][4], tap 3 multiplies the newest input
    const float* ssm_a = nullptr;      // [value_heads] stored = -exp(A_log)
    const float* dt_bias = nullptr;    // [value_heads]
    const float* norm_gain = nullptr;  // [value_dim] plain gain (not 1+w)
};

struct GdnState {
    float* conv = nullptr;   // [channels][3] last three conv inputs, oldest first
    float* delta = nullptr;  // [value_heads][key_dim][value_dim]  S[h][k][v]
};

// qkv: FP32 [rows][channels] = [q | k | v], OVERWRITTEN (conv+silu, l2norm of q (scaled 1/sqrt(key_dim)) and k, in place).
// z: [rows][value_width]; a_raw/b_raw: [rows][value_heads]; out: [rows][value_width]. key_dim and value_dim must be 128.
// Rows are processed sequentially in-kernel (row r sees the state after row r-1); the state is updated in place.
// beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias) (softplus: x > 20 -> x, else log1p(exp(x))); S *= exp(g);
// d = (v - S^T k) * beta; S += k (x) d; o = S^T q (k ascending, per v column sequential); out = gated_rmsnorm(o, z, swish).
cudaError_t gdn_decode(const GdnDims& d, const GdnLayer& layer, const GdnState& state, float* qkv, const float* z,
                       const float* a_raw, const float* b_raw, float* out, std::uint32_t rows, float eps,
                       cudaStream_t stream);

}  // namespace strix::cuda_core
