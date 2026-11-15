// RMSNorm (stored gain), gated RMSNorm, per-group l2norm, residual add. Reduction order documented in norms.hpp.
#include "strix/cuda_core/norms.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

enum class Mode : int { rms = 0, gated_swish = 1, gated_sigmoid = 2, l2 = 3 };

template <Mode M>
__global__ void __launch_bounds__(256)
    norm_kernel(const float* x, GroupedView xv, const float* gain, const float* z, GroupedView zv, float* y, GroupedView yv,
                float eps, float scale) {
    __shared__ float red[8];
    const std::uint32_t r = blockIdx.y, g = blockIdx.x, n = xv.n;
    const float* xp = x + static_cast<std::size_t>(r) * xv.rs() + static_cast<std::size_t>(g) * xv.gs();
    float* yp = y + static_cast<std::size_t>(r) * yv.rs() + static_cast<std::size_t>(g) * yv.gs();
    float p = 0.0f;
    for (std::uint32_t i = threadIdx.x; i < n; i += 256) {
        const float v = xp[i];
        p = __fadd_rn(p, __fmul_rn(v, v));
    }
    const float ss = block_sum256(p, red);
    float inv;
    if constexpr (M == Mode::l2) {
        inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    } else {
        inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
    }
    const float* zp = nullptr;
    if constexpr (M == Mode::gated_swish || M == Mode::gated_sigmoid)
        zp = z + static_cast<std::size_t>(r) * zv.rs() + static_cast<std::size_t>(g) * zv.gs();
    for (std::uint32_t i = threadIdx.x; i < n; i += 256) {
        float v = __fmul_rn(xp[i], inv);
        if constexpr (M == Mode::l2) {
            v = __fmul_rn(v, scale);
        } else {
            if (gain != nullptr) v = __fmul_rn(v, gain[i]);
            if constexpr (M == Mode::gated_swish) v = __fmul_rn(v, cc_silu(zp[i]));
            if constexpr (M == Mode::gated_sigmoid) v = __fmul_rn(v, cc_sigmoid(zp[i]));
        }
        yp[i] = v;
    }
}

bool same_shape(const GroupedView& a, const GroupedView& b) { return a.rows == b.rows && a.groups == b.groups && a.n == b.n; }

template <Mode M>
cudaError_t run(const float* x, const GroupedView& xv, const float* gain, const float* z, const GroupedView& zv, float* y,
                const GroupedView& yv, float eps, float scale, cudaStream_t s) {
    if (!same_shape(xv, yv) || xv.n == 0 || xv.rows == 0 || xv.groups == 0) return cudaErrorInvalidValue;
    if (xv.groups > 65535u * 1u && false) return cudaErrorInvalidValue;
    norm_kernel<M><<<dim3(xv.groups, xv.rows), 256, 0, s>>>(x, xv, gain, z, zv, y, yv, eps, scale);
    return cudaGetLastError();
}

__global__ void add_kernel(float* x, const float* d, std::uint32_t count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __fadd_rn(x[i], d[i]);
}

}  // namespace

cudaError_t rmsnorm(const float* x, const GroupedView& xv, const float* gain, float* y, const GroupedView& yv, float eps,
                    cudaStream_t stream) {
    return run<Mode::rms>(x, xv, gain, nullptr, GroupedView{}, y, yv, eps, 1.0f, stream);
}

cudaError_t gated_rmsnorm(const float* x, const GroupedView& xv, const float* gain, const float* z, const GroupedView& zv,
                          GateKind kind, float* y, const GroupedView& yv, float eps, cudaStream_t stream) {
    if (!same_shape(xv, zv)) return cudaErrorInvalidValue;
    return kind == GateKind::swish ? run<Mode::gated_swish>(x, xv, gain, z, zv, y, yv, eps, 1.0f, stream)
                                   : run<Mode::gated_sigmoid>(x, xv, gain, z, zv, y, yv, eps, 1.0f, stream);
}

cudaError_t l2norm(const float* x, const GroupedView& xv, float* y, const GroupedView& yv, float eps, float scale,
                   cudaStream_t stream) {
    return run<Mode::l2>(x, xv, nullptr, nullptr, GroupedView{}, y, yv, eps, scale, stream);
}

cudaError_t add_inplace(float* x, const float* d, std::uint32_t count, cudaStream_t stream) {
    add_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, d, count);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
