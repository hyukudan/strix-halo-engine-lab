// RMSNorm (stored gain), gated RMSNorm, per-group l2norm, residual add. Reduction order documented in norms.hpp.
#include "strix/cuda_core/norms.hpp"
#include "../../../core/pointwise.hpp"

#include "../common.cuh"
#include "../step_rows.cuh"
#include "row_math.cuh"

#include <type_traits>

namespace strix::cuda_core {
namespace {

enum class Mode : int { rms = 0, gated_swish = 1, gated_sigmoid = 2, l2 = 3, rms_bf16 = 4, rms_order_null = 5 };

template <Mode M>
using Gain = std::conditional_t<M == Mode::rms_bf16, std::uint16_t, float>;

// Same order as before (thread t adds x[t + 256 i] ascending); the NREG values of a thread stay in registers between the sum and the
// write (one global read per element instead of two). Padding terms are 0 (p + 0*0 == p for p >= 0).
template <Mode M, int NREG>
__global__ void __launch_bounds__(256)
    norm_kernel(const After<const float> x, GroupedView xv, const Gain<M>* gain, const After<const float> z, GroupedView zv,
                const After<float> y, GroupedView yv, float eps, float scale) {
#include "norm_body.inc"
}

template <unsigned Width, int NREG>
__global__ void __launch_bounds__(256)
    rms_bf16_step_kernel(const After<const float> x, GroupedView xv, const std::uint16_t* gain,
                         const After<float> y, float eps, cuda::core::StepRows<Width> binding) {
    if (!binding.valid(blockIdx.y)) return;
    constexpr auto M = Mode::rms_bf16;
    const GroupedView yv = xv;
    [[maybe_unused]] const GroupedView zv{};
    [[maybe_unused]] const After<const float> z{};
    [[maybe_unused]] constexpr float scale = 1.0F;
#include "norm_body.inc"
}

bool same_shape(const GroupedView& a, const GroupedView& b) { return a.rows == b.rows && a.groups == b.groups && a.n == b.n; }

template <Mode M>
cudaError_t run(const float* x, const GroupedView& xv, const Gain<M>* gain, const float* z, const GroupedView& zv, float* y,
                const GroupedView& yv, float eps, float scale, cudaStream_t s) {
    if (!same_shape(xv, yv) || xv.n == 0 || xv.rows == 0 || xv.groups == 0) return cudaErrorInvalidValue;
    if (xv.groups > 65535u * 1u && false) return cudaErrorInvalidValue;
    const dim3 grid(xv.groups, xv.rows);
    const std::uint32_t nreg = (xv.n + 255) / 256;
    if (nreg <= 1) launch_pdl(norm_kernel<M, 1>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 2) launch_pdl(norm_kernel<M, 2>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 4) launch_pdl(norm_kernel<M, 4>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 8) launch_pdl(norm_kernel<M, 8>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 16) launch_pdl(norm_kernel<M, 16>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 24) launch_pdl(norm_kernel<M, 24>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 32) launch_pdl(norm_kernel<M, 32>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else return cudaErrorInvalidValue;  // n <= 8192
    return cudaGetLastError();
}

__global__ void add_kernel(float* x, const float* d, std::uint32_t count) {
    core::pointwise::visit_elements<false>(count, core::pointwise::Add{x, d, x});
}

}  // namespace

cudaError_t rmsnorm(const float* x, const GroupedView& xv, const float* gain, float* y, const GroupedView& yv, float eps,
                    cudaStream_t stream) {
    return run<Mode::rms>(x, xv, gain, nullptr, GroupedView{}, y, yv, eps, 1.0f, stream);
}

cudaError_t rmsnorm_order_null(const float* x, const GroupedView& xv, const float* gain, float* y,
                              const GroupedView& yv, float eps, cudaStream_t stream) {
    return run<Mode::rms_order_null>(x, xv, gain, nullptr, GroupedView{}, y, yv, eps, 1.0f, stream);
}

cudaError_t rmsnorm_bf16(const float* x, const GroupedView& xv, const std::uint16_t* gain, float* y,
                         const GroupedView& yv, float eps, cudaStream_t stream) {
    if (!x || !y || xv.rows > 65535U || xv.n > 8192U || !(eps > 0.0f)) return cudaErrorInvalidValue;
    return run<Mode::rms_bf16>(x, xv, gain, nullptr, GroupedView{}, y, yv, eps, 1.0f, stream);
}

cudaError_t rmsnorm_bf16_step(const float* input, const std::uint16_t* gain, float* output,
    unsigned width, float epsilon, const exec::StepBlock* descriptor, dim3 grid, cudaStream_t stream) {
    const GroupedView view{grid.y, grid.x, width};
    return cuda::core::dispatch_step_rows(grid.y, [&]<unsigned Width> {
        const auto launch = [&]<int NREG> {
            rms_bf16_step_kernel<Width, NREG><<<grid, 256, 0, stream>>>(
                input, view, gain, output, epsilon, cuda::core::StepRows<Width>{descriptor});
        };
        const auto nreg = (width + 255U) / 256U;
        if (nreg <= 1) launch.template operator()<1>();
        else if (nreg <= 2) launch.template operator()<2>();
        else if (nreg <= 4) launch.template operator()<4>();
        else if (nreg <= 8) launch.template operator()<8>();
        else if (nreg <= 16) launch.template operator()<16>();
        else if (nreg <= 24) launch.template operator()<24>();
        else launch.template operator()<32>();
        return cudaGetLastError();
    });
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
