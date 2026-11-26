// NeoX rotate-half RoPE over the first rotary_dim dims. See rope.hpp.
#include "strix/device/decode_geometry.hpp"
#include "strix/cuda_core/rope.hpp"

#include "../common.cuh"
#include "row_math.cuh"

namespace strix::cuda_core {
namespace {

__global__ void rope_kernel(float* x, GroupedView v, std::uint32_t half, const float* cos_t, const float* sin_t,
                            const std::uint32_t* pos0p, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t r = blockIdx.y, h = blockIdx.x;
    const std::uint32_t pos = rm.pos ? rm.pos[r] : *pos0p + r;
    float* p = x + static_cast<std::size_t>(r) * v.rs() + static_cast<std::size_t>(h) * v.gs();
    const std::size_t tb = static_cast<std::size_t>(pos) * half;
    for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = cos_t[tb + i], s = sin_t[tb + i];
        const float x1 = p[i], x2 = p[i + half];
        p[i] = cuda::core::row_math::rotate_half(x1, x2, c, s, false);
        p[i + half] = cuda::core::row_math::rotate_half(x2, x1, c, s, true);
    }
}

}  // namespace

cudaError_t rope_apply(float* x, const GroupedView& view, std::uint32_t rotary_dim, const float* cos_table,
                       const float* sin_table, const std::uint32_t* pos0, cudaStream_t stream, const RowMap& rm) {
    if (rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > view.n) return cudaErrorInvalidValue;
    const device::LaunchGeometry geometry = device::decode_geometry::rotary(view.groups, view.rows);
    return launch_pdl(rope_kernel, geometry.grid<dim3>(), geometry.block<dim3>(), stream, x, view, rotary_dim / 2, cos_table, sin_table, pos0, rm);
}

}  // namespace strix::cuda_core
