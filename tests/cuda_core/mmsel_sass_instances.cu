// Compile as an sm_120a cubin against both the base and the candidate headers.
#include "cuda/core/route_project.cuh"
#include "cuda/core/route_softmax.cuh"
#include "moe_core/route_wave_kernel.hpp"

namespace strix::cuda::core::detail {

template __global__ void route_softmax_kernel<moe::RouteMath::gfx1151_v1>(
    const float*, unsigned, bool, std::uint32_t*, float*, moe::RouteStatus*, numeric::gfx1151::Tables);

template __global__ void route_sigmoid_projection_kernel<6144U, 128U, RouteStridedFmaXor32, moe::RouteCodyWaiteExpV1, 1U>(
    const float*, const float*, const float*, std::size_t, moe::RouteScorePlanes<1>);
template __global__ void route_sigmoid_projection_kernel<6144U, 128U, RouteStridedFmaXor32, moe::RouteCodyWaiteExpV1, 2U>(
    const float*, const float*, const float*, std::size_t, moe::RouteScorePlanes<2>);

}  // namespace strix::cuda::core::detail

namespace strix::moe_core {

struct MmselWavePolicy {
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::ordered_bits,
        moe::RouteOrder::selection, 32U, 8U};
};
struct MmselWaveMath {
    __device__ static float exp(float x) { return expf(x); }
};
using MmselWaveTraits = moe::RouteWaveTraits<true, true>;

template __global__ void route_wave_policy_entry<MmselWavePolicy, MmselWaveMath, MmselWaveTraits>(
    const float*, const float*, std::uint32_t*, float*, unsigned*, unsigned);
template __global__ void route_wave_step_entry<17U, MmselWavePolicy, MmselWaveMath, MmselWaveTraits>(
    const float*, const float*, std::uint32_t*, float*, unsigned*, const exec::StepBlock*);

}  // namespace strix::moe_core
