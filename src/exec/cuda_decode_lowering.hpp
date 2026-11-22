#pragma once

#include "catalog_cuda_decode.hpp"
#include "cuda_program_recipe.hpp"
#include "strix/core/rows.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#include "strix/device/runtime.hpp"
#include <cstring>
#endif

namespace strix::exec::cuda_decode {

struct CheckedWaveRoute {
    const float* logits;
    const float* bias;
    std::uint32_t* ids;
    float* weights;
    unsigned* failure;
};

#if defined(__CUDACC__) && !defined(__HIPCC__)
template <class Policy, class Math, class Traits>
[[nodiscard]] KernelBinding bind_checked_wave(const CheckedWaveRoute& args, std::string_view identity) {
    KernelBinding binding;
    binding.fn = +[](std::span<const std::byte> bytes, const Grid&, void* stream) {
        if (bytes.size() != sizeof(CheckedWaveRoute)) throw std::invalid_argument("invalid checked wave route binding");
        CheckedWaveRoute route{};
        std::memcpy(&route, bytes.data(), sizeof(route));
        device::native::check(moe_core::enqueue_route_wave<Policy, Math, Traits>(
            route.logits, route.bias, route.ids, route.weights, route.failure, 1U,
            static_cast<cudaStream_t>(stream)), "enqueue checked CUDA wave route");
    };
    binding.identity = binding.exact_identity = identity;
    binding.grid = {{1U, 1U, 1U}, {moe_core::kCheckedRouteRowsPerBlock * 32U, 1U, 1U}, 0U};
    const auto bytes = std::as_bytes(std::span(&args, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    binding.predicted.launches = 1U;
    return binding;
}
#endif

// Lower an incumbent homogeneous BF16 projection call. A mixed group issues
// its FP32 stripes first, in declaration order, then the MMA3 stripes.
template <class Add>
void lower_dense(const float* input, unsigned width, std::span<const core::Bf16RowProjection> projections,
                 bool fp32, const Add& add) {
    Bf16DenseGroup scalar{input, width, {}, 0, Bf16DenseMode::fp32};
    Bf16DenseGroup mma{input, width, {}, 0, Bf16DenseMode::mma3};
    if (projections.empty() || projections.size() > scalar.projections.size())
        throw std::invalid_argument("invalid CUDA decode projection count");
    for (const auto& projection : projections) {
        const bool tiled = !fp32 && width % core::rows::kBlockIn == 0 &&
                           projection.outputs % core::rows::kTileOut == 0;
        auto& group = tiled ? mma : scalar;
        group.projections[group.count++] = projection;
    }
    if (scalar.count) add(scalar, true);
    if (mma.count) {
        if (mma.projections[0].round_bf16) mma.mode = Bf16DenseMode::mma3_round;
        add(mma, true);
    }
}

template <class Add>
void lower_low_rank_controller(unsigned width, unsigned rank, float epsilon,
                               std::span<const float> query, std::span<const float> raw,
                               std::span<const std::span<const float>> bank, std::span<float> output,
                               std::span<float> scratch, const std::uint16_t* down,
                               const std::uint16_t* forget_weights, const std::uint16_t* gain_weights,
                               const std::uint16_t* norm, const Add& add) {
    const auto stride = 2U * width;
    auto* const message = scratch.data();
    auto* const forget = message + width;
    auto* const gain = forget + width;
    auto* const conditioned = gain + width;
    auto* const latent = conditioned + width;
    core::residual_mix::MixArgs mix{};
    mix.queries = query.data();
    mix.output[0] = message;
    mix.rows = mix.query_count = 1U;
    mix.width = width;
    mix.bank_count = static_cast<unsigned>(bank.size());
    mix.epsilon = epsilon;
    mix.normalized = true;
    for (std::size_t i = 0; i < bank.size(); ++i) mix.banks[i] = bank[i].data();
    add(mix, false);
    add(core::residual_mix::NormalizePairArgs{
        {raw.data(), forget, 1U, width, epsilon, width, stride, true},
        {message, gain, 1U, width, epsilon, width, stride, true}}, false);
    const std::array down_projection{core::Bf16RowProjection{down, rank, latent, true}};
    lower_dense(forget, stride, down_projection, false, add);
    const std::array gates{core::Bf16RowProjection{forget_weights, width, forget},
                           core::Bf16RowProjection{gain_weights, width, gain}};
    lower_dense(latent, rank, gates, false, add);
    add(core::residual_mix::ControllerArgs{raw.data(), message, forget, gain, conditioned, width}, false);
    add(RmsBf16{conditioned, norm, output.data(), 1U, width, epsilon}, true);
}

}  // namespace strix::exec::cuda_decode
