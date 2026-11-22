#pragma once

#include "attn_core/prepared_step.hpp"
#include "core/short_convolution.hpp"
#include "cuda/core/step_row_store.cuh"
#include "cuda_program_recipe.hpp"
#include "moe_core/microscaled_r1.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#include "strix/device/runtime.hpp"

#include <cstring>
#endif

namespace strix::exec::cuda_decode {

struct StepCheckedWaveRoute {
    const StepBlock* step;
    const float* logits;
    const float* bias;
    std::uint32_t* ids;
    float* weights;
    unsigned* failure;
};

template <bool GateUp>
struct StepPackedProjection {
    moe_core::microscaled_r1::StepArgs args;
    weight::microscaled::MatrixFormat format;
};

using StepPackedGateUp = StepPackedProjection<true>;
using StepPackedDownCombine = StepPackedProjection<false>;

[[nodiscard]] constexpr bool supports_wr2_width(unsigned rows) noexcept {
    return rows <= 16U && session::is_step_replay_width(rows);
}

// WR2 uses typed recipe binding. D4 selection cannot supply the device
// descriptor, stable state directory and policy-specific F3 thunk here.
struct BoundStepLeaf {
    KernelBinding binding;
    unsigned rows;

    void append_to(CudaProgramRecipe& recipe) && {
        recipe.kernel(std::move(binding), false, rows);
    }
};

[[nodiscard]] BoundStepLeaf bind_step(const StepPackedGateUp&, unsigned);
[[nodiscard]] BoundStepLeaf bind_step(const StepPackedDownCombine&, unsigned);

#if defined(__CUDACC__) && !defined(__HIPCC__)
}  // namespace strix::exec::cuda_decode

namespace strix::cuda_core {
template <unsigned Rows> cudaError_t prepared_step_kv(const attn_core::PreparedStepKv&, cudaStream_t);
template <unsigned Rows> cudaError_t prepared_step_flash(const attn_core::PreparedStepFlash&, cudaStream_t);
}  // namespace strix::cuda_core

namespace strix::exec::cuda_decode {
namespace wr2_detail {

template <class Select>
BoundStepLeaf by_width(unsigned rows, const Select& select) {
    switch (rows) {
    case 1U: return select.template operator()<1>();
    case 2U: return select.template operator()<2>();
    case 4U: return select.template operator()<4>();
    case 8U: return select.template operator()<8>();
    case 16U: return select.template operator()<16>();
    default: throw std::invalid_argument("unsupported WR2 width R=" + std::to_string(rows));
    }
}

template <class Args, auto Enqueue>
void launch(std::span<const std::byte> bytes, const Grid&, void* stream) {
    static_assert(std::is_trivially_copyable_v<Args>);
    if (bytes.size() != sizeof(Args)) throw std::invalid_argument("invalid WR2 recipe arguments");
    Args args{};
    std::memcpy(&args, bytes.data(), sizeof(args));
    device::native::check(Enqueue(args, static_cast<cudaStream_t>(stream)), "enqueue WR2 CUDA leaf");
}

template <unsigned Rows, auto Enqueue, class Args>
BoundStepLeaf bind(const Args& args, std::string_view family, Grid grid) {
    KernelBinding binding;
    binding.fn = launch<Args, Enqueue>;
    binding.identity = std::string(family) + "-x3-r" + std::to_string(Rows) + "-v1";
    binding.grid = grid;
    binding.predicted.launches = 1U;
    const auto bytes = std::as_bytes(std::span(&args, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    return {std::move(binding), Rows};
}

template <unsigned Rows, class Policy, class Math, class Traits>
cudaError_t route(const StepCheckedWaveRoute& a, cudaStream_t stream) {
    return moe_core::enqueue_route_wave_step<Rows, Policy, Math, Traits>(
        a.logits, a.bias, a.ids, a.weights, a.failure, a.step, stream);
}

}  // namespace wr2_detail

inline BoundStepLeaf bind_step(const attn_core::PreparedStepKv& a, unsigned rows) {
    return wr2_detail::by_width(rows, [&]<unsigned R>() {
        return wr2_detail::bind<R, cuda_core::prepared_step_kv<R>>(a, "cuda-sm120-kv-append-bf16",
            {{a.kv_heads, R, 1}, {a.width, 1, 1}, 0});
    });
}

inline BoundStepLeaf bind_step(const attn_core::PreparedStepFlash& a, unsigned rows) {
    constexpr unsigned shared = 2U * 32U * (4U * 128U + 32U + 16U);
    return wr2_detail::by_width(rows, [&]<unsigned R>() {
        return wr2_detail::bind<R, cuda_core::prepared_step_flash<R>>(a, "cuda-sm120-prepared-attention-bf16-pad2",
            {{a.heads, R, 1}, {256, 1, 1}, shared});
    });
}

inline BoundStepLeaf bind_step(const attn_core::PreparedStepFinite& a, unsigned rows) {
    if (!supports_wr2_width(rows) || a.rows.count != rows) throw std::invalid_argument("invalid WR2 finite width");
    const unsigned stride = std::max(a.width, a.relative_extent);
    const auto heads = std::size_t{rows} * a.heads;
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (stride == 0U || heads == 0U || heads > limit / stride) throw std::invalid_argument("invalid WR2 finite shape");
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((heads * stride - 1U) / 256U + 1U, 65535U));
    return wr2_detail::by_width(rows, [&]<unsigned R>() {
        return wr2_detail::bind<R, attn_core::enqueue_prepared_step_finite<R>>(a, "cuda-sm120-prepared-finite",
            {{blocks, 1, 1}, {256, 1, 1}, 0});
    });
}

inline BoundStepLeaf bind_step(const core::causal_conv::StepArgs& a, unsigned rows) {
    using namespace core::causal_conv;
    if (!core::valid_short_convolution_plan(a.plan)) throw std::invalid_argument("invalid WR2 convolution shape");
    const auto grid = launch_plan(a.plan, {rows, a.input, a.static_weights, a.dynamic_weights, a.pending, a.output});
    const Grid native{{grid.channel_blocks, 1, 1}, {grid.threads, 1, 1}, 0};
    return wr2_detail::by_width(rows, [&]<unsigned R>() {
        if (a.round_bf16) return wr2_detail::bind<R, enqueue_step<R, true>>(a, "cuda-sm120-causal-convolution-bf16", native);
        return wr2_detail::bind<R, enqueue_step<R, false>>(a, "cuda-sm120-causal-convolution-f32", native);
    });
}

// Position-indexed BF16 row store (drafter feature taps): one width-bound program per width, the positions
// come from the step descriptor, so the R = 1 decode graph and every verify graph reuse their capture.
inline BoundStepLeaf bind_step(const cuda::core::StepRowStoreBf16& a, unsigned rows) {
    if (!cuda::core::valid_step_row_store(a)) throw std::invalid_argument("invalid step row store");
    const unsigned blocks = cuda::core::step_row_store_blocks(a.width);
    return wr2_detail::by_width(rows, [&]<unsigned R>() {
        return wr2_detail::bind<R, cuda::core::enqueue_step_row_store_bf16<R>>(a, "cuda-sm120-step-row-store-bf16",
            {{blocks, R, 1}, {256, 1, 1}, 0});
    });
}

template <class Policy, class Math, class Traits>
BoundStepLeaf bind_step_route(const StepCheckedWaveRoute& a, unsigned rows, std::string_view policy_identity) {
    if (policy_identity.empty()) throw std::invalid_argument("missing WR2 route policy identity");
    constexpr unsigned waves = moe_core::kCheckedRouteRowsPerBlock;
    return wr2_detail::by_width(rows, [&]<unsigned R>() {
        return wr2_detail::bind<R, wr2_detail::route<R, Policy, Math, Traits>>(a, policy_identity,
            {{(R - 1U) / waves + 1U, 1, 1}, {waves * 32U, 1, 1}, 0});
    });
}

#endif

}  // namespace strix::exec::cuda_decode
