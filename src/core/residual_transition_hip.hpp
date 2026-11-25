#pragma once

// HIP body of the residual transition (residual_mix::TransitionArgs, contract and admission in core/residual_mix.hpp;
// the CUDA body is cuda/core/mix_cuda_transition.cuh). Its own header so that only a binder instantiates the kernel:
// the readout's mix_rows body, the RMS bf16 traits and the add expression come from the shared homes.
#if !defined(__HIPCC__)
#error "core/residual_transition_hip.hpp is the HIP body; include core/residual_mix.hpp for the contract"
#endif

#include "core/residual_mix.hpp"
#include "ops/norm/rms_bf16_rows.hpp"
#include "ops/norm/row_reduction.hpp"

namespace strix::core::residual_mix {
namespace detail {

// HIP body of TransitionArgs (admission: hip_transition_admits). One block in the paired-bank geometry of launch_mix
// for one row (kPairedBankThreads x bank_count threads, bank_count >= 2):
//   prologue  the AddArgs expression (__fadd_rn, then bf16_round when round_bf16) for every element, all operands read
//             (block barrier) before any store to the current bank, then a block barrier (the readout reads it back through the workgroup fence);
//   readout   mix_rows (one row), the incumbent paired-bank body (register form for width <= kMixRegisterWidth);
//   epilogue  per normed query, the register form of rmsnorm_strided_kernel<Bf16ScaleSqrtDivTraits, 256> (the
//             enqueue_rms_bf16_rows parallel leaf) at virtual lane y * kPairedBankThreads + x: the same lane chains,
//             weights, 256-lane LDS halving tree, factor and output expression.
inline constexpr unsigned kTransitionNormLanes = 256;
inline constexpr unsigned kTransitionNormRegs = kTransitionMaxWidth / kTransitionNormLanes;
inline constexpr unsigned kTransitionAddRegs = kTransitionMaxWidth / (2U * kPairedBankThreads);

template <bool Normalized>
static __global__ void __launch_bounds__(kPairedBankThreads * kMaxBanks) transition(TransitionArgs args) {
    using Traits = ops::norm::Bf16ScaleSqrtDivTraits;
    __shared__ float scratch[kPairedBankThreads * kMaxBanks];
    __shared__ float inverse[kMaxBanks];
    __shared__ float probability[kMaxQueries][kMaxBanks];
    static_assert(kTransitionNormLanes <= kPairedBankThreads * kMaxBanks);
    const unsigned threads = blockDim.x * blockDim.y;
    const unsigned lane = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned width = args.mix.width;
    if (args.add.lhs != nullptr) {
        // The contract lets the add operands alias any image, a shifted window of the current bank included, so every
        // operand is read before any store: the block's sums stay in registers across a barrier, then are stored.
        // threads >= 2 x kPairedBankThreads (admission: bank_count >= 2) covers width <= kTransitionMaxWidth.
        float sums[kTransitionAddRegs];
#pragma unroll
        for (unsigned k = 0; k < kTransitionAddRegs; ++k) {
            const unsigned i = lane + k * threads;
            sums[k] = 0.0F;
            if (i < width) {
                const float sum = __fadd_rn(args.add.lhs[i], args.add.rhs[i]);
                sums[k] = args.add.round_bf16 ? gi::bf16_round(sum) : sum;
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned k = 0; k < kTransitionAddRegs; ++k) {
            const unsigned i = lane + k * threads;
            if (i < width) args.add.output[i] = sums[k];
        }
        __syncthreads();
    }
    mix_rows<Normalized, true>(args.mix, 0, 1, scratch, inverse, probability);
    const bool norm_lane = lane < kTransitionNormLanes;
    for (unsigned q = 0; q < args.mix.query_count; ++q) {
        const TransitionNorm& norm = args.norm[q];
        if (norm.output == nullptr) continue;  // the same for every thread of the block
        const float* row = args.mix.output[q];
        float v[kTransitionNormRegs], weight[kTransitionNormRegs];
#pragma unroll
        for (unsigned k = 0; k < kTransitionNormRegs; ++k) {
            const unsigned i = lane + k * kTransitionNormLanes;
            v[k] = norm_lane && i < width ? row[i] : 0.0F;
        }
        float squares = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < kTransitionNormRegs; ++k) {
            const unsigned i = lane + k * kTransitionNormLanes;
            if (norm_lane && i < width)
                squares = ops::norm::square_step<ops::norm::square_form_of<Traits>()>(squares, v[k]);
        }
#pragma unroll
        for (unsigned k = 0; k < kTransitionNormRegs; ++k) {
            const unsigned i = lane + k * kTransitionNormLanes;
            weight[k] = 1.0F;
            if (norm.gain && norm_lane && i < width) weight[k] = Traits::load(norm.gain, 0, i);
        }
        const float sum = ops::norm::lds_halving_sum_lanes<kTransitionNormLanes>(scratch, squares, lane);
        const float factor = Traits::inv_sqrt(sum / static_cast<float>(width) + args.norm_epsilon);
#pragma unroll
        for (unsigned k = 0; k < kTransitionNormRegs; ++k) {
            const unsigned i = lane + k * kTransitionNormLanes;
            if (norm_lane && i < width)
                ops::norm::rmsnorm_emit<Traits>(norm.gain, 0, norm.output, 0, ops::norm::NoStore{}, i, v[k], factor,
                                                weight[k]);
        }
    }
}

}  // namespace detail

// One launch for the add -> bank readout -> layer-input RMS chain of one token row: every image the launch_add,
// launch_mix and enqueue_rms_bf16_rows (parallel) leaves store, with their bits (HIP transition fixture).
inline Error launch_transition(const TransitionArgs& args, Stream stream) {
    if (!hip_transition_admits(args)) return invalid_value;
    const dim3 threads(detail::kPairedBankThreads, args.mix.bank_count);
    if (args.mix.normalized) detail::transition<true><<<1, threads, 0, stream>>>(args);
    else detail::transition<false><<<1, threads, 0, stream>>>(args);
    return detail::last_error();
}

}  // namespace strix::core::residual_mix
