#pragma once

// B3 row-owner transition on sm_120: residual add prologue + bank readout + RMSNorm (BF16 gain, BF16 output) + a8g32-v1
// side output in ONE block per token. Contract and legality: core::residual_mix::TransitionArgs / fits (shared).
//
// Every stage repeats the operation order of the leaf it replaces, so each stored image holds those leaves' bits:
//   prologue  pointwise Add / AddRound: out[i] = (round)(__fadd_rn(lhs[i], rhs[i])), stored before any bank read;
//   scores    the tiled readout (mix_cuda_latency.cuh, bitwise with the K8c readout by the MIXLAT fixture): per bank
//             slice of 128 threads, virtual lanes x and x + 128 hold 16 strided values each, square / dot FMA chains in
//             ascending index, bank_halving_lane_zero tree, 1 / sqrtf(squares / width + epsilon), score * 1 / sqrtf(width);
//   softmax   one thread per query: running fmaxf, expf(score - max), ascending denominator, division;
//   output    value = fmaf(p_b, x_b * inverse_b, value) in ascending bank order, stored to the raw row;
//   norm      rmsnorm_bf16 (norms.cu norm_kernel<rms_bf16>): thread t of 256 adds __fmul_rn(x, x) of x[t + 256 k] with
//             __fadd_rn in ascending k, xor butterfly 16..1, the 8 warp sums in warp order, inverse
//             __fdiv_rn(1, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, n), eps))), y = bf16_rn(__fmul_rn(__fmul_rn(x, inv), gain));
//   a8g32     a8g32-v1 of y per 32 consecutive values, one value per lane (dense::detail::a8_group).
// Phases are separated by block barriers, which also order the prologue's global stores before the bank reads of the
// same block (the launch is one block).

#include "mix_cuda_latency.cuh"
#include "dense/a8_helpers.cuh"
#include "norms/row_math.cuh"

#include <cuda_bf16.h>

namespace strix::cuda::mix {
namespace detail {

inline constexpr unsigned kTransitionNormLanes = 256U;
inline constexpr unsigned kTransitionMinSlices = kTransitionNormLanes / kPairedLanes;
static_assert(kTransitionMinSlices * kPairedLanes == kTransitionNormLanes);
static_assert(residual::kTransitionMaxWidth == kLatencyMaxWidth);

[[nodiscard]] __host__ __device__ constexpr unsigned transition_slices(unsigned banks) {
    return banks < kTransitionMinSlices ? kTransitionMinSlices : banks;
}

// norm_kernel<rms_bf16> reduction of one row held in shared memory (lanes 0..255 of the block; every thread calls).
__device__ __forceinline__ float transition_square_sum(const float* row, unsigned width, unsigned tid, float* warps) {
    float p = 0.0F;
    if (tid < kTransitionNormLanes)
        for (unsigned idx = tid; idx < width; idx += kTransitionNormLanes) p = __fadd_rn(p, __fmul_rn(row[idx], row[idx]));
    // A thread past the row only adds __fmul_rn(0, 0) in the incumbent (p + 0 == p for p >= +0): skipped here.
    p = cuda_core::warp_sum_xor(p);
    if (tid < kTransitionNormLanes && (tid & 31U) == 0U) warps[tid >> 5U] = p;
    __syncthreads();
    float s = warps[0];
#pragma unroll
    for (unsigned w = 1; w < kTransitionNormLanes / 32U; ++w) s = __fadd_rn(s, warps[w]);
    __syncthreads();
    return s;
}

template <bool Normalized>
__global__ void __launch_bounds__(kPairedLanes * residual::kMaxBanks)
transition_kernel(const __grid_constant__ residual::TransitionArgs args) {
    namespace reference = residual::detail;
    static_assert(kLatencyMaxWidth <= reference::kMixRegisterWidth);
    __shared__ float scratch[residual::kMaxBanks][kPairedLanes];
    __shared__ float inverse[residual::kMaxBanks];
    __shared__ float probabilities[residual::kMaxQueries][residual::kMaxBanks];
    __shared__ float row[kLatencyMaxWidth];
    __shared__ float warps[kTransitionNormLanes / 32U];
    const auto& mix = args.mix;
    const unsigned lane = threadIdx.x, slice = threadIdx.y;
    const unsigned tid = slice * kPairedLanes + lane, threads = blockDim.y * kPairedLanes;
    const bool live = slice < mix.bank_count;
    const float width = static_cast<float>(mix.width);
    const float score_scale = 1.0F / sqrtf(width);
    cuda_core::pdl_wait();
    cuda_core::pdl_trigger();

    // Prologue: the current bank. Operands are read before any store of this block.
    if (args.add.lhs) {
        const auto& add = args.add;
        for (unsigned i = tid; i < mix.width; i += threads) {
            const float sum = __fadd_rn(add.lhs[i], add.rhs[i]);
            add.output[i] = add.round_bf16 ? reference::gi::bf16_round(sum) : sum;
        }
        __syncthreads();
    }

    // Scores: the tiled readout's slice schedule, once per bank, every query against the held bank values.
    float low[reference::kMixRegs], high[reference::kMixRegs];
    const float* source = live ? mix.banks[slice] : nullptr;
    if (live) {
        latency_load_strided(source, mix.width, lane, low);
        latency_load_strided(source, mix.width, lane + kPairedLanes, high);
    }
    float scale = 1.0F;
    if constexpr (!Normalized) {
        const float part = live ? latency_square_chain(low, mix.width, lane) +
                                      latency_square_chain(high, mix.width, lane + kPairedLanes) : 0.0F;
        const float squares = bank_halving_lane_zero(part, lane, scratch[slice]);
        if (live && lane == 0U) inverse[slice] = 1.0F / sqrtf(squares / width + mix.epsilon);
        __syncthreads();
        if (live) scale = inverse[slice];
    }
    for (unsigned q = 0; q < mix.query_count; ++q) {
        const float* query = mix.queries + std::size_t{q} * mix.width;
        float dot_low = 0.0F, dot_high = 0.0F;
        if (live) {
#pragma unroll
            for (unsigned k = 0; k < reference::kMixRegs; ++k) {
                const unsigned i = lane + k * kLanes;
                if (i < mix.width) dot_low = fmaf(query[i], bank_value<Normalized>(low[k], scale), dot_low);
                if (i + kPairedLanes < mix.width)
                    dot_high = fmaf(query[i + kPairedLanes], bank_value<Normalized>(high[k], scale), dot_high);
            }
        }
        const float dot = bank_halving_lane_zero(dot_low + dot_high, lane, scratch[slice]);
        if (live && lane == 0U) probabilities[q][slice] = dot * score_scale;
        // Retires the tree's scratch reads before the next query reuses it.
        __syncthreads();
    }
    if (tid < mix.query_count) {
        float* p = probabilities[tid];
        float maximum = p[0];
        for (unsigned b = 1; b < mix.bank_count; ++b) maximum = fmaxf(maximum, p[b]);
        float denominator = 0.0F;
        for (unsigned b = 0; b < mix.bank_count; ++b) {
            p[b] = expf(p[b] - maximum);
            denominator += p[b];
        }
        for (unsigned b = 0; b < mix.bank_count; ++b) p[b] /= denominator;
    }
    __syncthreads();

    // Readout and epilogue, one query at a time through the shared row.
    for (unsigned q = 0; q < mix.query_count; ++q) {
        const auto& norm = args.norm[q];
        float* out = mix.output[q];
        for (unsigned i = tid; i < mix.width; i += threads) {
            float values[residual::kMaxBanks];
#pragma unroll
            for (unsigned b = 0; b < residual::kMaxBanks; ++b)
                if (b < mix.bank_count) values[b] = mix.banks[b][i];
            float value = 0.0F;
#pragma unroll
            for (unsigned b = 0; b < residual::kMaxBanks; ++b)
                if (b < mix.bank_count)
                    value = fmaf(probabilities[q][b], bank_value<Normalized>(values[b], Normalized ? 1.0F : inverse[b]),
                                 value);
            out[i] = value;
            row[i] = value;
        }
        if (!norm.output) continue;  // uniform: the next query's stores reuse only this thread's row slots
        __syncthreads();
        const float squares = transition_square_sum(row, mix.width, tid, warps);
        const float inv = STRIX_CUDA_RMS_INVERSE_RN(squares, width, args.norm_epsilon);
        if (tid < kTransitionNormLanes) {
            for (unsigned base = 0; base < mix.width; base += kTransitionNormLanes) {
                const unsigned idx = base + tid;
                float y = 0.0F;
                if (idx < mix.width) {
                    y = __fmul_rn(row[idx], inv);
                    if (norm.gain) y = __fmul_rn(y, cuda_core::bf16_lo(norm.gain[idx]));
                    y = __bfloat162float(__float2bfloat16_rn(y));
                    norm.output[idx] = y;
                }
                // a8g32-v1: the 32 lanes of a warp hold 32 consecutive values (width % 32 == 0: uniform per warp).
                if (norm.codes && base + (tid & ~31U) < mix.width) {
                    float delta;
                    std::int32_t sum;
                    const auto code = ::strix::cuda::core::dense::detail::a8_group(y, delta, sum);
                    norm.codes[idx] = static_cast<std::int8_t>(code);
                    if ((tid & 31U) == 0U) {
                        norm.deltas[idx / 32U] = delta;
                        if (norm.sums) norm.sums[idx / 32U] = sum;
                    }
                }
            }
        }
        // The next query rewrites the shared row.
        __syncthreads();
    }
}

}  // namespace detail

struct TransitionGrid {
    unsigned threads, slices;
};

[[nodiscard]] inline TransitionGrid transition_grid(const ::strix::core::residual_mix::TransitionArgs& args) {
    if (!::strix::core::residual_mix::fits(args)) return {};
    return {detail::kPairedLanes, detail::transition_slices(args.mix.bank_count)};
}

// One block: every image the separate add, readout and norm leaves store, with their bits.
[[nodiscard]] inline cudaError_t enqueue_transition(const ::strix::core::residual_mix::TransitionArgs& args,
                                                    cudaStream_t stream) {
    const auto grid = transition_grid(args);
    if (grid.slices == 0U) return cudaErrorInvalidValue;
    const auto kernel = args.mix.normalized ? detail::transition_kernel<true> : detail::transition_kernel<false>;
    return cuda_core::launch_pdl(kernel, dim3(1U), dim3(grid.threads, grid.slices), stream, args);
}

}  // namespace strix::cuda::mix
