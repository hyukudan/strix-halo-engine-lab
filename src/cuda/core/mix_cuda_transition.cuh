#pragma once

// B3 row-owner transition on sm_120: residual add prologue + bank readout + RMSNorm (BF16 gain, BF16 output) + a8g32-v1
// side output in ONE launch per token, as ONE thread block cluster (plan: mix_cuda_latency_plan.hpp transition_plan).
// Contract and legality: core::residual_mix::TransitionArgs / fits (shared).
//
// Block r of the cluster owns tile r (kLatencyTile values) of the row. Every stage repeats the operation order of the
// leaf it replaces, so each stored image holds those leaves' bits:
//   prologue  pointwise Add / AddRound: (round)(__fadd_rn(lhs[i], rhs[i])); every block recomputes the values it reads
//             (the whole row for the scores, its tile for the readout) and the owner stores its tile;
//   scores    the tiled readout (mix_cuda_latency.cuh, bitwise with the K8c readout by the MIXLAT fixture), redundantly
//             in every block like the tiled kernel: per bank slice of 128 threads, virtual lanes x and x + 128 hold 16
//             strided values each, square / dot FMA chains in ascending index, bank_halving_lane_zero tree,
//             1 / sqrtf(squares / width + epsilon), score * 1 / sqrtf(width);
//   softmax   one thread per query: running fmaxf, expf(score - max), ascending denominator, division;
//   output    value = fmaf(p_b, x_b * inverse_b, value) in ascending bank order, for the block's tile, every query;
//   norm      rmsnorm_bf16 (norms.cu norm_kernel<rms_bf16>): lane t of 256 adds __fmul_rn(x, x) of x[t + 256 k] with
//             __fadd_rn in ascending k (x read from the owner's shared tile through distributed shared memory), xor
//             butterfly 16..1, the 8 warp sums in warp order, inverse
//             __fdiv_rn(1, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, n), eps))), y = bf16_rn(__fmul_rn(__fmul_rn(x, inv), gain))
//             for the block's tile;
//   a8g32     a8g32-v1 of y per 32 consecutive values, one value per lane (dense::detail::a8_group).
// Every global read of the launch precedes the first cluster barrier and every global store follows it, so the add
// operands may alias the current bank or a raw row (TransitionArgs::fits) across blocks.

#include "mix_cuda_latency.cuh"
#include "dense/a8_helpers.cuh"
#include "norms/row_math.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace strix::cuda::mix {
namespace detail {

static_assert(kTransitionNormLanes == 2U * kPairedLanes && kLatencyBankThreads == kPairedLanes);

// The prologue's value of the current bank at index i (the add leaf's stored bits).
__device__ __forceinline__ float transition_add_value(const residual::AddArgs& add, unsigned i) {
    const float sum = __fadd_rn(add.lhs[i], add.rhs[i]);
    return add.round_bf16 ? residual::detail::gi::bf16_round(sum) : sum;
}

// latency_load_strided of the current bank, computed from the prologue operands instead of read from its image.
__device__ __forceinline__ void transition_load_add(const residual::AddArgs& add, unsigned width, unsigned lane,
                                                    float (&v)[residual::detail::kMixRegs]) {
#pragma unroll
    for (unsigned k = 0; k < residual::detail::kMixRegs; ++k) {
        const unsigned i = lane + k * residual::detail::kThreads;
        v[k] = i < width ? transition_add_value(add, i) : 0.0F;
    }
}

template <bool Normalized>
__global__ void __launch_bounds__(kPairedLanes * residual::kMaxBanks)
transition_kernel(const __grid_constant__ residual::TransitionArgs args) {
    namespace reference = residual::detail;
    namespace cg = cooperative_groups;
    static_assert(kLatencyMaxWidth <= reference::kMixRegisterWidth);
    __shared__ float scratch[residual::kMaxBanks][kPairedLanes];
    __shared__ float inverse[residual::kMaxBanks];
    __shared__ float probabilities[residual::kMaxQueries][residual::kMaxBanks];
    __shared__ float rows[residual::kMaxQueries][kLatencyTile];  // this block's tile of every raw row
    __shared__ float current[kLatencyTile];                      // this block's tile of the prologue's store
    __shared__ float warps[residual::kMaxQueries][kTransitionNormLanes / 32U];
    const auto& mix = args.mix;
    const unsigned lane = threadIdx.x, slice = threadIdx.y;
    const unsigned tid = slice * kPairedLanes + lane, threads = blockDim.y * kPairedLanes;
    const bool live = slice < mix.bank_count;
    const bool prologue = args.add.lhs != nullptr;
    const unsigned last = mix.bank_count - 1U;
    const float width = static_cast<float>(mix.width);
    const float score_scale = 1.0F / sqrtf(width);
    const auto cluster = cg::this_cluster();
    const unsigned first = blockIdx.x * kLatencyTile;
    const unsigned end = min(first + kLatencyTile, mix.width);
    cuda_core::pdl_wait();
    cuda_core::pdl_trigger();

    // ---- reads only (no global store before the first cluster barrier) ----
    if (prologue)
        for (unsigned i = first + tid; i < end; i += threads) current[i - first] = transition_add_value(args.add, i);

    // Scores: the tiled readout's slice schedule, once per bank, every query against the held bank values.
    float low[reference::kMixRegs], high[reference::kMixRegs];
    if (live) {
        if (prologue && slice == last) {
            transition_load_add(args.add, mix.width, lane, low);
            transition_load_add(args.add, mix.width, lane + kPairedLanes, high);
        } else {
            latency_load_strided(mix.banks[slice], mix.width, lane, low);
            latency_load_strided(mix.banks[slice], mix.width, lane + kPairedLanes, high);
        }
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

    // Readout of this block's tile: each bank value is loaded once for every query.
    for (unsigned i = first + tid; i < end; i += threads) {
        float values[residual::kMaxBanks];
#pragma unroll
        for (unsigned b = 0; b < residual::kMaxBanks; ++b)
            if (b < mix.bank_count) values[b] = prologue && b == last ? current[i - first] : mix.banks[b][i];
        for (unsigned q = 0; q < mix.query_count; ++q) {
            float value = 0.0F;
#pragma unroll
            for (unsigned b = 0; b < residual::kMaxBanks; ++b)
                if (b < mix.bank_count)
                    value = fmaf(probabilities[q][b], bank_value<Normalized>(values[b], Normalized ? 1.0F : inverse[b]),
                                 value);
            rows[q][i - first] = value;
        }
    }
    // Every block has read every global operand; the raw tiles are visible to the cluster.
    cluster.sync();

    // ---- stores (the incumbent leaves' order per address: add, raw rows, then normalized rows) ----
    for (unsigned i = first + tid; i < end; i += threads) {
        if (prologue) args.add.output[i] = current[i - first];
        for (unsigned q = 0; q < mix.query_count; ++q) mix.output[q][i] = rows[q][i - first];
    }
    // RMS sums in the rms.bf16 lane order over the whole row (peer tiles through distributed shared memory).
    for (unsigned q = 0; q < mix.query_count; ++q) {
        if (!args.norm[q].output) continue;  // uniform
        float p = 0.0F;
        if (tid < kTransitionNormLanes)
            for (unsigned idx = tid; idx < mix.width; idx += kTransitionNormLanes) {
                const float* owner = cluster.map_shared_rank(&rows[q][0], transition_owner(idx));
                const float x = owner[transition_offset(idx)];
                p = __fadd_rn(p, __fmul_rn(x, x));
            }
        // A lane past the row only adds __fmul_rn(0, 0) in the incumbent (p + 0 == p for p >= +0): skipped here.
        p = cuda_core::warp_sum_xor(p);
        if (tid < kTransitionNormLanes && (tid & 31U) == 0U) warps[q][tid >> 5U] = p;
    }
    // This block no longer reads peer shared memory; its own tiles stay alive until the final wait.
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    __syncthreads();
    const unsigned span = end - first;
    for (unsigned q = 0; q < mix.query_count; ++q) {
        const auto& norm = args.norm[q];
        if (!norm.output) continue;  // uniform
        float s = warps[q][0];
#pragma unroll
        for (unsigned w = 1; w < kTransitionNormLanes / 32U; ++w) s = __fadd_rn(s, warps[q][w]);
        const float inv = STRIX_CUDA_RMS_INVERSE_RN(s, width, args.norm_epsilon);
        for (unsigned base = 0; base < span; base += threads) {
            const unsigned idx = first + base + tid;
            float y = 0.0F;
            if (idx < end) {
                y = __fmul_rn(rows[q][idx - first], inv);
                if (norm.gain) y = __fmul_rn(y, cuda_core::bf16_lo(norm.gain[idx]));
                y = __bfloat162float(__float2bfloat16_rn(y));
                norm.output[idx] = y;
            }
            // a8g32-v1: the 32 lanes of a warp hold 32 consecutive values (width % 32 == 0 and tiles start at
            // multiples of 32: uniform per warp).
            if (norm.codes && first + base + (tid & ~31U) < end) {
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
    // Peers may still read this block's tiles.
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

}  // namespace detail

struct TransitionGrid {
    unsigned threads, slices, tiles;
};

[[nodiscard]] inline TransitionGrid transition_grid(const ::strix::core::residual_mix::TransitionArgs& args) {
    if (!::strix::core::residual_mix::fits(args)) return {};
    const auto plan = transition_plan(args.mix.width, args.mix.bank_count);
    if (plan.tiles == 0U) return {};
    return {detail::kPairedLanes, plan.slices, plan.tiles};
}

// One cluster of grid.tiles blocks: every image the separate add, readout and norm leaves store, with their bits.
[[nodiscard]] inline cudaError_t enqueue_transition(const ::strix::core::residual_mix::TransitionArgs& args,
                                                    cudaStream_t stream) {
    const auto grid = transition_grid(args);
    if (grid.slices == 0U) return cudaErrorInvalidValue;
    const auto kernel = args.mix.normalized ? detail::transition_kernel<true> : detail::transition_kernel<false>;
    return cuda_core::launch_pdl_cluster(kernel, dim3(grid.tiles), dim3(grid.threads, grid.slices), grid.tiles, stream,
                                         args);
}

}  // namespace strix::cuda::mix
