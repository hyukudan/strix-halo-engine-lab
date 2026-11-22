#pragma once

// B3 row-owner transition on sm_120: residual add prologue + bank readout + RMSNorm (BF16 gain, BF16 output) + a8g32-v1
// side output in ONE launch per token, as ONE thread block cluster (plan: mix_cuda_latency_plan.hpp transition_plan).
// Contract and legality: core::residual_mix::TransitionArgs / fits (shared).
//
// Block r of the cluster owns tile r (kLatencyTile values) of the row. Every stage repeats the operation order of the
// leaf it replaces, so each stored image holds those leaves' bits:
//   prologue  pointwise Add / AddRound: (round)(__fadd_rn(lhs[i], rhs[i])); every block recomputes the values it reads
//             (the whole row for the scores, its tile for the readout) and the owner stores its tile;
//   staging   lhs, rhs and the query rows are copied into dynamic shared memory with cp.async at entry
//             (transition_stage): the 64-register cap of 1024-thread blocks would otherwise turn those reads into chains
//             of dependent memory round trips (the prologue slice and 4 query loads at a time). Values are unchanged;
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

#include <cstddef>

namespace strix::cuda::mix {
namespace detail {

static_assert(kTransitionNormLanes == 2U * kPairedLanes && kLatencyBankThreads == kPairedLanes);

inline constexpr unsigned kTransitionNormSpan = kLatencyMaxWidth / kTransitionNormLanes;  // RMS values per lane

// The add leaf's stored bits from its two operands.
__device__ __forceinline__ float transition_add_value(float lhs, float rhs, bool round) {
    const float sum = __fadd_rn(lhs, rhs);
    return round ? residual::detail::gi::bf16_round(sum) : sum;
}

// Issues the asynchronous copies of query rows q0 .. q0 + rows - 1 into the ring (row r at ring + r * width).
__device__ __forceinline__ void transition_stage_queries(const residual::MixArgs& mix, unsigned q0, unsigned rows,
                                                         float* ring, unsigned tid, unsigned threads) {
    for (unsigned r = 0; r < rows; ++r) {
        const float* query = mix.queries + std::size_t{q0 + r} * mix.width;
        float* const slot = ring + std::size_t{r} * mix.width;
        for (unsigned i = tid; i < mix.width; i += threads) cuda_core::cp_async_f32(slot + i, query + i);
    }
}

template <bool Normalized>
__global__ void __launch_bounds__(kPairedLanes * residual::kMaxBanks)
transition_kernel(const __grid_constant__ residual::TransitionArgs args, unsigned ring_rows) {
    namespace reference = residual::detail;
    namespace cg = cooperative_groups;
    static_assert(kLatencyMaxWidth <= reference::kMixRegisterWidth);
    __shared__ float scratch[residual::kMaxBanks][kPairedLanes];
    __shared__ float inverse[residual::kMaxBanks];
    __shared__ float probabilities[residual::kMaxQueries][residual::kMaxBanks];
    __shared__ float warps[residual::kMaxQueries][kTransitionNormLanes / 32U];
    extern __shared__ float stage[];  // transition_stage layout: raw tiles, add rows, query ring
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
    float* const raw = stage;                                                    // [query][kLatencyTile]
    float* const current = stage + std::size_t{mix.query_count} * kLatencyTile;  // lhs, then the current bank row
    float* const rhs = current + mix.width;
    float* const ring = current + (prologue ? 2U : 0U) * std::size_t{mix.width};
    cuda_core::pdl_wait();
    cuda_core::pdl_trigger();

    // ---- reads only (no global store before the first cluster barrier) ----
    // Every row read besides the banks is staged asynchronously first: one round trip, no register held.
    if (prologue)
        for (unsigned i = tid; i < mix.width; i += threads) {
            cuda_core::cp_async_f32(current + i, args.add.lhs + i);
            cuda_core::cp_async_f32(rhs + i, args.add.rhs + i);
        }
    transition_stage_queries(mix, 0U, min(ring_rows, mix.query_count), ring, tid, threads);

    // Scores: the tiled readout's slice schedule, once per bank, every query against the held bank values.
    float low[reference::kMixRegs], high[reference::kMixRegs];
    if (live && !(prologue && slice == last)) {
        latency_load_strided(mix.banks[slice], mix.width, lane, low);
        latency_load_strided(mix.banks[slice], mix.width, lane + kPairedLanes, high);
    }
    cuda_core::cp_async_wait_all();
    __syncthreads();
    if (prologue) {
        // The prologue's value of every row index (the add leaf's stored bits), in place of the staged lhs.
        for (unsigned i = tid; i < mix.width; i += threads)
            current[i] = transition_add_value(current[i], rhs[i], args.add.round_bf16);
        __syncthreads();
        if (live && slice == last) {
            latency_load_strided(current, mix.width, lane, low);
            latency_load_strided(current, mix.width, lane + kPairedLanes, high);
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
    for (unsigned q0 = 0; q0 < mix.query_count; q0 += ring_rows) {
        const unsigned rows = min(ring_rows, mix.query_count - q0);
        if (q0 != 0U) {
            // A later batch: the previous batch's ring reads retired at its last tree barrier.
            transition_stage_queries(mix, q0, rows, ring, tid, threads);
            cuda_core::cp_async_wait_all();
            __syncthreads();
        }
        for (unsigned r = 0; r < rows; ++r) {
            const unsigned q = q0 + r;
            const float* query = ring + std::size_t{r} * mix.width;
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
            // Retires the tree's scratch reads (and this query's ring reads) before the next query reuses them.
            __syncthreads();
        }
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
            if (b < mix.bank_count) values[b] = prologue && b == last ? current[i] : mix.banks[b][i];
        for (unsigned q = 0; q < mix.query_count; ++q) {
            float value = 0.0F;
#pragma unroll
            for (unsigned b = 0; b < residual::kMaxBanks; ++b)
                if (b < mix.bank_count)
                    value = fmaf(probabilities[q][b], bank_value<Normalized>(values[b], Normalized ? 1.0F : inverse[b]),
                                 value);
            raw[std::size_t{q} * kLatencyTile + (i - first)] = value;
        }
    }
    // Every block has read every global operand; the raw tiles are visible to the cluster.
    cluster.sync();

    // ---- stores (the incumbent leaves' order per address: add, raw rows, then normalized rows) ----
    for (unsigned i = first + tid; i < end; i += threads) {
        if (prologue) args.add.output[i] = current[i];
        for (unsigned q = 0; q < mix.query_count; ++q) mix.output[q][i] = raw[std::size_t{q} * kLatencyTile + (i - first)];
    }
    // RMS sums in the rms.bf16 lane order over the whole row (peer tiles through distributed shared memory). Each lane
    // loads its values first (one DSMEM round trip), then adds them in ascending order.
    for (unsigned q = 0; q < mix.query_count; ++q) {
        if (!args.norm[q].output) continue;  // uniform
        float p = 0.0F;
        if (tid < kTransitionNormLanes) {
            const float* tile = raw + std::size_t{q} * kLatencyTile;
            float x[kTransitionNormSpan];
#pragma unroll
            for (unsigned k = 0; k < kTransitionNormSpan; ++k) {
                const unsigned idx = tid + k * kTransitionNormLanes;
                const unsigned at = idx < mix.width ? idx : tid;  // a lane past the row reloads a live value, unused
                x[k] = cluster.map_shared_rank(tile, transition_owner(at))[transition_offset(at)];
            }
            // A lane past the row only adds __fmul_rn(0, 0) in the incumbent (p + 0 == p for p >= +0): skipped here.
#pragma unroll
            for (unsigned k = 0; k < kTransitionNormSpan; ++k)
                if (tid + k * kTransitionNormLanes < mix.width) p = __fadd_rn(p, __fmul_rn(x[k], x[k]));
        }
        p = cuda_core::warp_sum_xor(p);
        if (tid < kTransitionNormLanes && (tid & 31U) == 0U) warps[q][tid >> 5U] = p;
    }
    // This block no longer reads peer shared memory (every value it loaded is consumed above); its own tiles stay alive
    // until the final wait. Relaxed: nothing of this block is published to the peers here, so the arrive does not wait
    // for the global stores above (a release arrive is a GPU-scope memory barrier on them).
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    __syncthreads();
    const unsigned span = end - first;
    for (unsigned q = 0; q < mix.query_count; ++q) {
        const auto& norm = args.norm[q];
        if (!norm.output) continue;  // uniform
        const float* tile = raw + std::size_t{q} * kLatencyTile;
        float s = warps[q][0];
#pragma unroll
        for (unsigned w = 1; w < kTransitionNormLanes / 32U; ++w) s = __fadd_rn(s, warps[q][w]);
        const float inv = STRIX_CUDA_RMS_INVERSE_RN(s, width, args.norm_epsilon);
        for (unsigned base = 0; base < span; base += threads) {
            const unsigned idx = first + base + tid;
            float y = 0.0F;
            if (idx < end) {
                y = __fmul_rn(tile[idx - first], inv);
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
    unsigned ring;     // query rows staged per batch (transition_stage)
    std::size_t smem;  // dynamic shared memory per block
};

// Upper bound of the transition kernels' static shared memory: the declared arrays (scratch, inverse, probabilities,
// warps: 4640 B) plus the block's cluster reservation the compiler adds (sm_120a: 5632 / 5664 B as compiled), with
// margin. Only the device opt-in limit is read at run time.
inline constexpr std::size_t kTransitionStaticSharedBound =
    sizeof(float) * (::strix::core::residual_mix::kMaxBanks * detail::kPairedLanes + ::strix::core::residual_mix::kMaxBanks +
                     ::strix::core::residual_mix::kMaxQueries * ::strix::core::residual_mix::kMaxBanks +
                     ::strix::core::residual_mix::kMaxQueries * (kTransitionNormLanes / 32U)) +
    2048U;

// The launch plan, device-independent (bindings and host tests use it): every query row staged at entry.
[[nodiscard]] inline TransitionGrid transition_grid(const ::strix::core::residual_mix::TransitionArgs& args) {
    if (!::strix::core::residual_mix::fits(args)) return {};
    const auto plan = transition_plan(args.mix.width, args.mix.bank_count);
    if (plan.tiles == 0U) return {};
    const auto stage = transition_stage(args.mix.width, args.mix.query_count, args.add.lhs != nullptr,
                                        static_cast<std::size_t>(-1));
    if (stage.query_rows == 0U) return {};
    return {detail::kPairedLanes, plan.slices, plan.tiles, stage.query_rows, stage.bytes};
}

// One cluster of grid.tiles blocks: every image the separate add, readout and norm leaves store, with their bits. A
// device whose opt-in shared memory cannot hold every query row stages them in batches (transition_stage).
[[nodiscard]] inline cudaError_t enqueue_transition(const ::strix::core::residual_mix::TransitionArgs& args,
                                                    cudaStream_t stream) {
    auto grid = transition_grid(args);
    if (grid.slices == 0U) return cudaErrorInvalidValue;
    if (grid.smem + kTransitionStaticSharedBound > 48U * 1024U) {
        int device = 0, limit = 0;
        if (const auto status = cudaGetDevice(&device); status != cudaSuccess) return status;
        if (const auto status = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
            status != cudaSuccess)
            return status;
        const auto available = static_cast<std::size_t>(limit);
        if (grid.smem + kTransitionStaticSharedBound > available) {
            if (available <= kTransitionStaticSharedBound) return cudaErrorInvalidValue;
            const auto stage = transition_stage(args.mix.width, args.mix.query_count, args.add.lhs != nullptr,
                                                available - kTransitionStaticSharedBound);
            if (stage.query_rows == 0U) return cudaErrorInvalidValue;
            grid.ring = stage.query_rows;
            grid.smem = stage.bytes;
        }
    }
    const auto kernel = args.mix.normalized ? detail::transition_kernel<true> : detail::transition_kernel<false>;
    if (grid.smem + kTransitionStaticSharedBound > 48U * 1024U) {
        if (const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     static_cast<int>(grid.smem));
            status != cudaSuccess)
            return status;
    }
    return cuda_core::launch_pdl_cluster(kernel, dim3(grid.tiles), dim3(grid.threads, grid.slices), grid.tiles,
                                         grid.smem, stream, args, grid.ring);
}

}  // namespace strix::cuda::mix
