// qpf2k/attn microbench + fixtures: production QSA prefill kernels copied verbatim from
// src/hip_indexed_attention_launcher.hip (0998a0e5) and the qpf2k kernels (src/hip_qsa_prefill_flash.inc).
// qpf2k/attn microbench (standalone; build: amdclang++ -O3 -std=gnu++20 --offload-arch=gfx1151 -x hip).
// Times the QSA prefill part-2 kernels of one layer-chunk (T rows from position P) on synthetic data, the
// way enqueue_qsa_known_rows runs them for 64-row groups (pool, block scores, selection, batched union +
// attend), and checks the qpf2k kernels (src/hip_qsa_prefill_flash.inc) bitwise against the production ones.
// Pool/store kernels are shared directly with src/hip_qsa_prefill_stages.inc.
// The remaining production reference code below is copied verbatim from src/hip_indexed_attention_launcher.hip
// (0998a0e5); hip_qsa_fast_prefill.inc and hip_qsa_prefill_flash.inc are included from src/.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {
namespace {
constexpr std::uint32_t kThreads = 256;
enum class SmallWeightKind { f32, f16, bf16 };
struct GraphPositionKernelArguments { const std::uint64_t* position = nullptr; std::uint64_t context_tokens = 0; std::uint32_t* status = nullptr; };
struct GraphSelectionArguments { GraphPositionKernelArguments position; std::uint32_t ratio = 0; std::uint32_t block_topk = 0; };
__device__ bool graph_visible(GraphSelectionArguments dynamic, std::uint32_t& visible) {
    if (dynamic.position.position == nullptr) return true;
    const auto position = *dynamic.position.position;
    __shared__ std::uint32_t reject;
    if (threadIdx.x == 0) reject = atomicAdd(dynamic.position.status, 0U) != 0 || position >= dynamic.position.context_tokens;
    __syncthreads();
    if (reject) return false;
    visible = static_cast<std::uint32_t>(position + 1ULL);
    return true;
}
__device__ float bf16_to_float(std::uint16_t bits) { return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U); }
__device__ std::uint16_t float_to_bf16(float value) {
    const auto bits = __float_as_uint(value);
    const auto rounded = bits + 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}
__device__ float round_bf16(float value) { return bf16_to_float(float_to_bf16(value)); }
template <SmallWeightKind Kind> __device__ float load_small_weight(const void* weights, std::uint32_t index) {
    if constexpr (Kind == SmallWeightKind::f32) return static_cast<const float*>(weights)[index];
    else if constexpr (Kind == SmallWeightKind::f16) return __half2float(static_cast<const __half*>(weights)[index]);
    else return bf16_to_float(static_cast<const std::uint16_t*>(weights)[index]);
}

// ---- verbatim production kernels (hip_indexed_attention_launcher.hip) ----
constexpr std::uint32_t kSelectThreads = 1024;
constexpr std::uint32_t kSelectWaves = kSelectThreads / 32U;
__device__ __forceinline__ std::uint32_t select_key(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
__device__ __forceinline__ std::uint32_t select_scan(std::uint32_t value, std::uint32_t* waves, std::uint32_t& total) {
    const auto lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up(value, d, 32); if (lane >= d) value += up; }
    if (lane == 31U) waves[wave] = value;
    __syncthreads();
    if (wave == 0) {
        auto sum = waves[lane];
        for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up(sum, d, 32); if (lane >= d) sum += up; }
        waves[lane] = sum;
    }
    __syncthreads();
    total = waves[kSelectWaves - 1U];
    value += wave ? waves[wave - 1U] : 0U;
    __syncthreads();
    return value;
}
__device__ void select_blocks_parallel_body(float* block_scores, std::uint32_t num_blocks, std::uint32_t select_blocks,
    std::uint32_t compress_ratio, std::uint32_t visible_tokens, std::uint32_t* selected, GraphSelectionArguments dynamic) {
    const auto tid = static_cast<std::uint32_t>(threadIdx.x);
    if (dynamic.position.position != nullptr) {
        if (!graph_visible(dynamic, visible_tokens)) return;
        num_blocks = visible_tokens / compress_ratio;
        if (num_blocks <= dynamic.block_topk) {
            for (auto token = tid; token < visible_tokens; token += kSelectThreads) selected[token] = token;
            return;
        }
        select_blocks = dynamic.block_topk;
    }
    __shared__ std::uint32_t histogram[2048];
    __shared__ std::uint32_t waves[kSelectWaves];
    __shared__ std::uint32_t found_digit, found_need, found_count;
    std::uint32_t prefix = 0, mask = 0, need = select_blocks, zeros = 0;
    for (unsigned pass = 0; pass < 3U; ++pass) {
        const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
        const auto digits = pass == 2 ? 1024U : 2048U;
        for (auto i = tid; i < 2048U; i += kSelectThreads) histogram[i] = 0;
        __syncthreads();
        for (std::uint32_t base = 0; base < num_blocks; base += kSelectThreads) {
            const auto block = base + tid;
            const float value = block < num_blocks ? block_scores[block] : -INFINITY;
            const auto key = select_key(value);
            const bool counted = value > -INFINITY && (key & mask) == prefix;
            const auto digit = (key >> shift) & (digits - 1U);
            const auto votes = __ballot(counted);
            if (votes) {
                const auto leader = static_cast<unsigned>(__builtin_ctzll(votes));
                const auto lead = __shfl(digit, static_cast<int>(leader), 32);
                const auto same = __ballot(counted && digit == lead);
                if (counted && digit != lead) atomicAdd(&histogram[digit], 1U);
                else if (tid % 32U == leader) atomicAdd(&histogram[lead], static_cast<std::uint32_t>(__popcll(same)));
            }
        }
        __syncthreads();
        const auto high = 2U * tid + 1U < digits ? histogram[2U * tid + 1U] : 0U;
        const auto low = 2U * tid < digits ? histogram[2U * tid] : 0U;
        std::uint32_t counted_total = 0;
        const auto above = [&] { const auto inclusive = select_scan(high + low, waves, counted_total); return counted_total - inclusive; }();
        if (pass == 0 && counted_total <= need) { zeros = need - counted_total; need = counted_total; break; }
        if (above < need && need <= above + high) { found_digit = 2U * tid + 1U; found_need = need - above; found_count = high; }
        else if (above + high < need && need <= above + high + low) { found_digit = 2U * tid; found_need = need - above - high; found_count = low; }
        __syncthreads();
        prefix |= found_digit << shift; mask |= (digits - 1U) << shift; need = found_need;
        if (found_count == need) break;
    }
    for (auto token = tid; token < zeros * compress_ratio; token += kSelectThreads) selected[token] = token % compress_ratio;
    std::uint32_t greater_before = 0, equal_before = 0;
    for (std::uint32_t base = 0; base < num_blocks; base += 4U * kSelectThreads) {
        const auto first = base + 4U * tid;
        unsigned kind[4], greater = 0, equal = 0;
        for (unsigned j = 0; j < 4U; ++j) {
            const auto block = first + j;
            const float value = block < num_blocks ? block_scores[block] : -INFINITY;
            const auto masked = select_key(value) & mask;
            kind[j] = !(value > -INFINITY) ? 0U : masked > prefix ? 1U : masked == prefix ? 2U : 0U;
            greater += kind[j] == 1U; equal += kind[j] == 2U;
        }
        std::uint32_t tile = 0;
        const auto packed = select_scan((greater << 16U) | equal, waves, tile);
        auto g = greater_before + (packed >> 16U) - greater, e = equal_before + (packed & 0xffffU) - equal;
        for (unsigned j = 0; j < 4U; ++j) {
            if (kind[j] == 1U || (kind[j] == 2U && e < need)) {
                const auto block = first + j, slot = zeros + g + min(e, need);
                for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
                    selected[slot * compress_ratio + offset] = block * compress_ratio + offset;
                block_scores[block] = -INFINITY;
            }
            g += kind[j] == 1U; e += kind[j] == 2U;
        }
        greater_before += tile >> 16U; equal_before += tile & 0xffffU;
    }
    if (zeros && tid == 0) block_scores[0] = -INFINITY;
    const auto complete_tokens = num_blocks * compress_ratio, selected_complete = select_blocks * compress_ratio;
    for (auto token = complete_tokens + tid; token < visible_tokens; token += kSelectThreads)
        selected[selected_complete + token - complete_tokens] = token;
}
template <unsigned Rows> struct QsaRowsScoreViews {
    const float* index_query[Rows];
    float* block_scores[Rows];
    std::uint32_t blocks[Rows];
};
template <unsigned Rows> struct QsaRowsSelectViews {
    float* block_scores[Rows];
    std::uint32_t* selected[Rows];
    std::uint32_t visible[Rows];
};
template <unsigned Rows>
__global__ __launch_bounds__(kSelectThreads) void select_rows_kernel(QsaRowsSelectViews<Rows> views, std::uint32_t compress_ratio, std::uint32_t block_topk) {
    const auto r = blockIdx.x;
    const auto visible = views.visible[r];
    const auto complete = visible / compress_ratio;
    if (complete <= block_topk) {
        for (std::uint32_t index = threadIdx.x; index < visible; index += blockDim.x) views.selected[r][index] = index;
        return;
    }
    select_blocks_parallel_body(views.block_scores[r], complete, block_topk, compress_ratio, visible,
        views.selected[r], GraphSelectionArguments{});
}
template <unsigned D> __device__ __forceinline__ float qsa_lane_xor(float v) {
    if constexpr (D == 16U) return __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(v), __float_as_uint(v),
        0x76543210U, 0xfedcba98U, false, false));
    else return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x160 | static_cast<int>(D), 0xf, 0xf, false));
}
template <unsigned P, unsigned Step = 0U> __device__ __forceinline__ void qsa_scatter(float (&v)[P], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> Step) > 0U ? (P >> Step) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = qsa_lane_xor<D>(v[j]), high = qsa_lane_xor<D>(v[j + H]);
                v[j] = upper ? v[j + H] + high : v[j] + low;
            }
        } else v[0] += qsa_lane_xor<D>(v[0]);
        qsa_scatter<P, Step + 1U>(v, lane);
    }
}
#include "../src/hip_qsa_prefill_stages.inc"
template <unsigned Rows> struct QsaRowsInputViews {
    const std::uint16_t* input[Rows];
    const void* weights[5];
    std::uint32_t first[6]; // output-row prefix of the group's matrices
    float* output[5][Rows];
    std::uint32_t count;
};
template <unsigned Rows>
__device__ __forceinline__ unsigned qsa_rows_matrix(const QsaRowsInputViews<Rows>& views, std::uint32_t block, std::uint32_t& row) {
    unsigned m = 0;
    while (m + 1U < views.count && block >= views.first[m + 1U]) ++m;
    row = block - views.first[m];
    return m;
}

#include "../src/hip_qsa_prefill_index.inc"
// One wave scores RowsPerWave consecutive rows against 32 consecutive blocks (lane i keeps
// block base + i); index head_dim 128, 4 index heads.
template <unsigned Rows, unsigned RowsPerWave>
__global__ __launch_bounds__(256) void block_score_pooled_rows_kernel(QsaRowsScoreViews<Rows> views, const float* pooled_keys,
    std::uint32_t scored_blocks, std::uint32_t head_dim) {
#pragma clang fp contract(off)
    constexpr unsigned Heads = 4U, Dim = 128U; // head_dim == Dim (checked by the caller)
    const auto lane = threadIdx.x % 32U;
    const auto wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto chunks = (scored_blocks + 31U) / 32U;
    const auto r0 = wave / chunks * RowsPerWave, base = wave % chunks * 32U;
    if (r0 >= Rows) return;
    float q[RowsPerWave][Heads][4];
    std::uint32_t blocks[RowsPerWave];
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
        blocks[rr] = views.blocks[r0 + rr];
#pragma unroll
        for (unsigned h = 0; h < Heads; ++h)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) q[rr][h][j] = views.index_query[r0 + rr][h * Dim + lane + 32U * j];
    }
    float result[RowsPerWave];
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr) result[rr] = 0.0F;
    const auto last = min(32U, scored_blocks - base);
    for (unsigned i = 0; i < last; ++i) {
        const auto* key = pooled_keys + static_cast<std::uint64_t>(base + i) * Dim + lane;
        const float k0 = key[0], k1 = key[32], k2 = key[64], k3 = key[96];
#pragma unroll
        for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
            float partial[Heads];
#pragma unroll
            for (unsigned h = 0; h < Heads; ++h) {
                const float p0 = q[rr][h][0] * k0, p1 = q[rr][h][1] * k1, p2 = q[rr][h][2] * k2, p3 = q[rr][h][3] * k3;
                partial[h] = (p0 + p2) + (p1 + p3);
            }
            qsa_scatter<Heads, 0U>(partial, lane);
            // Lane 8h holds head h's sum.
            float total = 0.0F;
#pragma unroll
            for (unsigned h = 0; h < Heads; ++h) total += fmaxf(__shfl(partial[0], static_cast<int>(8U * h), 32), 0.0F);
            if (lane == i) result[rr] = total / sqrtf(static_cast<float>(head_dim));
        }
    }
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr)
        if (lane < last && base + lane < blocks[rr]) views.block_scores[r0 + rr][base + lane] = result[rr];
}
// ---- end verbatim ----
#include "../src/hip_qsa_fast_prefill.inc"
#include "../src/hip_qsa_fast_prefill_pipe.inc"
#include "../src/hip_qsa_fast_prefill_share.inc"
#include "../src/hip_qsa_fast_prefill_shadow.inc"
#include "../src/hip_qsa_fast_prefill_shshare.inc"
#include "../src/hip_qsa_fast_prefill_shpipe.inc"
#include "../src/hip_qsa_prefill_flash.inc"
} // namespace
} // namespace strix

