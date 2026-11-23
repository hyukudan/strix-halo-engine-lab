#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace select_qwen_baseline {

// Frozen from src/cuda/models/qwen/qwen_qsa.cu at 6fb09ae7c.
// Keep these reference bodies independent of the shared selection code.
inline constexpr unsigned kIndexHeads = 4U, kIndexDim = 128U, kIndexQWidth = 512U;
inline constexpr unsigned kRatio = 4U, kTopK = 512U, kMaxSelected = 2051U;

constexpr unsigned kFull = 0xffffffffU;
#define kInf __uint_as_float(0x7f800000U)
// Lane-xor reduce-scatter qsa_scatter<P> (H:3882-3913): step D = 16 (permlanex16 identity = lane ^ 16), 8, 4, 2, 1
// (DPP row_xmask = lane ^ D) halves the carried values, the lower lane keeping the lower half; each step adds the
// partials of lanes L and L ^ D (one addition: operand order irrelevant). Per value this is the 32-lane tree
// x_L + x_{L^16}, then ^8, ^4, ^2, ^1, i.e. lane 0's shfl_down 16..1 tree.
template <unsigned P, unsigned Step = 0U> __device__ __forceinline__ void scatter(float (&v)[P], unsigned lane) {
    if constexpr (Step < 5U) {
        constexpr unsigned D = 16U >> Step, Count = (P >> Step) > 0U ? (P >> Step) : 1U;
        if constexpr (Count > 1U) {
            constexpr unsigned H = Count / 2U;
            const bool upper = (lane & D) != 0U;
#pragma unroll
            for (unsigned j = 0; j < H; ++j) {
                const float low = __shfl_xor_sync(kFull, v[j], D), high = __shfl_xor_sync(kFull, v[j + H], D);
                v[j] = upper ? __fadd_rn(v[j + H], high) : __fadd_rn(v[j], low);
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(kFull, v[0], D));
        }
        scatter<P, Step + 1U>(v, lane);
    }
}
// block_score_pooled_rows_kernel<Rows,4> (H:4211-4259, `#pragma clang fp contract(off)`; S:322261). One warp scores 4
// consecutive rows against 32 consecutive blocks. Per (row, block, head): lane l, p_j = q[l+32j]*k[l+32j] (j=0..3),
// partial = (p0+p2)+(p1+p3), scatter<4>; total = 0 + relu(h0) + relu(h1) + relu(h2) + relu(h3) (head order),
// score = total / sqrtf(128). Written for blocks < complete(row) of rows with complete > 512.
constexpr unsigned kScoreRowsPerWave = 4;
__device__ __forceinline__ void block_score_body(const float* __restrict__ index_query, std::uint32_t rows,
    std::uint64_t first_position, const float* __restrict__ pooled_keys, std::uint32_t scored_blocks,
    float* __restrict__ block_scores, std::uint32_t max_blocks, std::uint32_t wave) {
    // wave = this warp's wave within the segment (blockIdx.x * 8 + warp, minus the segment's first wave)
    constexpr unsigned R = kScoreRowsPerWave;
    const unsigned lane = threadIdx.x & 31U;
    const auto chunks = (scored_blocks + 31U) / 32U;
    const auto r0 = wave / chunks * R, base = wave % chunks * 32U;
    if (r0 >= rows) return;
    float q[R][kIndexHeads][4];
    std::uint32_t blocks[R];
    bool any = false;
#pragma unroll
    for (unsigned rr = 0; rr < R; ++rr) {
        const auto r = r0 + rr;
        std::uint32_t complete = 0;
        if (r < rows) complete = static_cast<std::uint32_t>((first_position + r + 1U) / kRatio);
        blocks[rr] = complete > kTopK ? complete : 0U;
        any = any || base < blocks[rr];
#pragma unroll
        for (unsigned h = 0; h < kIndexHeads; ++h)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j)
                q[rr][h][j] = r < rows ? index_query[static_cast<std::uint64_t>(r) * kIndexQWidth + h * kIndexDim + lane + 32U * j] : 0.0F;
    }
    if (!any) return;  // uniform per warp; such a wave writes nothing in HIP either
    float result[R];
#pragma unroll
    for (unsigned rr = 0; rr < R; ++rr) result[rr] = 0.0F;
    const float scale = __fsqrt_rn(static_cast<float>(kIndexDim));
    const auto last = min(32U, scored_blocks - base);
    for (unsigned i = 0; i < last; ++i) {
        const auto* key = pooled_keys + static_cast<std::uint64_t>(base + i) * kIndexDim + lane;
        const float k0 = key[0], k1 = key[32], k2 = key[64], k3 = key[96];
#pragma unroll
        for (unsigned rr = 0; rr < R; ++rr) {
            float partial[kIndexHeads];
#pragma unroll
            for (unsigned h = 0; h < kIndexHeads; ++h) {
                const float p0 = __fmul_rn(q[rr][h][0], k0), p1 = __fmul_rn(q[rr][h][1], k1);
                const float p2 = __fmul_rn(q[rr][h][2], k2), p3 = __fmul_rn(q[rr][h][3], k3);
                partial[h] = __fadd_rn(__fadd_rn(p0, p2), __fadd_rn(p1, p3));
            }
            scatter<kIndexHeads>(partial, lane);
            float total = 0.0F;
#pragma unroll
            for (unsigned h = 0; h < kIndexHeads; ++h)
                total = __fadd_rn(total, fmaxf(__shfl_sync(kFull, partial[0], static_cast<int>(8U * h)), 0.0F));
            if (lane == i) result[rr] = __fdiv_rn(total, scale);
        }
    }
#pragma unroll
    for (unsigned rr = 0; rr < R; ++rr)
        if (lane < last && base + lane < blocks[rr]) block_scores[static_cast<std::uint64_t>(r0 + rr) * max_blocks + base + lane] = result[rr];
}
__global__ __launch_bounds__(256) void block_score_kernel(const float* __restrict__ index_query, std::uint32_t rows,
    std::uint64_t first_position, const float* __restrict__ pooled_keys, std::uint32_t scored_blocks,
    float* __restrict__ block_scores, std::uint32_t max_blocks) {
    block_score_body(index_query, rows, first_position, pooled_keys, scored_blocks, block_scores, max_blocks,
        blockIdx.x * 8U + threadIdx.x / 32U);
}
constexpr unsigned kTreeRows = 64, kTreeTile = 32, kTreeSeg = 256;
static_assert(kIndexHeads == 4U && kIndexDim == 128U, "block_score_tree_kernel maps 4 heads x 128 features");
__device__ __forceinline__ void block_score_tree_body(const float* __restrict__ index_query, std::uint32_t rows,
    std::uint64_t first_position, const float* __restrict__ pooled_keys, std::uint32_t scored_blocks,
    float* __restrict__ block_scores, std::uint32_t max_blocks, std::uint32_t tile_x, std::uint32_t tile_y) {
    __shared__ __align__(16) float sk[kTreeTile * kIndexDim];
    __shared__ float ss[kTreeRows][kTreeTile + 1U];
    const unsigned tid = threadIdx.x, lane = tid & 31U, rr = tid / 4U, h = tid & 3U;
    const std::uint32_t r0 = tile_x * kTreeRows, r = r0 + rr;
    if (r0 >= rows) return;  // uniform per CTA (multi-segment grids are sized by the largest segment)
    const std::uint32_t s0 = tile_y * kTreeSeg, s1 = min(scored_blocks, s0 + kTreeSeg);
    const std::uint32_t last_row = min(rows, r0 + kTreeRows) - 1U;  // complete() is nondecreasing in the row
    const auto limit_of = [&](std::uint32_t row) -> std::uint32_t {
        const auto complete = static_cast<std::uint32_t>((first_position + row + 1U) / kRatio);
        return complete > kTopK ? complete : 0U;
    };
    if (s0 >= s1 || s0 >= limit_of(last_row)) return;  // uniform per CTA, before any barrier
    float q[kIndexDim];
#pragma unroll
    for (unsigned d = 0; d < kIndexDim; d += 4U) {
        float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        if (r < rows) v = *reinterpret_cast<const float4*>(index_query + static_cast<std::uint64_t>(r) * kIndexQWidth + h * kIndexDim + d);
        q[d] = v.x; q[d + 1U] = v.y; q[d + 2U] = v.z; q[d + 3U] = v.w;
    }
    const float scale = __fsqrt_rn(static_cast<float>(kIndexDim));
    for (std::uint32_t b0 = s0; b0 < s1; b0 += kTreeTile) {
        const unsigned nb = min(kTreeTile, s1 - b0);
        __syncthreads();  // previous tile's sk / ss fully consumed
        for (unsigned e = tid; e < kTreeTile * kIndexDim / 4U; e += 256U) {
            const unsigned blk = e / (kIndexDim / 4U);
            float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (blk < nb) v = reinterpret_cast<const float4*>(pooled_keys + static_cast<std::uint64_t>(b0) * kIndexDim)[e];
            reinterpret_cast<float4*>(sk)[e] = v;
        }
        __syncthreads();
        for (unsigned i = 0; i < nb; ++i) {
            const float* k = sk + i * kIndexDim;
            float a[16];
#pragma unroll
            for (unsigned g = 0; g < 4U; ++g) {
                float v[8];  // v[0..3] = lanes 4g..4g+3, v[4..7] = lanes 4g+16..4g+19
#pragma unroll
                for (unsigned half = 0; half < 2U; ++half) {
                    const unsigned L = 4U * g + 16U * half;
                    const float4 k0 = *reinterpret_cast<const float4*>(k + L), k1 = *reinterpret_cast<const float4*>(k + L + 32U);
                    const float4 k2 = *reinterpret_cast<const float4*>(k + L + 64U), k3 = *reinterpret_cast<const float4*>(k + L + 96U);
                    const float kk0[4] = {k0.x, k0.y, k0.z, k0.w}, kk1[4] = {k1.x, k1.y, k1.z, k1.w};
                    const float kk2[4] = {k2.x, k2.y, k2.z, k2.w}, kk3[4] = {k3.x, k3.y, k3.z, k3.w};
#pragma unroll
                    for (unsigned c = 0; c < 4U; ++c) {
                        const float p0 = __fmul_rn(q[L + c], kk0[c]), p1 = __fmul_rn(q[L + c + 32U], kk1[c]);
                        const float p2 = __fmul_rn(q[L + c + 64U], kk2[c]), p3 = __fmul_rn(q[L + c + 96U], kk3[c]);
                        v[4U * half + c] = __fadd_rn(__fadd_rn(p0, p2), __fadd_rn(p1, p3));
                    }
                }
#pragma unroll
                for (unsigned c = 0; c < 4U; ++c) a[4U * g + c] = __fadd_rn(v[c], v[4U + c]);  // L + (L ^ 16)
            }
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l) a[l] = __fadd_rn(a[l], a[l + 8U]);  // ^8
#pragma unroll
            for (unsigned l = 0; l < 4U; ++l) a[l] = __fadd_rn(a[l], a[l + 4U]);  // ^4
#pragma unroll
            for (unsigned l = 0; l < 2U; ++l) a[l] = __fadd_rn(a[l], a[l + 2U]);  // ^2
            const float head_sum = __fadd_rn(a[0], a[1]);                         // ^1
            float total = 0.0F;
#pragma unroll
            for (unsigned hh = 0; hh < kIndexHeads; ++hh)
                total = __fadd_rn(total, fmaxf(__shfl_sync(kFull, head_sum, static_cast<int>((lane & ~3U) | hh)), 0.0F));
            if (h == 0U) ss[rr][i] = __fdiv_rn(total, scale);
        }
        __syncthreads();
        for (unsigned e = tid; e < kTreeRows * kTreeTile; e += 256U) {
            const unsigned row_in = e / kTreeTile, i = e % kTreeTile;
            const std::uint32_t row = r0 + row_in;
            if (row < rows && i < nb && b0 + i < limit_of(row))
                block_scores[static_cast<std::uint64_t>(row) * max_blocks + b0 + i] = ss[row_in][i];
        }
    }
}
__global__ __launch_bounds__(256, 1) void block_score_tree_kernel(const float* __restrict__ index_query, std::uint32_t rows,
    std::uint64_t first_position, const float* __restrict__ pooled_keys, std::uint32_t scored_blocks,
    float* __restrict__ block_scores, std::uint32_t max_blocks) {
    block_score_tree_body(index_query, rows, first_position, pooled_keys, scored_blocks, block_scores, max_blocks, blockIdx.x, blockIdx.y);
}
// select_rows_kernel<Rows> (H:3756-3766) -> select_blocks_parallel_body (H:1472-1557), 1024 threads per row: radix
// top-512 over the ordered key (11/11/10 bits, +0 == -0, NaN / -inf never counted), ties to the lowest block index,
// output blocks ascending (4 tokens each), the first `zeros` slots tokens 0..3 repeated when fewer than 512 finite
// scores (block 0 then set to -inf), then the partial tail tokens. Integer histogram atomics are order-free.
constexpr unsigned kSelectThreads = 1024, kSelectWaves = 32;
__device__ __forceinline__ std::uint32_t select_key(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
__device__ __forceinline__ std::uint32_t select_scan(std::uint32_t value, std::uint32_t* waves, std::uint32_t& total) {
    const auto lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up_sync(kFull, value, d); if (lane >= d) value += up; }
    if (lane == 31U) waves[wave] = value;
    __syncthreads();
    if (wave == 0) {
        auto sum = waves[lane];
        for (unsigned d = 1; d < 32U; d <<= 1U) { const auto up = __shfl_up_sync(kFull, sum, d); if (lane >= d) sum += up; }
        waves[lane] = sum;
    }
    __syncthreads();
    total = waves[kSelectWaves - 1U];
    value += wave ? waves[wave - 1U] : 0U;
    __syncthreads();
    return value;
}
__device__ __forceinline__ void select_kernel_body(float* __restrict__ all_scores, std::uint32_t max_blocks,
    std::uint64_t first_position, std::uint32_t* __restrict__ all_selected) {
    const auto r = blockIdx.x;
    const auto tid = threadIdx.x;
    const auto visible = static_cast<std::uint32_t>(first_position + r + 1U);
    const auto num_blocks = visible / kRatio;
    std::uint32_t* selected = all_selected + static_cast<std::uint64_t>(r) * kMaxSelected;
    if (num_blocks <= kTopK) {
        for (std::uint32_t index = tid; index < visible; index += kSelectThreads) selected[index] = index;
        return;
    }
    float* block_scores = all_scores + static_cast<std::uint64_t>(r) * max_blocks;
    __shared__ std::uint32_t histogram[2048];
    __shared__ std::uint32_t waves[kSelectWaves];
    __shared__ std::uint32_t found_digit, found_need, found_count;
    std::uint32_t prefix = 0, mask = 0, need = kTopK, zeros = 0;
    for (unsigned pass = 0; pass < 3U; ++pass) {
        const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
        const auto digits = pass == 2 ? 1024U : 2048U;
        for (auto i = tid; i < 2048U; i += kSelectThreads) histogram[i] = 0;
        __syncthreads();
        for (std::uint32_t base = 0; base < num_blocks; base += kSelectThreads) {
            const auto block = base + tid;
            const float value = block < num_blocks ? block_scores[block] : (-kInf);
            const auto key = select_key(value);
            const bool counted = value > (-kInf) && (key & mask) == prefix;
            if (counted) atomicAdd(&histogram[(key >> shift) & (digits - 1U)], 1U);
        }
        __syncthreads();
        const auto high = 2U * tid + 1U < digits ? histogram[2U * tid + 1U] : 0U;
        const auto low = 2U * tid < digits ? histogram[2U * tid] : 0U;
        std::uint32_t counted_total = 0;
        const auto inclusive = select_scan(high + low, waves, counted_total);
        const auto above = counted_total - inclusive;
        if (pass == 0 && counted_total <= need) { zeros = need - counted_total; need = counted_total; break; }
        if (above < need && need <= above + high) { found_digit = 2U * tid + 1U; found_need = need - above; found_count = high; }
        else if (above + high < need && need <= above + high + low) { found_digit = 2U * tid; found_need = need - above - high; found_count = low; }
        __syncthreads();
        prefix |= found_digit << shift; mask |= (digits - 1U) << shift; need = found_need;
        if (found_count == need) break;
    }
    for (auto token = tid; token < zeros * kRatio; token += kSelectThreads) selected[token] = token % kRatio;
    std::uint32_t greater_before = 0, equal_before = 0;
    for (std::uint32_t base = 0; base < num_blocks; base += 4U * kSelectThreads) {
        const auto first = base + 4U * tid;
        unsigned kind[4], greater = 0, equal = 0;
        for (unsigned j = 0; j < 4U; ++j) {
            const auto block = first + j;
            const float value = block < num_blocks ? block_scores[block] : (-kInf);
            const auto masked = select_key(value) & mask;
            kind[j] = !(value > (-kInf)) ? 0U : masked > prefix ? 1U : masked == prefix ? 2U : 0U;
            greater += kind[j] == 1U; equal += kind[j] == 2U;
        }
        std::uint32_t tile = 0;
        const auto packed = select_scan((greater << 16U) | equal, waves, tile);
        auto g = greater_before + (packed >> 16U) - greater, e = equal_before + (packed & 0xffffU) - equal;
        for (unsigned j = 0; j < 4U; ++j) {
            if (kind[j] == 1U || (kind[j] == 2U && e < need)) {
                const auto block = first + j, slot = zeros + g + min(e, need);
                for (std::uint32_t offset = 0; offset < kRatio; ++offset) selected[slot * kRatio + offset] = block * kRatio + offset;
                block_scores[block] = (-kInf);
            }
            g += kind[j] == 1U; e += kind[j] == 2U;
        }
        greater_before += tile >> 16U; equal_before += tile & 0xffffU;
    }
    if (zeros && tid == 0) block_scores[0] = (-kInf);
    const auto complete_tokens = num_blocks * kRatio, selected_complete = kTopK * kRatio;
    for (auto token = complete_tokens + tid; token < visible; token += kSelectThreads)
        selected[selected_complete + token - complete_tokens] = token;
}
__global__ __launch_bounds__(kSelectThreads) void select_kernel(float* __restrict__ all_scores, std::uint32_t max_blocks,
    std::uint64_t first_position, std::uint32_t* __restrict__ all_selected) {
    select_kernel_body(all_scores, max_blocks, first_position, all_selected);
}

#undef kInf

// Score fields of the private cohort descriptor and the two score wrappers,
// frozen from src/cuda/models/qwen/qwen_qsa.cu at 5b220d036.
inline constexpr unsigned kMaxSegments = 8;
struct SegmentMap {
    unsigned n = 0;
    unsigned first_row[kMaxSegments + 1] = {};
    std::uint64_t position[kMaxSegments] = {};
    const float* pooled[kMaxSegments] = {};
    unsigned scored[kMaxSegments] = {}, score_begin[kMaxSegments + 1] = {};
};

__device__ __forceinline__ const SegmentMap& map_of(const SegmentMap& m) { return m; }
__device__ __forceinline__ const SegmentMap& map_of(const SegmentMap* m) { return *m; }

template <class MapArg>
__global__ __launch_bounds__(256) void block_score_multi_kernel(const float* __restrict__ index_query, MapArg mapa,
    float* __restrict__ block_scores, std::uint32_t max_blocks) {
    const SegmentMap& map = map_of(mapa);
    const unsigned total = map.score_begin[map.n];
    for (unsigned wave = blockIdx.x * 8U + threadIdx.x / 32U; wave < total; wave += gridDim.x * 8U) {
        unsigned seg = 0;
        while (seg + 1U < map.n && wave >= map.score_begin[seg + 1U]) ++seg;
        const auto row = map.first_row[seg];
        block_score_body(index_query + static_cast<std::uint64_t>(row) * kIndexQWidth, map.first_row[seg + 1U] - row, map.position[seg],
            map.pooled[seg], map.scored[seg], block_scores + static_cast<std::uint64_t>(row) * max_blocks, max_blocks,
            wave - map.score_begin[seg]);
    }
}

template <class MapArg>
__global__ __launch_bounds__(256, 1) void block_score_tree_multi_kernel(const float* __restrict__ index_query, MapArg mapa,
    float* __restrict__ block_scores, std::uint32_t max_blocks) {
    const SegmentMap& map = map_of(mapa);
    const unsigned seg = blockIdx.z;
    const auto row = map.first_row[seg];
    block_score_tree_body(index_query + static_cast<std::uint64_t>(row) * kIndexQWidth, map.first_row[seg + 1U] - row, map.position[seg],
        map.pooled[seg], map.scored[seg], block_scores + static_cast<std::uint64_t>(row) * max_blocks, max_blocks, blockIdx.x, blockIdx.y);
}

} // namespace select_qwen_baseline
