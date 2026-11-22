#pragma once

#include "cuda/core/select_cuda.cuh"
#include "cuda/core/ptx.cuh"
#include "strix/weight/formats/scalar_float.hpp"

#include <cuda_bf16.h>
#include <mma.h>

namespace strix::cuda::core::select {
namespace detail {

inline constexpr unsigned score_threads = 128;
inline constexpr unsigned warp_mask = 0xffffffffU;
namespace codec = ::strix::weight::scalar_float;

__device__ __forceinline__ float sum_dot_terms(float (&terms)[32]) {
#pragma unroll
    for (unsigned stride = 16; stride; stride /= 2) {
#pragma unroll
        for (unsigned i = 0; i < stride; ++i) terms[i] = __fadd_rn(terms[i], terms[i + stride]);
    }
    return terms[0];
}

template <bool Strided, class Load>
__device__ __forceinline__ float tree_dot(const float* query, Load key) {
    float terms[32];
#pragma unroll
    for (unsigned i = 0; i < 32; ++i) {
        if constexpr (Strided) {
            const float a = __fmul_rn(query[i], key(i));
            const float b = __fmul_rn(query[i + 32], key(i + 32));
            const float c = __fmul_rn(query[i + 64], key(i + 64));
            const float d = __fmul_rn(query[i + 96], key(i + 96));
            terms[i] = __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
        } else {
            float sum = 0;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) sum = __fmaf_rn(query[4U * i + j], key(4U * i + j), sum);
            terms[i] = sum;
        }
    }
    return sum_dot_terms(terms);
}

__device__ __forceinline__ float fold_weighted(float sum, float dot, float weight) {
    return __fmaf_rn(__fmul_rn(weight, 0.17677669529663687F),
                     fmaxf(0.0F, __fmul_rn(dot, 0.088388347648318F)), sum);
}

__device__ __forceinline__ void store_score(const ScoreRow& row, unsigned index, float value) {
    row.scores[index] = value;
    if (row.ordered) row.ordered[index] = float_key(value);
    if (row.nonfinite && !isfinite(value)) atomicOr(row.nonfinite, 1U);
}

__device__ __forceinline__ int fp4_word(unsigned codes) {
    unsigned word = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const auto code = static_cast<std::uint8_t>((codes >> (4U * i)) & 15U);
        word |= (static_cast<unsigned>(codec::e2m1_doubled(code)) & 255U) << (8U * i);
    }
    return static_cast<int>(word);
}

__device__ inline float fp4_score(const ScoreRow& row, unsigned heads, unsigned index) {
    const auto* entry = row.packed_keys + std::size_t{index} * 68U;
    int keys[32];
    float scales[4];
#pragma unroll
    for (unsigned i = 0; i < 32; ++i) keys[i] = fp4_word(entry[2U * i] | (unsigned{entry[2U * i + 1U]} << 8U));
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) scales[i] = codec::e8m0(entry[64U + i]);
    const auto& query = *row.packed_query;
    float sum = 0;
    for (unsigned h = 0; h < heads; ++h) {
        float terms[4];
#pragma unroll
        for (unsigned g = 0; g < 4; ++g) {
            int dot = 0;
#pragma unroll
            for (unsigned j = 0; j < 8; ++j)
                dot = __dp4a(query.codes[h][g * 8U + j], keys[g * 8U + j], dot);
            terms[g] = __fmul_rn(__fmul_rn(__fmul_rn(static_cast<float>(dot), query.scales[h][g]), scales[g]), 0.25F);
        }
        const float dot = __fadd_rn(__fadd_rn(__fadd_rn(terms[0], terms[1]), terms[2]), terms[3]);
        sum = __fmaf_rn(fmaxf(dot, 0.0F), query.weights[h], sum);
    }
    return sum;
}

__device__ __forceinline__ void mma_fp8(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Same MMA fragments and four independent head folds as the incumbent's
// E4M3 score form. Input conversions and their scales are already staged.
__device__ inline void fp8_scores(const ScoreRow& row, unsigned base) {
    __shared__ __align__(16) unsigned keys[16][32];
    const unsigned lane = threadIdx.x;
    for (unsigned p = 0; p < 16; ++p) {
        keys[p][lane] = base + p < row.visible
            ? reinterpret_cast<const unsigned*>(row.packed_keys)[std::size_t{base + p} * 32U + lane] : 0U;
    }
    __syncwarp();
    unsigned a[4][4];
    for (unsigned k = 0; k < 4; ++k)
        ptx::ldsm_x4(a[k], &keys[lane & 15U][8U * k + 4U * (lane >> 4U)]);
    const unsigned q_col = lane & 3U, group = lane >> 2U;
    float lo = 0.0F, hi = 0.0F;
    for (unsigned h = 0; h < 32; h += 8) {
        const auto* q = reinterpret_cast<const unsigned*>(row.fp8_query->codes[h + group]);
        float c[4] = {};
        for (unsigned k = 0; k < 4; ++k) mma_fp8(c, a[k], q[8U * k + q_col], q[8U * k + 4U + q_col]);
        const float w0 = row.fp8_query->weights[h + 2U * q_col];
        const float w1 = row.fp8_query->weights[h + 2U * q_col + 1U];
        lo = __fmaf_rn(fmaxf(c[0], 0.0F), w0, lo);
        lo = __fmaf_rn(fmaxf(c[1], 0.0F), w1, lo);
        hi = __fmaf_rn(fmaxf(c[2], 0.0F), w0, hi);
        hi = __fmaf_rn(fmaxf(c[3], 0.0F), w1, hi);
    }
    lo = __fadd_rn(lo, __shfl_xor_sync(warp_mask, lo, 1));
    lo = __fadd_rn(lo, __shfl_xor_sync(warp_mask, lo, 2));
    hi = __fadd_rn(hi, __shfl_xor_sync(warp_mask, hi, 1));
    hi = __fadd_rn(hi, __shfl_xor_sync(warp_mask, hi, 2));
    const unsigned first = base + group, second = first + 8U;
    if (q_col == 0) {
        if (first < row.visible) store_score(row, first, __fmul_rn(lo, row.key_scales[first]));
        if (second < row.visible) store_score(row, second, __fmul_rn(hi, row.key_scales[second]));
    }
}

// The BF16 forms keep the incumbent's WMMA update order in each K=16 tile.
template <bool Residual>
__device__ void bf16_scores(const ScoreRow& row, unsigned base) {
    namespace w = nvcuda::wmma;
    using BF = __nv_bfloat16;
    __shared__ __align__(32) BF query[256], key[256], qlow[256], klow[256];
    __shared__ __align__(32) float dots[32 * 16];
    const unsigned lane = threadIdx.x;
    for (unsigned h = 0; h < 32; h += 16) {
        w::fragment<w::accumulator, 16, 16, 16, float> accum;
        w::fill_fragment(accum, 0.0F);
        for (unsigned k = 0; k < 128; k += 16) {
            for (unsigned i = lane; i < 256; i += 32) {
                const float q = row.query[(h + i / 16U) * 128U + k + i % 16U];
                const unsigned p = base + i % 16U;
                const float v = p < row.visible ? row.keys[std::size_t{p} * 128U + k + i / 16U] : 0.0F;
                query[i] = __float2bfloat16_rn(q);
                key[i] = __float2bfloat16_rn(v);
                if constexpr (Residual) {
                    qlow[i] = __float2bfloat16_rn(__fsub_rn(q, __bfloat162float(query[i])));
                    klow[i] = __float2bfloat16_rn(__fsub_rn(v, __bfloat162float(key[i])));
                }
            }
            __syncwarp();
            w::fragment<w::matrix_a, 16, 16, 16, BF, w::row_major> a;
            w::fragment<w::matrix_b, 16, 16, 16, BF, w::row_major> b;
            w::load_matrix_sync(a, query, 16);
            w::load_matrix_sync(b, key, 16);
            if constexpr (Residual) {
                decltype(a) al;
                decltype(b) bl;
                w::load_matrix_sync(al, qlow, 16);
                w::load_matrix_sync(bl, klow, 16);
                w::mma_sync(accum, al, b, accum);
                w::mma_sync(accum, a, bl, accum);
            }
            w::mma_sync(accum, a, b, accum);
            __syncwarp();
        }
        w::store_matrix_sync(dots + h * 16, accum, 16, w::mem_row_major);
    }
    __syncwarp();
    if (lane >= 16 || base + lane >= row.visible) return;
    float sum = 0;
    for (unsigned h = 0; h < 32; ++h) sum = fold_weighted(sum, dots[h * 16 + lane], row.weights[h]);
    store_score(row, base + lane, sum);
}

template <unsigned Lanes>
__device__ __forceinline__ float fold_block_max(float best) {
    for (unsigned stride = Lanes / 2U; stride; stride /= 2U) {
        const float value = __shfl_xor_sync(warp_mask, best, static_cast<int>(stride), Lanes);
        best = best < value ? value : best;
    }
    return best;
}

template <unsigned Lanes>
__device__ void block_score(const ScoreRow& row, ScoreSpec spec) {
    const unsigned blocks = row.visible == 0 ? 0 : 1U + (row.visible - 1U) / spec.block_width;
    const unsigned b = blockIdx.x / spec.heads, h = blockIdx.x % spec.heads;
    if (blocks <= spec.top_k || b >= blocks) return;
    const unsigned start = b * spec.block_width;
    const unsigned end = start + min(row.visible - start, spec.block_width);
    const unsigned lane = threadIdx.x;
    float best = __uint_as_float(0xff800000U);
    for (unsigned i = start + lane; lane < Lanes && i < end; i += Lanes) {
        const auto* key = row.bf16_keys + std::size_t{i} * 128U;
        const float value = __fmul_rn(tree_dot<false>(row.query + h * 128U,
            [&](unsigned j) { return __uint_as_float(unsigned{key[j]} << 16U); }), 0.08838834764831845F);
        best = best < value ? value : best;
    }
    best = fold_block_max<Lanes>(best);
    if (lane == 0) row.scores[std::size_t{h} * (gridDim.x / spec.heads) + b] = best;
}

template <ScoreForm Form>
__global__ void score_kernel(const ScoreRow* rows, ScoreSpec spec) {
    const auto row = rows[blockIdx.y];
    if constexpr (Form == ScoreForm::block_max || Form == ScoreForm::block_max_tiled) {
        block_score<Form == ScoreForm::block_max ? 32U : 8U>(row, spec);
    } else if constexpr (Form == ScoreForm::weighted_fp8) {
        const unsigned base = blockIdx.x * 16U;
        if (row.visible > spec.top_k && base < row.visible) fp8_scores(row, base);
    } else if constexpr (Form == ScoreForm::weighted_bf16 || Form == ScoreForm::weighted_bf16x3) {
        const unsigned base = blockIdx.x * 16U;
        if (row.visible > spec.top_k && base < row.visible)
            bf16_scores<Form == ScoreForm::weighted_bf16x3>(row, base);
    } else {
        const unsigned i = blockIdx.x * score_threads + threadIdx.x;
        if (i >= row.visible || row.visible <= spec.top_k) return;
        float sum = 0.0F;
        if constexpr (Form == ScoreForm::weighted_fp4) {
            sum = fp4_score(row, spec.heads, i);
        } else {
            const auto* key = row.keys + std::size_t{i} * 128U;
            for (unsigned h = 0; h < spec.heads; ++h) {
                const auto* query = row.query + h * 128U;
                if constexpr (Form == ScoreForm::relu_tree) {
                    const float dot = tree_dot<true>(query, [&](unsigned j) { return key[j]; });
                    sum = __fadd_rn(sum, fmaxf(dot, 0.0F));
                } else {
                    float dot = 0;
                    for (unsigned j = 0; j < 128; ++j) dot = __fmaf_rn(query[j], key[j], dot);
                    sum = fold_weighted(sum, dot, row.weights[h]);
                }
            }
            if constexpr (Form == ScoreForm::relu_tree) sum = __fdiv_rn(sum, __fsqrt_rn(128.0F));
        }
        store_score(row, i, sum);
    }
}

inline constexpr unsigned vectors_per_score_row = strided_score_width / 4U;
inline constexpr unsigned shared_key_stride = vectors_per_score_row + 1U;
inline constexpr unsigned tiled_score_threads = strided_score_tile_rows * strided_score_tile_lanes;
inline constexpr int tiled_score_smem = (strided_score_width * shared_key_stride +
    strided_score_tile_rows * vectors_per_score_row) * sizeof(float4);
inline constexpr float block_score_scale = 0.08838834764831845F;

__device__ __forceinline__ float widen_score_key(std::uint16_t value) {
    return __uint_as_float(unsigned{value} << 16U);
}

__device__ __forceinline__ bool score_visible_block(std::size_t start, std::size_t visible, unsigned top_k) {
    return start < visible && (visible + strided_score_width - 1U) / strided_score_width > top_k;
}

// Packed warp schedule from MiniMax RTX index_scores_kernel. Only cache
// addressing changes. Each warp stages one query and owns one output score.
__device__ __forceinline__ void score_strided_warps(const StridedScoreArgs& args, unsigned heads, unsigned top_k) {
    __shared__ float query[strided_score_warps][strided_score_width];
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const auto item = std::size_t{blockIdx.x} * strided_score_warps + warp;
    const auto row_head = item / args.max_groups;
    const auto row = row_head / heads;
    if (row >= args.row_count) return;
    const unsigned sequence = args.row_sequence[row];
    const auto origin = args.sequences[sequence];
    const auto visible = row - origin.first_row + origin.first_position + 1U;
    for (unsigned i = lane; i < strided_score_width; i += 32U)
        query[warp][i] = args.query[row_head * strided_score_width + i];
    __syncwarp();
    const auto start = (item % args.max_groups) * strided_score_width;
    if (!score_visible_block(start, visible, top_k)) return;

    const auto* keys = args.cache.base + args.cache.offset(sequence, args.layer, 0);
    float best = __uint_as_float(0xff800000U);
    for (auto pos = start + lane; pos < start + strided_score_width && pos < visible; pos += 32U) {
        const auto* key = keys + pos * args.cache.row_stride;
        const float value = __fmul_rn(tree_dot<false>(query[warp],
            [&](unsigned i) { return widen_score_key(key[i]); }), block_score_scale);
        best = best < value ? value : best;
    }
    best = fold_block_max<32>(best);
    if (lane == 0) args.scores[item] = best;
}

// Shared-key schedule from MiniMax RTX index_scores_tiled_kernel. The CTA
// owns one key block and up to 32 consecutive query rows from one sequence.
__device__ __forceinline__ void score_strided_shared_keys(const StridedScoreArgs& args, unsigned heads, unsigned top_k) {
    extern __shared__ float4 shared_score_vectors[];
    auto* keys = shared_score_vectors;
    auto* queries = keys + strided_score_width * shared_key_stride;
    const auto tile = args.tiles[blockIdx.x];
    const unsigned sequence = args.row_sequence[tile.first_row], head = blockIdx.y;
    const auto origin = args.sequences[sequence];
    const auto first_visible = std::size_t{tile.first_row} - origin.first_row + origin.first_position + 1U;
    const auto last_visible = first_visible + tile.rows - 1U;
    const auto start = std::size_t{blockIdx.z} * strided_score_width;
    if (!score_visible_block(start, last_visible, top_k)) return;

    const unsigned thread = threadIdx.x;
    const auto* cache = args.cache.base + args.cache.offset(sequence, args.layer, 0);
    const auto zero = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (unsigned i = thread; i < strided_score_width * vectors_per_score_row; i += tiled_score_threads) {
        const unsigned key = i / vectors_per_score_row, col = i % vectors_per_score_row;
        const auto pos = start + key;
        auto value = zero;
        if (pos < last_visible) {
            const auto* src = cache + pos * args.cache.row_stride + 4U * col;
            value = make_float4(widen_score_key(src[0]), widen_score_key(src[1]),
                                widen_score_key(src[2]), widen_score_key(src[3]));
        }
        keys[key * shared_key_stride + col] = value;
    }
    for (unsigned i = thread; i < strided_score_tile_rows * vectors_per_score_row; i += tiled_score_threads) {
        const unsigned r = i / vectors_per_score_row, col = i % vectors_per_score_row;
        const auto row_head = (std::size_t{tile.first_row} + r) * heads + head;
        queries[i] = r < tile.rows
            ? reinterpret_cast<const float4*>(args.query)[row_head * vectors_per_score_row + col] : zero;
    }
    __syncthreads();

    const unsigned r = thread / strided_score_tile_lanes, lane = thread % strided_score_tile_lanes;
    const auto visible = first_visible + r;
    const bool live = r < tile.rows && score_visible_block(start, visible, top_k);
    float best = __uint_as_float(0xff800000U);
    if (live) {
        for (unsigned key = lane; key < strided_score_width; key += strided_score_tile_lanes) {
            if (start + key >= visible) continue;
            float terms[32];
#pragma unroll
            for (unsigned col = 0; col < vectors_per_score_row; ++col) {
                const float4 q = queries[r * vectors_per_score_row + col], k = keys[key * shared_key_stride + col];
                float dot = __fmaf_rn(q.x, k.x, 0.0F);
                dot = __fmaf_rn(q.y, k.y, dot);
                dot = __fmaf_rn(q.z, k.z, dot);
                terms[col] = __fmaf_rn(q.w, k.w, dot);
            }
            const float value = __fmul_rn(sum_dot_terms(terms), block_score_scale);
            best = best < value ? value : best;
        }
    }
    best = fold_block_max<strided_score_tile_lanes>(best);
    if (lane == 0 && live)
        args.scores[((std::size_t{tile.first_row} + r) * heads + head) * args.max_groups + blockIdx.z] = best;
}

template <ScoreForm Form>
__global__ __launch_bounds__(Form == ScoreForm::block_max_tiled ? tiled_score_threads : 32U * strided_score_warps)
void score_strided_kernel(StridedScoreArgs args, unsigned heads, unsigned top_k) {
    if constexpr (Form == ScoreForm::block_max_tiled) score_strided_shared_keys(args, heads, top_k);
    else score_strided_warps(args, heads, top_k);
}

} // namespace detail

// max_groups is the score stride per head for block-max, otherwise the upper
// bound on visible groups. It is independent of the number visible in a row.
template <ScoreForm Form>
cudaError_t enqueue_scores(const ScoreRow* rows, unsigned row_count, unsigned max_groups, ScoreSpec spec, cudaStream_t stream) {
    if (row_count == 0 || max_groups == 0) return cudaSuccess;
    if (!rows || row_count > 65535U || spec.dim != 128 || spec.heads == 0 || spec.block_width == 0)
        return cudaErrorInvalidValue;
    constexpr bool block_max = Form == ScoreForm::block_max || Form == ScoreForm::block_max_tiled;
    constexpr bool tensor = Form == ScoreForm::weighted_bf16 || Form == ScoreForm::weighted_bf16x3 ||
                            Form == ScoreForm::weighted_fp8;
    if constexpr (Form == ScoreForm::weighted_fp4 || tensor) if (spec.heads != 32) return cudaErrorInvalidValue;
    if constexpr (block_max) if (max_groups > 2147483647U / spec.heads) return cudaErrorInvalidValue;
    constexpr unsigned group_tile = tensor ? 16U : detail::score_threads;
    const unsigned x = block_max ? max_groups * spec.heads : 1U + (max_groups - 1U) / group_tile;
    detail::score_kernel<Form><<<dim3(x, row_count), block_max || tensor ? 32U : detail::score_threads, 0, stream>>>(rows, spec);
    return cudaGetLastError();
}

// Set the dynamic shared-memory limit on the current device before launching
// the shared-key form, outside any per-layer loop or graph capture.
inline cudaError_t prepare_strided_score_tiles() {
    return cudaFuncSetAttribute(detail::score_strided_kernel<ScoreForm::block_max_tiled>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, detail::tiled_score_smem);
}

// StridedScoreArgs selects the cache accessor. The block_max_tiled trait
// uses one shared key block for 32 query rows with this accessor.
template <ScoreForm Form>
cudaError_t enqueue_scores(StridedScoreArgs args, ScoreSpec spec, cudaStream_t stream) {
    detail::StridedScoreGrid grid;
    if (!detail::pack_strided_score_grid<Form>(args, spec, grid)) return cudaErrorInvalidValue;
    if (grid.x == 0) return cudaSuccess;
    const dim3 blocks(grid.x, grid.y, grid.z);
    constexpr int smem = Form == ScoreForm::block_max_tiled ? detail::tiled_score_smem : 0;
    detail::score_strided_kernel<Form><<<blocks, grid.threads, smem, stream>>>(args, spec.heads, spec.top_k);
    return cudaGetLastError();
}

} // namespace strix::cuda::core::select
