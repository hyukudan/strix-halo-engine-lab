#pragma once
// F4 schedule form: chunked two-pass decode attention (scores + per-chunk maxima, then ordered partials merged in chunk
// order). This is a schedule, not a dtype trait: the key/value reader, the query preparation and the epilogue are
// policies, the reduction trees are the contract.
//
// Contract (bits depend on exactly this):
//   - a chunk is `kChunk` consecutive absolute positions; a block of kWaves waves walks j = wave, wave + kWaves, ... < kChunk;
//   - one score is the FP32 chain sum = sum + q[lane + 32 c] * k[lane + 32 c], c ascending from +0.0F (a separate
//     multiply and add, no fused multiply-add), then the policy's wave reduction (the engine's shfl_down tree);
//   - the chunk maximum scans the visible scores in ascending position order with `x > m ? x : m` from -infinity;
//   - ordered_sum adds terms in increasing index from +0.0F; Loads > 1 requests that many terms before adding them
//     (the same additions in the same order).
//   - Dim == 0 is the runtime-width score chain: lane partial sum = sum + q[i] * k[i], i = lane, lane + 32, ... < src.dim
//     (the same per-lane chain as Dim = src.dim when it is a multiple of 32; also valid for any other width);
//   - ordered_maximum scans chunk maxima in increasing index with `a > m ? a : m` from -infinity;
//   - chunk_denominator adds e[p - chunk_first] for p ascending in [begin, end) from +0.0F;
//   - chunk_numerators: per (column, head) chain acc = acc + e[h][p] * v_p[column] over present positions p ascending from
//     +0.0F, absent value rows skipped; Batch value rows are requested before their ordered additions (loads only).
// Policy `Source` (the only thing an engine supplies):
//   std::uint64_t first, last;            visible inclusive position range of the row
//   unsigned kv_offset;                   element offset of the key head inside a key row
//   unsigned dim;                         (Dim == 0 only) runtime head width
//   const float* key(std::uint64_t p);    key row of position p, nullptr when absent (score 0)
//   float reduce(float);                  wave reduction over the 32 lane partials, result on every lane
//   std::size_t index(std::uint64_t p);   score slot of position p
//   float finish(float);                  epilogue on lane 0 (scale, finite check) -> stored value
// Home: shared (attn_core): no model names; Dim is a template parameter.
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#define STRIX_CHUNK_DECODE_HD __host__ __device__
#define STRIX_CHUNK_DECODE_DEV __device__ __forceinline__
#define STRIX_CHUNK_DECODE_INLINE __host__ __device__ __forceinline__
#else
#include <bit>
#define STRIX_CHUNK_DECODE_HD
#define STRIX_CHUNK_DECODE_INLINE inline
#endif

namespace strix::attn_core::chunk_decode {

inline constexpr unsigned kChunk = 64U;
inline constexpr unsigned kWaves = 4U;
inline constexpr std::uint32_t kNegInfBits = 0xff800000U;

STRIX_CHUNK_DECODE_HD inline float neg_inf() {
#if defined(__HIPCC__) || defined(__HIP__)
    return __uint_as_float(kNegInfBits);
#else
    return std::bit_cast<float>(kNegInfBits);
#endif
}

// acc + a * b as a separate multiply and add (two roundings): the chain contract above is enforced here for every
// contraction setting that honours source pragmas (-ffp-contract=off / on / fast-honor-pragmas, the HIP default). Clang
// (HIP device and host): a scoped contraction pragma, the same instructions as a contraction-off translation unit. Other
// host compilers: the rounded product is materialised, so no fused multiply-add can form. An explicit
// -ffp-contract=fast on a HIP translation unit makes Clang ignore pragmas (and __fmul_rn / __fadd_rn are plain
// operations there): it is outside the binding contract, and an engine binds this form with a strict mode, as every
// exact engine target does. A register barrier would pin that setting too but changes the schedule of every bound kernel.
STRIX_CHUNK_DECODE_INLINE float mul_add_separate(float acc, float a, float b) {
#if defined(__clang__)
    {
#pragma clang fp contract(off)
        return acc + a * b;
    }
#else
    volatile float product = a * b;
    return acc + product;
#endif
}

// Ordered sum of base[c * step] for c = first..last, starting at +0 and adding in increasing c.
template <unsigned Loads>
STRIX_CHUNK_DECODE_HD inline float ordered_sum(const float* base, std::uint64_t first, std::uint64_t last, std::size_t step) {
    float sum = 0.0F;
    for (auto c = first; c <= last; c += Loads) {
#if defined(__HIPCC__) || defined(__HIP__)
        float x[Loads];
#pragma unroll
#else
        float x[Loads] = {};
#endif
        for (unsigned j = 0; j < Loads; ++j)
            if (c + j <= last)
                x[j] = base[(c + j) * step];
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
        for (unsigned j = 0; j < Loads; ++j)
            if (c + j <= last)
                sum = sum + x[j];
    }
    return sum;
}

// Maximum of the visible scores of one chunk, ascending position order. `score(j)` returns the score of chunk slot j.
template <class Score>
STRIX_CHUNK_DECODE_HD inline float chunk_maximum(std::uint64_t begin, std::uint64_t first, std::uint64_t last, Score&& score) {
    float m = neg_inf();
    for (unsigned j = 0; j < kChunk; ++j) {
        const auto p = begin + j;
        if (p >= first && p <= last) {
            const float x = score(j);
            m = x > m ? x : m;
        }
    }
    return m;
}

// Maximum of base[c * step] for c = first..last, increasing c, strict greater-than from -infinity.
STRIX_CHUNK_DECODE_HD inline float ordered_maximum(const float* base, std::uint64_t first, std::uint64_t last,
                                                   std::size_t step) {
    float m = neg_inf();
    for (auto c = first; c <= last; ++c) {
        const float a = base[c * step];
        m = a > m ? a : m;
    }
    return m;
}

// Denominator of one chunk partial: e[p - chunk_first] for p = begin..end-1, ascending from +0.
STRIX_CHUNK_DECODE_HD inline float chunk_denominator(const float* e, std::uint64_t chunk_first, std::uint64_t begin,
                                                     std::uint64_t end) {
    float denominator = 0.0F;
    for (auto p = begin; p < end; ++p)
        denominator = denominator + e[p - chunk_first];
    return denominator;
}

// Numerators of one chunk partial for Heads query heads that share each value load and Columns value columns
// (column c = column0 + c * column_step). e holds one row of coefficients per head, `pitch` floats apart, indexed by
// p - chunk_first. value(p) returns the value row of position p or nullptr (absent: the term is skipped). Every
// acc[c][h] starts at +0 and adds its present positions in increasing p; Batch only groups the loads.
template <unsigned Batch, unsigned Heads, unsigned Columns, class Value>
STRIX_CHUNK_DECODE_HD inline void chunk_numerators(const float* e, std::size_t pitch, std::uint64_t chunk_first,
                                                   std::uint64_t begin, std::uint64_t end, Value&& value,
                                                   unsigned column0, unsigned column_step,
                                                   float (&acc)[Columns][Heads]) {
    static_assert(Batch >= 1U && Batch <= 32U && Heads >= 1U && Columns >= 1U, "chunk_numerators shape");
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
    for (unsigned c = 0; c < Columns; ++c)
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
        for (unsigned h = 0; h < Heads; ++h)
            acc[c][h] = 0.0F;
    for (auto p = begin; p < end; p += Batch) {
#if defined(__HIPCC__) || defined(__HIP__)
        float v[Columns][Batch];
#else
        float v[Columns][Batch] = {};
#endif
        unsigned present = 0;
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
        for (unsigned j = 0; j < Batch; ++j)
            if (p + j < end) {
                const float* pointer = value(p + j);
                if (pointer) {
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
                    for (unsigned c = 0; c < Columns; ++c)
                        v[c][j] = pointer[column0 + c * column_step];
                    present |= 1U << j;
                }
            }
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
        for (unsigned j = 0; j < Batch; ++j)
            if (present & (1U << j))
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
                for (unsigned h = 0; h < Heads; ++h) {
                    const float coefficient = e[h * pitch + (p + j - chunk_first)];
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
                    for (unsigned c = 0; c < Columns; ++c)
                        acc[c][h] = mul_add_separate(acc[c][h], coefficient, v[c][j]);
                }
    }
}

// One head, one column: the numerator chain of feature `column`.
template <unsigned Batch, class Value>
STRIX_CHUNK_DECODE_HD inline float chunk_numerator(const float* e, std::uint64_t chunk_first, std::uint64_t begin,
                                                   std::uint64_t end, Value&& value, unsigned column) {
    float acc[1][1];
    chunk_numerators<Batch, 1U, 1U>(e, 0U, chunk_first, begin, end, value, column, 0U, acc);
    return acc[0][0];
}

// One lane's partial of one score (host-instantiable: the kernel body below calls it). `key` is the key row
// (nullptr when the row is absent: partial 0), `kv_offset` the element offset of the head inside it. Dim > 0: key[lane + 32 c] against query[lane + 32 c],
// c ascending, the key words loaded before the chain; Dim == 0: runtime width `dim`, i = lane, lane + 32, ... < dim.
template <unsigned Dim>
STRIX_CHUNK_DECODE_INLINE float score_lane(const float* query, const float* key, unsigned kv_offset, [[maybe_unused]] unsigned dim,
                                           unsigned lane) {
    float sum = 0.0F;
    if (key) {
        if constexpr (Dim == 0U) {
            for (unsigned i = lane; i < dim; i += 32U)
                sum = mul_add_separate(sum, query[i], key[kv_offset + i]);
        } else {
            float row[Dim / 32U];
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
            for (unsigned c = 0; c < Dim / 32U; ++c)
                row[c] = key[kv_offset + lane + 32U * c];
#if defined(__HIPCC__) || defined(__HIP__)
#pragma unroll
#endif
            for (unsigned c = 0; c < Dim / 32U; ++c)
                sum = mul_add_separate(sum, query[lane + 32U * c], row[c]);
        }
    }
    return sum;
}

#if defined(__HIPCC__) || defined(__HIP__)
// Body of one chunk-score block after the query row sits in LDS (`query[Dim]`) and the block has synchronised:
// scores of the visible positions into out[src.index(p)], then (Local) the maximum from `chunk_score` (LDS) or from `out`.
// Block of kWaves * 32 threads. `chunk_score` is read only when Local. Dim == 0: runtime width src.dim, `query` may be
// any readable row (global memory included).
template <unsigned Dim, bool Local, class Source>
STRIX_CHUNK_DECODE_DEV void score_chunk(const float* query, std::uint64_t begin, Source& src, float* out, float* chunk_score,
                                        float* maximum) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned j = wave; j < kChunk; j += kWaves) {
        const auto p = begin + j;
        if (p < src.first || p > src.last)
            continue;
        const auto* k = src.key(p);
        float sum;
        if constexpr (Dim == 0U)
            sum = score_lane<0U>(query, k, src.kv_offset, src.dim, lane);
        else
            sum = score_lane<Dim>(query, k, src.kv_offset, Dim, lane);
        sum = src.reduce(sum);
        if (lane == 0) {
            sum = src.finish(sum);
            out[src.index(p)] = sum;
            if constexpr (Local)
                chunk_score[j] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        *maximum = chunk_maximum(begin, src.first, src.last, [&](unsigned j) {
            return Local ? chunk_score[j] : out[src.index(begin + j)];
        });
}
#endif

}  // namespace strix::attn_core::chunk_decode
