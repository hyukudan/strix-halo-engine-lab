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
// Policy `Source` (the only thing an engine supplies):
//   std::uint64_t first, last;            visible inclusive position range of the row
//   unsigned kv_offset;                   element offset of the key head inside a key row
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
#else
#include <bit>
#define STRIX_CHUNK_DECODE_HD
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

#if defined(__HIPCC__) || defined(__HIP__)
// Body of one chunk-score block after the query row sits in LDS (`query[Dim]`) and the block has synchronised:
// scores of the visible positions into out[src.index(p)], then (Local) the maximum from `chunk_score` (LDS) or from `out`.
// Block of kWaves * 32 threads. `chunk_score` is read only when Local.
template <unsigned Dim, bool Local, class Source>
STRIX_CHUNK_DECODE_DEV void score_chunk(const float* query, std::uint64_t begin, Source& src, float* out, float* chunk_score,
                                        float* maximum) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned j = wave; j < kChunk; j += kWaves) {
        const auto p = begin + j;
        if (p < src.first || p > src.last)
            continue;
        const auto* k = src.key(p);
        float sum = 0.0F;
        if (k) {
            float key[Dim / 32U];
#pragma unroll
            for (unsigned c = 0; c < Dim / 32U; ++c)
                key[c] = k[src.kv_offset + lane + 32U * c];
#pragma unroll
            for (unsigned c = 0; c < Dim / 32U; ++c)
                sum = sum + query[lane + 32U * c] * key[c];
        }
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
