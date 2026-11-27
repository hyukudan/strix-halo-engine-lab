#pragma once
// Shared EXACT per-row preparation and list builder of an FP4 (E2M1) MLA selection indexer, the device statement of
// attn_core::select_indexed_rows_reference for the checkpoint contract. Per row at position p:
//   key   raw wk(x) -> layer norm (sequential sums) -> BF16 -> rope on the leading split halves (i, i + 32) ->
//         BF16 -> Hadamard -> BF16 -> E2M1 blocks of 32 with a power-of-two scale, stored as the dequantised FP32 row;
//   query raw wq_b(q_lora) per head: the same rotation, then quantise + unpack;
//   head weights  raw weights_proj(x) * scale; the prefill form keeps the kPrefillHeads heads of largest
//         (gate logit + bias) (ties: lower head) with weight logit * w, the decode form keeps all heads;
//   list  top kTopK of the scored part (shared radix select, ratio 1), ascending, then the window rows.
// The scan between them is the shared group_score_scan score_kernel (Key = float).
//
// Shape S (an engine adapter, all static constexpr unsigned): kDim (128: four values per lane of one wave),
// kHeads, kPrefillHeads, kHidden, kRopePairs (32: lanes 0..31 rotate (lane, lane + 32)).
// Geometry G: kTopK, kListMax, history(p) (rows below the window), taken(p), list_length(p) (constexpr, host = device).
#include "attn_core/group_score_scan.hpp"
#include "attn_core/select_radix3.hpp"
#include "core/hadamard.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::fp4_index {

__device__ __forceinline__ float bf16r(float x) { return weight::scalar_float::bf16_round(x); }

// Rotary on the leading split halves: pair (lane, lane + 32) is v[0], v[1]; v[2], v[3] pass.
__device__ __forceinline__ void rotate_pair(float (&v)[4], float c, float s) {
    const float a = v[0], b = v[1];
    v[0] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    v[1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}

// BF16, Hadamard (ascending strides), BF16, then the dequantised E2M1 block round trip; block j = v[j] over the lanes.
__device__ __forceinline__ void hadamard_quantise(float (&v)[4], unsigned lane) {
    namespace sf = weight::scalar_float;
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) v[j] = bf16r(v[j]);
    core::hadamard128(v, lane);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) v[j] = bf16r(v[j]);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        float amax = fabsf(v[j]);
#pragma unroll
        for (unsigned o = 16U; o > 0U; o >>= 1U) amax = fmaxf(amax, __shfl_xor(amax, static_cast<int>(o), 32));
        amax = fmaxf(amax, 7.05296657e-38F);
        const int exponent = sf::log2_ceil_positive_normal(__fmul_rn(amax, 1.0F / 6.0F));
        const float scale = sf::pow2f(exponent);
        const float q = __fdiv_rn(v[j], scale);
        const std::uint8_t code = sf::e2m1_encode(fminf(fmaxf(q, -6.0F), 6.0F));
        v[j] = __fmul_rn(sf::e2m1(code), sf::e8m0(static_cast<std::uint8_t>(exponent + 127)));
    }
}

// ---- key: [rows] raw wk outputs -> dequantised FP32 key rows at positions first .. first + rows. grid rows, block 32 --
template <class S>
__global__ __launch_bounds__(32) void key_kernel(const float* raw, const float* gain, const float* bias, float epsilon,
                                                 const float* cos, const float* sin, unsigned first, float* keys) {
    static_assert(S::kDim == 128U && S::kRopePairs == 32U, "one wave, four values per lane, rope on (lane, lane + 32)");
    constexpr unsigned kDim = S::kDim, kRopePairs = S::kRopePairs;
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    __shared__ float x[kDim];
    for (unsigned j = 0; j < 4U; ++j) x[lane + 32U * j] = raw[std::size_t{row} * kDim + lane + 32U * j];
    __syncwarp();
    float mean = 0.0F, inv = 0.0F;
    if (lane == 0U) {  // the host's sequential sums
        float sum = 0.0F;
        for (unsigned i = 0; i < kDim; ++i) sum = __fadd_rn(sum, x[i]);
        mean = __fdiv_rn(sum, static_cast<float>(kDim));
        float square = 0.0F;
        for (unsigned i = 0; i < kDim; ++i) {
            const float d = __fsub_rn(x[i], mean);
            square = __fmaf_rn(d, d, square);
        }
        inv = __fdiv_rn(1.0F, __fsqrt_rn(__fadd_rn(__fdiv_rn(square, static_cast<float>(kDim)), epsilon)));
    }
    mean = __shfl(mean, 0, 32);
    inv = __shfl(inv, 0, 32);
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        const unsigned i = lane + 32U * j;
        v[j] = bf16r(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x[i], mean), inv), gain[i]), bias[i]));
    }
    const std::size_t at = std::size_t{first + row} * kRopePairs + lane;
    rotate_pair(v, cos[at], sin[at]);
    hadamard_quantise(v, lane);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) keys[std::size_t{first + row} * kDim + lane + 32U * j] = v[j];
}

// ---- query: [rows][kHeads][kDim] raw wq_b outputs -> dequantised queries. grid (rows, kHeads), block 32 --------------
template <class S>
__global__ __launch_bounds__(32) void query_kernel(const float* raw, const float* cos, const float* sin, unsigned first,
                                                   float* queries) {
    static_assert(S::kDim == 128U && S::kRopePairs == 32U, "one wave, four values per lane, rope on (lane, lane + 32)");
    constexpr unsigned kDim = S::kDim, kRopePairs = S::kRopePairs;
    const unsigned row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const std::size_t base = (std::size_t{row} * S::kHeads + head) * kDim;
    float v[4];
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) v[j] = raw[base + lane + 32U * j];
    const std::size_t at = std::size_t{first + row} * kRopePairs + lane;
    rotate_pair(v, cos[at], sin[at]);
    hadamard_quantise(v, lane);
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) queries[base + lane + 32U * j] = v[j];
}

// ---- head weights. grid rows, block kHeads (one thread per head) ---------------------------------------------------
// raw: [rows][kHeads] weights_proj outputs. Prefill form: gate logit h = fma chain over the kHidden inputs ascending
// (gate_t is the gate matrix transposed to [input][head]); rank = logit + bias; head h is kept iff fewer than
// kPrefillHeads heads rank above it (equal rank: lower head first); kept weight = logit * (raw * scale).
template <class S>
__global__ __launch_bounds__(S::kHeads) void weights_kernel(const float* x, const float* raw, const float* gate_t,
                                                            const float* bias, float scale, bool prefill, float* out) {
    constexpr unsigned kHeads = S::kHeads, kHidden = S::kHidden;
    const unsigned row = blockIdx.x, h = threadIdx.x;
    const float w = __fmul_rn(raw[row * kHeads + h], scale);
    if (!prefill) {
        out[row * kHeads + h] = w;
        return;
    }
    __shared__ float xs[kHidden];
    __shared__ float rank[kHeads];
    for (unsigned i = h; i < kHidden; i += kHeads) xs[i] = x[std::size_t{row} * kHidden + i];
    __syncthreads();
    float logit = 0.0F;
    for (unsigned i = 0; i < kHidden; ++i) logit = __fmaf_rn(gate_t[std::size_t{i} * kHeads + h], xs[i], logit);
    const float mine = __fadd_rn(logit, bias[h]);
    rank[h] = mine;
    __syncthreads();
    unsigned above = 0;
    for (unsigned g = 0; g < kHeads; ++g) above += (rank[g] > mine) || (rank[g] == mine && g < h);
    out[row * kHeads + h] = above < S::kPrefillHeads ? __fmul_rn(logit, w) : 0.0F;
}

// ---- the exact scan (shared score_kernel) over the rows below the window ---------------------------------------------
struct ScanRows {
    unsigned rows, first;
    const float* keys;
};
template <class S, class G> struct Scan {
    using Rows = ScanRows;
    using Key = float;
    static constexpr unsigned kHeads = S::kHeads, kDim = S::kDim;
    __device__ __forceinline__ static unsigned rows(const Rows& d) { return d.rows; }
    __device__ __forceinline__ static const float* keys(const Rows& d, unsigned) { return d.keys; }
    __device__ __forceinline__ static unsigned pools(const Rows& d, unsigned r) { return G::history(d.first + r); }
    __device__ __forceinline__ static float weight(float raw) { return raw; }
    __device__ __forceinline__ static float scaled(float dot) { return dot; }
    __device__ __forceinline__ static std::uint32_t key_of(float s) { return __float_as_uint(s); }  // read back as float
};

// ---- the list: radix select (shared select_blocks, ratio 1) + the window. grid rows, block 1024 ----------------------
struct ListRows {
    unsigned first, stride;  // first position, score row stride
    float* scores;
    std::uint32_t* ids;      // [rows][G::kListMax]
    std::uint32_t* counts;   // [rows]
};
template <class G> __global__ __launch_bounds__(1024) void list_kernel(ListRows d) {
    const unsigned r = blockIdx.x, p = d.first + r, tid = threadIdx.x;
    const unsigned hist = G::history(p), take = G::taken(p);
    std::uint32_t* ids = d.ids + std::size_t{r} * G::kListMax;
    if (hist > 0U)
        select::radix3::select_blocks<1024U>(d.scores + std::size_t{r} * d.stride, hist, G::kTopK, 1U, hist, ids,
                                             select::radix3::NoControl{});
    for (unsigned j = tid; j < p + 1U - hist; j += 1024U) ids[take + j] = hist + j;
    if (tid == 0U) d.counts[r] = G::list_length(p);
}

// The selected list of every row of the window for attend (dense_decode selector: bound(row), at(row, slot)).
template <class G> struct IdList {
    const std::uint32_t* ids;
    const std::uint32_t* counts;
    __device__ unsigned bound(unsigned row) const { return counts[row] - 1U; }
    __device__ unsigned at(unsigned row, unsigned slot) const { return ids[std::size_t{row} * G::kListMax + slot]; }
};

// Launch the scan and the list for rows at positions first .. first + rows (the keys of those rows are stored).
template <class S, class G>
inline hipError_t select_rows(const float* keys, const float* queries, const float* weights, unsigned first, unsigned rows,
                              unsigned stride, float* scores, std::uint32_t* hist0, std::uint32_t* ids,
                              std::uint32_t* counts, hipStream_t stream) {
    const unsigned widest = G::history(first + rows - 1U);
    if (widest > 0U) {
        const ScanRows d{rows, first, keys};
        group_scan::score_kernel<Scan<S, G>><<<dim3((widest + group_scan::kThreads - 1U) / group_scan::kThreads, rows),
                                               group_scan::kThreads, 0, stream>>>(
            d, queries, weights, reinterpret_cast<std::uint32_t*>(scores), stride, hist0);
        if (const hipError_t e = hipGetLastError(); e != hipSuccess) return e;
    }
    list_kernel<G><<<rows, 1024, 0, stream>>>(ListRows{first, stride, scores, ids, counts});
    return hipGetLastError();
}

}  // namespace strix::attn_core::fp4_index
