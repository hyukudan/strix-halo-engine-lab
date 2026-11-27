#pragma once

// Block-diffusion drafter (the DFlash family): the per-element row functors of the device forward (grouped two-tap
// convolution, embedding rows, residual, canonical top-k of the target logits, selector edge scores and greedy path).
// Home: shared (strix::spec::dflash::row_ops), host + device: the device forward (spec/dflash_resident.hpp) launches them
// through core::pointwise::elements, the host test (tests/spec_dflash_rows_tests.cpp) runs the same bodies on the CPU
// against the host reference's order.

#include "strix/attn/cache_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DFLASH_HD __host__ __device__
#else
#define STRIX_DFLASH_HD
#endif

namespace strix::spec::dflash::row_ops {

using weight::scalar_float::bf16_round;

inline constexpr unsigned kMaxTopK = 16U;      // selector candidates per row (registers of the top-k functors)
inline constexpr unsigned kTopChunks = 1024U;  // first top-k stage: interleaved chunks per row (coalesced reads)

STRIX_DFLASH_HD inline float decode_bf16(std::uint16_t v) {
    return attn::cache_rows::decode(v);
}

// Canonical order of the top-k: larger value first, then the lower id (-0 == +0; NaN never enters).
STRIX_DFLASH_HD inline bool better(float a, std::uint32_t ia, float b, std::uint32_t ib) {
    return a > b || (a == b && ia < ib);
}

template <unsigned K>
struct TopList {
    float value[K];
    std::uint32_t id[K];
    STRIX_DFLASH_HD void clear() {
        for (unsigned i = 0; i < K; ++i) {
            value[i] = -std::numeric_limits<float>::infinity();
            id[i] = 0xffffffffU;
        }
    }
    STRIX_DFLASH_HD void offer(float v, std::uint32_t token, unsigned k) {
        if (!better(v, token, value[k - 1U], id[k - 1U])) return;
        unsigned at = k - 1U;
        while (at > 0U && better(v, token, value[at - 1U], id[at - 1U])) {
            value[at] = value[at - 1U];
            id[at] = id[at - 1U];
            --at;
        }
        value[at] = v;
        id[at] = token;
    }
};

// Stage 1: thread (row, c) keeps the top-k of logits[row][c + i * kTopChunks] (coalesced across c).
struct TopChunks {
    const float* logits;   // [rows][vocab]
    unsigned vocab, k;
    float* values;         // [rows][kTopChunks][k]
    std::uint32_t* ids;
    unsigned* failure;     // raised on a non-finite logit
    STRIX_DFLASH_HD void operator()(std::size_t i) const {
        const auto row = static_cast<unsigned>(i / kTopChunks), chunk = static_cast<unsigned>(i % kTopChunks);
        TopList<kMaxTopK> list;
        list.clear();
        const float* in = logits + std::size_t{row} * vocab;
        bool bad = false;
        for (unsigned j = chunk; j < vocab; j += kTopChunks) {
            const float v = in[j];
            if (!(v - v == 0.0F)) { bad = true; continue; }
            list.offer(v == 0.0F ? 0.0F : v, j, k);
        }
        if (bad) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            atomicOr(failure, 1U);
#else
            *failure |= 1U;
#endif
        }
        const auto base = (std::size_t{row} * kTopChunks + chunk) * k;
        for (unsigned j = 0; j < k; ++j) {
            values[base + j] = list.value[j];
            ids[base + j] = list.id[j];
        }
    }
};

// Merge stage: thread (row, m) merges `fan` consecutive lists of k (row has `lists` lists) into one list of k.
struct MergeTop {
    const float* in_values;
    const std::uint32_t* in_ids;
    unsigned lists, fan, k;
    float* out_values;
    std::uint32_t* out_ids;
    STRIX_DFLASH_HD void operator()(std::size_t i) const {
        const auto outs = lists / fan;
        const auto row = static_cast<unsigned>(i / outs), m = static_cast<unsigned>(i % outs);
        TopList<kMaxTopK> list;
        list.clear();
        const auto first = (std::size_t{row} * lists + std::size_t{m} * fan) * k;
        for (unsigned j = 0; j < fan * k; ++j) list.offer(in_values[first + j], in_ids[first + j], k);
        const auto base = (std::size_t{row} * outs + m) * k;
        for (unsigned j = 0; j < k; ++j) {
            out_values[base + j] = list.value[j];
            out_ids[base + j] = list.id[j];
        }
    }
};

// Grouped two-tap convolution over the block rows: out_r[c] = sum_{t <= r, t < taps}
// (base[side][t][c] + delta_r[(side * taps + t) * groups + c / group]) * in_{r - t}[c].
struct GroupedConv {
    const float* input;          // [rows][width]
    const std::uint16_t* base;   // [2][taps][width] BF16
    const float* delta;          // [rows][2 * taps * groups]
    float* output;               // [rows][width], distinct from input
    unsigned width, taps, group, side;
    STRIX_DFLASH_HD void operator()(std::size_t i) const {
        const auto r = static_cast<unsigned>(i / width), c = static_cast<unsigned>(i % width);
        const auto groups = width / group;
        const auto stride = 2U * taps * groups;
        float acc = 0.0F;
        for (unsigned t = 0; t < taps && t <= r; ++t) {
            const float coef = decode_bf16(base[(std::size_t{side} * taps + t) * width + c]) +
                               delta[std::size_t{r} * stride + (side * taps + t) * groups + c / group];
            acc = acc + coef * input[std::size_t{r - t} * width + c];
        }
        output[i] = bf16_round(acc);
    }
};

// Embedding rows of the block: row 0 the anchor, the others the fill token (BF16 table rows widened).
struct EmbedBlock {
    const std::uint16_t* table;  // [vocab][width] BF16
    std::uint32_t anchor, fill;
    unsigned width;
    float* output;               // [rows][width]
    STRIX_DFLASH_HD void operator()(std::size_t i) const {
        const auto r = static_cast<unsigned>(i / width), c = static_cast<unsigned>(i % width);
        output[i] = decode_bf16(table[std::size_t{r == 0U ? anchor : fill} * width + c]);
    }
};

struct AddRound {
    float* x;
    const float* branch;
    STRIX_DFLASH_HD void operator()(std::size_t i) const { x[i] = bf16_round(x[i] + branch[i]); }
};

// Selector edge scores s[j][p][c] = cap(unary[j][c] + sum_q pred[prev][q] h[j][q] succ[cand[j][c]][q]),
// prev = the anchor at step 0 (p = 0 only), else cand[j - 1][p].
struct EdgeScores {
    const float* unary;               // [k][K]
    const std::uint32_t* cand;        // [k][K]
    const float* hidden;              // [k][rank]
    const std::uint16_t* pred;        // [vocab][rank] BF16
    const std::uint16_t* succ;        // [vocab][rank] BF16
    std::uint32_t anchor, vocab;
    unsigned K, rank;
    float softcap;
    bool cap_before;
    float* scores;                    // [k][K][K]
    STRIX_DFLASH_HD void operator()(std::size_t i) const {
        const auto c = static_cast<unsigned>(i % K), p = static_cast<unsigned>((i / K) % K), j = static_cast<unsigned>(i / (K * K));
        if (j == 0U && p != 0U) {
            scores[i] = -std::numeric_limits<float>::infinity();
            return;
        }
        const std::uint32_t prev = j == 0U ? (anchor < vocab ? anchor : vocab - 1U) : cand[std::size_t{j - 1U} * K + p];
        const std::uint32_t next = cand[std::size_t{j} * K + c];
        const std::uint16_t* a = pred + std::size_t{prev} * rank;
        const std::uint16_t* b = succ + std::size_t{next} * rank;
        const float* h = hidden + std::size_t{j} * rank;
        float dot = 0.0F;
        for (unsigned q = 0; q < rank; ++q) dot = dot + (decode_bf16(a[q]) * h[q]) * decode_bf16(b[q]);
        float u = unary[std::size_t{j} * K + c];
        if (softcap > 0.0F && cap_before) u = softcap * std::tanh(u / softcap);
        float e = u + dot;
        if (softcap > 0.0F && !cap_before) e = softcap * std::tanh(e / softcap);
        scores[i] = e;
    }
};

// Greedy path over the edge scores (one thread): the argmax per step from the chosen predecessor (ties: lower index).
// Without a selector (K = 1 candidates per row) it is the per-row argmax.
struct GreedyPath {
    const float* scores;           // [k][K][K]
    const std::uint32_t* cand;     // [k][K]
    unsigned steps, K;
    std::uint32_t* picks;          // [k] token ids
    STRIX_DFLASH_HD void operator()(std::size_t) const {
        unsigned prev = 0U;
        for (unsigned j = 0; j < steps; ++j) {
            const float* row = scores + (std::size_t{j} * K + (j == 0U ? 0U : prev)) * K;
            unsigned best = 0U;
            for (unsigned c = 1; c < K; ++c)
                if (row[c] > row[best]) best = c;
            prev = best;
            picks[j] = cand[std::size_t{j} * K + best];
        }
    }
};

}  // namespace strix::spec::dflash::row_ops
