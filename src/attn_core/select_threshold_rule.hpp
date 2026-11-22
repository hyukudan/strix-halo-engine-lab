#pragma once
// Host-neutral statement of the "Threshold" select schedule (select_threshold.hpp): the order key, the radix threshold
// (both schedules, with and without the whole-bin early end), the ordered compaction, the candidate-block pass and the
// two-level slice / merge form, as scalar code in the device code's pass order. Integer only. Used by the CPU test to
// show that every form selects the k largest keys, equal keys by the smaller position, in ascending position.
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix::attn_core::select::threshold::rule {

constexpr std::uint32_t order_key(float x) {
    const std::uint32_t b = std::bit_cast<std::uint32_t>(x);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}

struct Threshold {
    std::uint32_t thr = 0;   // keys above thr are taken
    std::uint32_t need = 0;  // then the first `need` keys equal to thr, in position order
};

// requires n > k > 0. WholeBinEnd = the default device schedule; false = the legacy schedule (four full passes).
template <bool WholeBinEnd>
constexpr Threshold radix_threshold(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k) {
    std::uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        std::uint32_t hist[256] = {};
        for (std::uint32_t i = 0; i < n; ++i)
            if ((keys[i] & mask) == prefix) ++hist[(keys[i] >> shift) & 255U];
        std::uint32_t cum = 0;
        unsigned d = 255;
        while (cum + hist[d] < rem) cum += hist[d--];   // the bin holding the rem-th largest remaining key
        const std::uint32_t low = prefix | (static_cast<std::uint32_t>(d) << shift);
        mask |= 255U << shift;
        if (WholeBinEnd && hist[d] == rem - cum && low != 0U) return {low - 1U, 0U};
        prefix = low;
        rem -= cum;
    }
    return {prefix, rem};
}

// f(i, selected, out) for every i < n, in position order; out = number of selected positions before i.
template <class F>
constexpr void ordered_select(const std::uint32_t* keys, std::uint32_t n, Threshold t, F&& f) {
    std::uint32_t eq_rank = 0, out = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const bool eq = keys[i] == t.thr;
        const bool sel = keys[i] > t.thr || (eq && eq_rank < t.need);
        f(i, sel, out);
        eq_rank += eq ? 1U : 0U;
        out += sel ? 1U : 0U;
    }
}

// Top-k positions of keys[0, n) in ascending order into idx; returns the count (n when n <= k).
template <bool WholeBinEnd>
constexpr std::uint32_t select_positions(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, std::uint32_t* idx) {
    if (n <= k) {
        for (std::uint32_t t = 0; t < n; ++t) idx[t] = t;
        return n;
    }
    ordered_select(keys, n, radix_threshold<WholeBinEnd>(keys, n, k), [&](std::uint32_t i, bool s, std::uint32_t o) {
        if (s) idx[o] = i;
    });
    return k;
}

// Candidate source: block maxima keys (the newest block's key is +inf) into bkeys; cand[b] = 1 for the selected blocks
// (every block above -inf when nb <= cand_blocks).
template <bool WholeBinEnd>
constexpr void candidate_blocks(const float* scores, std::uint32_t n, std::uint32_t block_size, std::uint32_t cand_blocks,
                                std::uint32_t* bkeys, std::uint8_t* cand) {
    if (n == 0U) return;
    const std::uint32_t nb = (n + block_size - 1) / block_size, last = (n - 1) / block_size;
    const float inf = std::numeric_limits<float>::infinity();
    for (std::uint32_t b = 0; b < nb; ++b) {
        float m = -inf;
        for (std::uint32_t t = b * block_size; t < (b + 1) * block_size && t < n; ++t) m = scores[t] > m ? scores[t] : m;
        bkeys[b] = order_key(b == last ? inf : m);
    }
    const std::uint32_t ninf_key = order_key(-inf);
    if (nb <= cand_blocks) {
        for (std::uint32_t b = 0; b < nb; ++b) cand[b] = bkeys[b] > ninf_key ? 1U : 0U;
        return;
    }
    ordered_select(bkeys, nb, radix_threshold<WholeBinEnd>(bkeys, nb, cand_blocks), [&](std::uint32_t b, bool s, std::uint32_t) {
        cand[b] = s && bkeys[b] > ninf_key ? 1U : 0U;
    });
}

// Two-level form: per slice of `slice` positions its top-k (keys, positions), padded with key 0 to k, then the top-k of
// the survivors in slice order. scratch_keys / scratch_pos hold ceil(n / slice) x k entries.
template <bool WholeBinEnd>
constexpr std::uint32_t select_two_level(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, std::uint32_t slice,
                                         std::uint32_t* scratch_keys, std::uint32_t* scratch_pos, std::uint32_t* idx) {
    if (n <= k) return select_positions<WholeBinEnd>(keys, n, k, idx);
    const std::uint32_t slices = (n + slice - 1) / slice;
    for (std::uint32_t s = 0; s < slices; ++s) {
        const std::uint32_t t0 = s * slice, m = n - t0 < slice ? n - t0 : slice;
        std::uint32_t* ok = scratch_keys + static_cast<std::size_t>(s) * k;
        std::uint32_t* op = scratch_pos + static_cast<std::size_t>(s) * k;
        if (m <= k) {
            for (std::uint32_t i = 0; i < k; ++i) { ok[i] = i < m ? keys[t0 + i] : 0U; op[i] = t0 + i; }
            continue;
        }
        ordered_select(keys + t0, m, radix_threshold<WholeBinEnd>(keys + t0, m, k), [&](std::uint32_t i, bool sel, std::uint32_t o) {
            if (sel) { ok[o] = keys[t0 + i]; op[o] = t0 + i; }
        });
    }
    const std::uint32_t total = slices * k;
    ordered_select(scratch_keys, total, radix_threshold<WholeBinEnd>(scratch_keys, total, k), [&](std::uint32_t i, bool s, std::uint32_t o) {
        if (s) idx[o] = scratch_pos[i];
    });
    return k;
}

} // namespace strix::attn_core::select::threshold::rule
