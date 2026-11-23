#pragma once
// Host-neutral statement of the "Radix3" select schedule (select_radix3.hpp): the zero-folding order key, the three radix
// passes (11, 11, 10 bits) with the early end on a whole bin, the ordered compaction (greater | equal), the zero fill and
// the block -> token expansion with the partial-block tail, plus the candidate-group form (maxima of groups of 4 blocks,
// then the same passes over the groups' blocks). Scalar code in the device code's pass order; integer only. Used by the
// CPU test to show every form emits the k first blocks in (key descending, block ascending) order, in ascending block order.
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strix::attn_core::select::radix3::rule {

constexpr std::uint32_t order_key(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) == 0U) bits = 0U;
    return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

constexpr bool counts(float v) { return v > -std::numeric_limits<float>::infinity(); }

struct Prefix {
    std::uint32_t prefix = 0, mask = 0, need = 0;
    std::uint32_t counted = 0;  // number of values > -inf (set when fewer than need: everything counted is taken)
    bool enough = true;
};

// The passes over values v[0 .. n): (prefix, mask, need) of the need-th largest key; enough == false when fewer than need
// values are > -inf (prefix = mask = 0 then).
inline Prefix radix_prefix(const float* v, std::uint32_t n, std::uint32_t k) {
    Prefix r;
    r.need = k;
    for (unsigned pass = 0; pass < 3U; ++pass) {
        const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
        const std::uint32_t digits = pass == 2 ? 1024U : 2048U;
        std::vector<std::uint32_t> hist(digits, 0U);
        std::uint32_t total = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto key = order_key(v[i]);
            if (counts(v[i]) && (key & r.mask) == r.prefix) { ++hist[(key >> shift) & (digits - 1U)]; ++total; }
        }
        if (pass == 0 && total <= r.need) { r.counted = total; r.enough = total == r.need; return r; }
        std::uint32_t above = 0;
        for (std::uint32_t d = digits; d-- > 0;) {
            if (above < r.need && r.need <= above + hist[d]) {
                r.prefix |= d << shift; r.mask |= (digits - 1U) << shift;
                r.need -= above;
                if (hist[d] == r.need) return r;  // the bin is taken whole
                break;
            }
            above += hist[d];
        }
    }
    return r;
}

// Ordered compaction: positions (blocks) selected under (prefix, mask, need), ascending.
inline std::vector<std::uint32_t> compact(const float* v, std::uint32_t n, const Prefix& p) {
    std::vector<std::uint32_t> out;
    std::uint32_t equal = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!counts(v[i])) continue;
        const auto masked = order_key(v[i]) & p.mask;
        if (masked > p.prefix) out.push_back(i);
        else if (masked == p.prefix && equal < p.need) { out.push_back(i); ++equal; }
    }
    return out;
}

// Token ids of select_blocks (one row, eager control). selected is sized k * ratio + (visible % ratio).
inline std::vector<std::uint32_t> select_tokens(const std::vector<float>& scores, std::uint32_t k, std::uint32_t ratio,
                                                std::uint32_t visible) {
    const std::uint32_t blocks = visible / ratio;
    std::vector<std::uint32_t> out;
    if (blocks <= k) { for (std::uint32_t t = 0; t < visible; ++t) out.push_back(t); return out; }
    auto p = radix_prefix(scores.data(), blocks, k);
    std::uint32_t zeros = 0;
    if (!p.enough) { zeros = p.need - p.counted; p.need = p.counted; }
    for (std::uint32_t z = 0; z < zeros * ratio; ++z) out.push_back(z % ratio);
    for (const auto b : compact(scores.data(), blocks, p))
        for (std::uint32_t o = 0; o < ratio; ++o) out.push_back(b * ratio + o);
    for (std::uint32_t t = blocks * ratio; t < visible; ++t) out.push_back(t);
    return out;
}

// Candidate-group form: group maxima (groups of `group` blocks, fmax, -inf past the row) -> prefix over the maxima ->
// groups whose maximum key >= prefix are candidates (ascending) -> the same passes over their blocks. Same tokens as
// select_tokens. Returns false (and the caller runs select_tokens) when fewer than k finite maxima or > cap candidate groups.
inline bool select_tokens_candidates(const std::vector<float>& scores, std::uint32_t k, std::uint32_t ratio, std::uint32_t visible,
                                     std::uint32_t group, std::uint32_t cap_groups, std::vector<std::uint32_t>& out) {
    const std::uint32_t blocks = visible / ratio;
    const std::uint32_t groups = (blocks + group - 1U) / group;
    std::vector<float> maxima(groups, -std::numeric_limits<float>::infinity());
    for (std::uint32_t b = 0; b < blocks; ++b) {
        auto& m = maxima[b / group];
        if (scores[b] > m) m = scores[b];
    }
    const auto pm = radix_prefix(maxima.data(), groups, k);
    if (!pm.enough) return false;
    std::vector<std::uint32_t> cand_group;
    for (std::uint32_t g = 0; g < groups; ++g)
        if (counts(maxima[g]) && (order_key(maxima[g]) & pm.mask) >= pm.prefix) cand_group.push_back(g);
    if (cand_group.size() > cap_groups) return false;
    std::vector<float> cand;
    std::vector<std::uint32_t> blk;
    for (const auto g : cand_group)
        for (std::uint32_t j = 0; j < group; ++j) {
            const auto b = g * group + j;
            cand.push_back(b < blocks ? scores[b] : -std::numeric_limits<float>::infinity());
            blk.push_back(b);
        }
    const auto n = static_cast<std::uint32_t>(cand.size());
    const auto p = radix_prefix(cand.data(), n, k);
    out.clear();
    for (const auto i : compact(cand.data(), n, p))
        for (std::uint32_t o = 0; o < ratio; ++o) out.push_back(blk[i] * ratio + o);
    for (std::uint32_t t = blocks * ratio; t < visible; ++t) out.push_back(t);
    return true;
}

}  // namespace strix::attn_core::select::radix3::rule
