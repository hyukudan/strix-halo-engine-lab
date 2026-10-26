#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>

namespace strix {
// Largest T of one tiled known-prompt prefill block (--known-prefix-prefill-rows). Every T-row
// prefill operator works in row groups or 64-bit row strides; the routed MoE CSR of T * top_k
// assignments is built by one CTA (its wave counters bound T * top_k, see build_prefill_csr_waves).
inline constexpr std::uint32_t known_prefill_max_rows = 8192;
inline std::string known_prefill_rows_range() { return "16.." + std::to_string(known_prefill_max_rows); }
// STRIX_QPF2K_SCHED block plans (bit 4 merged tails, bit 16 balanced blocks). A row's arithmetic depends on the launches
// it lands in: dense projections of < 32-row launches take the dense A8 form, and the QSA part-1 input projections run
// a block's rows in 256/128-row FP16 launches but its last (rows mod 128) rows in 64/32/16-row dense A8 groups
// (hip_indexed_attention_launcher.hip enqueue_qsa_prefill_rows). A block whose rows are a multiple of 128 puts every row
// on the FP16 path whatever its size; the T blocks (T a multiple of 128) and the 1024/256 tails are such blocks, the 64
// and 16-row tails are not. So a plan is bitwise equal to the ladder when it only regroups the ladder's
// multiple-of-128 units into other multiple-of-128 blocks and leaves every other unit (and the serial suffix) as is.
inline constexpr std::uint64_t known_prefill_balanced_granule = 128U;
// The next balanced block for m remaining rows on a T-row owner (0: none; the rest runs as the ladder runs it).
inline std::uint64_t known_prefill_balanced_rows(std::uint64_t m, std::uint64_t t) {
    constexpr auto g = known_prefill_balanced_granule;
    if (m < g || t < g) return 0U;
    const auto k = (m + t - 1U) / t;
    return std::min({((m + k - 1U) / k + g - 1U) / g * g, t / g * g, m / g * g});
}
// Bit 16 eligibility: T a multiple of 128 and the ladder's multiple-of-128 units (T blocks, then leading tails) leave
// exactly m mod 128 rows to its other units (which then run identically after the balanced blocks).
inline bool known_prefill_balanced_eligible(std::uint64_t m, std::uint64_t t, std::span<const std::uint32_t> tails) {
    constexpr auto g = known_prefill_balanced_granule;
    if (m < g || !t || t % g) return false;
    std::uint64_t rest = m % t;
    for (const auto rows : tails) {
        if (!rows || rest < rows) continue;
        if (rows % g) break;            // the first used unit off the 128 grid: it and the rest run as the ladder runs them
        rest %= rows;
    }
    return rest == m % g;
}
// Bit 4: on the m rows left after the T blocks (m < T), the ladder's leading tail units that are multiples of 128 rows
// merged into ONE partial block of the T owner; the first used unit off the 128 grid, the smaller ones and the serial
// suffix stay as the ladder runs them. Returns the merged rows, or 0 when fewer than two units would merge.
inline std::uint64_t known_prefill_merged_tail_rows(std::uint64_t m, std::uint64_t t, std::span<const std::uint32_t> tails) {
    if (m >= t) return 0U;
    std::uint64_t merged = 0, units = 0;
    for (const auto rows : tails) {
        if (!rows || m < rows) continue;
        if (rows % known_prefill_balanced_granule) break;
        while (m >= rows) { m -= rows; merged += rows; ++units; }
    }
    return units >= 2U ? merged : 0U;
}
// Whether tail unit `rows` of the ladder for the m rows left after the T blocks is merged by bit 4 (S4 prunes its owner).
inline bool known_prefill_tail_merged(std::uint64_t m, std::uint64_t t, std::span<const std::uint32_t> tails, std::uint32_t unit) {
    if (!known_prefill_merged_tail_rows(m, t, tails)) return false;
    for (const auto rows : tails) {
        if (!rows || m < rows) continue;
        if (rows % known_prefill_balanced_granule) return false;
        if (rows == unit) return true;
        m %= rows;
    }
    return false;
}
} // namespace strix
