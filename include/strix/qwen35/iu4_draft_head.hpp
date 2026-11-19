#pragma once
// Draft-only subset of an IU4 g64 output head (host plan; strix/spec/draft_rows.hpp picks the rows).
// Layout of the shared IU4 pack (src/core/iu4.hpp, src/core/dense_linear_iu4.hpp): N*K/2 bytes of tiles
// [K/16][N/16][16 columns][8 bytes], then FP16 scales [N][K/64]. One DraftRows group of 16 token ids = one tile column, so a
// subset head is a byte gather (no re-quantisation): for every K/16 tile row, the source tile column g moves to compact
// column slot s, and the 16 scale rows of g move to rows [16 s, 16 s + 16). Each compact column then holds exactly the bytes of
// its source column, so the dense IU4 kernel gives it the same weights; the draft argmax is mapped back to token ids by the
// slot table. Drafts only steer acceptance: the verify window emits the target's own greedy tokens either way.
// A compact head is laid out for a fixed column count (tile stride = columns / 16): when the subset outgrows it, the
// caller picks a new count (iu4_draft_columns) and re-gathers every slot.
// Home: shared host plan (no device code, no model names); bound by the qwen35 HIP adapter (HipEngine::set_draft_rows).
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::qwen35 {

inline constexpr std::uint32_t kIu4DraftGroup = 16U;     // ids per DraftRows group = columns per IU4 tile
inline constexpr std::uint32_t kIu4DraftHeadroom = 1024U;  // spare columns at a re-layout (context admissions)

// One strided byte copy: `height` rows of `width` bytes; height 1 = a plain copy.
struct Iu4Copy {
    std::size_t src = 0, dst = 0, width = 0, height = 1, spitch = 0, dpitch = 0;
};

inline constexpr std::size_t iu4_head_bytes(std::uint32_t columns, std::uint32_t inputs) {
    return std::size_t(columns) * inputs / 2U + std::size_t(columns) * (inputs / 64U) * 2U;
}

// Column count of a compact head that holds `rows` ids: rows + headroom, rounded up to the kernel's 64-column block,
// capped at the source column count (a multiple of 64).
inline std::uint32_t iu4_draft_columns(std::uint32_t rows, std::uint32_t source_columns) {
    if (source_columns % 64U != 0U) throw std::invalid_argument("iu4 draft head: source columns must be a multiple of 64");
    const std::uint64_t want = (std::uint64_t(rows) + kIu4DraftHeadroom + 63U) / 64U * 64U;
    return want < source_columns ? std::uint32_t(want) : source_columns;
}

// Copies that move slots [from, to) of `groups` (slot -> source group) from a source head of `src_columns` into a compact
// head of `dst_columns`. Runs of consecutive source groups at consecutive slots become one copy per region.
inline std::vector<Iu4Copy> iu4_group_gather(std::uint32_t inputs, std::uint32_t src_columns, std::uint32_t dst_columns,
                                             std::span<const std::uint32_t> groups, std::uint32_t from, std::uint32_t to) {
    if (inputs == 0U || inputs % 64U != 0U || src_columns % kIu4DraftGroup != 0U || dst_columns % kIu4DraftGroup != 0U)
        throw std::invalid_argument("iu4 draft head: K must be a multiple of 64, columns a multiple of 16");
    if (to > groups.size() || from > to || std::size_t(to) * kIu4DraftGroup > dst_columns)
        throw std::invalid_argument("iu4 draft head: slots exceed the compact head");
    constexpr std::size_t tile = kIu4DraftGroup * 8U;  // bytes of one 16-column x 16-K tile
    const std::size_t scale_group = std::size_t(kIu4DraftGroup) * (inputs / 64U) * 2U;  // scale bytes of 16 columns
    const std::size_t src_scales = std::size_t(src_columns) * inputs / 2U, dst_scales = std::size_t(dst_columns) * inputs / 2U;
    std::vector<Iu4Copy> plan;
    for (std::uint32_t a = from; a < to;) {
        if (std::size_t(groups[a] + 1U) * kIu4DraftGroup > src_columns) throw std::invalid_argument("iu4 draft head: group past the head");
        std::uint32_t b = a + 1U;
        while (b < to && groups[b] == groups[b - 1U] + 1U && std::size_t(groups[b] + 1U) * kIu4DraftGroup <= src_columns) ++b;
        const std::size_t n = b - a;
        plan.push_back({std::size_t(groups[a]) * tile, std::size_t(a) * tile, n * tile, inputs / 16U,
                        std::size_t(src_columns / kIu4DraftGroup) * tile, std::size_t(dst_columns / kIu4DraftGroup) * tile});
        plan.push_back({src_scales + std::size_t(groups[a]) * scale_group, dst_scales + std::size_t(a) * scale_group,
                        n * scale_group, 1U, 0U, 0U});
        a = b;
    }
    return plan;
}

}  // namespace strix::qwen35
