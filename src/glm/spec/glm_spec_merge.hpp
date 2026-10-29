#pragma once
// HIP-free speculative wire layout and rank-ordered host merge.
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace strix::glm::spec {
inline constexpr std::uint32_t kEligibilityMerge = 0x53504543U;
inline constexpr std::uint32_t kConfigurationMismatchMerge = 0x53504544U;


struct MergeRow {
    float value = 0;
    std::uint32_t id = 0, flag = 0;
};
struct MergeCtl {
    std::uint32_t cycle = 0, step = 0, kind = 0, replay_flag = 0;
    std::uint32_t table_version = 0;
    float table[11] = {};  // plain_ms, verify_ms[1..8], extension_ms, catchup_ms
    float mtp_max = 0, mtp_sum = 0;
};
static_assert(sizeof(MergeRow) == 12 && sizeof(MergeCtl) == 72);
static_assert(offsetof(MergeRow, value) == 0 && offsetof(MergeRow, id) == 4 && offsetof(MergeRow, flag) == 8);
static_assert(offsetof(MergeCtl, cycle) == 0 && offsetof(MergeCtl, step) == 4 && offsetof(MergeCtl, kind) == 8);
static_assert(offsetof(MergeCtl, replay_flag) == 12 && offsetof(MergeCtl, table_version) == 16);
static_assert(offsetof(MergeCtl, table) == 20 && offsetof(MergeCtl, mtp_max) == 64 && offsetof(MergeCtl, mtp_sum) == 68);
static_assert(std::is_trivially_copyable_v<MergeRow> && std::is_trivially_copyable_v<MergeCtl>);
static_assert(std::is_standard_layout_v<MergeRow> && std::is_standard_layout_v<MergeCtl>);
inline constexpr std::size_t kMergeBytes = sizeof(MergeCtl) + 8 * sizeof(MergeRow);
static_assert(kMergeBytes == 168);

// Candidates are NaN-free (the head's C5 checkpoint owns non-finite detection).
void merge_rank_order(std::span<MergeRow> mine, MergeCtl& ctl, std::span<const MergeRow> peer,
                      const MergeCtl& peer_ctl, std::uint32_t rank);

}  // namespace strix::glm::spec
