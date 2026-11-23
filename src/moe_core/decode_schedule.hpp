#pragma once
#include <cstddef>
namespace strix::moe_core {
// A schedule describes arithmetic partitions, not a performance qualification.
template <unsigned GateSplits, unsigned DownSplits, unsigned SplitChannels, unsigned Waves>
struct DecodeSchedule {
    static_assert(GateSplits > 0 && DownSplits > 0);
    static_assert(SplitChannels % 128U == 0 && Waves > 0);
    static constexpr unsigned kGateSplits = GateSplits;
    static constexpr unsigned kDownSplits = DownSplits;
    static constexpr unsigned kSplitChannels = SplitChannels;
    static constexpr unsigned kWaves = Waves;
    static constexpr bool kAscendingSplitSums = true;
    static constexpr bool kHalfMaterializedDownInput = true;
};
// Output frame of the down projection over 256-column groups. Part 0 = every group (the incumbent launch); Part 1 / 2 =
// the first / second half of the groups (a 2-way column split): launch group g owns group g + kFirstGroup and stores
// its columns at row stride Hidden / 2 relative to the half's first column. Element arithmetic and order are Part 0's.
template <unsigned Part, unsigned Hidden, unsigned GroupWidth = 256U>
struct DownFrame {
    static_assert(Part <= 2U && GroupWidth > 0U && Hidden % GroupWidth == 0U);
    static_assert(Part == 0U || Hidden % (2U * GroupWidth) == 0U);
    static constexpr unsigned kGroups = Hidden / GroupWidth;
    static constexpr unsigned kLaunchGroups = Part == 0U ? kGroups : kGroups / 2U;
    static constexpr unsigned kFirstGroup = Part == 2U ? kGroups / 2U : 0U;
    static constexpr unsigned kFirstColumn = kFirstGroup * GroupWidth;
    static constexpr unsigned kRowStride = Part == 0U ? Hidden : Hidden / 2U;
    static constexpr unsigned group(unsigned launch_group) { return launch_group + kFirstGroup; }
    static constexpr std::size_t out_index(unsigned row, unsigned column) {
        return static_cast<std::size_t>(row) * kRowStride + (column - kFirstColumn);
    }
};
}  // namespace strix::moe_core
