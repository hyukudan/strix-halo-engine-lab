#pragma once

#include "strix/exec/descriptor.hpp"
#include "strix/spec/block_drafter.hpp"

#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_STEP_SCATTER_HD __host__ __device__
#else
#define STRIX_STEP_SCATTER_HD
#endif

// Keep the incumbent scratch-view names used by CUDA kernel signatures.
namespace strix::cuda::core::step_inputs {

inline constexpr auto kRows = exec::kStepMaxRows;
inline constexpr auto kTreeWords = 1U + 2U * kRows;

// Legacy scratch views for consumers that have yet to read the X3 rows directly.
// Device launches bind device pointers, stable across replay.
struct LinearOutput {
    std::uint32_t* position;  // scalar position of the anchor row
    std::uint32_t* ids;       // kRows
};

struct Linear { LinearOutput out; };

struct TreeMetadata {
    std::uint32_t leaf;
    std::uint32_t self_kv[kRows];  // absolute physical index, separate from StepRow::position
};

struct Tree {
    LinearOutput out;
    const TreeMetadata* metadata;
    std::uint32_t* words;  // [leaf mask][kRows absolute positions][kRows self K/V indices]
};

struct Feed {
    const std::uint32_t* ids;
    const std::uint32_t* indices;  // feed indices are independent of StateTable slots and h.feed_mask
};

struct FedRow {
    LinearOutput out;
    Feed feed;  // one required device-feed index. Only ids[0] is written.
};

struct FedRows {
    LinearOutput out;
    Feed feed;  // one index per valid row. DraftFeed::kHost selects StepRow::token.
};

struct SlotGroups {
    std::uint32_t row0[kRows], rows[kRows], slot[kRows];  // indexed by group, zero-filled tail
};

struct Slots {
    const SlotGroups* groups;
    std::uint32_t *ids, *position, *slot, *group_row0, *group_rows, *group_slot;  // kRows each
};

namespace detail {

// Projection to scratch only. Padding is initialized for the legacy consumers;
// StateTable writes and RowOut status are the responsibility of the downstream program.
template <class View>
STRIX_STEP_SCATTER_HD inline void scatter_row(const exec::StepBlock& step, View v, std::uint32_t i) {
    if constexpr (std::is_same_v<View, FedRow>) {
        if (i == 0U && exec::step_row_valid(step, i)) {
            *v.out.position = exec::step_row(step, i).position;
            v.out.ids[0] = v.feed.ids[v.feed.indices[0]];
        }
    } else {
        const bool valid = exec::step_row_valid(step, i);
        const auto& row = exec::step_row(step, i);
        auto id = valid ? row.token : 0U;
        const auto position = valid ? row.position : 0U;
        if constexpr (std::is_same_v<View, Slots>) {
            v.ids[i] = id;
            v.position[i] = position;
            v.slot[i] = valid ? row.slot : 0U;
            v.group_row0[i] = v.groups->row0[i];
            v.group_rows[i] = v.groups->rows[i];
            v.group_slot[i] = v.groups->slot[i];
        } else {
            if constexpr (std::is_same_v<View, FedRows>) {
                if (valid) {
                    const auto index = v.feed.indices[i];
                    if (index != spec::DraftFeed::kHost) id = v.feed.ids[index];
                }
            } else if constexpr (std::is_same_v<View, Tree>) {
                if (i == 0U) v.words[0] = v.metadata->leaf;
                v.words[1U + i] = position;
                v.words[1U + kRows + i] = valid ? v.metadata->self_kv[i] : 0U;
            } else {
                static_assert(std::is_same_v<View, Linear>);
            }
            if (i == 0U) *v.out.position = position;
            v.out.ids[i] = id;
        }
    }
}

}  // namespace detail

}  // namespace strix::cuda::core::step_inputs

#undef STRIX_STEP_SCATTER_HD
