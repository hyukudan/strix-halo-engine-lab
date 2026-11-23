#pragma once

#include "strix/exec/descriptor.hpp"
#include "strix/state/table.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_DECODE_STATE_HD __host__ __device__
#else
#define STRIX_DECODE_STATE_HD
#endif

namespace strix::core::decode_state {

// The directory dimensions and layout are captured. Entry contents and the
// pending images are read from device storage, so graph replay can switch slots.
struct SlotView {
    state::StateTable table;
    unsigned instance;

    [[nodiscard]] STRIX_DECODE_STATE_HD const state::SlotEntry* entry(unsigned slot) const noexcept {
        return slot < table.slots ? table.entries + std::size_t{instance} * table.slots + slot : nullptr;
    }
    [[nodiscard]] bool valid() const noexcept {
        return table.entries && instance < table.states && table.slots != 0U;
    }
};

struct PositionView {
    SlotView slots;
    std::uint64_t row_bytes;
    unsigned ring;            // Zero for dense or paged storage.
    bool paged;

    [[nodiscard]] STRIX_DECODE_STATE_HD std::uint64_t address(const state::SlotEntry& entry,
                                                           unsigned position) const noexcept {
        if (paged) {
            const auto* pages = reinterpret_cast<const std::uint64_t*>(entry.base);
            return pages[position / state::page_rows] + (position % state::page_rows) * row_bytes;
        }
        return entry.base + (ring ? position % ring : position) * row_bytes;
    }
    [[nodiscard]] bool valid(std::uint64_t payload) const noexcept {
        return slots.valid() && row_bytes >= payload && row_bytes % 16U == 0U && (!paged || ring == 0U);
    }
};

[[nodiscard]] STRIX_DECODE_STATE_HD inline bool accepts(const state::SlotEntry* entry,
                                                       unsigned position) noexcept {
    return entry && entry->base && position >= entry->frontier && position < entry->capacity;
}

// The caller owns the committed prefix. Siblings at the same position live
// in separate pending rows, and only an inclusive ancestor can supply a key.
[[nodiscard]] STRIX_DECODE_STATE_HD constexpr unsigned pending_row(const exec::StepBlock& step,
    unsigned query, unsigned position) noexcept {
    const auto& row = step.row[query];
    for (unsigned i = 0; i <= query; ++i) {
        if ((row.ancestors & (1U << i)) && exec::step_row_valid(step, i) &&
            step.row[i].slot == row.slot && step.row[i].position == position) return i;
    }
    return exec::kStepNoParent;
}

[[nodiscard]] STRIX_DECODE_STATE_HD constexpr unsigned parent_row(const exec::StepBlock& step,
                                                               unsigned row) noexcept {
    const auto parent = step.row[row].parent;
    return parent < row && exec::step_row_valid(step, parent) && step.row[parent].slot == step.row[row].slot &&
        step.row[row].position != 0U && step.row[parent].position == step.row[row].position - 1U
        ? parent : exec::kStepNoParent;
}

}  // namespace strix::core::decode_state

#undef STRIX_DECODE_STATE_HD
