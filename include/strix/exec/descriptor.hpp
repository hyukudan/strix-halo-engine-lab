#pragma once

#include "strix/session/step_replay.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_EXEC_STEP_HD __host__ __device__
#else
#define STRIX_EXEC_STEP_HD
#endif

namespace strix::exec {

// Captured graphs and program bindings must agree on this ABI.
inline constexpr char kStepLayoutId[] = "strix.exec.step.v1";
inline constexpr std::uint32_t kStepMaxRows = 32;
inline constexpr std::uint32_t kStepMaxStops = 4;
inline constexpr std::uint32_t kStepNoParent = 0xffffffffU;

enum StepRowFlag : std::uint32_t {
    step_valid = 1U << 0U,
    step_sample = 1U << 1U,
    step_pending_state = 1U << 2U,
    step_group_close = 1U << 3U,
    step_prefix_select = 1U << 4U,
    step_emit_topn = 1U << 5U,
};

struct StepRow {
    std::uint32_t token;
    std::uint32_t position;
    std::uint32_t slot;       // StateTable session lane.
    std::uint32_t parent;     // Earlier row index or kStepNoParent.
    std::uint32_t ancestors;  // Visible in-step rows, indexed by bit.
    std::uint32_t flags;
    std::uint32_t rng_lo, rng_hi;
};

struct StepHeader {
    std::uint32_t rows;  // Program width, including padding.
    std::uint32_t valid_mask, step, feed_mask;
    std::uint32_t sampling_policy;
    // The reserved words carry the count and device address in a fixed word layout.
    std::uint32_t stop_count, stop_table_lo, stop_table_hi;
};

struct StepBlock {
    StepHeader h;
    StepRow row[kStepMaxRows];
};

struct StepStopTable {
    std::uint32_t ids[kStepMaxStops];
};

enum class RowStatus : std::uint32_t { picked = 0, masked = 1, stop = 2 };

struct RowOut {
    std::uint32_t id;
    float top1, top2;
    RowStatus status;
};

static_assert(std::is_standard_layout_v<StepRow> && std::is_trivial_v<StepRow> &&
              sizeof(StepRow) == 32 && alignof(StepRow) == 4);
static_assert(offsetof(StepRow, token) == 0 && offsetof(StepRow, position) == 4 &&
              offsetof(StepRow, slot) == 8 && offsetof(StepRow, parent) == 12 &&
              offsetof(StepRow, ancestors) == 16 && offsetof(StepRow, flags) == 20 &&
              offsetof(StepRow, rng_lo) == 24 && offsetof(StepRow, rng_hi) == 28);
static_assert(std::is_standard_layout_v<StepHeader> && std::is_trivial_v<StepHeader> &&
              sizeof(StepHeader) == 32 && alignof(StepHeader) == 4);
static_assert(offsetof(StepHeader, rows) == 0 && offsetof(StepHeader, valid_mask) == 4 &&
              offsetof(StepHeader, step) == 8 && offsetof(StepHeader, feed_mask) == 12 &&
              offsetof(StepHeader, sampling_policy) == 16 && offsetof(StepHeader, stop_count) == 20 &&
              offsetof(StepHeader, stop_table_lo) == 24 && offsetof(StepHeader, stop_table_hi) == 28);
static_assert(std::is_standard_layout_v<StepBlock> && std::is_trivial_v<StepBlock> &&
              sizeof(StepBlock) == 1056 && alignof(StepBlock) == 4);
static_assert(offsetof(StepBlock, h) == 0 && offsetof(StepBlock, row) == 32);
static_assert(std::is_standard_layout_v<StepStopTable> && std::is_trivial_v<StepStopTable> &&
              sizeof(StepStopTable) == 16 && alignof(StepStopTable) == 4 && offsetof(StepStopTable, ids) == 0);
static_assert(std::is_standard_layout_v<RowOut> && std::is_trivial_v<RowOut> &&
              sizeof(RowOut) == 16 && alignof(RowOut) == 4);
static_assert(offsetof(RowOut, id) == 0 && offsetof(RowOut, top1) == 4 &&
              offsetof(RowOut, top2) == 8 && offsetof(RowOut, status) == 12);

// Launch geometry supplies an index within the block. A padded row can be
// read, but state writes and picks must first pass step_row_valid().
[[nodiscard]] STRIX_EXEC_STEP_HD constexpr const StepRow& step_row(const StepBlock& block,
                                                                   std::uint32_t index) noexcept {
    return block.row[index];
}

[[nodiscard]] STRIX_EXEC_STEP_HD constexpr bool step_row_valid(const StepBlock& block,
                                                              std::uint32_t index) noexcept {
    return index < kStepMaxRows && index < block.h.rows &&
           (block.h.valid_mask & (std::uint32_t{1} << index)) != 0 &&
           (step_row(block, index).flags & step_valid) != 0;
}

[[nodiscard]] STRIX_EXEC_STEP_HD constexpr session::StepDescriptor step_input(const StepRow& row) noexcept {
    return {row.position, row.token};
}

constexpr void set_step_stop_address(StepHeader& header, std::uint64_t address) noexcept {
    header.stop_table_lo = static_cast<std::uint32_t>(address);
    header.stop_table_hi = static_cast<std::uint32_t>(address >> 32U);
}

[[nodiscard]] STRIX_EXEC_STEP_HD constexpr std::uint64_t step_stop_address(const StepHeader& header) noexcept {
    return std::uint64_t{header.stop_table_lo} | (std::uint64_t{header.stop_table_hi} << 32U);
}

// Use the device table resolved from step_stop_address() on device, or its
// host mirror in advance_step(). A zero count permits a null table address.
[[nodiscard]] STRIX_EXEC_STEP_HD constexpr bool step_stops_on(const StepHeader& header,
                                                             const StepStopTable& table,
                                                             std::uint32_t id) noexcept {
    for (std::uint32_t i = 0; i < header.stop_count && i < kStepMaxStops; ++i) {
        if (table.ids[i] == id) return true;
    }
    return false;
}

struct StepAdvance {
    StepBlock next;
    RowOut readback[kStepMaxRows];  // The first next.h.rows entries are in row order.
    // Each set bit publishes next.row[index].position as the exclusive
    // frontier for next.row[index].slot. Unset bits publish no state.
    std::uint32_t commit_mask;
};

// Consume one complete decode step of independent session lanes. Each active
// row appends its input and commits position + 1, even when its pick is a stop.
// Only feeding, non-stopped rows remain active in the next step. In a replay
// chain, use the returned descriptor for each successive readback, including
// after a stop. Even if later work ran, its picks for retired lanes are masked
// and its state is excluded by the last published frontier.
[[nodiscard]] StepAdvance advance_step(const StepBlock& current, std::span<const RowOut> picks,
                                       const StepStopTable& host_stops);

}  // namespace strix::exec

#undef STRIX_EXEC_STEP_HD
