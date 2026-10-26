#pragma once

#include "strix/execution_runtime.hpp"
#include "strix/launch_dispatch.hpp"

#include <array>
#include <span>

namespace strix {

// Batched forms of one-row launchers for the tiled prefill row loops (STRIX_PREFILL_PIPE bit 16): each runs the
// same per-element arithmetic as one launch per row, over rows whose buffers sit at one fixed pitch, after the
// same per-row admission. A rows function returns false before enqueueing anything when its rows do not
// qualify; the caller then launches them row by row. Registered once by the launcher's translation unit
// during static initialization; looked up by the one-row function.
using ExecutionRowsLaunchFunction = bool (*)(std::span<const ExecutionDispatchContext>, void* user_data);

struct RowsLaunchEntry {
    ExecutionLaunchFunction one_row = nullptr;
    ExecutionRowsLaunchFunction rows = nullptr;
};

inline std::array<RowsLaunchEntry, 8>& rows_launch_table() {
    static std::array<RowsLaunchEntry, 8> table{};
    return table;
}

inline bool register_rows_launch(ExecutionLaunchFunction one_row, ExecutionRowsLaunchFunction rows) {
    for (auto& entry : rows_launch_table()) {
        if (entry.one_row == one_row) return entry.rows == rows;
        if (!entry.one_row) { entry = {one_row, rows}; return true; }
    }
    return false;
}

[[nodiscard]] inline ExecutionRowsLaunchFunction rows_launch_for(ExecutionLaunchFunction one_row) {
    if (!one_row) return nullptr;
    for (const auto& entry : rows_launch_table()) if (entry.one_row == one_row) return entry.rows;
    return nullptr;
}

} // namespace strix
