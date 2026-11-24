#pragma once
// Backend-neutral measured plan data for the start order of the R1 rows launch (moe_core/microscaled_r1.hpp
// gate_up_rows): per (device, expert bank, row count) the last-level-cache budget a tile's span of weight bytes must
// fit. The tile only reorders CTAs of the same kernel, so a plan changes the schedule, never the bits.
// An entry covers exactly the workload it was measured on: device (architecture name without feature suffix +
// multiprocessor count), bank (experts, hidden, intermediate, top_k), the measured packed formats and one budget per
// row count. A budget of 0, a row count past the table, an unmeasured format or any other key keeps the incumbent
// (block, unit) start order.
// Home: shared (host shared policy): no backend, no model names. HIP binds it in moe_core/microscaled_r1.hpp
// (current_rows_tiles); the tables sit here side by side, as in strix/attn_row_share.hpp.
#include "strix/weight/formats/microscaled.hpp"

#include <cstdint>
#include <string_view>

namespace strix::r1_rows_tiles {

// Row counts 0 .. 8 of one rows launch (microscaled_r1::kMaxRows = 8).
inline constexpr unsigned table_rows = 9U;

constexpr unsigned format_bit(weight::microscaled::MatrixFormat format) {
    return 1U << static_cast<unsigned>(format);
}

struct DeviceTable {
    std::string_view arch;                  // e.g. "gfx1151" (HIP gcnArchName up to ':')
    unsigned multiprocessors;               // HIP multiProcessorCount (WGPs on RDNA: 20 for a 40-CU gfx1151)
    unsigned experts, hidden, intermediate, top_k;
    unsigned formats;                       // format_bit() of every measured format
    std::uint64_t cache_bytes[table_rows];  // per row count; 0 = incumbent order (measured slower or not measured)
};

inline constexpr std::uint64_t mib(std::uint64_t n) { return n << 20U; }
inline constexpr unsigned kMeasuredFormats = format_bit(weight::microscaled::MatrixFormat::nvfp4) |
                                             format_bit(weight::microscaled::MatrixFormat::mxfp8);

// gfx1151 with 20 WGPs (32 MiB last-level cache), 128 experts, hidden 2304 x intermediate 1152, top-8, NVFP4 and
// MXFP8. No-model fixture tests/moe_core/microscaled_r1_hip.hip (rows 1 / 2 / 4 / 8, uniform / skewed / collide
// routes, digests equal): with a 24 MiB budget rows 2 is slower (kept on the incumbent order), rows 4 and 8 faster
// (gate_up NVFP4 rows 8 -12.8 %, MXFP8 rows 4 skewed -6.9 %). Rows 1 runs the one-row kernel. Rows 3 / 5 / 6 / 7 were
// not measured and keep the incumbent order.
inline constexpr DeviceTable gate_up_tables[] = {
    {"gfx1151", 20U, 128U, 2304U, 1152U, 8U, kMeasuredFormats, {0U, 0U, 0U, 0U, mib(24U), 0U, 0U, 0U, mib(24U)}},
};

// The measured budget of (architecture name as the runtime reports it, e.g. "gfx1151:sramecc-:xnack-",
// multiprocessor count, format, bank, row count), or 0 (incumbent order).
constexpr std::uint64_t gate_up_cache_bytes(std::string_view arch, unsigned multiprocessors,
                                            weight::microscaled::MatrixFormat format, unsigned experts,
                                            unsigned hidden, unsigned intermediate, unsigned top_k, unsigned rows) {
    const auto name = arch.substr(0U, arch.find(':'));
    if (rows >= table_rows) return 0U;
    for (const auto& table : gate_up_tables) {
        if (table.arch == name && table.multiprocessors == multiprocessors && table.experts == experts &&
            table.hidden == hidden && table.intermediate == intermediate && table.top_k == top_k &&
            (table.formats & format_bit(format)) != 0U) {
            return table.cache_bytes[rows];
        }
    }
    return 0U;
}

}  // namespace strix::r1_rows_tiles
