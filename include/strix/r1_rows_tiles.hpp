#pragma once
// Backend-neutral measured plan data for the start order of the R1 rows launch (moe_core/microscaled_r1.hpp
// gate_up_rows): per (device, expert bank, row count) the last-level-cache budget a tile's span of weight bytes must
// fit. The tile only reorders CTAs of the same kernel, so a plan changes the schedule, never the bits.
// No row is compiled in: the measured rows come from the device declaration (strix-devices-v1 `[r1_rows_tiles.<name>]`,
// strix/device/declaration.hpp), which the start gate publishes for the process (publish()). An entry covers exactly
// the workload it was measured on: device (architecture name without feature suffix + the multiprocessor count as the
// runtime reports it), bank (experts, hidden, intermediate, top_k), the measured packed formats and one budget per row
// count. No published table, a budget of 0, a row count past the entry, an unmeasured format or any other key keeps
// the incumbent (block, unit) start order.
// Home: shared (host shared policy): no backend, no model names, no device rows. HIP binds it in
// moe_core/microscaled_r1.hpp (current_rows_cache_bytes).
#include "strix/weight/formats/microscaled.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <forward_list>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::r1_rows_tiles {

// Row counts 0 .. 16 of one rows launch (microscaled_r1::kMaxLaunchRows = 16).
inline constexpr unsigned table_rows = 17U;

constexpr unsigned format_bit(weight::microscaled::MatrixFormat format) {
    return 1U << static_cast<unsigned>(format);
}
// The declaration's name of a packed format ("nvfp4", "mxfp8"), or 0 when it names none.
constexpr unsigned format_bit(std::string_view name) {
    if (name == "nvfp4") return format_bit(weight::microscaled::MatrixFormat::nvfp4);
    if (name == "mxfp8") return format_bit(weight::microscaled::MatrixFormat::mxfp8);
    return 0U;
}

struct DeviceTable {
    std::string name;  // the declaration's table key
    std::string arch;  // architecture name as the runtime reports it, feature suffix removed ("gfx1151")
    unsigned multiprocessors{};  // multiprocessor count as the runtime reports it (HIP: WGPs on RDNA, 20 on gfx1151)
    unsigned experts{}, hidden{}, intermediate{}, top_k{};
    unsigned formats{};                                   // format_bit() of every measured format
    std::array<std::uint64_t, table_rows> cache_bytes{};  // per row count; 0 = incumbent order
};

// The measured budget in `tables` for (architecture name as the runtime reports it, e.g. "gfx1151:sramecc-:xnack-",
// multiprocessor count, format, bank, row count), or 0 (incumbent order).
inline std::uint64_t gate_up_cache_bytes(std::span<const DeviceTable> tables, std::string_view arch,
                                         unsigned multiprocessors, weight::microscaled::MatrixFormat format,
                                         unsigned experts, unsigned hidden, unsigned intermediate, unsigned top_k,
                                         unsigned rows) {
    const auto name = arch.substr(0U, arch.find(':'));
    if (rows >= table_rows || name.empty()) return 0U;
    for (const auto& table : tables) {
        if (table.arch == name && table.multiprocessors == multiprocessors && table.experts == experts &&
            table.hidden == hidden && table.intermediate == intermediate && table.top_k == top_k &&
            (table.formats & format_bit(format)) != 0U) {
            return table.cache_bytes[rows];
        }
    }
    return 0U;
}

namespace detail {
// Every published table stays alive for the process: a launch may still read one while a later gate publishes.
struct Published {
    std::mutex mu;
    std::forward_list<std::vector<DeviceTable>> kept;
    std::atomic<const std::vector<DeviceTable>*> current{nullptr};
};
inline Published& published() {
    static Published p;
    return p;
}
}  // namespace detail

// The process's tables (the start gate's declaration); empty before any publish.
inline std::span<const DeviceTable> process_tables() {
    const auto* tables = detail::published().current.load(std::memory_order_acquire);
    return tables ? std::span<const DeviceTable>(*tables) : std::span<const DeviceTable>();
}
// Written by the start gate (device/gate.cpp start_gate) from the loaded declaration; a no-model fixture may publish
// the entry it measures.
inline void publish(std::vector<DeviceTable> tables) {
    auto& p = detail::published();
    std::lock_guard<std::mutex> lock(p.mu);
    p.kept.push_front(std::move(tables));
    p.current.store(&p.kept.front(), std::memory_order_release);
}

}  // namespace strix::r1_rows_tiles
