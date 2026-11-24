#pragma once
// Backend-neutral measured plan data for the start order of the R1 rows launch (moe_core/microscaled_r1.hpp
// gate_up_rows): per (device, expert bank, row count) the last-level-cache budget a tile's span of weight bytes must
// fit. The tile only reorders CTAs of the same kernel, so a plan changes the schedule, never the bits.
// No row is compiled in: the measured rows come from the device declaration (strix-devices-v1 `[r1_rows_tiles.<name>]`,
// strix/device/declaration.hpp). The start gate keeps them on the binding it resolves (device::Bound); the backend's
// binder publishes them for the runtime device it bound, once it owns it (publish()). Each bound device has its own
// tables: a start for another device, or a start the backend refuses, never changes them. An entry covers exactly
// the workload it was measured on: device (architecture name without feature suffix + the multiprocessor count as the
// runtime reports it), bank (experts, hidden, intermediate, top_k), the measured packed formats and one budget per row
// count. No published table, a budget of 0, a row count past the entry, an unmeasured format or any other key keeps
// the incumbent (block, unit) start order.
// Home: shared (host shared policy): no backend, no model names, no device rows. HIP binds it in
// moe_core/microscaled_r1.hpp (current_rows_cache_bytes).
#include "strix/weight/formats/microscaled.hpp"

#include <array>
#include <atomic>
#include <cstddef>
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

// A bound runtime device: the backend runtime and the device index the runtime gives the launching thread.
enum class Runtime : unsigned { hip = 0U, cuda = 1U };
inline constexpr unsigned runtimes = 2U;
inline constexpr int ordinals = 64;  // device indices with a table slot; a higher index reads no table (incumbent)
struct Binding {
    Runtime runtime{Runtime::hip};
    int ordinal{};
};

namespace detail {
// Every published table stays alive for the process: a launch may still read one while a later bind publishes.
struct Published {
    std::mutex mu;
    std::forward_list<std::vector<DeviceTable>> kept;
    std::array<std::array<std::atomic<const std::vector<DeviceTable>*>, ordinals>, runtimes> current{};
};
inline Published& published() {
    static Published p;
    return p;
}
inline std::atomic<const std::vector<DeviceTable>*>* slot(Binding binding) {
    const auto runtime = static_cast<unsigned>(binding.runtime);
    if (runtime >= runtimes || binding.ordinal < 0 || binding.ordinal >= ordinals) return nullptr;
    return &published().current[runtime][static_cast<std::size_t>(binding.ordinal)];
}
}  // namespace detail

// The tables of one bound device (its binding's declaration); empty before that device is bound.
inline std::span<const DeviceTable> binding_tables(Binding binding) {
    const auto* s = detail::slot(binding);
    const auto* tables = s ? s->load(std::memory_order_acquire) : nullptr;
    return tables ? std::span<const DeviceTable>(*tables) : std::span<const DeviceTable>();
}
// Written by the backend binder once it owns the device (device::publish_bound_plans from the HIP / CUDA gate), with the
// binding's tables (empty without a declaration, so no table of an earlier start on that device survives); a no-model
// fixture may publish the entry it measures for its device. Returns false (nothing published) for an index past the
// slots.
inline bool publish(Binding binding, std::vector<DeviceTable> tables) {
    auto* s = detail::slot(binding);
    if (!s) return false;
    auto& p = detail::published();
    std::lock_guard<std::mutex> lock(p.mu);
    p.kept.push_front(std::move(tables));
    s->store(&p.kept.front(), std::memory_order_release);
    return true;
}

}  // namespace strix::r1_rows_tiles
