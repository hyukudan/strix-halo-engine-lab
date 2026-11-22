#pragma once

#include "strix/exec/catalog.hpp"

#include <limits>

namespace strix::cuda::mix {
namespace detail {

using namespace exec;

inline constexpr LauncherRef kReadoutLauncher{
    "strix::cuda::mix::enqueue", "src/cuda/core/mix_cuda.cuh", "bank_softmax.f32.native",
    std::numeric_limits<std::uint32_t>::max()};

inline EntryCost readout_cost(const LaunchSite& site, const device::Profile&) {
    return {std::numeric_limits<double>::infinity(), site.weight_bytes, site.state_bytes, site.macs, 1U};
}

inline void readout_workspace(const LaunchSite&, WorkspaceNeed& need) { need = {}; }

template <std::uint32_t FirstRow, std::uint32_t LastRow>
EntryLaunch bind_readout(const LaunchSite& site, EntryBinder& binder) {
    if (!binder.resolve || site.rows < FirstRow || site.rows > LastRow) return {};
    return binder.resolve(binder.context, kReadoutLauncher, site);
}

template <RowsClass Rows, std::uint32_t FirstRow, std::uint32_t LastRow>
constexpr KernelEntry readout_entry() {
    return {.key = {graph::Op::mix, "residual.bank", {}, OperandClass::f32_fma, Rows},
            .arch = device::ArchMask::sm120, .identity = "cuda-sm120-residual-bank-softmax-v1",
            .exact = lever::Exactness::bitwise, .cost = readout_cost, .workspace = readout_workspace,
            .bind = bind_readout<FirstRow, LastRow>, .launcher = &kReadoutLauncher,
            .reference = ReferenceClass::base};
}

inline constexpr KernelEntry kReadoutEntries[] = {
    readout_entry<RowsClass::r1, 1U, 1U>(),
    readout_entry<RowsClass::r2_8, 2U, 8U>(),
    readout_entry<RowsClass::r9_32, 9U, 32U>(),
    readout_entry<RowsClass::tall, 33U, kReadoutLauncher.max_rows>(),
};

} // namespace detail

// D4 fragment. The executor can add these static rows to its CUDA catalogue.
[[nodiscard]] inline std::span<const exec::KernelEntry> catalog_entries() noexcept {
    return detail::kReadoutEntries;
}

} // namespace strix::cuda::mix
