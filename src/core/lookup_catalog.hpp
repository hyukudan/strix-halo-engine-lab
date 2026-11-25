#pragma once

#include "core/lookup_plan.hpp"
#include "strix/exec/catalog.hpp"

#include <array>
#include <utility>

namespace strix::cuda::core::lookup {
namespace detail {

using namespace exec;

struct CatalogForm {
    FormId ids, step;
    ::strix::weight::FormatId weight;
    std::string_view identity;
};

inline constexpr CatalogForm kForms[] = {
    {"embedding.bf16_to_f32", "embedding.bf16_to_f32.x3", {"plain.bf16", 1}, "cuda-sm120-lookup-bf16-f32-v1"},
    {"embedding.f32", "embedding.f32.x3", {"plain.f32", 1}, "cuda-sm120-lookup-f32-v1"},
    {"embedding.f16_to_f32", "embedding.f16_to_f32.x3", {"plain.f16", 1}, "cuda-sm120-lookup-f16-f32-v1"},
    {"embedding.f16_payload_to_f32", "embedding.f16_payload_to_f32.x3", {"plain.f16", 1}, "cuda-sm120-lookup-f16-payload-f32-v1"},
    {"embedding.q8_0_to_f32", "embedding.q8_0_to_f32.x3", {"gguf.q8_0", 1}, "cuda-sm120-lookup-q8-f32-v1"},
    {"embedding.q8_0_to_bf16", "embedding.q8_0_to_bf16.x3", {"gguf.q8_0", 1}, "cuda-sm120-lookup-q8-bf16-v1"},
    {"embedding.q8_0_to_bf16_row_zero", "embedding.q8_0_to_bf16_row_zero.x3", {"gguf.q8_0", 1}, "cuda-sm120-lookup-q8-bf16-row-zero-v1"},
    {"embedding.nvfp4_tile_to_f32", "embedding.nvfp4_tile_to_f32.x3", {"gguf.nvfp4_tile.strix", 1}, "cuda-sm120-lookup-nvfp4-tile-f32-v1"},
};

inline EntryCost cost(const LaunchSite& site, const device::Profile&) {
    return {std::numeric_limits<double>::infinity(), site.weight_bytes, site.state_bytes, 0U, 1U};
}

inline void workspace(const LaunchSite&, WorkspaceNeed& need) { need = {}; }

template <std::size_t I, bool Step>
inline constexpr LauncherRef kLauncher{
    "strix::cuda::core::lookup::enqueue", "src/cuda/core/lookup_cuda.cu", Step ? kForms[I].step : kForms[I].ids,
    Step ? kStepMaxRows : kMaxIdRows};

template <std::size_t I, bool Step, std::uint32_t First, std::uint32_t Last>
EntryLaunch bind(const LaunchSite& site, EntryBinder& binder) {
    if (!binder.resolve || site.rows < First || site.rows > Last) return {};
    return binder.resolve(binder.context, kLauncher<I, Step>, site);
}

template <std::size_t I, bool Step, RowsClass Rows, std::uint32_t First, std::uint32_t Last>
constexpr KernelEntry entry() {
    const auto& launcher = kLauncher<I, Step>;
    return {.key = {graph::Op::gather, launcher.mode, kForms[I].weight, OperandClass::f32_fma, Rows},
            .arch = device::ArchMask::sm120, .identity = kForms[I].identity, .exact = lever::Exactness::bitwise,
            .cost = cost, .workspace = workspace, .bind = bind<I, Step, First, Last>, .launcher = &launcher};
}

template <std::size_t... I>
constexpr auto entries(std::index_sequence<I...>) {
    return std::array<KernelEntry, sizeof...(I) * 7U>{
        entry<I, false, RowsClass::r1, 1U, 1U>()...,
        entry<I, false, RowsClass::r2_8, 2U, 8U>()...,
        entry<I, false, RowsClass::r9_32, 9U, 32U>()...,
        entry<I, false, RowsClass::tall, 33U, kMaxIdRows>()...,
        entry<I, true, RowsClass::r1, 1U, 1U>()...,
        entry<I, true, RowsClass::r2_8, 2U, 8U>()...,
        entry<I, true, RowsClass::r9_32, 9U, kStepMaxRows>()...,
    };
}

inline constexpr auto kEntries = entries(std::make_index_sequence<std::size(kForms)>{});

}  // namespace detail

// D4 fragment. Adopt the BF16 R1 device-ID row in place of E1's existing key.
[[nodiscard]] inline std::span<const exec::KernelEntry> catalog_entries() noexcept { return detail::kEntries; }

}  // namespace strix::cuda::core::lookup
