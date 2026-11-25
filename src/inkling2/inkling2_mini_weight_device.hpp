#pragma once

#include "core/resident_row_plan.hpp"
#include "inkling2_mini_weight_options.hpp"
#include "strix/device/runtime.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

#if defined(__HIPCC__)
#include "device/hip_cost_identity.hpp"
#include "strix/device/hip_enumerator.hpp"
#include "strix/device/hip_gate.hpp"
#include <hip/hip_runtime_api.h>
#else
#include "device/cuda_cost_identity.hpp"
#include "strix/device/cuda_enumerator.hpp"
#endif

namespace strix::inkling2 {

// The scope a load-time weight decision row is matched against: the current device's architecture, the toolkit
// this binary was built with, the resident Q8_0 consumers' implementation identity and the raw checkpoint SHA-256
// (shared persistent stat-keyed cache: only the first FAST run of a checkpoint reads it whole). The resolver sets
// the regime of each lookup (kMiniWeightLoadRegimes). Hash only for explicit qualification or when a committed FULL
// row of this backend, device, toolkit and consumer identity could select for `extent` (its context band holds the
// instance's context and its gain's lane band its lanes): a gate-only row, or a gain outside its band, never selects,
// so it never costs a whole-checkpoint read (which also evicts the page cache of running servers on a node).
// Owns the strings; view() builds the scope on use, so copies and moves never leave a view dangling.
// device_class: the architecture (the profile ladder's device class); weight_device_class: the weight-load key, the
// shared cost-device identity of the current device (device/{hip,cuda}_cost_identity.hpp: architecture, product name,
// compute units, memory, clock, runtime and driver), so a residency receipt never crosses products or clock limits.
struct MiniWeightScope {
    weight::Backend backend{};
    std::string device_class, weight_device_class, compiler_id;
    std::string_view regime;
    weight::Digest256 build{}, checkpoint{};
    [[nodiscard]] weight::WeightLoadScope view() const {
        return {backend, weight_device_class, compiler_id, regime, build, checkpoint};
    }
};

#if defined(__HIPCC__)
inline constexpr auto kMiniWeightBackend = weight::Backend::hip;
#else
inline constexpr auto kMiniWeightBackend = weight::Backend::cuda;
#endif

inline MiniWeightScope mini_weight_scope(const std::string& model, profile::Name tier,
                                         const MiniWeightOptions& options, weight::WeightLoadExtent extent) {
    const bool qualify = options.qualify_q8 || options.qualify != nullptr;
    const bool row_requested = mini_may_load_lowbit_row(tier, options, kMiniWeightBackend, extent);
    const bool compact_tier = tier == profile::Name::fast || tier == profile::Name::turbo;
    const bool needed = compact_tier && !options.disable_q8 && (qualify || row_requested ||
        std::ranges::any_of(kMiniWeightLoadDecisions, [](const auto& row) { return row.scope.backend == kMiniWeightBackend; }));
    MiniWeightScope out;
    out.backend = kMiniWeightBackend;
    if (!needed && (!compact_tier || kMiniProfile.ladder.empty())) return out;
    int ordinal = 0;
#if defined(__HIPCC__)
    if (hipGetDevice(&ordinal) != hipSuccess) return out;
    const auto visible = device::HipEnumerator{}.enumerate();
    const bool lowbit = row_requested || (options.qualify && (mini_lowbit_storage(options.qualify->target) ||
                                                               mini_lowbit_storage(options.qualify->head_target())));
    // The historical low-bit receipt used the device compiler's major and minor version.
    out.compiler_id = lowbit ? "hip-clang-" + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__)
                             : weight::hip_clang_compiler_id(__clang_version__);
    const auto identity = weight::implementation_identity(lowbit
        ? mini_lowbit_consumers(options.qualify ? options.qualify->head_target() : weight::kQ4G64WeightStorage)
        : std::span<const weight::ArithmeticId>(core::kResidentQ8ArithmeticHip));
    if (lowbit) out.regime = kMiniWeightRegime;
#else
    if (cudaGetDevice(&ordinal) != cudaSuccess) return out;
    const auto visible = device::CudaEnumerator{}.enumerate();
    out.compiler_id = "cuda-" + std::to_string(CUDART_VERSION / 1000) + "." + std::to_string(CUDART_VERSION % 1000 / 10);
    const auto identity = weight::implementation_identity(core::kResidentQ8ArithmeticCuda);
#endif
    for (const auto& device : visible)
        if (device.ordinal == ordinal) out.device_class = device.arch;
    // Empty when the runtime cannot describe the device: no weight row matches then.
#if defined(__HIPCC__)
    // The explicitly enabled calibrated row retains the scope of its historical gfx1151 receipt.
    out.weight_device_class = row_requested ? out.device_class : device::hip_cost_device_identity();
#else
    out.weight_device_class = device::cuda_cost_device_identity();
#endif
    if (!needed) return out;
    out.build = identity;
    const auto row_matches = [&out, extent](const weight::WeightLoadDecision& row) {
        return row.scope.backend == out.backend && row.scope.device_class == out.weight_device_class &&
               row.scope.compiler_id == out.compiler_id && row.scope.build == out.build &&
               weight::weight_load_gain_selects(row, extent);
    };
    if (!qualify && (row_requested ? std::ranges::none_of(kMiniLowBitDecisions, row_matches)
                                  : std::ranges::none_of(kMiniWeightLoadDecisions, row_matches))) return out;
    out.checkpoint = weight::Digest256{sha256_file_content(model).bytes};
    return out;
}

// The low-bit resident routes run on the switch points declared for the bound device (strix/device/lowbit_rows.hpp,
// bound by the shared device gate). A load that admitted a low-bit storage binds its device through that gate before
// it loads (STRIX_DEVICES / STRIX_DEVICE, as every bound binary); a server bound it already (same binding, no-op).
// Without a declaration the lone integrated device binds implicitly and runs one route.
inline void bind_mini_lowbit_device(const weight::WeightLoadPlan& plan) {
#if defined(__HIPCC__)
    if (mini_lowbit_storage(plan.storage) || mini_lowbit_storage(plan.head_storage))
        static_cast<void>(device::bind_hip());
#else
    static_cast<void>(plan);
#endif
}

inline std::uint32_t mini_device_weight_slots(std::uint64_t device_limit, std::uint64_t kv_limit,
                                              std::uint64_t slot_bytes, unsigned requested, bool calibrated) {
    const auto memory = device::native::memory_info();
    return mini_weight_slot_capacity(memory.free, memory.total, device_limit, kv_limit, slot_bytes, requested, calibrated);
}

inline void require_mini_calibrated_cli_slot(const MiniLowBitRows& lowbit, std::uint64_t slot_bytes) {
    if (!lowbit.admitted) return;
    std::uint64_t device_limit = 0;
#if defined(__HIPCC__)
    if (const auto* bound = device::hip_bound()) device_limit = bound->budget.device_bytes;
#endif
    static_cast<void>(mini_device_weight_slots(device_limit, 0, slot_bytes, 1U, true));
}

template <class Load>
auto load_mini_weights(const std::string& model, weight::WeightLoadPlan& plan, profile::Name tier,
                       const MiniWeightOptions& options, const weight::WeightLoadScope& scope,
                       weight::WeightLoadExtent extent, Load load) {
    const bool lowbit = mini_lowbit_storage(plan.storage) || mini_lowbit_storage(plan.head_storage);
    if (!lowbit)
        return load_mini_weight_pack(plan, tier, options, scope, extent, {}, nullptr, std::cerr, load);
    bind_mini_lowbit_device(plan);
    const bool measured = !options.qualify && !options.qualify_q8;
    std::string path;
    if (const char* supplied = envcfg::text("STRIX_LOWBIT_PACK"); supplied && *supplied) path = supplied;
    else if (measured) path = (std::filesystem::path(model).parent_path() / kMiniLowBitPackFile).string();
    auto incumbent_scope = scope;
    incumbent_scope.regime = {};
#if defined(__HIPCC__)
    // WeightLoadScope holds views: the id must outlive the fallback plan that copies this scope.
    static const std::string compiler_id = weight::hip_clang_compiler_id(__clang_version__);
    incumbent_scope.compiler_id = compiler_id;
    incumbent_scope.build = weight::implementation_identity(core::kResidentQ8ArithmeticHip);
#endif
    return load_mini_weight_pack(plan, tier, options, incumbent_scope, extent, path,
                                  measured ? &kMiniLowBitPackIdentity : nullptr, std::cerr, load);
}

} // namespace strix::inkling2
