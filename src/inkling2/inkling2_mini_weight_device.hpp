#pragma once

#include "core/resident_row_plan.hpp"
#include "inkling2_mini_weight_options.hpp"
#include "strix/model_content_hash.hpp"

#if defined(__HIPCC__)
#include "strix/device/hip_enumerator.hpp"
#include <hip/hip_version.h>
#else
#include "strix/device/cuda_enumerator.hpp"
#endif

namespace strix::inkling2 {

// The scope a load-time weight decision row is matched against: the current device's architecture, the toolkit
// this binary was built with, the resident regime, the resident Q8_0 consumers' implementation identity and the raw
// checkpoint SHA-256 (shared persistent stat-keyed cache: only the first FAST run of a checkpoint reads it whole).
// Hash only for explicit qualification or when a default decision row can match.
// Owns the strings; view() builds the scope on use, so copies and moves never leave a view dangling.
struct MiniWeightScope {
    weight::Backend backend{};
    std::string device_class, compiler_id;
    std::string_view regime;
    weight::Digest256 build{}, checkpoint{};
    [[nodiscard]] weight::WeightLoadScope view() const {
        return {backend, device_class, compiler_id, regime, build, checkpoint};
    }
};

#if defined(__HIPCC__)
inline constexpr auto kMiniWeightBackend = weight::Backend::hip;
#else
inline constexpr auto kMiniWeightBackend = weight::Backend::cuda;
#endif

inline MiniWeightScope mini_weight_scope(const std::string& model, profile::Name tier,
                                         const MiniWeightOptions& options) {
    const bool compact_tier = tier == profile::Name::fast || tier == profile::Name::turbo;
    const bool needed = compact_tier && !options.disable_q8 &&
        (options.qualify_q8 || !kMiniWeightLoadDecisions.empty());
    MiniWeightScope out;
    out.backend = kMiniWeightBackend;
    if (!needed) return out;
    int ordinal = 0;
#if defined(__HIPCC__)
    if (hipGetDevice(&ordinal) != hipSuccess) return out;
    const auto visible = device::HipEnumerator{}.enumerate();
    out.compiler_id = "hip-" + std::to_string(HIP_VERSION_MAJOR) + "." + std::to_string(HIP_VERSION_MINOR);
    const auto identity = weight::implementation_identity(core::kResidentQ8ArithmeticHip);
#else
    if (cudaGetDevice(&ordinal) != cudaSuccess) return out;
    const auto visible = device::CudaEnumerator{}.enumerate();
    out.compiler_id = "cuda-" + std::to_string(CUDART_VERSION / 1000) + "." + std::to_string(CUDART_VERSION % 1000 / 10);
    const auto identity = weight::implementation_identity(core::kResidentQ8ArithmeticCuda);
#endif
    for (const auto& device : visible)
        if (device.ordinal == ordinal) out.device_class = device.arch;
    out.regime = kMiniResidentRegime;
    out.build = identity;
    out.checkpoint = weight::Digest256{sha256_file_content(model).bytes};
    return out;
}

} // namespace strix::inkling2
