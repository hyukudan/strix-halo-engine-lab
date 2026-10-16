#pragma once
#include "strix/hip_shared_expert_launcher.hpp"
#include <cstddef>
#include <stdexcept>
namespace strix {
struct HipSharedServingStaging {
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0, allocation_generation = 0;
    std::uint32_t capacity_rows = 0;
};
[[nodiscard]] inline std::uint64_t hip_shared_serving_staging_bytes(std::uint32_t rows) {
    if (rows != 2U && rows != 4U) throw std::invalid_argument("shared serving staging requires two or four rows");
    return 15360ULL * rows; // Three 256-aligned H2560/M640 row payloads.
}
// Independent original C1 contexts; arbitrary per-request positions. No window
// identity, same-state assertion or consecutive-position reinterpretation.
// Pure admission then explicit-stream gather/three existing row projections/
// row-local activation+scalar/scatter. Owner retains every request and staging
// through quiescence. No allocation/sync/status publication in either entry.
void shared_expert_serving_cohort_preflight(std::span<const ExecutionDispatchContext>,
    const HipSharedExpertLauncherConfig&,HipSharedServingStaging);
void shared_expert_serving_cohort_launch(std::span<const ExecutionDispatchContext>,
    const HipSharedExpertLauncherConfig&,HipSharedServingStaging);
} // namespace strix
