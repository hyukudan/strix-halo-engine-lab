#pragma once
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
#include <cstdint>
namespace strix::model_residency_diagnostic {
// Dedicated test executable only: no environment or production hook.
enum class Fault { none, allocation, copy, device_alias, unregister, arena_alignment, alias_alignment, staging_allocation, staging_post_allocate };
struct FaultPlan {
    Fault fault = Fault::none;
    std::uint64_t occurrence = 1;
    // Dedicated fixture only; zero retains the production 32 MiB capacity cap.
    std::uint64_t staging_capacity_bytes = 0;
};
struct Stats {
    std::uint64_t staging_allocations = 0, acquired_staging_allocations = 0;
    std::uint64_t staging_frees = 0, released_staging_allocations = 0;
    std::uint64_t staging_copy_chunks = 0, maximum_staging_copy_bytes = 0;
    std::uint64_t registrations = 0, alias_queries = 0, allocations = 0, copies = 0;
    std::uint64_t unregisters = 0, frees = 0, acquired_allocations = 0, released_allocations = 0;
    std::uint64_t acquired_registrations = 0, released_registrations = 0, quarantined_mappings = 0;
};
void set_fault_plan(FaultPlan = {});
[[nodiscard]] Stats stats() noexcept;
void retire_quarantined_mappings();
}
#endif
