#pragma once
#include "strix/linux_uma_observation.hpp"
#include <functional>
#include <optional>
#include <vector>

namespace strix {
using PreflightNumber = std::optional<std::uint64_t>;
struct PreflightScalar {
    PreflightNumber value;
    LinuxUmaReadStatus status = LinuxUmaReadStatus::unavailable;
};
struct PreflightPaths { std::filesystem::path root = "/"; };
enum class PreflightThp { unavailable, always, madvise, never };
struct PreflightLimit { PreflightNumber bytes; bool infinite = false; };
struct PreflightProcessFacts {
    PreflightNumber page_size, self_pid;
    PreflightLimit memlock_soft, memlock_hard;
    // A DRM card index is an explicit mapping, not a HIP ordinal.
    std::optional<unsigned> drm_card;
    std::optional<bool> uma;
    std::optional<std::vector<unsigned>> allowed_nodes;
};
using PreflightFactProvider = std::function<PreflightProcessFacts()>;
struct PreflightInput { PreflightNumber G, H, R, L, C; };
struct PreflightBuddy {
    PreflightScalar blocks, units_2m;
};
struct PreflightVmstat { PreflightScalar stall, success, fail; };
struct PreflightObservation {
    PreflightScalar mem_total, mem_available, gtt_total, gtt_used, vram_total, vram_used, other_clients;
    PreflightBuddy buddy;
    PreflightVmstat compact;
    PreflightThp thp = PreflightThp::unavailable;
    PreflightProcessFacts process;
    bool device_ambiguous = false;
    PreflightNumber amdgpu_cards;
    std::uint64_t begin_ns = 0, end_ns = 0, process_identity = 0;
};
enum class PreflightWarning { unavailable, gtt_plan_pressure, host_plan_pressure, preexisting_gtt_use,
    allocation_owner_unresolved, firmware_reserve_review, contiguous_supply_risk,
    startup_compaction_pressure, pin_limit_risk };
struct PreflightAssessment {
    PreflightInput plan;
    PreflightNumber gtt_free, gtt_deficit, host_deficit, needed_2m_units, stall_delta, elapsed_ns;
    std::vector<PreflightWarning> warnings;
    // Facts worth reading but not pressure: never alert or fail on these alone.
    std::vector<PreflightWarning> advisories;
};
[[nodiscard]] LinuxUmaObservation parse_preflight_meminfo(std::string_view);
[[nodiscard]] PreflightScalar parse_preflight_scalar(std::string_view);
[[nodiscard]] PreflightBuddy parse_preflight_buddyinfo(std::string_view, PreflightNumber page_size,
    const std::optional<std::vector<unsigned>>& allowed_nodes = std::nullopt);
[[nodiscard]] PreflightVmstat parse_preflight_vmstat(std::string_view);
[[nodiscard]] PreflightThp parse_preflight_thp(std::string_view);
[[nodiscard]] PreflightProcessFacts preflight_process_facts();
[[nodiscard]] PreflightObservation collect_preflight(const PreflightPaths& = {},
    const PreflightFactProvider& = preflight_process_facts) noexcept;
[[nodiscard]] PreflightAssessment assess_preflight(const PreflightObservation&, const PreflightInput& = {},
    const PreflightObservation* previous = nullptr);
[[nodiscard]] std::string render_preflight(const PreflightObservation&, const PreflightAssessment&);
// Startup-only wrapper; disabled preflight performs no collection or provider calls.
// Inspection runs (--print-effective-config) pass inspection=true: their combined
// output carries exactly one configuration object, so nothing is emitted there.
struct StartupPreflight {
    PreflightObservation observation;
    PreflightAssessment assessment;
};
std::optional<StartupPreflight> emit_qwen_startup_diagnostics(bool inspection = false, const PreflightPaths& = {},
    const PreflightFactProvider& = preflight_process_facts) noexcept;
} // namespace strix
