#pragma once
#include "strix/execution_tensor_access.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace strix::qsa_input_prefetch {
inline constexpr std::string_view kernel_suffix = ".prefetch8.virtual256.q8-qsa-in";
inline constexpr std::string_view projection_base_id = "qsa.linear.legacy256";
inline constexpr std::string_view projection_prefetch_id =
    "qsa.linear.prefetch8.virtual256.q8-qsa-in";
struct RuntimeDispatchStats {
    // Host submissions: eager calls and graph capture, not device graph replay.
    std::atomic<std::uint64_t> dispatched_projections{0};
};
struct Projection {
    std::string tensor_name;
    std::uint32_t layer = 0;
    bool eligible = false;
};
struct Plan {
    std::vector<std::uint32_t> layers;
    std::vector<Projection> projections;
    std::uint64_t eligible_steps = 0, fallback_steps = 0, eligible_projections = 0,
                  fallback_projections = 0;
};
[[nodiscard]] bool eligible(const ManifestTensor &) noexcept;
[[nodiscard]] std::uint64_t required_weight_bytes(std::uint64_t width, std::uint64_t outputs);
[[nodiscard]] Plan build_plan(const ModelContract &, const ModelManifest &);
void validate_tensor_access(const Plan &, const ModelManifest &, const SerialExecutionPlan &,
                            const ExecutionTensorAccessPlan &);
void validate_layers(const ModelContract &, bool enabled, std::span<const std::uint32_t>);
[[nodiscard]] bool selected(bool enabled, std::span<const std::uint32_t>,
                            std::optional<std::uint32_t>) noexcept;
} // namespace strix::qsa_input_prefetch
