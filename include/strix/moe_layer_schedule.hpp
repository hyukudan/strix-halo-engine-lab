#pragma once

#include "strix/moe_plan.hpp"

#include <cstdint>
#include <vector>

namespace strix {

enum class MoeLayerScheduleStage : std::uint8_t {
    routed_experts = 0,
    shared_expert = 1,
    combine = 2,
};

struct MoeLayerScheduleNode {
    MoeLayerScheduleStage stage =
        MoeLayerScheduleStage::routed_experts;
    std::vector<MoeLayerScheduleStage> dependencies;
};

struct MoeLayerSchedulePlan {
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t qualification_identity = 0;
    MoeExpertExecutionStrategy routed_strategy =
        MoeExpertExecutionStrategy::DirectAssignment;
    bool routed_shared_parallel = true;
    double estimated_layer_us = 0.0;
    std::vector<MoeLayerScheduleNode> nodes;
    std::uint64_t schedule_identity = 0;
};

[[nodiscard]] MoeLayerSchedulePlan
build_qualified_moe_layer_schedule(
    const MoeRouteRequest& request,
    std::uint64_t active_runtime_compatibility_identity,
    const MoeQualifiedLayerPolicy& policy);

}  // namespace strix
