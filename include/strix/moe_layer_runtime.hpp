#pragma once

#include "strix/moe_layer_schedule.hpp"

#include <cstdint>
#include <vector>

namespace strix {

class MoeLayerRuntimeBackend {
public:
    virtual ~MoeLayerRuntimeBackend() = default;

    virtual void start_routed(
        MoeExpertExecutionStrategy strategy) = 0;
    virtual void finish_routed() = 0;

    virtual void start_shared() = 0;
    virtual void finish_shared() = 0;

    virtual void combine() = 0;
};

struct MoeLayerRuntimeResult {
    std::uint64_t schedule_identity = 0;
    bool routed_shared_parallel = false;
    MoeExpertExecutionStrategy routed_strategy =
        MoeExpertExecutionStrategy::DirectAssignment;
    std::vector<MoeLayerScheduleStage> completed_stages;
};

[[nodiscard]] MoeLayerRuntimeResult
execute_qualified_moe_layer_schedule(
    const MoeLayerSchedulePlan& plan,
    std::uint64_t active_runtime_compatibility_identity,
    MoeLayerRuntimeBackend& backend);

}  // namespace strix
