#include "strix/moe_layer_runtime.hpp"

#include <stdexcept>

namespace strix {
namespace {

void validate_schedule(
    const MoeLayerSchedulePlan& plan,
    std::uint64_t active_runtime_compatibility_identity) {
    if (plan.schedule_identity == 0 ||
        plan.qualification_identity == 0 ||
        plan.runtime_compatibility_identity == 0 ||
        active_runtime_compatibility_identity == 0 ||
        plan.runtime_compatibility_identity !=
            active_runtime_compatibility_identity) {
        throw std::invalid_argument(
            "MoE runtime schedule identity/runtime binding is invalid");
    }
    if (plan.nodes.size() != 3 ||
        plan.nodes[0].stage !=
            MoeLayerScheduleStage::routed_experts ||
        plan.nodes[1].stage !=
            MoeLayerScheduleStage::shared_expert ||
        plan.nodes[2].stage !=
            MoeLayerScheduleStage::combine) {
        throw std::invalid_argument(
            "MoE runtime schedule must contain routed/shared/combine stages");
    }

    if (!plan.nodes[0].dependencies.empty()) {
        throw std::invalid_argument(
            "routed expert stage must not depend on another MoE stage");
    }

    const auto& shared_dependencies =
        plan.nodes[1].dependencies;
    if (plan.routed_shared_parallel) {
        if (!shared_dependencies.empty()) {
            throw std::invalid_argument(
                "parallel shared expert stage must not depend on routed experts");
        }
    } else {
        if (shared_dependencies.size() != 1 ||
            shared_dependencies[0] !=
                MoeLayerScheduleStage::routed_experts) {
            throw std::invalid_argument(
                "serialized shared expert stage must depend on routed experts");
        }
    }

    const auto& combine_dependencies =
        plan.nodes[2].dependencies;
    if (combine_dependencies.size() != 2 ||
        combine_dependencies[0] !=
            MoeLayerScheduleStage::routed_experts ||
        combine_dependencies[1] !=
            MoeLayerScheduleStage::shared_expert) {
        throw std::invalid_argument(
            "MoE combine stage must depend on routed and shared experts");
    }
}

}  // namespace

MoeLayerRuntimeResult execute_qualified_moe_layer_schedule(
    const MoeLayerSchedulePlan& plan,
    std::uint64_t active_runtime_compatibility_identity,
    MoeLayerRuntimeBackend& backend) {
    validate_schedule(
        plan,
        active_runtime_compatibility_identity);

    MoeLayerRuntimeResult result;
    result.schedule_identity =
        plan.schedule_identity;
    result.routed_shared_parallel =
        plan.routed_shared_parallel;
    result.routed_strategy =
        plan.routed_strategy;
    result.completed_stages.reserve(3);

    if (plan.routed_shared_parallel) {
        backend.start_routed(
            plan.routed_strategy);
        backend.start_shared();

        backend.finish_routed();
        result.completed_stages.push_back(
            MoeLayerScheduleStage::routed_experts);

        backend.finish_shared();
        result.completed_stages.push_back(
            MoeLayerScheduleStage::shared_expert);
    } else {
        backend.start_routed(
            plan.routed_strategy);
        backend.finish_routed();
        result.completed_stages.push_back(
            MoeLayerScheduleStage::routed_experts);

        backend.start_shared();
        backend.finish_shared();
        result.completed_stages.push_back(
            MoeLayerScheduleStage::shared_expert);
    }

    backend.combine();
    result.completed_stages.push_back(
        MoeLayerScheduleStage::combine);
    return result;
}

}  // namespace strix
