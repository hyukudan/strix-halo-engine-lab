#include "strix/moe_layer_schedule.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_identity(
    const MoeLayerSchedulePlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, plan.runtime_compatibility_identity);
    hash_u64(hash, plan.qualification_identity);
    hash_u64(hash, static_cast<std::uint64_t>(plan.routed_strategy));
    hash_u64(hash, plan.routed_shared_parallel ? 1ULL : 0ULL);
    hash_u64(hash, plan.nodes.size());
    for (const auto& node : plan.nodes) {
        hash_u64(hash, static_cast<std::uint64_t>(node.stage));
        hash_u64(hash, node.dependencies.size());
        for (const auto dependency : node.dependencies) {
            hash_u64(hash, static_cast<std::uint64_t>(dependency));
        }
    }
    return hash;
}

}  // namespace

MoeLayerSchedulePlan
build_qualified_moe_layer_schedule(
    const MoeRouteRequest& request,
    std::uint64_t active_runtime_compatibility_identity,
    const MoeQualifiedLayerPolicy& policy) {
    const auto decision =
        select_qualified_moe_layer_execution_strategy(
            request,
            active_runtime_compatibility_identity,
            policy);

    MoeLayerSchedulePlan plan;
    plan.runtime_compatibility_identity =
        active_runtime_compatibility_identity;
    plan.qualification_identity =
        policy.qualification_identity;
    plan.routed_strategy =
        decision.strategy;
    plan.routed_shared_parallel =
        decision.routed_shared_parallel;
    plan.estimated_layer_us =
        decision.strategy ==
                MoeExpertExecutionStrategy::GroupedExpertMajor
            ? decision.grouped_layer_estimated_us
            : decision.direct_layer_estimated_us;

    if (!std::isfinite(plan.estimated_layer_us) ||
        plan.estimated_layer_us < 0.0) {
        throw std::invalid_argument(
            "qualified MoE layer schedule has invalid estimated latency");
    }

    plan.nodes.push_back(
        MoeLayerScheduleNode{
            MoeLayerScheduleStage::routed_experts,
            {},
        });

    MoeLayerScheduleNode shared;
    shared.stage =
        MoeLayerScheduleStage::shared_expert;
    if (!plan.routed_shared_parallel) {
        shared.dependencies.push_back(
            MoeLayerScheduleStage::routed_experts);
    }
    plan.nodes.push_back(std::move(shared));

    plan.nodes.push_back(
        MoeLayerScheduleNode{
            MoeLayerScheduleStage::combine,
            {
                MoeLayerScheduleStage::routed_experts,
                MoeLayerScheduleStage::shared_expert,
            },
        });

    plan.schedule_identity =
        build_identity(plan);
    if (plan.schedule_identity == 0) {
        throw std::logic_error(
            "qualified MoE layer schedule identity must be non-zero");
    }
    return plan;
}

}  // namespace strix
