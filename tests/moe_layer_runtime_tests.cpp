#include "strix/moe_layer_runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct RecordingBackend final
    : strix::MoeLayerRuntimeBackend {
    std::vector<std::string> events;
    strix::MoeExpertExecutionStrategy strategy =
        strix::MoeExpertExecutionStrategy::DirectAssignment;

    void start_routed(
        strix::MoeExpertExecutionStrategy value) override {
        strategy = value;
        events.push_back("start_routed");
    }
    void finish_routed() override {
        events.push_back("finish_routed");
    }
    void start_shared() override {
        events.push_back("start_shared");
    }
    void finish_shared() override {
        events.push_back("finish_shared");
    }
    void combine() override {
        events.push_back("combine");
    }
};

strix::MoeLayerSchedulePlan make_plan(
    bool parallel) {
    strix::MoeLayerSchedulePlan plan;
    plan.runtime_compatibility_identity =
        0x1111ULL;
    plan.qualification_identity =
        0x2222ULL;
    plan.routed_strategy =
        strix::MoeExpertExecutionStrategy::
            GroupedExpertMajor;
    plan.routed_shared_parallel =
        parallel;
    plan.schedule_identity =
        parallel ? 0x3333ULL : 0x4444ULL;

    plan.nodes.push_back({
        strix::MoeLayerScheduleStage::
            routed_experts,
        {},
    });
    plan.nodes.push_back({
        strix::MoeLayerScheduleStage::
            shared_expert,
        parallel
            ? std::vector<
                  strix::MoeLayerScheduleStage>{}
            : std::vector<
                  strix::MoeLayerScheduleStage>{
                  strix::MoeLayerScheduleStage::
                      routed_experts,
              },
    });
    plan.nodes.push_back({
        strix::MoeLayerScheduleStage::combine,
        {
            strix::MoeLayerScheduleStage::
                routed_experts,
            strix::MoeLayerScheduleStage::
                shared_expert,
        },
    });
    return plan;
}

}  // namespace

int main() {
    {
        const auto plan =
            make_plan(true);
        RecordingBackend backend;
        const auto result =
            strix::execute_qualified_moe_layer_schedule(
                plan,
                0x1111ULL,
                backend);

        expect(
            backend.events ==
                std::vector<std::string>({
                    "start_routed",
                    "start_shared",
                    "finish_routed",
                    "finish_shared",
                    "combine",
                }),
            "parallel MoE runtime starts both branches before joining them");
        expect(
            backend.strategy ==
                strix::MoeExpertExecutionStrategy::
                    GroupedExpertMajor &&
                result.completed_stages.size() == 3,
            "MoE runtime forwards qualified routed strategy and completes all stages");
    }

    {
        const auto plan =
            make_plan(false);
        RecordingBackend backend;
        static_cast<void>(
            strix::execute_qualified_moe_layer_schedule(
                plan,
                0x1111ULL,
                backend));
        expect(
            backend.events ==
                std::vector<std::string>({
                    "start_routed",
                    "finish_routed",
                    "start_shared",
                    "finish_shared",
                    "combine",
                }),
            "serialized MoE runtime finishes routed branch before shared branch");
    }

    {
        auto plan =
            make_plan(true);
        RecordingBackend backend;
        bool foreign_rejected = false;
        try {
            static_cast<void>(
                strix::execute_qualified_moe_layer_schedule(
                    plan,
                    0x9999ULL,
                    backend));
        } catch (const std::invalid_argument&) {
            foreign_rejected = true;
        }
        expect(
            foreign_rejected &&
                backend.events.empty(),
            "foreign runtime identity is rejected before any MoE launch");

        plan.nodes[2].dependencies.clear();
        bool malformed_rejected = false;
        try {
            static_cast<void>(
                strix::execute_qualified_moe_layer_schedule(
                    plan,
                    0x1111ULL,
                    backend));
        } catch (const std::invalid_argument&) {
            malformed_rejected = true;
        }
        expect(
            malformed_rejected &&
                backend.events.empty(),
            "malformed MoE dependency schedule is rejected before launch");
    }

    if (failures == 0) {
        std::cout
            << "MoE layer runtime tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
