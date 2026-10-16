#pragma once
#include "strix/model_runtime_binding.hpp"
#include "strix/moe_token_schedule.hpp"
#include "strix/serial_request.hpp"
#include "strix/speculation_controller.hpp"
#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>

namespace strix {
struct MoeTokenCostContext {
    ModelContentIdentity model_content;
    std::uint64_t execution_identity = 0, kernel_plan_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0, resolved_profile_identity = 0;
    std::uint64_t position = 0, context_capacity = 0, committed_tokens = 0, vocab_size = 0;
    std::uint32_t eos_token_id = 0;
    bool cancelled = false, ready = true;
};
struct ObservedMoeTokenCostPolicy {
    std::uint32_t bootstrap_samples_per_arm = 2, reexplore_interval = 32;
    double switch_hysteresis_fraction = 0.03;
    std::uint64_t context_bucket_tokens = 256;
};
struct MoeTokenCostArmStats {
    std::uint64_t selected = 0, observed = 0, censored = 0;
    double complete_wall_seconds = 0.0;
};
struct MoeTokenCostStats {
    // Index 0 serial-expanded, index 1 fork/join. Counts aggregate the three
    // separate intent controllers; costs never compete across output intents.
    std::array<MoeTokenCostArmStats, 2> arms{};
    std::uint64_t selection_changes = 0, key_resets = 0;
    MoeTokenSchedule last_selected = MoeTokenSchedule::fork_join;
    bool saturated = false;
};
// Explicit experimental online scheduling heuristic, NOT paired performance or
// numerical qualification. Borrowed inner/context/sink owners must outlive it.
// One serialized caller; no request batching or asynchronous publication.
// Provider describes actual session state; sink selects the SAME retained MoE
// dispatcher for exactly one target. Selection never changes output intent,
// tensor layout, variant IDs, arithmetic or serial lineage.
// Wall measurements include context lookup/mode selection, native completion
// and requested output delivery. Pure observation bookkeeping is outside them.
// EOS is censored when a greedy selection exists; state/full intents do not
// invent a selection solely for scheduling. Boundary/error/cancel samples never
// train the controller; after-publication observer failures are benign censorship.
class MoeTokenCostBackend final : public SerialGreedyStepBackend {
public:
    using ContextProvider = std::function<MoeTokenCostContext()>;
    using ModeSink = std::function<void(MoeTokenSchedule)>;
    MoeTokenCostBackend(SerialGreedyStepBackend&, ContextProvider, ModeSink,
                       ObservedMoeTokenCostPolicy = {});
    std::vector<float> execute(std::uint32_t, std::uint64_t) override;
    std::uint32_t execute_greedy(std::uint32_t, std::uint64_t, std::uint64_t) override;
    void execute_state(std::uint32_t, std::uint64_t) override;
    std::optional<GreedyPrefixRestore> try_restore_greedy_prefix(const SerialGreedyRequest&) override;
    void capture_greedy_prefix(std::span<const std::uint32_t>, std::uint32_t, std::uint64_t) override;
    [[nodiscard]] MoeTokenCostStats stats() const noexcept { return stats_; }
private:
    struct IntentState {
        explicit IntentState(ObservedSpeculationPolicy p) : controller(p) {}
        ObservedSpeculationController controller;
        MoeTokenCostContext key;
        std::uint64_t bucket = 0;
        bool initialized = false;
    };
    struct Observation {
        MoeTokenCostContext before;
        std::size_t intent = 0;
        std::uint32_t arm = 0;
    };
    Observation begin(std::size_t, std::uint32_t, std::uint64_t, std::uint64_t);
    void finish(const Observation&, double, std::optional<std::uint32_t>) noexcept;
    void censor(const Observation&) noexcept;
    SerialGreedyStepBackend& inner_;
    ContextProvider context_;
    ModeSink sink_;
    ObservedMoeTokenCostPolicy policy_;
    std::array<IntentState, 3> intents_;
    MoeTokenCostStats stats_;
    bool active_ = false, have_selection_ = false;
};
} // namespace strix
