#pragma once
#include "strix/serving_cohort.hpp"
#include <cstdint>
#include <functional>
#include <memory>
namespace strix {
class HipServingGreedyBackend;
// One layer-major C1 target per independent session. Only the three canonical
// shared-Q8 projections decode weights across rows; all other operators retain
// their original per-request R1 dispatch and serial arithmetic lineage.
// Setup/execute/destruction are externally serialized and cannot run inside an
// unrelated active capture/scope. Each request and observer retains its actual
// owners through the complete call; no per-request owner is retained afterward.
class HipServingCohort final : public ServingCohortExecutor {
public:
    using Resolver = std::function<HipServingGreedyBackend*(SerialGreedyStepBackend&)>;
    HipServingCohort(std::shared_ptr<const void> shared_model_owner, std::uint32_t max_rows = 4,
                    Resolver resolver = {});
    ~HipServingCohort();
    HipServingCohort(const HipServingCohort&) = delete;
    HipServingCohort& operator=(const HipServingCohort&) = delete;
    // Exact device staging payload only. HIP stream/event driver overhead is
    // unmeasured and must remain covered by the caller's explicit headroom.
    [[nodiscard]] static std::uint64_t requested_reservation_bytes(std::uint32_t max_rows);
    [[nodiscard]] bool compatible(std::span<SerialGreedyStepBackend* const>) const noexcept override;
    void execute(std::span<ServingCohortTarget>) override;
    struct Stats { std::uint64_t cohorts=0, targets=0, shared_tensor_steps=0, fallback_r1_steps=0; bool saturated=false; };
    [[nodiscard]] Stats stats() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
