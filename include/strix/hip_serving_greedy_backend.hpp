#pragma once

#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/serial_request.hpp"
#include "strix/serial_session.hpp"

#include <memory>

namespace strix {
class HipServingPrefixCache;

// One independently owned C1 session, transferable between scheduler workers
// only while idle. The scheduler owns greedy/EOS/capacity/cancellation policy.
// This backend never batches unrelated sessions or predicts a second input.
//
// actual_owner retains the exact original model/state/arena/PLE/stream/status
// allocations and resource descriptors. Immutable weights may be shared;
// mutable state, scratch, PLE and the D1 record MUST be exclusive to this session.
// D1 is required; there is no shadow, identity retag or status clear. Its record
// must be initialized to zero before construction and remain sticky thereafter.
// The stream is explicitly owned by actual_owner, not created or destroyed here.
//
// Calls establish the captured original device on their worker thread. No two
// threads may access this instance/session/owners concurrently. A submitted-call
// failure permanently disables the backend and poisons/taints its session after
// proving quiescence. Output completion/finite checks occur BEFORE token commit.
// Constructor/admission is not native or model qualification evidence.
class HipServingGreedyBackend final : public SerialGreedyStepBackend {
public:
    HipServingGreedyBackend(CompiledSerialExecution serial_snapshot, ModelContract model,
        HipLauncherCatalogConfig config, const ExecutionRuntimeResources& original_resources,
        SerialSessionContext& session, HipLaunchContext eager_scope,
        std::shared_ptr<const void> actual_owner, bool prepared_eager = false,
        std::shared_ptr<HipServingPrefixCache> prefix_cache = {});
    ~HipServingGreedyBackend();
    // Pure requested selector-pool payload, with overflow/zero checks. Excludes
    // allocator/stream overhead, model/state/scratch and response storage.
    [[nodiscard]] static std::uint64_t output_reservation_bytes(std::uint32_t vocabulary_size);
    HipServingGreedyBackend(const HipServingGreedyBackend&) = delete;
    HipServingGreedyBackend& operator=(const HipServingGreedyBackend&) = delete;
    HipServingGreedyBackend(HipServingGreedyBackend&&) = delete;
    HipServingGreedyBackend& operator=(HipServingGreedyBackend&&) = delete;

    [[nodiscard]] std::vector<float> execute(std::uint32_t input_token,
                                           std::uint64_t position) override;
    [[nodiscard]] std::uint32_t execute_greedy(std::uint32_t input_token,
        std::uint64_t position, std::uint64_t vocab_size) override;
    // Omits only the terminal state-free head via GenerationOutputDispatcher;
    // persistent state/PLE/D1 completion and lineage commit remain unchanged.
    void execute_state(std::uint32_t input_token, std::uint64_t position) override;
    [[nodiscard]] std::optional<GreedyPrefixRestore> try_restore_greedy_prefix(const SerialGreedyRequest&) override;
    void capture_greedy_prefix(std::span<const std::uint32_t>,std::uint32_t pending,std::uint64_t vocabulary) override;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] int device_index() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix
