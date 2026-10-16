#pragma once

#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/serial_request.hpp"
#include "strix/serial_session.hpp"
#include "strix/serving_cohort.hpp"
#include "strix/known_prefix_prefill.hpp"

#include <memory>

namespace strix {
class HipServingPrefixCache;
class HipSetupQuiescenceGuard;
struct HipServingPrefillPlan {
    std::uint32_t rows=0;
    CompiledKnownPrefixWindow state_window,terminal_window;
    std::optional<CompiledKnownPrefixPrefill> state_tiled,terminal_tiled;
    std::uint64_t workspace_bytes=0,alignment=256,metadata_payload_bytes=0;
    std::uint64_t runtime_payload_bytes=0,per_request_reservation_bytes=0;
    // Metadata covers known arena/vector payload, not allocator RSS or all
    // catalog/driver overhead; caller's explicit serving headroom remains.
};
struct HipServingPrefillStats {
    std::uint64_t chunks=0,consumed_rows=0,state_only_chunks=0,terminal_chunks=0;
    std::uint64_t r2_dispatches=0,shared_rows_dispatches=0,gdn_rows_dispatches=0,omitted_lm_heads=0;
    bool saturated=false;
};
// Borrowed only by the single cohort owner while this request is leased idle.
// Original C1 identities are never retagged to a known-window identity.
struct HipServingCohortMember {
    const CompiledSerialExecution* serial = nullptr;
    const ModelContract* model = nullptr;
    const HipLauncherCatalogConfig* config = nullptr;
    const ExecutionRuntimeResources* resources = nullptr;
    SerialSessionContext* session = nullptr;
    const CompiledLaunchPlan* launches = nullptr;
    DirectLaunchDispatcher* dispatcher = nullptr;
    HipLaunchContext scope;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    const HipSharedExpertLauncherConfig* shared_config=nullptr;
    const HipSharedExpertLauncherConfig* shared_fallback_config=nullptr;
};

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
        std::shared_ptr<HipServingPrefixCache> prefix_cache = {},
        std::shared_ptr<const HipServingPrefillPlan> prefill = {});
    ~HipServingGreedyBackend();
    // Pure requested selector-pool payload, with overflow/zero checks. Excludes
    // allocator/stream overhead, model/state/scratch and response storage.
    [[nodiscard]] static std::uint64_t output_reservation_bytes(std::uint32_t vocabulary_size);
    // Pure source planning; no HIP calls or allocation of native pools. The
    // caller charges the returned shared metadata and per-request payload
    // BEFORE an accepted factory may construct the backend.
    [[nodiscard]] static HipServingPrefillPlan prepare_prefill(const ModelContract&,
        const CompiledSerialExecution&,const ModelManifest&,const ExecutionTensorAccessPlan&,
        std::uint32_t rows,std::uint64_t workspace_budget,std::uint64_t metadata_payload_budget);
    [[nodiscard]] std::uint32_t prefill_chunk_rows() const noexcept override;
    [[nodiscard]] std::optional<std::uint32_t> execute_prefill_chunk(std::span<const std::uint32_t>,
        std::uint64_t position,std::uint64_t vocabulary,bool terminal_output) override;
    [[nodiscard]] HipServingPrefillStats prefill_stats() const noexcept;
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
private:
    friend class HipServingCohort;
    // Internal production seam for HipServingCohort. Caller excludes all other
    // accesses until completion/drain. Admission does not consume or publish.
    struct CohortKey { std::uint64_t execution_identity=0; const ModelRuntimeBinding* weights=nullptr;
        const SerialSessionContext* session=nullptr; int device=-1; };
    [[nodiscard]] CohortKey cohort_key() const noexcept;
    [[nodiscard]] HipServingCohortMember admit_cohort_target(const SerialGreedyTargetInput&);
    void begin_cohort_target() noexcept;
    [[nodiscard]] std::optional<std::uint32_t> complete_cohort_target(const SerialGreedyTargetInput&);
    void publish_cohort_target(const SerialGreedyTargetInput&,std::optional<std::uint32_t>) noexcept;
    void poison_cohort_target() noexcept;
public:
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] int device_index() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix
