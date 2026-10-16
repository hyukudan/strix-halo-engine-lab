#include "strix/speculation_executor.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeAllocator final
    : public strix::RuntimeMemoryAllocator {
public:
    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        const auto result = align(next_device, alignment);
        next_device =
            result +
            static_cast<std::uintptr_t>(bytes + alignment);
        return result;
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        const auto result = align(next_host, alignment);
        next_host =
            result +
            static_cast<std::uintptr_t>(bytes + alignment);
        return result;
    }

    void free_device(
        std::uintptr_t,
        std::uint64_t) noexcept override {}

    void free_host(
        std::uintptr_t,
        std::uint64_t) noexcept override {}

private:
    static std::uintptr_t align(
        std::uintptr_t value,
        std::uint64_t alignment) {
        const auto remainder =
            value %
            static_cast<std::uintptr_t>(alignment);
        return remainder == 0
            ? value
            : value +
                  static_cast<std::uintptr_t>(alignment - remainder);
    }

    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(0xd0000000ULL);
    std::uintptr_t next_host =
        static_cast<std::uintptr_t>(0xe0000000ULL);
};

class CountingTransfer final
    : public strix::SessionSnapshotTransferBackend {
public:
    int captures = 0;
    int restores = 0;
    int fail_restore_at = -1;
    void restored() {
        if (restores++ == fail_restore_at) throw std::runtime_error("injected partial restore failure");
    }

    void capture_device(
        std::uintptr_t,
        std::uintptr_t,
        std::uint64_t) override {
        ++captures;
    }

    void capture_host(
        std::uintptr_t,
        std::uintptr_t,
        std::uint64_t) override {
        ++captures;
    }

    void restore_device(
        std::uintptr_t,
        std::uintptr_t,
        std::uint64_t) override {
        restored();
    }

    void restore_host(
        std::uintptr_t,
        std::uintptr_t,
        std::uint64_t) override {
        restored();
    }
};

class FakeDraftBackend final
    : public strix::GreedyDraftProposalBackend {
public:
    std::vector<std::uint32_t> proposed_tokens;
    int calls = 0;
    std::uint32_t seen_max_depth = 0;

    int reconcile_calls = 0;
    bool fail_reconcile = false;
    std::vector<std::uint32_t> reconciled_tokens;

    std::vector<std::uint32_t>
    propose_candidates(
        const strix::SerialSessionContext&,
        const strix::DraftProposalStateContext& proposal_state,
        std::uint32_t max_candidate_depth) override {
        ++calls;
        seen_max_depth = max_candidate_depth;
        expect(proposal_state.phase ==
                   strix::DraftProposalStatePhase::proposal_active,
               "draft backend sees active proposal state");
        return proposed_tokens;
    }

    void reconcile_after_target(
        const strix::DraftProposalStateContext& proposal_state,
        const strix::SerialSessionContext&,
        std::span<const std::uint32_t> emitted_tokens) override {
        ++reconcile_calls;
        expect(proposal_state.phase ==
                   strix::DraftProposalStatePhase::reconciliation_required,
               "draft backend reconciles only dirty proposal state");
        if (fail_reconcile) {
            throw std::runtime_error(
                "injected draft reconciliation failure");
        }
        reconciled_tokens.assign(
            emitted_tokens.begin(),
            emitted_tokens.end());
    }
};

class FakeSpeculationBackend final
    : public strix::GreedySpeculationExecutionBackend {
public:
    std::vector<std::uint32_t> target_tokens;
    bool fail_verify = false;
    bool publish_lineage = false;
    bool native_failure = false;
    strix::ExecutionMemoryBinding* mutate_scratch = nullptr;
    int fail_replay_after = -1;
    int verify_calls = 0;
    int replay_calls = 0;

    std::vector<std::uint32_t>
    verify_target_candidates(
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources&,
        std::span<const std::uint32_t> candidates) override {
        ++verify_calls;
        if (native_failure) {
            session.health = strix::SerialSessionHealth::poisoned;
            session.arithmetic_lineage.tainted = true;
            throw std::runtime_error("injected native completion failure");
        }
        if (fail_verify) {
            throw std::runtime_error(
                "injected verify failure");
        }

        if (!candidates.empty()) {
            const auto depth =
                static_cast<std::uint64_t>(
                    candidates.size());
            if (publish_lineage) session.arithmetic_lineage = strix::prepare_prefill_chunk_lineage(
                session.arithmetic_lineage, session.next_position, session.next_position+depth, 0x1234, 0x5678, 0x9abc);
            session.next_position += depth;
            session.committed_tokens += depth;
            for (auto& generation :
                 session.state.generations) {
                ++generation;
            }
        }
        if (mutate_scratch) ++mutate_scratch->allocation_generation;
        return target_tokens;
    }

    void replay_target_token(
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources&,
        std::uint32_t) override {
        if (fail_replay_after >= 0 &&
            replay_calls == fail_replay_after) {
            ++replay_calls;
            throw std::runtime_error(
                "injected replay failure");
        }
        ++replay_calls;
        if (publish_lineage) session.arithmetic_lineage = strix::prepare_serial_row_lineage(
            session.arithmetic_lineage,session.next_position,session.execution_identity,0x7777);
        ++session.next_position;
        ++session.committed_tokens;
        for (auto& generation :
             session.state.generations) {
            ++generation;
        }
    }
};

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

struct Fixture {
    strix::ModelContract model =
        strix::ModelContract::qwen38_flash_next();
    strix::CompiledSerialExecution compiled;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionTensorAccessPlan tensor_access;
    strix::SerialSessionStateLayout state_layout;
    strix::SessionStateMemoryBinding state_memory;
    strix::ExecutionMemoryBinding scratch;
    strix::ExecutionRuntimeResources resources;
    strix::SerialSessionContext session;

    Fixture() {
        compiled.execution_identity =
            0x1111ULL;
        compiled.arena.alignment = 256;
        compiled.arena.total_bytes = 4096;

        strix::ManifestTensor tensor;
        tensor.name = "fixture.weight";
        tensor.dimensions = {64, 64};
        tensor.ggml_type = 1;
        tensor.byte_size = 8192;
        tensor.absolute_offset = 4096;
        tensor.role =
            strix::TensorRole::dense_projection;
        manifest.architecture = "fixture";
        manifest.tensors.push_back(tensor);
        manifest.total_tensor_bytes =
            tensor.byte_size;
        manifest.layout_identity =
            strix::model_manifest_layout_identity(
                manifest);

        weights.manifest_layout_identity =
            manifest.layout_identity;
        weights.model_content_identity.bytes[0] =
            0x42;
        weights.tensors.push_back(
            {
                0,
                strix::RuntimeTensorStorage::resident,
                static_cast<std::uintptr_t>(
                    0xf0000000ULL),
                tensor.byte_size,
                1,
                0,
            });

        tensor_access.manifest_layout_identity =
            manifest.layout_identity;
        tensor_access.plan_identity =
            0x2222ULL;

        strix::SerialSessionStateLayoutRequest state_request;
        state_request.context_tokens = 128;
        state_layout =
            strix::build_serial_session_state_layout(
                model,
                state_request);

        state_memory.layout_identity =
            state_layout.layout_identity;
        state_memory.device_base =
            static_cast<std::uintptr_t>(
                0x100000000ULL);
        state_memory.device_bytes =
            state_layout.device_total_bytes;
        state_memory.device_allocation_generation = 1;
        state_memory.host_base =
            static_cast<std::uintptr_t>(
                0x200000000ULL);
        state_memory.host_bytes =
            state_layout.host_total_bytes;
        state_memory.host_allocation_generation = 2;

        scratch.execution_identity =
            compiled.execution_identity;
        scratch.arena_base =
            static_cast<std::uintptr_t>(
                0x300000000ULL);
        scratch.arena_bytes =
            compiled.arena.total_bytes;
        scratch.allocation_generation = 3;

        resources.scratch = &scratch;
        resources.model_manifest = &manifest;
        resources.model_weights = &weights;
        resources.tensor_access = &tensor_access;
        resources.session_state_layout =
            &state_layout;
        resources.session_state_memory =
            &state_memory;

        const auto ids =
            strix::build_runtime_execution_identities(
                compiled.execution_identity,
                compiled.arena,
                resources);

        session.execution_identity =
            compiled.execution_identity;
        session.state_layout_identity =
            state_layout.layout_identity;
        session.runtime_compatibility_identity =
            ids.compatibility_identity;
        session.next_position = 10;
        session.committed_tokens = 10;
        session.state.valid =
            all_state();
        session.state.generations.fill(5);
    }
};

}  // namespace

int main() {
    FakeAllocator allocator;
    strix::AllocationGenerationSource generations;
    CountingTransfer transfer;

    {
        Fixture fixture;
        FakeSpeculationBackend backend;
        backend.target_tokens = {
            10, 20, 99, 77, 88};
        const std::vector<std::uint32_t> candidates{
            10, 20, 30, 40};

        strix::GreedySpeculationExecutionRequest request;
        request.candidate_tokens = candidates;
        request.token_budget = 5;

        const auto result =
            strix::execute_greedy_speculation_cycle(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                backend);

        expect(result.snapshot_captured &&
                   result.restored_pre_window &&
                   result.replayed_tokens == 2,
               "partial match captures, restores and replays accepted prefix");
        expect(result.plan.emitted_tokens ==
                   std::vector<std::uint32_t>({10, 20, 99}),
               "executor returns accepted prefix plus target bonus");
        expect(fixture.session.next_position == 12 &&
                   fixture.session.committed_tokens == 12 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready,
               "partial-match cycle leaves target state after accepted candidates only");
        expect(backend.verify_calls == 1 &&
                   backend.replay_calls == 2,
               "partial-match cycle verifies once and replays exact accepted count");
    }

    {
        Fixture fixture;
        FakeSpeculationBackend backend;
        backend.target_tokens = {
            1, 2, 3, 4};
        const std::vector<std::uint32_t> candidates{
            1, 2, 3};

        strix::GreedySpeculationExecutionRequest request;
        request.candidate_tokens = candidates;
        request.token_budget = 4;

        const auto result =
            strix::execute_greedy_speculation_cycle(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                backend);

        expect(result.snapshot_captured &&
                   !result.restored_pre_window &&
                   result.replayed_tokens == 0,
               "all-match cycle retains verified target state");
        expect(fixture.session.next_position == 13 &&
                   fixture.session.committed_tokens == 13,
               "all-match target state advances through every candidate only");
    }

    {
        Fixture fixture;
        FakeSpeculationBackend backend;
        backend.target_tokens = {42};

        strix::GreedySpeculationExecutionRequest request;
        request.token_budget = 1;

        const auto result =
            strix::execute_greedy_speculation_cycle(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                backend);

        expect(!result.snapshot_captured &&
                   !result.restored_pre_window &&
                   fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10,
               "depth-zero cycle does not mutate target state");
    }

    {
        Fixture fixture;
        FakeSpeculationBackend backend;
        backend.fail_verify = true;
        backend.target_tokens = {
            1, 2, 3};
        const std::vector<std::uint32_t> candidates{
            1, 2};

        strix::GreedySpeculationExecutionRequest request;
        request.candidate_tokens = candidates;
        request.token_budget = 3;

        bool failed = false;
        try {
            static_cast<void>(
                strix::execute_greedy_speculation_cycle(
                    fixture.model,
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    request,
                    allocator,
                    generations,
                    transfer,
                    backend));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        expect(failed &&
                   fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready,
               "verify failure restores pre-window session when snapshot exists");
    }

    {
        Fixture fixture;
        FakeSpeculationBackend backend;
        backend.target_tokens = {
            7, 88, 77};
        backend.fail_replay_after = 0;
        const std::vector<std::uint32_t> candidates{
            7, 8};

        strix::GreedySpeculationExecutionRequest request;
        request.candidate_tokens = candidates;
        request.token_budget = 3;

        bool failed = false;
        try {
            static_cast<void>(
                strix::execute_greedy_speculation_cycle(
                    fixture.model,
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    request,
                    allocator,
                    generations,
                    transfer,
                    backend));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        expect(failed &&
                   fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10 &&
                   fixture.session.health ==
                       strix::SerialSessionHealth::ready,
               "replay failure rolls session back to pre-window snapshot");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        draft.proposed_tokens = {10, 99};
        FakeSpeculationBackend backend;
        backend.target_tokens = {10, 20, 30};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 2;
        request.token_budget = 3;

        const auto result =
            strix::execute_greedy_speculation_with_draft(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                draft_state,
                draft,
                backend);

        expect(result.proposal_depth_limit == 2 &&
                   draft.calls == 1 &&
                   draft.reconcile_calls == 1 &&
                   draft.seen_max_depth == 2 &&
                   draft_state.phase ==
                       strix::DraftProposalStatePhase::synchronized &&
                   result.proposed_tokens ==
                       std::vector<std::uint32_t>({10, 99}),
               "draft wrapper obtains bounded candidate tokens before target verification");
        expect(result.target_cycle.plan.emitted_tokens ==
                   std::vector<std::uint32_t>({10, 20}) &&
                   result.target_cycle.replayed_tokens == 1 &&
                   fixture.session.next_position == 11,
               "draft wrapper preserves target-owned greedy acceptance and replay");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        draft.proposed_tokens = {1, 2};
        FakeSpeculationBackend backend;
        backend.target_tokens = {42};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 0;
        request.token_budget = 1;

        const auto result =
            strix::execute_greedy_speculation_with_draft(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                draft_state,
                draft,
                backend);

        expect(result.proposal_depth_limit == 0 &&
                   draft.calls == 0 &&
                   draft.reconcile_calls == 0 &&
                   result.proposed_tokens.empty() &&
                   result.target_cycle.plan.emitted_tokens ==
                       std::vector<std::uint32_t>({42}),
               "depth-zero wrapper bypasses drafter and uses target baseline");
        expect(fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10,
               "depth-zero draft wrapper leaves target state unchanged");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        draft.proposed_tokens = {1, 2, 3};
        FakeSpeculationBackend backend;
        backend.target_tokens = {1, 2, 3};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 2;
        request.token_budget = 3;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_greedy_speculation_with_draft(
                    fixture.model,
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    request,
                    allocator,
                    generations,
                    transfer,
                    draft_state,
                    draft,
                    backend));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected &&
                   backend.verify_calls == 0 &&
                   draft_state.phase ==
                       strix::DraftProposalStatePhase::poisoned &&
                   fixture.session.next_position == 10 &&
                   fixture.session.committed_tokens == 10,
               "over-depth draft result fails before target verification");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        draft.proposed_tokens = {10, 99};
        draft.fail_reconcile = true;
        FakeSpeculationBackend backend;
        backend.target_tokens = {10, 20, 30};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 2;
        request.token_budget = 3;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_greedy_speculation_with_draft(
                    fixture.model,
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    request,
                    allocator,
                    generations,
                    transfer,
                    draft_state,
                    draft,
                    backend));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        expect(rejected &&
                   draft_state.phase ==
                       strix::DraftProposalStatePhase::poisoned,
               "draft reconciliation failure poisons proposal state");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        strix::poison_draft_proposal_state(
            draft_state);
        FakeSpeculationBackend backend;
        backend.target_tokens = {42};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 0;
        request.token_budget = 1;

        const auto result =
            strix::execute_greedy_speculation_with_draft(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                draft_state,
                draft,
                backend);

        expect(result.target_cycle.plan.emitted_tokens ==
                   std::vector<std::uint32_t>({42}) &&
                   draft.calls == 0 &&
                   draft_state.phase ==
                       strix::DraftProposalStatePhase::poisoned,
               "depth-zero fallback remains available with poisoned drafter");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        ++draft_state.synchronized_target_position;
        FakeSpeculationBackend backend;

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 1;
        request.token_budget = 2;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_greedy_speculation_with_draft(
                    fixture.model,
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    request,
                    allocator,
                    generations,
                    transfer,
                    draft_state,
                    draft,
                    backend));
        } catch (const std::logic_error&) {
            rejected = true;
        }
        expect(rejected &&
                   draft.calls == 0 &&
                   backend.verify_calls == 0,
               "stale draft state fails before proposal or target verification");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        draft.proposed_tokens = {10};
        FakeSpeculationBackend backend;
        backend.target_tokens = {10, 20};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 4;
        request.token_budget = 2;

        const auto result =
            strix::execute_greedy_speculation_with_draft(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                draft_state,
                draft,
                backend);

        expect(result.proposal_depth_limit == 1 &&
                   draft.seen_max_depth == 1,
               "draft depth is clamped to token budget minus target bonus");
    }

    {
        Fixture fixture;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        FakeSpeculationBackend backend;

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 2;
        request.token_budget = 0;

        bool rejected = false;
        try {
            static_cast<void>(
                strix::execute_greedy_speculation_with_draft(
                    fixture.model,
                    fixture.compiled,
                    fixture.resources,
                    fixture.session,
                    request,
                    allocator,
                    generations,
                    transfer,
                    draft_state,
                    draft,
                    backend));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected &&
                   draft.calls == 0 &&
                   draft_state.phase ==
                       strix::DraftProposalStatePhase::synchronized,
               "invalid token budget fails before draft proposal state changes");
    }

    {
        Fixture fixture;
        fixture.session.next_position =
            fixture.state_layout.context_tokens - 1;
        fixture.session.committed_tokens =
            fixture.session.next_position;
        FakeDraftBackend draft;
        auto draft_state =
            strix::make_draft_proposal_state(
                0x5555ULL,
                fixture.session);
        draft.proposed_tokens = {10};
        FakeSpeculationBackend backend;
        backend.target_tokens = {10, 20};

        strix::GreedyDraftExecutionRequest request;
        request.max_candidate_depth = 4;
        request.token_budget = 5;

        const auto result =
            strix::execute_greedy_speculation_with_draft(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                draft_state,
                draft,
                backend);

        expect(result.proposal_depth_limit == 1 &&
                   draft.seen_max_depth == 1,
               "draft depth is clamped to remaining allocated target context");
    }

    // Transaction-owned producer history is republished only after successful
    // raw-state restoration; byte restore alone cannot heal a native failure.
    for (const unsigned accepted : {0U,1U,2U}) {
        Fixture fixture;
        for(std::uint64_t pos=0;pos<fixture.session.next_position;++pos)
            fixture.session.arithmetic_lineage=strix::prepare_serial_row_lineage(
                fixture.session.arithmetic_lineage,pos,fixture.compiled.execution_identity,0x7777);
        const auto original=fixture.session.arithmetic_lineage;
        FakeSpeculationBackend backend;
        backend.publish_lineage=true;
        backend.target_tokens=accepted==0U ? std::vector<std::uint32_t>{9,8,7} :
            accepted==1U ? std::vector<std::uint32_t>{1,9,7} : std::vector<std::uint32_t>{1,2,7};
        const std::vector<std::uint32_t> candidates{1,2};
        strix::GreedySpeculationExecutionRequest request;
        request.candidate_tokens=candidates;request.token_budget=3;
        CountingTransfer local_transfer;
        const auto result=strix::execute_greedy_speculation_cycle(fixture.model,fixture.compiled,
            fixture.resources,fixture.session,request,allocator,generations,local_transfer,backend);
        expect(strix::arithmetic_lineage_valid(fixture.session.arithmetic_lineage,10U+accepted),
            "accepted/replayed target has complete valid producer history");
        expect(fixture.session.next_position==10U+accepted && result.replayed_tokens==(accepted==2U?0U:accepted),
            "rollback follows accepted input count and leaves emitted bonus unconsumed");
        if(accepted==0U) expect(fixture.session.arithmetic_lineage.intervals==original.intervals &&
            fixture.session.arithmetic_lineage.history_identity==original.history_identity &&
            local_transfer.restores==local_transfer.captures && local_transfer.restores>=7,
            "zero-match rollback restores every state range and exact original history");
        if(accepted==1U) expect(fixture.session.arithmetic_lineage.intervals.back().producer==strix::ArithmeticProducer::serial_row &&
            fixture.session.arithmetic_lineage.intervals.back().end==11,
            "one-match rollback appends actual serial replay history rather than verifier chunk");
        if(accepted==2U) expect(fixture.session.arithmetic_lineage.intervals.back().producer==strix::ArithmeticProducer::prefill_chunk &&
            local_transfer.restores==0,"all-match retains the actual verified chunk producer");
    }
    for(const bool native_failure : {false,true}) {
        Fixture fixture;
        for(std::uint64_t pos=0;pos<fixture.session.next_position;++pos)
            fixture.session.arithmetic_lineage=strix::prepare_serial_row_lineage(
                fixture.session.arithmetic_lineage,pos,fixture.compiled.execution_identity,0x7777);
        FakeSpeculationBackend backend;backend.publish_lineage=true;backend.native_failure=native_failure;
        backend.target_tokens={9,8,7};
        const std::vector<std::uint32_t> candidates{1,2};
        strix::GreedySpeculationExecutionRequest request;request.candidate_tokens=candidates;request.token_budget=3;
        CountingTransfer local_transfer;
        if(!native_failure) local_transfer.fail_restore_at=2;
        bool threw=false;
        try { (void)strix::execute_greedy_speculation_cycle(fixture.model,fixture.compiled,fixture.resources,
            fixture.session,request,allocator,generations,local_transfer,backend); }
        catch(const std::runtime_error&) { threw=true; }
        expect(threw && fixture.session.health==strix::SerialSessionHealth::poisoned && fixture.session.arithmetic_lineage.tainted,
            "partial-transfer or native failure cannot be healed by restoring snapshot metadata");
        if(native_failure) expect(local_transfer.restores==local_transfer.captures,
            "native failure still restores all state bytes while preserving poison");
        else expect(local_transfer.restores==3,"partial restore stops at failed transfer before publishing history");
    }

    {
        Fixture fixture;
        FakeSpeculationBackend backend;backend.target_tokens={9,8,7};backend.mutate_scratch=&fixture.scratch;
        const std::vector<std::uint32_t> candidates{1,2};
        strix::GreedySpeculationExecutionRequest request;request.candidate_tokens=candidates;request.token_budget=3;
        CountingTransfer local_transfer;
        bool rejected=false;
        try { (void)strix::execute_greedy_speculation_cycle(fixture.model,fixture.compiled,fixture.resources,
            fixture.session,request,allocator,generations,local_transfer,backend); }
        catch(const std::invalid_argument&) { rejected=true; }
        expect(rejected && local_transfer.restores==0 && fixture.session.health==strix::SerialSessionHealth::poisoned &&
            fixture.session.arithmetic_lineage.tainted,"changed allocation generation cannot redirect snapshot restore or republish producer history");
    }

    {
        Fixture fixture;
        fixture.session.next_position=fixture.state_layout.context_tokens;
        fixture.session.committed_tokens=fixture.session.next_position;
        FakeDraftBackend draft;
        auto draft_state=strix::make_draft_proposal_state(0x5555ULL,fixture.session);
        FakeSpeculationBackend backend;backend.target_tokens={42};
        strix::GreedyDraftExecutionRequest request;request.max_candidate_depth=2;request.token_budget=3;
        CountingTransfer local_transfer;
        const auto result=strix::execute_greedy_speculation_with_draft(fixture.model,fixture.compiled,fixture.resources,
            fixture.session,request,allocator,generations,local_transfer,draft_state,draft,backend);
        expect(result.proposal_depth_limit==0 && result.target_cycle.plan.emitted_tokens==std::vector<std::uint32_t>{42} &&
            fixture.session.next_position==fixture.state_layout.context_tokens && draft.calls==0 &&
            backend.verify_calls==1 && backend.replay_calls==0 && local_transfer.captures==0,
            "exact-capacity frontier emits cached bonus at depth zero without another input or snapshot");
        ++fixture.session.next_position;
        bool rejected=false;
        try { (void)strix::execute_greedy_speculation_with_draft(fixture.model,fixture.compiled,fixture.resources,
            fixture.session,request,allocator,generations,local_transfer,draft_state,draft,backend); }
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && backend.verify_calls==1 && draft.calls==0,"frontier beyond capacity rejects before callbacks");
    }

    if (failures == 0) {
        std::cout << "speculation executor tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
