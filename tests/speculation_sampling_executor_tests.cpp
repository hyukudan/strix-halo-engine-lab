#include "strix/speculation_sampling_executor.hpp"

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
        const auto value =
            align(next_device, alignment);
        next_device =
            value +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return value;
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        const auto value =
            align(next_host, alignment);
        next_host =
            value +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return value;
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
        const auto a =
            static_cast<std::uintptr_t>(
                alignment);
        const auto remainder =
            value % a;
        return remainder == 0
            ? value
            : value + (a - remainder);
    }

    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(
            0xd0000000ULL);
    std::uintptr_t next_host =
        static_cast<std::uintptr_t>(
            0xe0000000ULL);
};

class CountingTransfer final
    : public strix::SessionSnapshotTransferBackend {
public:
    int captures = 0;
    int restores = 0;

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
        ++restores;
    }
    void restore_host(
        std::uintptr_t,
        std::uintptr_t,
        std::uint64_t) override {
        ++restores;
    }
};

class FakeBackend final
    : public strix::SampledSpeculationExecutionBackend {
public:
    std::vector<std::vector<double>> target;
    bool fail_verify = false;
    bool poison_on_verify = false;
    bool mutate_zero_lineage = false;
    bool wrong_replay_identity = false;
    int fail_replay_after = -1;
    int verify_calls = 0;
    int replay_calls = 0;

    std::vector<std::vector<double>>
    verify_target_distributions(
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources&,
        std::span<const std::uint32_t> candidates) override {
        ++verify_calls;
        if (fail_verify) {
            if (poison_on_verify) {
                session.health = strix::SerialSessionHealth::poisoned;
                session.arithmetic_lineage = {};
                session.arithmetic_lineage.tainted = true;
            }
            throw std::runtime_error(
                "injected sampled verify failure");
        }
        if (!candidates.empty()) {
            session.arithmetic_lineage = {};
            const auto depth =
                static_cast<std::uint64_t>(
                    candidates.size());
            session.next_position +=
                depth;
            session.committed_tokens +=
                depth;
            for (auto& generation :
                 session.state.generations) {
                ++generation;
            }
        }
        if (mutate_zero_lineage) {
            session.arithmetic_lineage = {};
        }
        return target;
    }

    void replay_target_token(
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources&,
        std::uint32_t) override {
        if (fail_replay_after >= 0 &&
            replay_calls ==
                fail_replay_after) {
            ++replay_calls;
            throw std::runtime_error(
                "injected sampled replay failure");
        }
        ++replay_calls;
        if (wrong_replay_identity) {
            ++session.execution_identity;
        }
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
        strix::ModelContract::
            qwen38_flash_next();
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
            strix::TensorRole::
                dense_projection;
        manifest.architecture = "fixture";
        manifest.tensors.push_back(
            tensor);
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
                strix::RuntimeTensorStorage::
                    resident,
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

        strix::SerialSessionStateLayoutRequest
            state_request;
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
        state_memory.device_allocation_generation =
            1;
        state_memory.host_base =
            static_cast<std::uintptr_t>(
                0x200000000ULL);
        state_memory.host_bytes =
            state_layout.host_total_bytes;
        state_memory.host_allocation_generation =
            2;

        scratch.execution_identity =
            compiled.execution_identity;
        scratch.arena_base =
            static_cast<std::uintptr_t>(
                0x300000000ULL);
        scratch.arena_bytes =
            compiled.arena.total_bytes;
        scratch.allocation_generation = 3;

        resources.scratch =
            &scratch;
        resources.model_manifest =
            &manifest;
        resources.model_weights =
            &weights;
        resources.tensor_access =
            &tensor_access;
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

strix::SpeculationRngIdentity rng() {
    return {1234, 0x3333ULL, 0x4444ULL};
}

}  // namespace

int main() try {
    FakeAllocator allocator;
    strix::AllocationGenerationSource
        generations;
    CountingTransfer transfer;

    {
        Fixture fixture;
        fixture.session.next_position = 1;
        fixture.session.committed_tokens = 1;
        FakeBackend backend;
        backend.target = {{0x1.e36df62d0e73fp-2, 0x1.0e4904e978c60p-1, 0x1p-53, 0x1p-53}};
        strix::SampledSpeculationExecutionRequest request;
        request.rng = {1234, 0x1111ULL, 0x2222ULL};
        request.probability_identity = strix::SpeculationProbabilityIdentity::binary64_v1;
        const auto result = strix::execute_sampled_speculation_cycle(fixture.model, fixture.compiled,
            fixture.resources, fixture.session, request, allocator, generations, transfer, backend);
        expect(result.plan.emitted_tokens == std::vector<std::uint32_t>{1}, "executor forwards the binary64 identity");
    }

    {
        Fixture fixture;
        FakeBackend backend;
        backend.target = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
        };
        const std::vector<std::uint32_t>
            candidates{0, 0};
        const std::vector<std::vector<double>>
            draft{
                {1.0, 0.0},
                {1.0, 0.0},
            };

        strix::SampledSpeculationExecutionRequest
            request;
        request.candidate_tokens =
            candidates;
        request.draft_probabilities =
            draft;
        request.rng = rng();
        request.token_budget = 3;

        const auto result =
            strix::execute_sampled_speculation_cycle(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                backend);

        expect(
            result.snapshot_captured &&
                result.restored_pre_window &&
                result.replayed_tokens == 1 &&
                result.plan.emitted_tokens ==
                    std::vector<std::uint32_t>(
                        {0, 1}),
            "sampled rejection restores and replays exact accepted prefix");
        expect(
            fixture.session.next_position == 11 &&
                fixture.session.committed_tokens == 11 &&
                fixture.session.health ==
                    strix::SerialSessionHealth::ready,
            "sampled rejection leaves target state after accepted candidates only");
    }

    {
        Fixture fixture;
        FakeBackend backend;
        backend.target = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        };
        const std::vector<std::uint32_t>
            candidates{0, 1};
        const std::vector<std::vector<double>>
            draft{
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
            };

        strix::SampledSpeculationExecutionRequest
            request;
        request.candidate_tokens =
            candidates;
        request.draft_probabilities =
            draft;
        request.rng = rng();
        request.token_budget = 3;

        const auto result =
            strix::execute_sampled_speculation_cycle(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                backend);

        expect(
            result.snapshot_captured &&
                !result.restored_pre_window &&
                result.replayed_tokens == 0 &&
                result.plan.emitted_tokens ==
                    std::vector<std::uint32_t>(
                        {0, 1, 2}),
            "all-accepted sampled cycle retains verified target state and emits bonus");
        expect(
            fixture.session.next_position == 12 &&
                fixture.session.committed_tokens == 12,
            "all-accepted sampled state remains advanced through candidates");
    }

    {
        Fixture fixture;
        FakeBackend backend;
        backend.target = {
            {0.0, 1.0},
        };
        const std::vector<std::vector<double>>
            draft;

        strix::SampledSpeculationExecutionRequest
            request;
        request.draft_probabilities =
            draft;
        request.rng = rng();
        request.token_budget = 1;

        const auto result =
            strix::execute_sampled_speculation_cycle(
                fixture.model,
                fixture.compiled,
                fixture.resources,
                fixture.session,
                request,
                allocator,
                generations,
                transfer,
                backend);

        expect(
            !result.snapshot_captured &&
                !result.restored_pre_window &&
                result.plan.emitted_tokens ==
                    std::vector<std::uint32_t>(
                        {1}) &&
                fixture.session.next_position == 10 &&
                fixture.session.committed_tokens == 10,
            "depth-zero sampled cycle is snapshot-free and does not mutate target state");
    }

    {
        Fixture fixture;
        FakeBackend backend;
        backend.fail_verify = true;
        backend.target = {
            {1.0, 0.0},
            {0.0, 1.0},
        };
        const std::vector<std::uint32_t>
            candidates{0};
        const std::vector<std::vector<double>>
            draft{{1.0, 0.0}};

        strix::SampledSpeculationExecutionRequest
            request;
        request.candidate_tokens =
            candidates;
        request.draft_probabilities =
            draft;
        request.rng = rng();
        request.token_budget = 2;

        bool failed = false;
        try {
            static_cast<void>(
                strix::execute_sampled_speculation_cycle(
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
        expect(
            failed &&
                fixture.session.next_position == 10 &&
                fixture.session.committed_tokens == 10 &&
                fixture.session.health ==
                    strix::SerialSessionHealth::ready,
            "sampled verify failure restores the pre-window state");
    }

    {
        Fixture fixture;
        FakeBackend backend;
        backend.target = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
        };
        backend.fail_replay_after = 0;
        const std::vector<std::uint32_t>
            candidates{0, 0};
        const std::vector<std::vector<double>>
            draft{
                {1.0, 0.0},
                {1.0, 0.0},
            };

        strix::SampledSpeculationExecutionRequest
            request;
        request.candidate_tokens =
            candidates;
        request.draft_probabilities =
            draft;
        request.rng = rng();
        request.token_budget = 3;

        bool failed = false;
        try {
            static_cast<void>(
                strix::execute_sampled_speculation_cycle(
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
        expect(
            failed &&
                fixture.session.next_position == 10 &&
                fixture.session.committed_tokens == 10 &&
                fixture.session.health ==
                    strix::SerialSessionHealth::ready,
            "sampled replay failure rolls back to the pre-window snapshot");
    }

    {
        Fixture fixture;
        auto& session = fixture.session;
        auto& lineage = session.arithmetic_lineage;
        lineage.intervals.push_back({0, session.next_position, strix::ArithmeticProducer::serial_row,
                                     session.execution_identity, 1, session.execution_identity});
        lineage.history_identity = strix::arithmetic_history_identity(lineage.intervals);
        const auto original_lineage = lineage;
        FakeBackend backend;
        backend.target = {{0.0, 1.0}, {1.0, 0.0}};
        const std::vector<std::uint32_t> candidates{0};
        const std::vector<std::vector<double>> draft{{1.0, 0.0}};
        const strix::SampledSpeculationExecutionRequest request{candidates, draft, rng(), 2, {}};
        const auto execute = [&](const strix::SampledSpeculationExecutionRequest& chosen) {
            return strix::execute_sampled_speculation_cycle(fixture.model, fixture.compiled,
                fixture.resources, session, chosen, allocator, generations, transfer, backend);
        };
        const auto restored = execute(request);
        expect(restored.restored_pre_window && restored.replayed_tokens == 0 &&
                   lineage.history_identity == original_lineage.history_identity &&
                   lineage.intervals == original_lineage.intervals,
               "zero-prefix rollback restores producer lineage with the persistent bytes");

        const auto committed = session.committed_tokens;
        session.committed_tokens = UINT64_MAX;
        const int calls = backend.verify_calls;
        bool overflow = false;
        try {
            static_cast<void>(execute(request));
        } catch (const std::overflow_error&) {
            overflow = true;
        }
        expect(overflow && backend.verify_calls == calls && session.committed_tokens == UINT64_MAX,
               "verifier committed frontier overflow is detected before callbacks");
        session.committed_tokens = committed;

        backend.fail_verify = true;
        backend.poison_on_verify = true;
        bool failed = false;
        try {
            static_cast<void>(execute(request));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        expect(failed && session.next_position == 10 &&
                   session.health == strix::SerialSessionHealth::poisoned && lineage.tainted,
               "restoring bytes after verifier poison retains the failure");
    }

    {
        Fixture fixture;
        auto& session = fixture.session;
        session.next_position = fixture.resources.session_state_layout->context_tokens;
        auto& lineage = session.arithmetic_lineage;
        lineage.intervals.push_back({0, session.next_position, strix::ArithmeticProducer::serial_row,
                                     session.execution_identity, 1, session.execution_identity});
        lineage.history_identity = strix::arithmetic_history_identity(lineage.intervals);
        FakeBackend backend;
        backend.target = {{0.0, 1.0}};
        strix::SampledSpeculationExecutionRequest request;
        request.rng = rng();
        const auto execute = [&] {
            return strix::execute_sampled_speculation_cycle(fixture.model, fixture.compiled,
                fixture.resources, session, request, allocator, generations, transfer, backend);
        };
        const auto result = execute();
        expect(result.plan.emitted_tokens == std::vector<std::uint32_t>{1} &&
                   !result.snapshot_captured && session.next_position == 128,
               "depth-zero cycle samples cached logits at the allocated input frontier");

        backend.mutate_zero_lineage = true;
        bool rejected = false;
        try {
            static_cast<void>(execute());
        } catch (const std::logic_error&) {
            rejected = true;
        }
        expect(rejected && session.health == strix::SerialSessionHealth::poisoned && lineage.tainted,
               "snapshot-free verifier lineage mutation poisons the session");
    }

    {
        Fixture fixture;
        FakeBackend backend;
        backend.target = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}};
        backend.wrong_replay_identity = true;
        const std::vector<std::uint32_t> candidates{0, 0};
        const std::vector<std::vector<double>> draft{{1.0, 0.0}, {1.0, 0.0}};
        const strix::SampledSpeculationExecutionRequest request{candidates, draft, rng(), 3, {}};
        bool rejected = false;
        try {
            static_cast<void>(strix::execute_sampled_speculation_cycle(fixture.model, fixture.compiled,
                fixture.resources, fixture.session, request, allocator, generations, transfer, backend));
        } catch (const std::logic_error&) {
            rejected = true;
        }
        expect(rejected && fixture.session.health == strix::SerialSessionHealth::poisoned,
               "replay identity mutation is rejected and failed rollback poisons the session");
    }

    if (failures == 0) {
        std::cout
            << "sampled speculation executor tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected sampled executor exception: " << error.what() << '\n';
    return 1;
}
