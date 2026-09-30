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

class FakeSpeculationBackend final
    : public strix::GreedySpeculationExecutionBackend {
public:
    std::vector<std::uint32_t> target_tokens;
    bool fail_verify = false;
    int fail_replay_after = -1;
    int verify_calls = 0;
    int replay_calls = 0;

    std::vector<std::uint32_t>
    verify_target_candidates(
        strix::SerialSessionContext& session,
        const strix::ExecutionRuntimeResources&,
        std::span<const std::uint32_t> candidates) override {
        ++verify_calls;
        if (fail_verify) {
            throw std::runtime_error(
                "injected verify failure");
        }

        if (!candidates.empty()) {
            const auto depth =
                static_cast<std::uint64_t>(
                    candidates.size());
            session.next_position += depth;
            session.committed_tokens += depth;
            for (auto& generation :
                 session.state.generations) {
                ++generation;
            }
        }
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

    if (failures == 0) {
        std::cout << "speculation executor tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
