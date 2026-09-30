#include "strix/compiled_speculation_backend.hpp"

#include "strix/execution_binding.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

strix::kernels::KernelVariant variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    strix::kernels::ExecutionPhase phase,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant result;
    result.id = std::move(id);
    result.family = family;
    result.phase = phase;
    result.shape = shape;
    result.wave =
        strix::kernels::WaveSize::wave32;
    result.resources.workgroup_threads = 256;
    result.numerics.input_precision = "bf16";
    result.numerics.accumulator_precision = "fp32";
    result.numerics.output_precision = "bf16";
    result.numerics.row_invariant = true;
    result.numerics.deterministic = true;
    result.qualification =
        strix::kernels::QualificationStatus::
            oracle_matched;
    result.graph_safe = true;
    result.input_layout = "fixture";
    result.output_layout = "fixture";
    result.compiler_identity = "fixture";
    return result;
}

strix::kernels::KernelRegistry registry_for(
    const strix::ModelContract& model,
    std::uint32_t rows,
    strix::kernels::ExecutionPhase phase) {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);

    strix::ExecutionBindingRequest request;
    request.logical_rows = rows;
    request.request_concurrency = 1;

    std::set<std::string> unique;
    for (const auto& step :
         execution.steps) {
        if (step.op ==
            strix::SerialOp::position_prepare) {
            continue;
        }
        const auto family =
            strix::operator_family_for(
                step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model,
                step.op,
                request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(
                    *family)) +
            "." +
            std::to_string(
                shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }
        registry.add(
            variant(
                (phase ==
                         strix::kernels::ExecutionPhase::
                             decode
                     ? "decode."
                     : "verify.") +
                    key,
                *family,
                phase,
                shape));
    }
    return registry;
}

struct CountingDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t calls = 0;

    void dispatch_kernel(
        const strix::ExecutionDispatchContext&) override {
        ++calls;
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext&) override {
        ++calls;
    }
};

struct FixedCollector final
    : strix::GreedyTargetTokenCollector {
    std::uint32_t base = 10;
    std::vector<std::uint32_t> suffix{
        20, 30};

    std::uint32_t collect_base_target_token(
        const strix::CompiledSerialExecution&,
        const strix::ExecutionRuntimeResources&,
        const strix::SerialSessionContext&) override {
        return base;
    }

    std::vector<std::uint32_t>
    collect_verified_target_tokens(
        const strix::CompiledSpeculativeVerifyExecution&,
        const strix::ExecutionRuntimeResources&,
        const strix::SerialSessionContext&) override {
        return suffix;
    }
};

class FakeAllocator final
    : public strix::RuntimeMemoryAllocator {
public:
    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        const auto address =
            align(next_device, alignment);
        next_device =
            address +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return address;
    }

    std::uintptr_t allocate_host(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        const auto address =
            align(next_host, alignment);
        next_host =
            address +
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return address;
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
            static_cast<std::uintptr_t>(
                alignment);
        return remainder == 0
            ? value
            : value +
                  static_cast<std::uintptr_t>(
                      alignment - remainder);
    }

    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(
            0x400000000ULL);
    std::uintptr_t next_host =
        static_cast<std::uintptr_t>(
            0x500000000ULL);
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

struct Fixture {
    strix::ModelContract model =
        strix::ModelContract::qwen38_flash_next();

    strix::CompiledSerialExecution serial;
    strix::CompiledSpeculativeVerifyExecution verify;

    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::SerialSessionStateLayout state_layout;
    strix::SessionStateMemoryBinding state_memory;

    strix::ExecutionTensorAccessPlan serial_access;
    strix::ExecutionTensorAccessPlan verify_access;

    strix::ExecutionMemoryBinding serial_scratch;
    strix::ExecutionMemoryBinding verify_scratch;

    strix::ExecutionRuntimeResources serial_resources;
    strix::ExecutionRuntimeResources verify_resources;

    strix::SerialSessionContext session;

    Fixture() {
        strix::ExecutionBindingRequest serial_binding;
        serial_binding.logical_rows = 1;
        serial_binding.request_concurrency = 1;

        strix::ExecutionBufferRequest serial_buffers;
        serial_buffers.logical_rows = 1;

        const auto serial_registry =
            registry_for(
                model,
                1,
                strix::kernels::ExecutionPhase::decode);

        serial =
            strix::compile_serial_execution(
                model,
                serial_registry,
                serial_binding,
                serial_buffers,
                256);

        strix::SpeculativeVerifyBindingRequest verify_binding;
        verify_binding.request_concurrency = 1;
        strix::ExecutionBufferRequest verify_buffers;
        verify_buffers.logical_rows = 2;

        const auto verify_registry =
            registry_for(
                model,
                2,
                strix::kernels::ExecutionPhase::
                    speculative_verify);

        verify =
            strix::compile_speculative_verify_execution(
                model,
                2,
                verify_registry,
                verify_binding,
                verify_buffers,
                256);

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
                    0x600000000ULL),
                tensor.byte_size,
                1,
                0,
            });

        strix::SerialSessionStateLayoutRequest state_request;
        state_request.context_tokens = 64;
        state_layout =
            strix::build_serial_session_state_layout(
                model,
                state_request);

        state_memory.layout_identity =
            state_layout.layout_identity;
        state_memory.device_base =
            static_cast<std::uintptr_t>(
                0x610000000ULL);
        state_memory.device_bytes =
            state_layout.device_total_bytes;
        state_memory.device_allocation_generation = 2;
        state_memory.host_base =
            static_cast<std::uintptr_t>(
                0x620000000ULL);
        state_memory.host_bytes =
            state_layout.host_total_bytes;
        state_memory.host_allocation_generation = 3;

        serial_access.manifest_layout_identity =
            manifest.layout_identity;
        serial_access.plan_identity =
            0x7001ULL;
        verify_access.manifest_layout_identity =
            manifest.layout_identity;
        verify_access.plan_identity =
            0x7002ULL;

        serial_scratch.execution_identity =
            serial.execution_identity;
        serial_scratch.arena_base =
            static_cast<std::uintptr_t>(
                0x630000000ULL);
        serial_scratch.arena_bytes =
            serial.arena.total_bytes;
        serial_scratch.allocation_generation = 4;

        verify_scratch.execution_identity =
            verify.execution_identity;
        verify_scratch.arena_base =
            static_cast<std::uintptr_t>(
                0x640000000ULL);
        verify_scratch.arena_bytes =
            verify.arena.total_bytes;
        verify_scratch.allocation_generation = 5;

        serial_resources.scratch =
            &serial_scratch;
        serial_resources.model_manifest =
            &manifest;
        serial_resources.model_weights =
            &weights;
        serial_resources.tensor_access =
            &serial_access;
        serial_resources.session_state_layout =
            &state_layout;
        serial_resources.session_state_memory =
            &state_memory;

        verify_resources.scratch =
            &verify_scratch;
        verify_resources.model_manifest =
            &manifest;
        verify_resources.model_weights =
            &weights;
        verify_resources.tensor_access =
            &verify_access;
        verify_resources.session_state_layout =
            &state_layout;
        verify_resources.session_state_memory =
            &state_memory;

        const auto ids =
            strix::build_runtime_execution_identities(
                serial.execution_identity,
                serial.arena,
                serial_resources);

        session.execution_identity =
            serial.execution_identity;
        session.state_layout_identity =
            state_layout.layout_identity;
        session.runtime_compatibility_identity =
            ids.compatibility_identity;
        session.next_position = 10;
        session.committed_tokens = 10;
        session.state.valid = all_state();
        session.state.generations.fill(5);
    }
};

}  // namespace

int main() {
    Fixture fixture;
    CountingDispatcher serial_dispatcher;
    CountingDispatcher verify_dispatcher;
    FixedCollector collector;

    strix::CompiledGreedySpeculationBackend backend(
        fixture.serial,
        fixture.serial_resources,
        serial_dispatcher,
        fixture.verify,
        fixture.verify_resources,
        verify_dispatcher,
        collector);

    FakeAllocator allocator;
    strix::AllocationGenerationSource generations;
    CountingTransfer transfer;

    const std::vector<std::uint32_t> candidates{
        10, 99};
    strix::GreedySpeculationExecutionRequest request;
    request.candidate_tokens = candidates;
    request.token_budget = 3;

    const auto result =
        strix::execute_greedy_speculation_cycle(
            fixture.model,
            fixture.serial,
            fixture.serial_resources,
            fixture.session,
            request,
            allocator,
            generations,
            transfer,
            backend);

    expect(result.plan.emitted_tokens ==
               std::vector<std::uint32_t>({10, 20}),
           "compiled backend feeds base+verify target tokens into greedy acceptance");
    expect(result.replayed_tokens == 1 &&
               fixture.session.next_position == 11 &&
               fixture.session.committed_tokens == 11,
           "compiled backend replays accepted prefix through exact serial path");
    expect(verify_dispatcher.calls ==
               fixture.verify.topology.steps.size(),
           "compiled backend executes full verify DAG once");
    expect(serial_dispatcher.calls ==
               fixture.serial.execution.steps.size(),
           "accepted replay executes one complete serial target token");

    auto changed_verify_resources =
        fixture.verify_resources;
    auto changed_weights =
        fixture.weights;
    ++changed_weights.model_content_identity.bytes[0];
    changed_verify_resources.model_weights =
        &changed_weights;

    bool mismatch_rejected = false;
    try {
        strix::CompiledGreedySpeculationBackend invalid(
            fixture.serial,
            fixture.serial_resources,
            serial_dispatcher,
            fixture.verify,
            changed_verify_resources,
            verify_dispatcher,
            collector);
        static_cast<void>(invalid);
    } catch (const std::invalid_argument&) {
        mismatch_rejected = true;
    }
    expect(mismatch_rejected,
           "compiled speculation backend rejects serial/verify checkpoint mismatch");

    if (failures == 0) {
        std::cout << "compiled speculation backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
