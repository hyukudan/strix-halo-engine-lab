#include "strix/compiled_speculative_verify.hpp"

#include "strix/execution_binding.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/runtime_launchers.hpp"
#include "strix/speculative_verify_runtime.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>

#include <set>
#include <string>
#include <utility>

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

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds = 0.0) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::
            speculative_verify;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::
            oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "verify";
    variant.output_layout = "verify";
    variant.compiler_identity = "fixture";
    variant.measured_seconds =
        measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    std::uint32_t depth,
    std::string suffix = "", bool parallel_router = false, bool ranked_router = false, bool wave_router = false) {
    strix::kernels::KernelRegistry registry;
    const auto verify =
        strix::build_speculative_verify_plan(
            model,
            depth);

    strix::ExecutionBindingRequest shape_request;
    shape_request.logical_rows = depth;
    shape_request.request_concurrency = 1;

    std::set<std::string> unique;
    for (const auto& step :
         verify.steps) {
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
                shape_request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(
                shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "verify." + key + suffix +
                    (step.op == strix::SerialOp::moe_route && wave_router ? ".wave.parallel" :
                     step.op == strix::SerialOp::moe_route && ranked_router ? ".ranked.parallel" :
                     step.op == strix::SerialOp::moe_route && parallel_router ? ".parallel" : ""),
                *family,
                shape));
    }
    return registry;
}


struct KnownFixture {
    strix::CompiledSerialExecution serial;
    strix::ModelManifest manifest;
    strix::ExecutionTensorAccessPlan access;
};
KnownFixture make_known_fixture(const strix::ModelContract& model) {
    KnownFixture fixture;
    fixture.serial.execution = strix::build_serial_decode_execution_plan(model);
    fixture.serial.io = strix::build_serial_execution_io_plan(fixture.serial.execution);
    strix::ExecutionBufferRequest buffers;
    buffers.q4k_a8g32_gate_up = true;
    buffers.grouped_exact_q51_down = true;
    buffers.q51_slot_parallel_down = true;
    fixture.serial.arena = strix::build_serial_execution_arena_plan(model, buffers,
        fixture.serial.execution, fixture.serial.io);
    fixture.serial.execution_identity = 0x51515231;
    fixture.serial.kernel_plan_identity = 0x51514b31;
    fixture.serial.vocab_size = model.vocab_size;
    fixture.serial.max_position_embeddings = model.max_position_embeddings;
    fixture.manifest.layout_identity = 17;
    fixture.access.manifest_layout_identity = 17;
    fixture.access.plan_identity = 29;
    std::uint64_t offset = 256;
    for (const auto& step : fixture.serial.execution.steps) {
        strix::ExecutionKernelBinding binding;
        binding.step_id = step.id; binding.op = step.op;
        binding.family = strix::operator_family_for(step.op);
        if (step.op == strix::SerialOp::position_prepare) {
            binding.host_or_runtime_step = true;
            ++fixture.serial.bindings.host_or_runtime_steps;
        } else {
            binding.kernel_id = "fixture.serial." + std::to_string(step.id);
            ++fixture.serial.bindings.bound_steps;
        }
        strix::ExecutionTensorAccess tensors;
        tensors.step_id = step.id; tensors.op = step.op;
        const auto tensor = [&](std::string suffix, std::vector<std::uint64_t> dimensions, std::uint32_t type) {
            strix::ManifestTensor value;
            value.name = "blk." + std::to_string(*step.layer) + "." + suffix;
            value.role = suffix.starts_with("attn_") ? strix::TensorRole::attention_projection :
                suffix.starts_with("ssm_") ? strix::TensorRole::recurrent_state : strix::TensorRole::dense_projection;
            value.layer = step.layer; value.dimensions = std::move(dimensions); value.ggml_type = type;
            value.absolute_offset = offset;
            value.byte_size = type == 8 ? value.dimensions[0] / 32U * 34U * value.dimensions[1] : value.dimensions[0] * 4U;
            offset += (value.byte_size + 255U) / 256U * 256U;
            tensors.tensor_indices.push_back(static_cast<std::uint32_t>(fixture.manifest.tensors.size()));
            fixture.manifest.tensors.push_back(std::move(value));
        };
        if (step.op == strix::SerialOp::shared_expert) {
            binding.kernel_id = "hip.shared_expert.bf16.h2560.m640.r1.c1";
            tensor("ffn_gate_shexp.weight", {2560,640}, 8);
            tensor("ffn_up_shexp.weight", {2560,640}, 8);
            tensor("ffn_down_shexp.weight", {640,2560}, 8);
            tensor("ffn_gate_inp_shexp.weight", {2560}, 0);
        } else if (step.op == strix::SerialOp::gated_delta_net) {
            binding.kernel_id = "hip.gdn.bf16.h2560.kh16.vh48.d128.r1.c1.scalar.wave8-input-v1.wave8-q8-output-v1.prefetch8.virtual256.q8-input.virtual256.q8-output";
            tensor("attn_qkv.weight", {2560,10240}, 8);
            tensor("attn_gate.weight", {2560,6144}, 8);
            tensor("ssm_beta.weight", {2560,48}, 8);
            tensor("ssm_alpha.weight", {2560,48}, 8);
            tensor("ssm_out.weight", {6144,2560}, 8);
        } else if (step.op == strix::SerialOp::ple) {
            // Only metadata is retained. A start-fault backend below never
            // accesses this table or claims a real PLE numerical baseline.
            const auto hash = strix::build_ple_hash_contract(model);
            strix::ManifestTensor table;
            table.name = "fixture.ngram_table.weight";
            table.role = strix::TensorRole::ngram_table;
            table.dimensions = {hash.head_dim, hash.minimum_table_rows};
            table.ggml_type = 1; table.absolute_offset = offset;
            table.byte_size = hash.row_bytes * hash.minimum_table_rows;
            tensors.tensor_indices.push_back(static_cast<std::uint32_t>(fixture.manifest.tensors.size()));
            fixture.manifest.tensors.push_back(std::move(table));
        }
        fixture.serial.bindings.bindings.push_back(std::move(binding));
        if (!tensors.tensor_indices.empty()) fixture.access.accesses.push_back(std::move(tensors));
    }
    return fixture;
}
// Host flow witnesses, not numerical/GPU qualification. Canonical topology and
// real dispatch/resource resolution surround deliberately tiny state recurrences.
constexpr std::uint64_t recording_qsa_identity = 0x5445535451534131ULL;
constexpr std::array<strix::PersistentState, 7> mock_state_flags{
    strix::PersistentState::attention_kv, strix::PersistentState::recurrent_matrix,
    strix::PersistentState::convolution_history, strix::PersistentState::convolution_history,
    strix::PersistentState::sparse_index, strix::PersistentState::ngram_history,
    strix::PersistentState::position_history};
constexpr std::array<strix::SessionStateComponent, 7> mock_components{
    strix::SessionStateComponent::attention_kv, strix::SessionStateComponent::recurrent_matrix,
    strix::SessionStateComponent::gdn_convolution, strix::SessionStateComponent::ple_convolution,
    strix::SessionStateComponent::sparse_index, strix::SessionStateComponent::ngram_history,
    strix::SessionStateComponent::position_history};

std::uint64_t read_word(std::uintptr_t address) {
    std::uint64_t word = 0;
    std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
    return word;
}
void write_word(std::uintptr_t address, std::uint64_t word) {
    std::memcpy(reinterpret_cast<void*>(address), &word, sizeof(word));
}
bool mock_has(strix::PersistentState set, strix::PersistentState flag) {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(flag)) != 0;
}
struct WindowTrace {
    std::uint32_t calls = 0;
    std::uint32_t admissions = 0;
    bool fail_kernel = false;
    bool fail_qsa = false;
    strix::SessionStateMemoryBinding* rebind_after_qsa = nullptr;
    std::uint32_t rebindings = 0;
    std::vector<std::uint64_t> qsa_positions;
    std::vector<std::uint64_t> ple_previous;
};
void recording_window_preflight(const strix::ExecutionDispatchContext& context, void* opaque) {
    ++static_cast<WindowTrace*>(opaque)->admissions;
    if (context.logical_rows != 1 || context.request_concurrency != 1 || !context.resources)
        throw std::invalid_argument("mock R1 preflight requires a complete one-row binding");
    for (std::size_t i = 0; i < context.reads.size(); ++i) (void)strix::resolve_dispatch_read(context, i);
    for (std::size_t i = 0; i < context.writes.size(); ++i) (void)strix::resolve_dispatch_write(context, i);
    for (std::size_t i = 0; i < context.temporaries.size(); ++i) (void)strix::resolve_dispatch_temporary(context, i);
    for (std::size_t i = 0; i < mock_components.size(); ++i)
        if (mock_has(context.reads_state | context.writes_state, mock_state_flags[i]))
            (void)strix::resolve_dispatch_state(context, mock_components[i]);
}
void throwing_window_preflight(const strix::ExecutionDispatchContext&, void* opaque) {
    ++static_cast<WindowTrace*>(opaque)->admissions;
    throw std::runtime_error("late row1 terminal metadata admission failure");
}
class StartFaultLookup final : public strix::PleLookupBackend {
public:
    explicit StartFaultLookup(bool zero) : zero_(zero) {}
    std::uint32_t starts = 0, cancels = 0;
    std::uint64_t implementation_identity() const noexcept override { return 0x54455354504c4531ULL; }
    strix::PleLookupHandle start(const strix::PleLookupRequest&) override {
        ++starts;
        if (zero_) return 0;
        throw std::runtime_error("lookup start failure before mutation");
    }
    void finish(strix::PleLookupHandle, std::span<std::byte>) override {
        throw std::logic_error("a failed start has no finishable handle");
    }
    void cancel(strix::PleLookupHandle) noexcept override { ++cancels; }
private:
    bool zero_;
};
void recording_window_kernel(const strix::ExecutionDispatchContext& context, void* opaque) {
    auto& trace = *static_cast<WindowTrace*>(opaque);
    ++trace.calls;
    if (context.logical_rows != 1 || context.request_concurrency != 1)
        throw std::logic_error("mock requires the incumbent R1 context");
    if (mock_has(context.reads_state, strix::PersistentState::position_history)) {
        const auto position = strix::resolve_dispatch_state(context,
            strix::SessionStateComponent::position_history);
        const auto word = read_word(position.address);
        if (word != context.position) throw std::logic_error("row reader observed future position");
        trace.qsa_positions.push_back(word);
        if (context.op == strix::SerialOp::indexed_attention && trace.rebind_after_qsa &&
            trace.rebindings == 0) {
            ++trace.rebind_after_qsa->host_allocation_generation;
            ++trace.rebindings;
        }
    }
    if (trace.fail_kernel && context.position == 1 && context.op == strix::SerialOp::gated_delta_net)
        throw std::runtime_error("injected row1 kernel failure");
    if (trace.fail_qsa && context.position == 1 && context.op == strix::SerialOp::indexed_attention)
        throw std::runtime_error("injected row1 QSA failure");
    std::uint64_t value = 101U + context.input_token_id + 17U * context.position + context.step_id;
    // Snapshot all input words before writing any potentially reused arena slot.
    for (std::size_t i = 0; i < context.reads.size(); ++i)
        value = value * 1099511628211ULL + read_word(strix::resolve_dispatch_read(context, i).address);
    for (std::size_t i = 0; i < mock_components.size() - 1; ++i) {
        if (!mock_has(context.writes_state, mock_state_flags[i])) continue;
        if (mock_components[i] == strix::SessionStateComponent::gdn_convolution &&
            context.op != strix::SerialOp::gated_delta_net) continue;
        if (mock_components[i] == strix::SessionStateComponent::ple_convolution &&
            context.op != strix::SerialOp::ple) continue;
        const auto state = strix::resolve_dispatch_state(context, mock_components[i]);
        const auto lane = static_cast<std::uint64_t>(context.layer.value_or(0));
        const auto address = state.address + lane * sizeof(std::uint64_t);
        const auto previous = read_word(address);
        if (mock_components[i] == strix::SessionStateComponent::ngram_history)
            trace.ple_previous.push_back(previous);
        value = value * 1099511628211ULL + previous;
        write_word(address, previous * 33U + context.input_token_id + context.position + 1U);
    }
    for (std::size_t i = 0; i < context.writes.size(); ++i)
        write_word(strix::resolve_dispatch_write(context, i).address, value + i);
}
void recording_r2_preflight(std::span<const strix::ExecutionDispatchContext, 2> rows, void*) {
    if (rows[0].step_id != rows[1].step_id || rows[1].position != rows[0].position + 1 ||
        rows[0].execution_identity != rows[1].execution_identity)
        throw std::invalid_argument("mock R2 admission needs adjacent rows of one step");
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.reads.size(); ++i) (void)strix::resolve_dispatch_read(row, i);
        for (std::size_t i = 0; i < row.writes.size(); ++i) (void)strix::resolve_dispatch_write(row, i);
    }
}
void recording_r2_launch(std::span<const strix::ExecutionDispatchContext, 2> rows, void* opaque) {
    recording_window_kernel(rows[0], opaque);
    recording_window_kernel(rows[1], opaque);
}

struct KnownRuntimeFixture {
    KnownFixture source;
    strix::CompiledKnownPrefixWindow window;
    std::vector<std::uint64_t> serial_pool, window_pool, state_words;
    strix::ExecutionMemoryBinding serial_memory, window_memory;
    strix::SerialSessionStateLayout layout;
    strix::SessionStateMemoryBinding state_memory;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionRuntimeResources resources;
    strix::SerialSessionContext session;
    WindowTrace trace;
    strix::ExecutionLauncherRegistry registry;
    strix::CompiledLaunchPlan serial_launch;
    std::array<strix::CompiledLaunchPlan, 2> row_launch;
    std::array<std::uint64_t, 1> allowed_readers{recording_qsa_identity};
    std::array<std::uint32_t, 2> tokens{7, 11};
    std::uint32_t quiesces = 0;

    explicit KnownRuntimeFixture(const strix::ModelContract& model, bool promote = false)
        : source(make_known_fixture(model)),
          window(strix::compile_known_prefix_window(model, source.serial, source.manifest,
              source.access, {promote, promote})),
          serial_pool(static_cast<std::size_t>((source.serial.arena.total_bytes + 7U) / 8U)),
          window_pool(static_cast<std::size_t>((window.arena.physical.total_bytes + 7U) / 8U)),
          state_words(7U * 48U, 0) {
        serial_memory = {source.serial.execution_identity, reinterpret_cast<std::uintptr_t>(serial_pool.data()),
            source.serial.arena.total_bytes, 1};
        window_memory = {window.window_execution_identity, reinterpret_cast<std::uintptr_t>(window_pool.data()),
            window.arena.physical.total_bytes, 2};
        layout.context_tokens = 64; layout.layout_identity = 0x535441544531;
        layout.host_total_bytes = state_words.size() * sizeof(std::uint64_t);
        for (std::size_t i = 0; i < mock_components.size(); ++i)
            layout.allocations.push_back({mock_components[i], mock_state_flags[i],
                strix::SessionStateStorage::host, i * 48U * sizeof(std::uint64_t),
                48U * sizeof(std::uint64_t), false});
        state_memory.layout_identity = layout.layout_identity;
        state_memory.host_base = reinterpret_cast<std::uintptr_t>(state_words.data());
        state_memory.host_bytes = layout.host_total_bytes; state_memory.host_allocation_generation = 3;
        weights.manifest_layout_identity = source.manifest.layout_identity;
        weights.model_content_identity.bytes[0] = 0x42;
        for (std::size_t i = 0; i < source.manifest.tensors.size(); ++i)
            weights.tensors.push_back({static_cast<std::uint32_t>(i), strix::RuntimeTensorStorage::resident,
                0x10000000U + source.manifest.tensors[i].absolute_offset,
                source.manifest.tensors[i].byte_size, 4, 0});
        resources = {&serial_memory, &source.manifest, &weights, &source.access, &layout, &state_memory};
        strix::register_default_runtime_launchers(registry);
        std::set<std::string> registered;
        for (const auto& binding : source.serial.bindings.bindings) {
            if (binding.host_or_runtime_step || !registered.insert(*binding.kernel_id).second) continue;
            registry.add_kernel_launcher({*binding.kernel_id,
                binding.op == strix::SerialOp::indexed_attention ? recording_qsa_identity : 0x544553544b45524eULL,
                &recording_window_kernel, &trace, &recording_window_preflight});
        }
        serial_launch = strix::compile_launch_plan(source.serial, registry);
        for (std::size_t r = 0; r < 2; ++r)
            row_launch[r] = strix::compile_launch_plan(window.row_plan[r], registry);
        reset();
    }
    void reset() {
        std::fill(serial_pool.begin(), serial_pool.end(), 0);
        std::fill(window_pool.begin(), window_pool.end(), 0);
        std::fill(state_words.begin(), state_words.end(), 0);
        trace = {}; quiesces = 0; session = {};
        session.execution_identity = source.serial.execution_identity;
        session.state_layout_identity = layout.layout_identity;
        session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
            source.serial.execution_identity, source.serial.arena, resources).compatibility_identity;
        for (const auto flag : mock_state_flags) session.state.valid = session.state.valid | flag;
    }
    strix::KnownWindowExecutionHooks hooks() {
        return {[] {}, [] {}, [] {}, [this] { ++quiesces; }, allowed_readers};
    }
    strix::KnownWindowResult run(strix::KnownWindowExecutionHooks callbacks) {
        const std::array<strix::KnownWindowR2Entry, 2> entries{{
            {strix::KnownWindowR2Variant::shared_q8_virtual256_r2_v1, &trace,
                &recording_r2_preflight, &recording_r2_launch},
            {strix::KnownWindowR2Variant::gdn_q8_virtual256_r2_v1, &trace,
                &recording_r2_preflight, &recording_r2_launch}}};
        return strix::execute_known_prefix_window_eager(window, source.serial, resources,
            window_memory, session, tokens, {&row_launch[0], &row_launch[1]}, entries, callbacks);
    }
    std::uint64_t serial_token(std::uint32_t token) {
        strix::DirectLaunchDispatcher dispatcher(serial_launch);
        (void)strix::execute_serial_token(source.serial, session, resources, token, dispatcher);
        const auto use = source.serial.io.steps.back().writes.front();
        return read_word(strix::resolve_execution_buffer(source.serial.execution_identity,
            source.serial.arena, serial_memory, use).address);
    }
    std::uintptr_t position_address() const {
        return strix::resolve_session_state(layout, state_memory,
            strix::SessionStateComponent::position_history).address;
    }
};

void known_window_runtime_tests(const strix::ModelContract& model) {
    KnownRuntimeFixture fixture(model);
    const std::array<std::uint64_t, 2> expected{fixture.serial_token(7), fixture.serial_token(11)};
    const auto expected_state = fixture.state_words;
    const auto expected_generations = fixture.session.state.generations;
    const auto expected_ple = fixture.trace.ple_previous;
    const auto third = fixture.serial_token(13);
    const auto third_state = fixture.state_words;
    for (const bool promote : {false, true}) {
        KnownRuntimeFixture candidate(model, promote);
        const auto result = candidate.run(candidate.hooks());
        expect(read_word(result.logits[0].address) == expected[0] &&
            read_word(result.logits[1].address) == expected[1] &&
            result.logits[0].address != result.logits[1].address,
            "known R2/all-off host flow preserves both serial row outputs in distinct storage");
        expect(candidate.state_words == expected_state &&
            candidate.session.state.generations == expected_generations &&
            candidate.trace.ple_previous == expected_ple,
            "known layer-major flow preserves seven state recurrences, six generations and PLE row order");
        expect(candidate.session.next_position == 2 && candidate.session.committed_tokens == 2 &&
            read_word(candidate.position_address()) == 1 && candidate.quiesces == 0 &&
            candidate.session.arithmetic_lineage.intervals.size() == 1 &&
            (promote ? result.r2_dispatches == 84 : result.r2_dispatches == 0),
            "known window commits only after completion and records a genuine chunk lineage");
        expect(candidate.serial_token(13) == third && candidate.state_words == third_state &&
            candidate.session.arithmetic_lineage.intervals.size() == 2,
            "serial continuation after a known chunk preserves state and appends serial lineage");
    }
    // A direct mutant witness: absent host materialization makes row0 observe
    // row1's position even though all ordinary kernel arguments still say row0.
    fixture.reset(); write_word(fixture.position_address(), 1);
    strix::ExecutionDispatchContext reader;
    reader.resources = &fixture.resources; reader.position = 0;
    reader.reads_state = strix::PersistentState::position_history;
    bool caught = false;
    try { recording_window_kernel(reader, &fixture.trace); }
    catch (const std::logic_error&) { caught = true; }
    expect(caught, "position reader witness detects omitted row0 materialization");

    // Call the actual fixed-position launcher, independently of the recorder.
    fixture.reset();
    auto position_context = reader;
    position_context.op = strix::SerialOp::position_prepare;
    position_context.step_id = 1;
    const auto position_launcher = strix::make_position_prepare_launcher();
    auto position_words = fixture.state_words;
    for (const std::uint64_t position : {7ULL, 8ULL}) {
        position_context.position = position;
        position_launcher.preflight(position_context, position_launcher.user_data);
        position_launcher.function(position_context, position_launcher.user_data);
        const auto position_index = static_cast<std::size_t>(
            (fixture.position_address() - fixture.state_memory.host_base) / sizeof(std::uint64_t));
        position_words[position_index] = position;
        expect(read_word(fixture.position_address()) == position && fixture.state_words == position_words &&
            fixture.trace.calls == 0,
            "actual position_prepare writes p then p+1 and preserves every surrounding state word");
    }

    const auto reject = [&](const std::function<void(KnownRuntimeFixture&, strix::KnownWindowExecutionHooks&)>& mutate,
                            const char* message, const char* expected_error = nullptr) {
        KnownRuntimeFixture candidate(model);
        auto callbacks = candidate.hooks(); mutate(candidate, callbacks);
        const auto words = candidate.state_words;
        const auto health = candidate.session.health;
        const auto position = candidate.session.next_position;
        const auto committed = candidate.session.committed_tokens;
        bool rejected = false;
        std::string error;
        try { (void)candidate.run(callbacks); }
        catch (const std::exception& exception) { rejected = true; error = exception.what(); }
        expect(rejected && candidate.trace.calls == 0 && candidate.state_words == words &&
            candidate.session.health == health && candidate.session.next_position == position &&
            candidate.session.committed_tokens == committed && candidate.quiesces == 0 &&
            (!expected_error || error == expected_error), message);
    };
    reject([](auto& f, auto&) { ++f.window.arena.row_stride.front(); }, "G0 rejects changed row stride before mutation");
    reject([](auto& f, auto&) { f.window.row_plan[1].arena = f.window.row_plan[0].arena; }, "G0 rejects aliased row views");
    reject([](auto& f, auto&) { f.window_memory.allocation_generation = 0; }, "G0 rejects stale scratch generation");
    reject([](auto& f, auto&) { f.window_memory.arena_bytes = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects an overflowing full scratch owner extent");
    reject([](auto& f, auto&) { f.serial_memory.arena_bytes = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects an overflowing original serial owner extent");
    reject([](auto& f, auto&) { ++f.window.window_execution_identity; }, "G0 rejects changed window identity");
    reject([](auto& f, auto&) { ++f.session.execution_identity; },
        "G0 rejects a session belonging to a different serial execution before dispatch",
        "known window serial session execution identity mismatch");
    reject([](auto& f, auto&) { f.layout.context_tokens = 1; }, "G0 rejects insufficient two-row capacity");
    reject([](auto& f, auto&) { f.tokens[1] = static_cast<std::uint32_t>(f.source.serial.vocab_size); },
        "G0 rejects an out-of-vocabulary row1 token");
    reject([](auto& f, auto&) { f.session.committed_tokens = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects committed-token overflow");
    reject([](auto& f, auto&) { f.session.health = strix::SerialSessionHealth::poisoned; }, "G0 refuses a poisoned session");
    reject([](auto& f, auto&) { f.session.arithmetic_lineage.tainted = true; }, "G0 refuses tainted lineage");
    reject([](auto& f, auto&) {
        f.session.arithmetic_lineage = strix::prepare_serial_row_lineage({}, 0,
            f.source.serial.execution_identity, f.source.serial.kernel_plan_identity);
    }, "G0 rejects producer history whose endpoint differs from the session position");
    reject([](auto& f, auto&) { f.session.state.generations[0] = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects persistent generation overflow");
    reject([](auto&, auto& h) { h.preflight_scope = [] { throw std::runtime_error("scope fault"); }; },
        "owned-scope admission failure dispatches nothing and preserves ready state");
    reject([](auto&, auto& h) { h.preflight_plan = [] { throw std::runtime_error("deep plan fault"); }; },
        "deep R1 admission failure dispatches nothing and preserves ready state");
    reject([](auto& f, auto&) { ++f.row_launch[1].execution_identity; }, "G0 rejects an unrelated row launch plan");
    reject([](auto& f, auto&) { f.row_launch[1].steps.back().preflight = nullptr; },
        "G0 refuses a row-loop launcher without a pure metadata preflight");
    reject([](auto& f, auto&) { f.row_launch[1].steps.back().preflight = &throwing_window_preflight; },
        "late row1 head preflight rejects before any row0 launch or position write");
    reject([](auto&, auto& h) { h.position_reader_implementations = {}; }, "G0 refuses unadmitted position readers");
    reject([](auto& f, auto&) { f.row_launch[0].steps[1].function = &recording_window_kernel; },
        "G0 rejects a fake position handler even with the incumbent implementation ID");
    reject([](auto& f, auto&) {
        const auto verify = strix::make_position_verify_window_launcher();
        auto& position = f.row_launch[1].steps[1];
        position.function = verify.function;
        position.implementation_identity = verify.implementation_identity;
        // Retain the pure metadata callback so C2, rather than a missing
        // preflight, must reject the actual verify-window implementation.
    }, "G0 rejects the actual verify-window position launcher at step1 before either row runs",
        "known window position-history reader implementation is not admitted");
    reject([](auto& f, auto&) {
        const auto qsa = std::find_if(f.source.serial.execution.steps.begin(), f.source.serial.execution.steps.end(),
            [](const auto& step) { return step.op == strix::SerialOp::indexed_attention; });
        if (qsa == f.source.serial.execution.steps.end()) throw std::logic_error("fixture requires a QSA step");
        f.window.mode[qsa->id] = strix::KnownWindowStepMode::r2_gdn_staged;
        f.window.r2_variant[qsa->id] = strix::KnownWindowR2Variant::gdn_q8_virtual256_r2_v1;
    }, "G0 window validation rejects a forged QSA position-reader R2 entry before callbacks",
        "known window GDN variant/source ID mismatch");
    reject([](auto& f, auto&) {
        write_word(f.position_address(), 0x123456789abcdef0ULL);
        auto& position = f.layout.allocations.back();
        f.state_memory.device_base = f.position_address();
        f.state_memory.device_bytes = position.bytes;
        f.state_memory.device_allocation_generation = 9;
        f.layout.device_total_bytes = position.bytes;
        position.storage = strix::SessionStateStorage::device;
        position.offset = 0;
    }, "G0 rejects device position storage with zero dispatches, ready session and unchanged sentinel",
        "known window requires writable bounded host position history");
    reject([](auto& f, auto&) {
        write_word(f.position_address(), 0x123456789abcdef0ULL);
        f.layout.allocations.back().bytes = sizeof(std::uint64_t) - 1U;
    }, "G0 rejects a short host position view with zero dispatches, ready session and unchanged sentinel",
        "known window requires writable bounded host position history");
    reject([](auto& f, auto&) {
        write_word(f.position_address(), 0x123456789abcdef0ULL);
        f.state_memory.host_base = 0;
    }, "G0 rejects a null host position pool with zero dispatches, ready session and unchanged sentinel",
        "host session-state pool has a null base");

    {
        KnownRuntimeFixture candidate(model);
        const auto generations = candidate.session.state.generations;
        const auto valid = candidate.session.state.valid;
        const auto generation = candidate.state_memory.host_allocation_generation;
        const auto compatibility = candidate.session.runtime_compatibility_identity;
        candidate.trace.rebind_after_qsa = &candidate.state_memory;
        std::string error;
        try { (void)candidate.run(candidate.hooks()); }
        catch (const std::invalid_argument& exception) { error = exception.what(); }
        expect(error == "known window host position binding changed after admission" &&
            candidate.trace.admissions > 0 && candidate.trace.calls > 0 && candidate.trace.rebindings == 1 &&
            candidate.trace.qsa_positions == std::vector<std::uint64_t>{0} &&
            candidate.state_memory.host_allocation_generation == generation + 1 &&
            candidate.session.health == strix::SerialSessionHealth::poisoned && candidate.quiesces == 1 &&
            candidate.session.arithmetic_lineage.tainted && candidate.session.arithmetic_lineage.intervals.empty() &&
            candidate.session.state.generations == generations && candidate.session.state.valid == valid &&
            candidate.session.next_position == 0 && candidate.session.committed_tokens == 0 &&
            candidate.session.runtime_compatibility_identity == compatibility && read_word(candidate.position_address()) == 0,
            "a generation swap inside QSA is re-resolved before row1, poisons and drains once without metadata publication");
    }
    for (const bool zero : {false, true}) {
        KnownRuntimeFixture candidate(model);
        StartFaultLookup lookup(zero);
        const auto hash = strix::build_ple_hash_contract(model);
        strix::PleLookupPrefetchState prefetch;
        candidate.resources.ple_lookup = &lookup;
        candidate.resources.ple_hash_contract = &hash;
        candidate.resources.ple_prefetch = &prefetch;
        candidate.session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
            candidate.source.serial.execution_identity, candidate.source.serial.arena,
            candidate.resources).compatibility_identity;
        const auto words = candidate.state_words;
        caught = false;
        try { (void)candidate.run(candidate.hooks()); } catch (const std::runtime_error&) { caught = true; }
        expect(caught && lookup.starts == 1 && lookup.cancels == 0 && !prefetch.active &&
            candidate.trace.calls == 0 && candidate.state_words == words && candidate.quiesces == 0 &&
            candidate.session.health == strix::SerialSessionHealth::ready &&
            !candidate.session.arithmetic_lineage.tainted && candidate.session.committed_tokens == 0,
            "PLE start throw/zero handle preserves ready state without cancelling an unknown handle");
    }
    for (const bool completion_fault : {false, true}) {
        KnownRuntimeFixture candidate(model);
        auto callbacks = candidate.hooks();
        if (completion_fault) callbacks.complete = [&candidate] {
            expect(candidate.session.committed_tokens == 0 && candidate.session.next_position == 0,
                "completion observes uncommitted window metadata");
            throw std::runtime_error("completion fault");
        };
        else candidate.trace.fail_kernel = true;
        caught = false;
        try { (void)candidate.run(callbacks); } catch (const std::runtime_error&) { caught = true; }
        expect(caught && candidate.trace.calls > 0 && candidate.session.health == strix::SerialSessionHealth::poisoned &&
            candidate.session.arithmetic_lineage.tainted && candidate.session.committed_tokens == 0 &&
            candidate.session.next_position == 0 && candidate.quiesces == 1,
            "mid-dispatch/completion failure poisons, taints and drains exactly once without metadata commit");
    }
    {
        KnownRuntimeFixture candidate(model);
        candidate.trace.fail_qsa = true;
        const auto generations = candidate.session.state.generations;
        caught = false;
        try { (void)candidate.run(candidate.hooks()); }
        catch (const std::runtime_error& exception) {
            caught = std::string(exception.what()) == "injected row1 QSA failure";
        }
        expect(caught && candidate.trace.qsa_positions == std::vector<std::uint64_t>{0, 1} &&
            candidate.session.health == strix::SerialSessionHealth::poisoned &&
            candidate.session.arithmetic_lineage.tainted && candidate.session.arithmetic_lineage.intervals.empty() &&
            candidate.session.state.generations == generations && candidate.session.committed_tokens == 0 &&
            candidate.session.next_position == 0 && candidate.quiesces == 1,
            "an actual row1 QSA dispatch failure poisons and drains once without publishing generations or chunk metadata");
    }
}

void known_window_tests(const strix::ModelContract& model) {
    const auto fixture = make_known_fixture(model);
    const auto off = strix::compile_known_prefix_window(model, fixture.serial, fixture.manifest, fixture.access);
    expect(off.r2_steps == 0 && off.row_loop_steps == fixture.serial.execution.steps.size(),
        "known window default never promotes a serial kernel");
    expect(off.logical_projection_weight_bytes_avoided == 0 && off.projection_launches_saved == 0,
        "all-off row loops claim no logical projection savings");
    const auto on = strix::compile_known_prefix_window(model, fixture.serial, fixture.manifest, fixture.access, {true,true});
    expect(on.r2_steps == 84 && on.runtime_row_steps == 1 &&
        on.window_execution_identity != fixture.serial.execution_identity &&
        on.window_execution_identity != off.window_execution_identity,
        "known window closed table promotes 48 shared and 36 GDN steps with a distinct identity");
    expect(on.projection_launches_saved == 48U * 3U + 36U * 5U &&
        on.logical_projection_weight_bytes_avoided ==
            48ULL * (3ULL * (2560ULL / 32ULL * 34ULL * 640ULL)) +
            36ULL * (2560ULL / 32ULL * 34ULL * (10240ULL + 6144ULL + 48ULL + 48ULL) +
                6144ULL / 32ULL * 34ULL * 2560ULL),
        "logical savings derive only from the verified three/five selected Q8 projections");
    for (std::size_t row = 0; row < 2; ++row) {
        expect(on.row_plan[row].execution_identity == on.window_execution_identity &&
            on.row_plan[row].kernel_plan_identity == fixture.serial.kernel_plan_identity &&
            on.row_plan[row].logical_rows == 1 && !on.row_plan[row].whole_plan_graph_safe,
            "known window row clones retain serial kernel contract with window memory identity");
        for (std::size_t i = 0; i < fixture.serial.bindings.bindings.size(); ++i)
            expect(on.row_plan[row].bindings.bindings[i].kernel_id == fixture.serial.bindings.bindings[i].kernel_id,
                "known window never retags a serial KernelVariant ID");
    }
    for (std::size_t i = 0; i < on.arena.physical.allocations.size(); ++i) {
        const auto* original = strix::find_arena_allocation(fixture.serial.arena, on.arena.physical.allocations[i].use);
        expect(original && on.arena.row_view[0].allocations[i].bytes == original->bytes &&
            on.arena.row_view[1].allocations[i].offset - on.arena.row_view[0].allocations[i].offset == on.arena.row_stride[i],
            "R2 derives row strides from actual R1 payloads including the A8/Q51 workspace");
    }
    const auto reject = [&](auto changed, const char* message) {
        bool caught = false;
        try { strix::validate_compiled_known_prefix_window(changed, fixture.serial); }
        catch (const std::invalid_argument&) { caught = true; }
        expect(caught, message);
    };
    auto changed = on; changed.arena.row_stride[0] += 256;
    reject(changed, "known window rejects a mutated public stride");
    changed = on; changed.arena.row_view[1].allocations[0].offset = changed.arena.row_view[0].allocations[0].offset;
    reject(changed, "known window rejects row1/row0 physical alias");
    changed = on; changed.row_plan[1].bindings.bindings.back().kernel_id = "mutated.head";
    reject(changed, "known window rejects a mutated row kernel ID");
    changed = on; ++changed.row_plan[1].io.hidden_slots;
    reject(changed, "known window rejects mutated row IO metadata");
    changed = on; ++changed.r2_steps;
    reject(changed, "known window rejects fabricated promotion counters");
    changed = on; ++changed.logical_projection_weight_bytes_avoided;
    reject(changed, "known window rejects fabricated logical weight savings");
    changed = on; ++changed.projection_launches_saved;
    reject(changed, "known window rejects fabricated projection launch savings");
    changed = on; changed.mode[0] = strix::KnownWindowStepMode::r2_shared_q8;
    reject(changed, "known window rejects an R2 mode on a runtime step");
    auto unsupported = fixture;
    for (auto& binding : unsupported.serial.bindings.bindings)
        if (binding.op == strix::SerialOp::gated_delta_net) *binding.kernel_id += ".unknown";
    const auto fallback = strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access, {true,true});
    expect(fallback.r2_steps == 48, "unknown GDN serial IDs remain explicit R1 row loops");
    unsupported = fixture; unsupported.manifest.tensors[0].ggml_type = 0;
    const auto typed_fallback = strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access, {true,true});
    expect(typed_fallback.r2_steps == 83, "a non-Q8 matrix cannot promote the corresponding R2 step");
    unsupported = fixture; unsupported.manifest.tensors.back().byte_size -= 1;
    bool caught = false;
    try { (void)strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access, {true,true}); }
    catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "a late malformed Q8 tensor rejects before runtime dispatch");
    unsupported = fixture; unsupported.serial.device_graph_inputs = true; caught = false;
    try { (void)strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access); }
    catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "known window rejects a graph source");
    strix::ArithmeticLineage lineage;
    lineage = strix::prepare_prefill_chunk_lineage(lineage, 0, 2, on.window_execution_identity, on.window_kernel_plan_identity, on.window_execution_identity);
    lineage = strix::prepare_prefill_chunk_lineage(lineage, 2, 4, on.window_execution_identity, on.window_kernel_plan_identity, on.window_execution_identity);
    expect(lineage.intervals.size() == 2 && strix::arithmetic_lineage_valid(lineage,4) &&
        !strix::arithmetic_lineage_serial_only(lineage, fixture.serial.execution_identity,4),
        "prefill chunks retain boundaries and cannot forge a serial producer history");
    lineage = strix::prepare_serial_row_lineage(lineage,4,fixture.serial.execution_identity,fixture.serial.kernel_plan_identity);
    expect(lineage.intervals.size() == 3 && strix::arithmetic_lineage_valid(lineage,5),
        "a serial continuation may append to a complete known-prefix chunk history");
    expect(strix::prepare_prefill_chunk_lineage({}, 7,9,1,2,3).intervals.empty(),
        "prefill lineage never invents an unknown prior history");
    lineage.tainted = true; caught = false;
    try { (void)strix::prepare_prefill_chunk_lineage(lineage,5,7,1,2,3); }
    catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "tainted lineage cannot become a clean chunk producer");
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SpeculativeVerifyBindingRequest binding;
    binding.request_concurrency = 1;

    strix::ExecutionBufferRequest buffers;
    buffers.logical_rows = 4;

    const auto registry =
        make_registry(model, 4);
    const auto compiled =
        strix::compile_speculative_verify_execution(
            model,
            4,
            registry,
            binding,
            buffers,
            256);

    expect(compiled.candidate_depth == 4 &&
               compiled.bindings.missing_steps == 0 &&
               compiled.execution_identity != 0 &&
               compiled.kernel_plan_identity != 0,
           "compiled verify freezes depth, bindings and identities");
    // Captured from the pinned, pre-router implementation with this fixture.
    expect(compiled.execution_identity == 2522750290423016878ULL &&
               compiled.kernel_plan_identity == 2260442504783589592ULL &&
               compiled.arena.total_bytes == 3993600ULL,
           "default speculative identities and arena retain the pre-router receipt");
    expect(compiled.arena.total_bytes > 0 &&
               compiled.sequential_window_steps == 37 &&
               compiled.causal_window_steps == 12,
           "compiled verify includes D-row arena and stateful window counts");
    expect(!compiled.whole_plan_graph_safe,
           "whole verify plan remains non-graph-safe while dynamic/runtime steps exist");

    const auto compiled_again =
        strix::compile_speculative_verify_execution(
            model,
            4,
            registry,
            binding,
            buffers,
            256);
    expect(compiled_again.execution_identity ==
               compiled.execution_identity,
           "verify execution identity is reproducible");

    const auto different_registry =
        make_registry(
            model,
            4,
            ".alternate");
    const auto different =
        strix::compile_speculative_verify_execution(
            model,
            4,
            different_registry,
            binding,
            buffers,
            256);
    expect(different.execution_identity !=
               compiled.execution_identity,
           "verify kernel identity change invalidates execution identity");

    auto wrong_rows = buffers;
    wrong_rows.logical_rows = 2;
    bool rows_rejected = false;
    try {
        static_cast<void>(
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                binding,
                wrong_rows,
                256));
    } catch (const std::invalid_argument&) {
        rows_rejected = true;
    }
    expect(rows_rejected,
           "verify compile rejects buffer rows that differ from depth");

    auto wrong_concurrency = binding;
    wrong_concurrency.request_concurrency = 2;
    bool concurrency_rejected = false;
    try {
        static_cast<void>(
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                wrong_concurrency,
                buffers,
                256));
    } catch (const std::invalid_argument&) {
        concurrency_rejected = true;
    }
    expect(concurrency_rejected,
           "initial verify compile keeps request concurrency at one");

    auto router_binding = binding;
    auto router_buffers = buffers;
    router_binding.parallel_moe_router = true;
    router_buffers.parallel_moe_router = true;
    const auto router_registry = make_registry(model, 4, "", true);
    const auto router_compiled = strix::compile_speculative_verify_execution(
        model, 4, router_registry, router_binding, router_buffers);
    expect(router_compiled.execution_identity != compiled.execution_identity &&
               router_compiled.kernel_plan_identity != compiled.kernel_plan_identity &&
               router_compiled.io.router_workspace_slots == 1,
           "parallel router has admitted workspace and distinct speculative identities");
    bool router_mismatch = false;
    try { static_cast<void>(strix::compile_speculative_verify_execution(model, 4,
                                      router_registry, router_binding, buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "verify compilation rejects router request/admission mismatch");
    router_mismatch = false;
    try { static_cast<void>(strix::compile_speculative_verify_execution(model, 4,
                                      registry, router_binding, router_buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "verify compilation rejects legacy router under parallel admission");
    router_mismatch = false;
    try { static_cast<void>(strix::compile_speculative_verify_execution(model, 4,
                                      router_registry, binding, buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "verify compilation rejects parallel router under default admission");
    auto ranked_binding = router_binding; ranked_binding.ranked_moe_router_topk = true;
    const auto ranked_registry = make_registry(model, 4, "", true, true);
    const auto ranked_compiled = strix::compile_speculative_verify_execution(model, 4, ranked_registry, ranked_binding, router_buffers);
    expect(ranked_compiled.execution_identity != router_compiled.execution_identity &&
               ranked_compiled.kernel_plan_identity != router_compiled.kernel_plan_identity &&
               ranked_compiled.arena.total_bytes == router_compiled.arena.total_bytes,
           "ranked speculative binding changes identities with unchanged arena");
    bool rejected_ranked = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, router_registry, ranked_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_ranked = true; }
    expect(rejected_ranked, "ranked speculative admission rejects parallel-only kernel");
    rejected_ranked = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, ranked_registry, router_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_ranked = true; }
    expect(rejected_ranked, "parallel speculative admission rejects ranked kernel");
    auto wave_binding = router_binding; wave_binding.wave_moe_router_topk = true;
    const auto wave_registry = make_registry(model, 4, "", true, false, true);
    const auto wave_compiled = strix::compile_speculative_verify_execution(model, 4, wave_registry, wave_binding, router_buffers);
    expect(wave_compiled.execution_identity != ranked_compiled.execution_identity &&
               wave_compiled.kernel_plan_identity != ranked_compiled.kernel_plan_identity &&
               wave_compiled.arena.total_bytes == router_compiled.arena.total_bytes,
           "wave speculative identities change without changing arena");
    bool rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, router_registry, wave_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "wave speculative admission rejects parallel-only binding");
    rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, wave_registry, router_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "parallel speculative admission rejects wave binding");
    auto invalid_wave = wave_binding; invalid_wave.ranked_moe_router_topk = true;
    rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, wave_registry, invalid_wave, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "wave and ranked speculative modes mutually exclude");
    invalid_wave = wave_binding; invalid_wave.parallel_moe_router = false;
    rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, wave_registry, invalid_wave, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "wave speculative mode requires parallel topology");
    auto invalid_ranked = binding; invalid_ranked.ranked_moe_router_topk = true;
    rejected_ranked = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, registry, invalid_ranked, buffers); }
    catch (const std::invalid_argument&) { rejected_ranked = true; }
    expect(rejected_ranked, "ranked speculative mode requires parallel topology");
    router_binding.parallel_moe_router = false;
    router_buffers.parallel_moe_router = false;
    const auto explicit_default = strix::compile_speculative_verify_execution(model, 4,
                                      registry, router_binding, router_buffers);
    expect(explicit_default.execution_identity == compiled.execution_identity &&
               explicit_default.kernel_plan_identity == compiled.kernel_plan_identity &&
               explicit_default.arena.total_bytes == compiled.arena.total_bytes,
           "explicit disabled router preserves speculative default identities and arena");

    auto q4_buffers=buffers; q4_buffers.logical_rows=1; q4_buffers.q4k_a8g32_gate_up=true;
    bool q4_rejected=false;
    try { (void)strix::compile_speculative_verify_execution(model,1,make_registry(model,1),binding,q4_buffers); }
    catch(const std::invalid_argument& error) {
        q4_rejected=std::string(error.what()).find("Q4K A8 runtime does not support speculative verify") != std::string::npos;
    }
    expect(q4_rejected,"Q4K A8 explicitly rejects speculative depth1 before alternate admission");
    q4_buffers.q4k_a8g32_gate_up=false; q4_rejected=false;
    try { (void)strix::compile_speculative_verify_execution(model,1,
        make_registry(model,1,".q4k-a8g32-gateup-v2"),binding,q4_buffers); }
    catch(const std::invalid_argument& error) {
        q4_rejected=std::string(error.what()).find("Q4K A8 runtime does not support speculative verify") != std::string::npos;
    }
    expect(q4_rejected,"speculative depth1 rejects quantized kernel even without mode admission");


    auto hc_binding=binding;hc_binding.lane_chain_hyper_output=true;bool hc_rejected=false;
    try{(void)strix::compile_speculative_verify_execution(model,4,registry,hc_binding,buffers);}catch(const std::invalid_argument&){hc_rejected=true;}
    expect(hc_rejected,"lane-chain HC rejects compiled speculative execution");
    auto deferred=binding;deferred.deferred_moe_status=true;
    bool deferred_rejected=false;
    try{(void)strix::compile_speculative_verify_execution(model,4,registry,deferred,buffers);}
    catch(const std::invalid_argument&){deferred_rejected=true;}
    expect(deferred_rejected,"deferred MoE explicitly rejects speculative verification");

    known_window_tests(model);
    known_window_runtime_tests(model);

    if (failures == 0) {
        std::cout << "compiled speculative verify tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
