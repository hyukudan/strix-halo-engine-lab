#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace strix;
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Span { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
struct Owner {
    HipSetupQuiescenceGuard quiescence;
    hipStream_t stream{};
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    Owner() { quiescence.run([&] { check(hipStreamCreate(&stream), "create known-window stream"); }); }
    ~Owner() {
        quiescence.recover_or_fail_stop();
        const auto cleanup = [](hipError_t error) { if (error != hipSuccess) std::_Exit(3); };
        for (const auto graph : graphs) cleanup(hipGraphDestroy(graph));
        for (const auto allocation : allocations) cleanup(hipFree(allocation));
        cleanup(hipStreamDestroy(stream));
    }
    Span allocate(std::uint64_t bytes) {
        // The long-context QSA bench owns whole-session state (~7 GiB at 256K).
        const auto bound = std::getenv("STRIX_QSA_LONG_BENCH") ? (1ULL << 34U) : (1ULL << 30U);
        require(bytes && bytes < bound, "fixture allocation exceeds its bound");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, static_cast<std::size_t>(bytes + 512U)), "allocate guarded known-window fixture");
        try { allocations.push_back(pointer); }
        catch (...) { quiescence.recover_or_fail_stop(); (void)hipFree(pointer); throw; }
        check(hipMemset(pointer, 0xa5, static_cast<std::size_t>(bytes + 512U)), "poison fixture guards");
        return {reinterpret_cast<std::uintptr_t>(pointer) + 256U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "complete known-window fixture"); quiescence.require_original(); }
};
std::vector<unsigned char> read(Span span) {
    std::vector<unsigned char> bytes(static_cast<std::size_t>(span.bytes));
    check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(span.address), bytes.size(), hipMemcpyDeviceToHost), "read fixture bytes");
    return bytes;
}
void write(Span span, const std::vector<unsigned char>& bytes) {
    require(bytes.size() == span.bytes, "fixture upload extent differs");
    check(hipMemcpy(reinterpret_cast<void*>(span.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "upload fixture bytes");
}
void guards(Span span) {
    for (const auto address : {span.address - 256U, span.address + span.bytes}) {
        const auto bytes = read({address, 256U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == 0xa5; }), "fixture exterior canary changed");
    }
}
template<class T> std::vector<unsigned char> bytes_of(const std::vector<T>& values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
struct Q8 { __half scale; std::int8_t codes[32]; };
static_assert(sizeof(Q8) == 34);
std::vector<unsigned char> q8(std::uint32_t width, std::uint32_t outputs, unsigned salt) {
    std::vector<Q8> blocks(static_cast<std::size_t>(width / 32U) * outputs);
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        blocks[index].scale = __float2half(0.0005F * static_cast<float>(1U + (index + salt) % 5U));
        for (unsigned code = 0; code < 32U; ++code)
            blocks[index].codes[code] = static_cast<std::int8_t>(static_cast<int>((index * 7U + code * 3U + salt) % 13U) - 6);
    }
    return bytes_of(blocks);
}
std::vector<unsigned char> f32(std::size_t size, float base, float variation = 0.0F) {
    std::vector<float> values(size);
    for (std::size_t index = 0; index < size; ++index)
        values[index] = base + variation * static_cast<float>(static_cast<int>(index % 7U) - 3);
    return bytes_of(values);
}
struct Weight { Span device; std::vector<unsigned char> original; };
struct Result {
    std::array<std::vector<unsigned char>, 2> output, temporary;
    std::vector<unsigned char> matrix, convolution;
};

// One real selected operator, two C1 row views, original canonical model
// geometry. No decoder/session execution is inferred from this component test.
struct Fixture {
    Owner owner; // retained until every stream operation and borrowed owner is drained
    const bool gdn;
    const bool f32_gates;
    const std::uint32_t context_capacity;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch, state;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    SerialSessionStateLayout state_layout;
    SessionStateMemoryBinding state_memory;
    std::vector<unsigned char> host_state;
    std::vector<unsigned char> initial_state, seed;
    std::array<ExecutionRuntimeResources, 2> resources;
    std::array<ExecutionDispatchContext, 2> rows;
    std::array<ExecutionBufferAddress, 2> input{}, output{}, temporary{};
    HipSharedExpertLauncherConfig shared_config;
    HipGdnLauncherConfig gdn_config;
    std::string selected_id;
    ExecutionLauncher selected;
    Span matrix, convolution;

    explicit Fixture(bool is_gdn, bool shared_prefetch = false, bool mixed_gates = false,std::uint32_t context_tokens=4U)
        : gdn(is_gdn), f32_gates(mixed_gates),context_capacity(context_tokens) {
        const auto wanted = gdn ? SerialOp::gated_delta_net : SerialOp::shared_expert;
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [&](const auto& value) { return value.op == wanted && value.layer == 0U; });
        require(found != execution.steps.end(), "canonical layer-zero operator missing");
        step = *found;
        const auto& step_io = io.steps[step.id];
        require(step_io.reads.size() == 1U && step_io.writes.size() == 1U && step_io.temporaries.size() == 1U,
            "fixture must represent one selected row IO contract");
        const std::uint64_t workspace_bytes = gdn ? (10240ULL + 6144ULL + 96ULL + 6144ULL) * 4U : 640ULL * 8U;
        serial.alignment = 256U;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes});
            serial.total_bytes = offset + bytes;
        };
        allocation(step_io.reads[0], 2560U * 2U);
        allocation(step_io.writes[0], 2560U * 2U);
        allocation(step_io.temporaries[0], workspace_bytes);
        serial.total_bytes = (serial.total_bytes + 255U) / 256U * 256U;
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e523201ULL, scratch.address, scratch.bytes, 19U};
        manifest.architecture = "qwen4_exp";
        if (gdn) setup_gdn(); else setup_shared(shared_prefetch);
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e523202ULL;
        for (unsigned row = 0; row < 2U; ++row) {
            resources[row] = {&memory, &manifest, &runtime, &access};
            if (gdn) {
                resources[row].session_state_layout = &state_layout;
                resources[row].session_state_memory = &state_memory;
            }
            auto& context = rows[row];
            context.execution_identity = memory.execution_identity;
            context.step_id = step.id; context.op = step.op; context.layer = step.layer;
            context.kernel_id = selected_id;
            context.reads = step_io.reads; context.writes = step_io.writes; context.temporaries = step_io.temporaries;
            context.arena = &window.row_view[row]; context.memory = &memory; context.resources = &resources[row];
            context.session_state_layout = gdn ? &state_layout : nullptr;
            context.session_state_memory = gdn ? &state_memory : nullptr;
            context.reads_state = step.reads_state; context.writes_state = step.writes_state;
            context.position = row; context.input_token_id = 17U + row;
            input[row] = resolve_dispatch_read(context, 0);
            output[row] = resolve_dispatch_write(context, 0);
            temporary[row] = resolve_dispatch_temporary(context, 0);
        }
    }
    void add(std::string name, TensorRole role, std::uint32_t type,
             std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes) {
        ManifestTensor tensor;
        tensor.name = std::move(name); tensor.role = role; tensor.layer = step.layer;
        tensor.ggml_type = type; tensor.dimensions = std::move(dimensions); tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes += tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    }
    void setup_shared(bool prefetch) {
        add("blk.0.shared_expert.gate_proj.weight", TensorRole::dense_projection, 8U, {2560, 640}, q8(2560, 640, 1));
        add("blk.0.shared_expert.up_proj.weight", TensorRole::dense_projection, 8U, {2560, 640}, q8(2560, 640, 2));
        add("blk.0.shared_expert.down_proj.weight", TensorRole::dense_projection, 8U, {640, 2560}, q8(640, 2560, 3));
        add("blk.0.shared_expert_gate.weight", TensorRole::dense_projection, 0U, {2560}, f32(2560, 0.0002F, 0.00003F));
        access.accesses.push_back({step.id, step.op, 0U, {0, 1, 2, 3}});
        shared_config.hidden_size = 2560U; shared_config.intermediate_size = 640U;
        shared_config.prefetch_q8 = prefetch;
        if (prefetch) {
            shared_config.prefetch_eligible_layers = {0U};
            shared_config.prefetch_stats = std::make_shared<shared_expert_prefetch::RuntimeDispatchStats>();
        }
        kernels::KernelRegistry registry;
        ExecutionBindingRequest binding;
        binding.prefetch_q8_shared_expert = prefetch;
        binding.shared_expert_prefetch_layers = shared_config.prefetch_eligible_layers;
        register_hip_shared_expert_kernel_variant(model, binding, shared_config, registry);
        require(registry.variants().size() == 1U, "shared serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_shared_expert_launcher(selected_id, &shared_config);
    }
    void setup_gdn() {
        add("blk.0.attn_qkv.weight", TensorRole::attention_projection, 8U, {2560, 10240}, q8(2560, 10240, 4));
        add("blk.0.attn_gate.weight", TensorRole::attention_projection, 8U, {2560, 6144}, q8(2560, 6144, 5));
        add("blk.0.ssm_conv1d.weight", TensorRole::recurrent_state, 0U, {4, 10240}, f32(4U * 10240U, 0.2F, 0.015F));
        add("blk.0.ssm_dt.bias", TensorRole::recurrent_state, 0U, {48}, f32(48U, -0.1F, 0.01F));
        add("blk.0.ssm_a", TensorRole::recurrent_state, 0U, {48}, f32(48U, -0.25F, 0.01F));
        add("blk.0.ssm_beta.weight", TensorRole::recurrent_state, f32_gates ? 0U : 8U, {2560, 48},
            f32_gates ? f32(2560U * 48U, 0.0005F, 0.0001F) : q8(2560, 48, 6));
        add("blk.0.ssm_alpha.weight", TensorRole::recurrent_state, f32_gates ? 0U : 8U, {2560, 48},
            f32_gates ? f32(2560U * 48U, -0.0004F, 0.0001F) : q8(2560, 48, 7));
        add("blk.0.ssm_norm.weight", TensorRole::recurrent_state, 0U, {128}, f32(128U, 0.75F, 0.01F));
        add("blk.0.ssm_out.weight", TensorRole::recurrent_state, 8U, {6144, 2560}, q8(6144, 2560, 8));
        access.accesses.push_back({step.id, step.op, 0U, {2, 3, 4, 5, 6, 7, 8}});
        access.accesses.push_back({step.id, step.op, 1U, {0, 1}});
        SerialSessionStateLayoutRequest request; request.context_tokens = context_capacity;
        state_layout = build_serial_session_state_layout(model, request);
        state = owner.allocate(state_layout.device_total_bytes);
        host_state.resize(static_cast<std::size_t>(state_layout.host_total_bytes));
        state_memory = {state_layout.layout_identity, state.address, state.bytes, 23U,
            reinterpret_cast<std::uintptr_t>(host_state.data()), state_layout.host_total_bytes, 29U};
        initial_state.resize(static_cast<std::size_t>(state.bytes), 0U);
        const auto m = resolve_gdn_layer_state(state_layout, state_memory, model, 0U, SessionStateComponent::recurrent_matrix);
        const auto c = resolve_gdn_layer_state(state_layout, state_memory, model, 0U, SessionStateComponent::gdn_convolution);
        matrix = {m.address, m.bytes}; convolution = {c.address, c.bytes};
        const auto initial_matrix = f32(static_cast<std::size_t>(matrix.bytes / 4U), 0.002F, 0.0001F);
        std::memcpy(initial_state.data() + (matrix.address - state.address), initial_matrix.data(), initial_matrix.size());
        std::vector<std::uint16_t> initial_conv(static_cast<std::size_t>(convolution.bytes / 2U));
        for (std::size_t i = 0; i < initial_conv.size(); ++i) initial_conv[i] = bf16(0.01F * static_cast<float>(1U + i % 9U));
        std::memcpy(initial_state.data() + (convolution.address - state.address), initial_conv.data(), initial_conv.size() * 2U);
        gdn_config.model = model; gdn_config.virtual_q8_input_projections = true;
        gdn_config.prefetch_q8_input_projections = true; gdn_config.wave8_q8_input_projections = true;
        gdn_config.virtual_q8_output_projection = true; gdn_config.wave8_q8_output_projection = true;
        ExecutionBindingRequest binding;
        binding.wave8_q8_gdn_input = true; binding.wave8_q8_gdn_output = true;
        kernels::KernelRegistry registry;
        register_hip_gdn_kernel_variant(model, binding, gdn_config, registry);
        require(registry.variants().size() == 1U, "GDN serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_gdn_launcher(selected_id, &gdn_config);
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    void preflight(std::span<const ExecutionDispatchContext, 2> contexts) const {
        if (gdn && f32_gates) gdn_known_window_r2_f32_gates_preflight(contexts, gdn_config);
        else if (gdn) gdn_known_window_r2_preflight(contexts, gdn_config);
        else shared_expert_known_window_r2_preflight(contexts, shared_config);
    }
    void launch(std::span<const ExecutionDispatchContext, 2> contexts) const {
        if (gdn && f32_gates) gdn_known_window_r2_f32_gates_launch(contexts, gdn_config);
        else if (gdn) gdn_known_window_r2_launch(contexts, gdn_config);
        else shared_expert_known_window_r2_launch(contexts, shared_config);
    }
    void prepare(unsigned flavor, bool swapped = false) {
        seed.assign(static_cast<std::size_t>(scratch.bytes), 0xa5);
        for (unsigned row = 0; row < 2U; ++row) {
            const auto source_row = swapped ? 1U - row : row;
            std::vector<std::uint16_t> values(2560U);
            constexpr std::array<std::uint16_t, 12> edge{0, 0x8000, 1, 0x8001, 0x007f, 0x807f,
                0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x7f81, 0xff83};
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (flavor == 4U) values[i] = i % 32U == 3U ? static_cast<std::uint16_t>(source_row ? 0xff83U : 0x7f81U) :
                    i % 32U == 7U ? static_cast<std::uint16_t>(source_row ? 0xffc3U : 0x7fc1U) :
                    bf16((i == 255U || i == 256U || i == 2559U) ? 2.0F :
                        (i % 2U ? -1.0F : 1.0F) * (source_row ? 0.125F : 0.25F));
                else if (flavor == 1U) values[i] = edge[(i + source_row * 3U) % 6U];
                else if (flavor == 3U) values[i] = edge[(i + source_row * 5U) % edge.size()];
                else if (flavor == 2U) values[i] = bf16((i % 2U ? -1.0F : 1.0F) * (source_row ? 0.125F : 0.25F));
                else values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + source_row * 11U) % 29U) - 14) * (source_row ? -0.043F : 0.031F));
            }
            std::memcpy(seed.data() + (input[row].address - scratch.address), values.data(), values.size() * 2U);
        }
    }
    void reset() { write(scratch, seed); if (gdn) write(state, initial_state); }
    void readonly() const {
        const auto actual = read(scratch);
        auto expected = seed;
        for (const auto& writes : {output, temporary})
            for (const auto view : writes)
                std::copy_n(actual.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address),
                    static_cast<std::size_t>(view.bytes), expected.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address));
        require(actual == expected, "readonly row input or interior arena canary changed");
        guards(scratch);
        for (const auto& weight : weights) {
            require(read(weight.device) == weight.original, "known-window weights changed");
            guards(weight.device);
        }
        if (gdn) {
            auto expected_state = initial_state;
            const auto actual_state = read(state);
            for (const auto view : {matrix, convolution})
                std::copy_n(actual_state.begin() + static_cast<std::ptrdiff_t>(view.address - state.address),
                    static_cast<std::size_t>(view.bytes), expected_state.begin() + static_cast<std::ptrdiff_t>(view.address - state.address));
            require(actual_state == expected_state, "GDN modified another layer or state component");
            guards(state);
        }
    }
    Result run(bool candidate) {
        reset();
        const auto before = shared_config.prefetch_stats ?
            shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) : 0U;
        {
            const HipLaunchScope owned(scope());
            if (candidate) { preflight(rows); launch(rows); }
            else for (const auto& row : rows) selected.function(row, selected.user_data);
        }
        owner.finish();
        if (shared_config.prefetch_stats)
            require(shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) - before == (candidate ? 3U : 6U),
                "shared R2 actual projection submission count differs");
        Result result;
        for (unsigned row = 0; row < 2U; ++row) {
            result.output[row] = read({output[row].address, output[row].bytes});
            result.temporary[row] = read({temporary[row].address, temporary[row].bytes});
        }
        if (gdn) { result.matrix = read(matrix); result.convolution = read(convolution); }
        readonly();
        return result;
    }
};
void same(const Result& left, const Result& right) {
    require(left.output == right.output, "R2 output rows differ bitwise from two selected serial calls");
    require(left.temporary == right.temporary, "R2 complete row temporaries differ bitwise from selected serial calls");
    require(left.matrix == right.matrix, "R2 complete recurrent matrix differs from serial after two rows");
    require(left.convolution == right.convolution, "R2 complete convolution history differs from serial after two rows");
}
void finite(const Result& result) {
    const auto bf16_values = [](const std::vector<unsigned char>& bytes) {
        require(bytes.size() % 2U == 0U, "BF16 result has partial element");
        for (std::size_t offset = 0; offset < bytes.size(); offset += 2U) {
            std::uint16_t bits{};
            std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
            require((bits & 0x7f80U) != 0x7f80U, "finite fixture produced a nonfinite BF16 value");
        }
    };
    for (const auto& output : result.output) bf16_values(output);
    bf16_values(result.convolution);
    require(result.matrix.size() % 4U == 0U, "FP32 matrix has partial element");
    for (std::size_t offset = 0; offset < result.matrix.size(); offset += 4U) {
        float value{};
        std::memcpy(&value, result.matrix.data() + offset, sizeof(value));
        require(std::isfinite(value), "finite fixture produced a nonfinite recurrent state");
    }
}
template<class Work> void invalid(Work&& work) {
    bool rejected = false;
    try { work(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid known-window admission was accepted");
}
void capture_rejection(Fixture& fixture, std::span<const ExecutionDispatchContext, 2> rows) {
    check(hipStreamBeginCapture(fixture.owner.stream, hipStreamCaptureModeThreadLocal), "begin rejection capture");
    std::exception_ptr error;
    try { const HipLaunchScope owned(fixture.scope()); invalid([&] { fixture.launch(rows); }); }
    catch (...) { error = std::current_exception(); }
    hipGraph_t graph{};
    check(hipStreamEndCapture(fixture.owner.stream, &graph), "end rejection capture");
    require(graph != nullptr, "rejection capture did not return an empty graph");
    try { fixture.owner.graphs.push_back(graph); }
    catch (...) { (void)hipGraphDestroy(graph); throw; }
    std::size_t nodes = 0;
    check(hipGraphGetNodes(graph, nullptr, &nodes), "count rejection capture nodes");
    require(nodes == 0U, "rejected known-window call enqueued work");
    if (error) std::rethrow_exception(error);
}
void rejections(Fixture& fixture) {
    fixture.prepare(0U); fixture.reset();
    const auto original_arena = read(fixture.scratch);
    const auto original_state = fixture.gdn ? read(fixture.state) : std::vector<unsigned char>{};
    const auto original_count = fixture.shared_config.prefetch_stats ?
        fixture.shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) : 0U;
    const auto bad = [&](auto mutate) {
        auto rows = fixture.rows;
        mutate(rows);
        { const HipLaunchScope owned(fixture.scope()); invalid([&] { fixture.preflight(rows); }); }
        capture_rejection(fixture, rows);
        fixture.owner.finish();
        require(read(fixture.scratch) == original_arena, "rejected R2 call changed scratch bytes");
        if (fixture.gdn) require(read(fixture.state) == original_state, "rejected R2 call changed state bytes");
    };
    bad([](auto& rows) { rows[1].execution_identity ^= 1U; });
    bad([](auto& rows) { rows[1].position += 1U; });
    bad([](auto& rows) { rows[1].logical_rows = 2U; });
    bad([](auto& rows) { rows[1].temporaries = {}; });
    const auto same_values_other_object = fixture.memory;
    bad([&](auto& rows) { rows[1].memory = &same_values_other_object; });
    auto stale = fixture.memory; stale.allocation_generation = 0U;
    auto stale_resources = fixture.resources[1]; stale_resources.scratch = &stale;
    bad([&](auto& rows) { rows[1].memory = &stale; rows[1].resources = &stale_resources; });
    auto alias = fixture.window.row_view[1];
    for (auto& allocation : alias.allocations)
        if (allocation.use == fixture.rows[1].writes[0]) allocation.offset = fixture.input[0].address - fixture.scratch.address;
    bad([&](auto& rows) { rows[1].arena = &alias; });
    auto stride = fixture.window.row_view[1];
    for (auto& allocation : stride.allocations)
        if (allocation.use == fixture.rows[1].writes[0]) allocation.offset += 16U;
    bad([&](auto& rows) { rows[1].arena = &stride; });
    const auto weight_generation = fixture.runtime.tensors[0].allocation_generation;
    fixture.runtime.tensors[0].allocation_generation = 0U;
    bad([](auto&) {});
    fixture.runtime.tensors[0].allocation_generation = weight_generation;
    if (fixture.gdn) {
        const auto same_state_other_object = fixture.state_memory;
        auto changed_resources = fixture.resources[1];
        changed_resources.session_state_memory = &same_state_other_object;
        bad([&](auto& rows) { rows[1].resources = &changed_resources; });
    }
    // Valid metadata still cannot enter the eager-only wrapper during capture.
    capture_rejection(fixture, fixture.rows);
    fixture.owner.finish();
    require(read(fixture.scratch) == original_arena, "capture rejection changed scratch");
    if (fixture.gdn) require(read(fixture.state) == original_state, "capture rejection changed state");
    if (fixture.shared_config.prefetch_stats)
        require(fixture.shared_config.prefetch_stats->dispatched_projections.load(std::memory_order_relaxed) == original_count,
            "rejected shared R2 call incremented submission count");
    fixture.readonly();
}

// Mutate only the selected original-format Q8 owner. Raw HALF bits are copied,
// never converted on the CPU: signaling/quiet payloads and signed subnormals
// reach the actual device decoder. Keep the readonly oracle in step with the
// intentional mutation, then restore both host oracle and device bytes even
// when a bitwise comparison throws. Drain before any restoration can overwrite
// a weight still borrowed by an unsuccessful launch.
template<class Work> void adversarial_q8_weight(Fixture& fixture, unsigned index,
    std::uint32_t width, std::uint32_t outputs, Work&& work) {
    auto& weight = fixture.weights[index];
    auto saved = weight.original;
    auto mutated = saved;
    const auto blocks = width / 32U;
    require(width % 32U == 0U && blocks && outputs >= 24U &&
        mutated.size() == static_cast<std::size_t>(blocks) * outputs * sizeof(Q8),
        "adversarial Q8 fixture shape differs from original owner");
    constexpr std::array<std::uint16_t, 12> scales{0x0000, 0x8000, 0x0001, 0x8001,
        0x03ff, 0x83ff, 0x7c00, 0xfc00, 0x7e11, 0xfe35, 0x7c03, 0xfc07};
    constexpr std::array<std::int8_t, 8> codes{0, 1, -1, 127, -128, 7, -7, 0};
    for (std::size_t pattern = 0; pattern < scales.size(); ++pattern) {
        for (unsigned end = 0; end < 2U; ++end) {
            const auto feature = end ? static_cast<std::size_t>(outputs) - 1U - pattern : pattern;
            const auto block = (pattern * 7U + end * 13U) % blocks;
            const auto offset = (feature * blocks + block) * sizeof(Q8);
            std::memcpy(mutated.data() + offset, &scales[pattern], sizeof(scales[pattern]));
            for (unsigned code = 0; code < 32U; ++code) {
                const auto value = codes[(code + pattern + end) % codes.size()];
                std::memcpy(mutated.data() + offset + 2U + code, &value, sizeof(value));
            }
        }
    }
    weight.original.swap(mutated);
    const auto restore = [&] {
        weight.original.swap(saved);
        fixture.owner.quiescence.run([&] { write(weight.device, weight.original); });
        require(read(weight.device) == weight.original, "original Q8 owner was not restored exactly");
        guards(weight.device);
    };
    try {
        write(weight.device, weight.original);
        work();
        fixture.owner.finish();
    } catch (...) {
        const auto failure = std::current_exception();
        fixture.owner.quiescence.recover_or_fail_stop();
        restore();
        std::rethrow_exception(failure);
    }
    restore();
}

// Observe complete projection intermediates too: comparing only the final BF16
// operator output could hide a changed input projection behind later rounding.
void gdn_projection_rows(Fixture& fixture) {
    using View = GdnKnownWindowProjectionView;
    using Direction = GdnKnownWindowProjectionDirection;
    using Mode = GdnKnownWindowProjectionMode;
    struct Case { unsigned weight; std::uint32_t width, outputs; Direction direction; };
    const std::array<Case, 5> cases{{{0U, 2560U, 10240U, Direction::bf16_to_fp32},
        {1U, 2560U, 6144U, Direction::bf16_to_fp32}, {5U, 2560U, 48U, Direction::bf16_to_fp32},
        {6U, 2560U, 48U, Direction::bf16_to_fp32}, {8U, 6144U, 2560U, Direction::fp32_to_bf16}}};
    for (const auto item : cases) {
        const bool input_bf16 = item.direction == Direction::bf16_to_fp32;
        const auto in_bytes = static_cast<std::uint64_t>(item.width) * (input_bf16 ? 2U : 4U);
        const auto out_bytes = static_cast<std::uint64_t>(item.outputs) * (input_bf16 ? 4U : 2U);
        const std::array<Span, 2> inputs{fixture.owner.allocate(in_bytes), fixture.owner.allocate(in_bytes)};
        const std::array<Span, 2> outputs{fixture.owner.allocate(out_bytes), fixture.owner.allocate(out_bytes)};
        const std::array<View, 2> input_views{{{inputs[0].address, inputs[0].bytes}, {inputs[1].address, inputs[1].bytes}}};
        const std::array<View, 2> output_views{{{outputs[0].address, outputs[0].bytes}, {outputs[1].address, outputs[1].bytes}}};
        const auto weight = fixture.weights[item.weight].device;
        const auto compare = [&](unsigned flavor) {
            std::array<std::vector<unsigned char>, 2> original;
            for (unsigned row = 0; row < 2U; ++row) {
                std::vector<std::uint16_t> half(item.width);
                std::vector<float> full(item.width);
                constexpr std::array<std::uint16_t, 12> edge{0, 0x8000, 1, 0x8001, 0x7f80, 0xff80,
                    0x7fc1, 0xffc3, 0x7f81, 0xff83, 0x7fc5, 0xffc7};
                for (std::uint32_t i = 0; i < item.width; ++i) {
                    const auto bits = flavor == 4U ?
                        (i % 32U == 3U ? static_cast<std::uint16_t>(row ? 0xff83U : 0x7f81U) :
                         i % 32U == 7U ? static_cast<std::uint16_t>(row ? 0xffc3U : 0x7fc1U) :
                         bf16((i % 2U ? -1.0F : 1.0F) * (row ? 0.125F : 0.25F))) :
                        flavor == 3U ? edge[(i + row * 3U) % edge.size()] :
                        flavor == 1U ? edge[(i + row) % 4U] :
                        bf16((flavor == 2U ? (i % 2U ? -0.25F : 0.25F) :
                            static_cast<float>(static_cast<int>((i * 3U + row * 7U) % 23U) - 11) * 0.031F));
                    half[i] = bits;
                    full[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
                }
                original[row] = input_bf16 ? bytes_of(half) : bytes_of(full);
                write(inputs[row], original[row]);
            }
            const auto run = [&](Mode mode) {
                for (const auto output : outputs)
                    check(hipMemset(reinterpret_cast<void*>(output.address), 0xa5, static_cast<std::size_t>(output.bytes)), "poison raw projection output");
                gdn_known_window_r2_projection_probe(item.direction, mode, input_views,
                    {weight.address, weight.bytes}, output_views, item.width, item.outputs, fixture.owner.stream);
                fixture.owner.finish();
                std::array<std::vector<unsigned char>, 2> result;
                for (unsigned row = 0; row < 2U; ++row) {
                    result[row] = read(outputs[row]);
                    require(read(inputs[row]) == original[row], "raw projection mutated its independent row input");
                    guards(inputs[row]); guards(outputs[row]);
                }
                return result;
            };
            const auto reference = run(Mode::incumbent_row_loop);
            require(run(Mode::rows2) == reference, "GDN R2 complete projection intermediates differ bitwise");
            require(run(Mode::rows2) == reference, "GDN R2 projection is not bitwise repeatable");
            require(read(weight) == fixture.weights[item.weight].original, "raw projection changed packed weights");
            guards(weight);
        };
        for (unsigned flavor = 0; flavor < 4U; ++flavor) compare(flavor);
        // Exactly one additional decoder-header pattern per actual projection,
        // with independent cancellation/NaN-bearing row inputs. Full F32/BF16
        // payload equality remains mandatory, including every NaN payload bit.
        adversarial_q8_weight(fixture, item.weight, item.width, item.outputs, [&] { compare(4U); });
    }
}

// Typed stage observation keeps NaN payloads visible before dt/sigmoid overwrite
// beta/gate. Synthetic canonical geometry only, not real-weight/model evidence.
void gdn_typed_input_stage(Fixture& fixture) {
    constexpr std::uint64_t prefix_bytes = 16528ULL * sizeof(float);
    const auto run = [&](GdnKnownWindowProjectionMode mode) {
        fixture.reset();
        {
            const HipLaunchScope scope(fixture.scope());
            gdn_known_window_r2_input_stage_probe(fixture.rows, fixture.gdn_config,
                fixture.f32_gates ? GdnKnownWindowFormat::q8_rows2_f32_gates : GdnKnownWindowFormat::q8_all5, mode);
        }
        fixture.owner.finish();
        auto expected = fixture.seed;
        std::array<std::vector<unsigned char>,2> result;
        for (unsigned row = 0; row < 2U; ++row) {
            require(fixture.temporary[row].bytes >= prefix_bytes, "typed input prefix exceeds temporary");
            result[row] = read({fixture.temporary[row].address, prefix_bytes});
            std::copy(result[row].begin(), result[row].end(), expected.begin() +
                static_cast<std::ptrdiff_t>(fixture.temporary[row].address - fixture.scratch.address));
        }
        require(read(fixture.scratch) == expected,
            "input-stage probe touched core, row outputs, input or non-prefix scratch bytes");
        require(read(fixture.state) == fixture.initial_state, "input-stage probe touched persistent state");
        fixture.readonly();
        return result;
    };
    for (unsigned flavor = 0; flavor < 5U; ++flavor) {
        fixture.prepare(flavor);
        const auto reference = run(GdnKnownWindowProjectionMode::incumbent_row_loop);
        require(run(GdnKnownWindowProjectionMode::rows2) == reference,
            "typed R2 complete 16528-F32 input prefix differs bitwise from both incumbent rows");
        require(run(GdnKnownWindowProjectionMode::rows2) == reference,
            "typed R2 input prefix is not bitwise repeatable");
    }
}

template<class Work> void adversarial_f32_gates(Fixture& fixture, Work&& work) {
    require(fixture.f32_gates, "F32 adversary requires actual F32 gate owners");
    const std::array<std::vector<unsigned char>,2> saved{fixture.weights[5].original, fixture.weights[6].original};
    constexpr std::array<std::uint32_t,16> patterns{0x00000000U,0x80000000U,0x00000001U,0x80000001U,
        0x007fffffU,0x807fffffU,0x00800000U,0x80800000U,0x7f800000U,0xff800000U,
        0x7fc00001U,0xffc12345U,0x7f800001U,0xff812345U,0x7f7fffffU,0xff7fffffU};
    constexpr std::array<unsigned,4> outputs{0,1,46,47}, features{0,255,256,2559};
    for (unsigned gate = 0; gate < 2U; ++gate) {
        auto& target = fixture.weights[5U + gate];
        require(target.original.size() == 491520U, "F32 gate fixture extent differs");
        for (unsigned o = 0; o < outputs.size(); ++o)
            for (unsigned f = 0; f < features.size(); ++f) {
                const auto bits = patterns[(o * 4U + f + gate * 5U) % patterns.size()];
                const auto offset = (static_cast<std::size_t>(outputs[o]) * 2560U + features[f]) * 4U;
                std::memcpy(target.original.data() + offset, &bits, sizeof(bits));
            }
        for (const auto output : outputs) {
            constexpr std::array<std::uint32_t,2> cancellation{0x3f800000U,0xbf800000U};
            std::memcpy(target.original.data() + (static_cast<std::size_t>(output) * 2560U + 64U) * 4U,
                cancellation.data(), sizeof(cancellation));
        }
    }
    const auto restore = [&] {
        for (unsigned gate = 0; gate < 2U; ++gate) {
            fixture.weights[5U + gate].original = saved[gate];
            fixture.owner.quiescence.run([&] { write(fixture.weights[5U + gate].device, saved[gate]); });
        }
    };
    try {
        for (const auto index : {5U,6U}) write(fixture.weights[index].device, fixture.weights[index].original);
        work(); fixture.owner.finish();
    } catch (...) {
        const auto error = std::current_exception();
        fixture.owner.quiescence.recover_or_fail_stop(); restore(); std::rethrow_exception(error);
    }
    restore();
}

void wrong_gdn_entry(Fixture& fixture) {
    fixture.prepare(0U); fixture.reset();
    const auto scratch = read(fixture.scratch), state = read(fixture.state);
    const auto call = [&](bool launch) {
        if (fixture.f32_gates) {
            if (launch) gdn_known_window_r2_launch(fixture.rows, fixture.gdn_config);
            else gdn_known_window_r2_preflight(fixture.rows, fixture.gdn_config);
        } else {
            if (launch) gdn_known_window_r2_f32_gates_launch(fixture.rows, fixture.gdn_config);
            else gdn_known_window_r2_f32_gates_preflight(fixture.rows, fixture.gdn_config);
        }
    };
    {
        const HipLaunchScope scope(fixture.scope());
        std::string message;
        try { call(false); } catch (const std::invalid_argument& error) { message = error.what(); }
        require(message == (fixture.f32_gates ? "GDN known-window R2 requires all five Q8_0 projections" :
            "GDN known-window R2 F32-gate variant requires Q8_0 qkv/z/out and F32 beta/alpha"),
            "wrong closed GDN entry did not preserve its exact format rejection");
    }
    check(hipStreamBeginCapture(fixture.owner.stream, hipStreamCaptureModeThreadLocal), "capture wrong typed GDN entry");
    std::exception_ptr error;
    try { const HipLaunchScope scope(fixture.scope()); invalid([&] { call(true); }); }
    catch (...) { error = std::current_exception(); }
    hipGraph_t graph{};
    check(hipStreamEndCapture(fixture.owner.stream, &graph), "end wrong typed GDN entry capture");
    require(graph != nullptr, "wrong typed GDN entry did not return graph");
    try { fixture.owner.graphs.push_back(graph); } catch (...) { (void)hipGraphDestroy(graph); throw; }
    std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "wrong typed GDN entry nodes");
    require(nodes == 0U, "wrong typed GDN entry enqueued capture nodes");
    if (error) std::rethrow_exception(error);
    fixture.owner.finish();
    require(read(fixture.scratch) == scratch && read(fixture.state) == state,
        "wrong typed GDN entry changed scratch or state");
}

void mixed_gdn_rejections(Fixture& fixture) {
    fixture.prepare(0U); fixture.reset();
    const auto before_scratch = read(fixture.scratch), before_state = read(fixture.state);
    const auto original_manifest = fixture.manifest;
    const auto original_runtime = fixture.runtime;
    const auto restore = [&] { fixture.manifest = original_manifest; fixture.runtime = original_runtime; };
    for (unsigned fault = 0; fault < 14U; ++fault) {
        restore();
        auto contexts = fixture.rows;
        auto other_runtime = fixture.runtime;
        auto other_resources = fixture.resources[1];
        if (fault == 0U || fault == 1U) {
            const auto index = fault == 0U ? 5U : 6U;
            fixture.manifest.tensors[index].ggml_type = 8U;
            fixture.manifest.tensors[index].byte_size = 2560ULL / 32U * 34U * 48U;
        }
        if (fault == 2U || fault == 3U) fixture.manifest.tensors[5].ggml_type = fault == 2U ? 1U : 30U;
        if (fault == 4U) {
            fixture.manifest.tensors[0].ggml_type = 0U;
            fixture.manifest.tensors[0].byte_size = 2560ULL * 10240U * 4U;
        }
        if (fault == 5U) fixture.runtime.tensors[5].bytes = 491520U - 4U;
        if (fault == 6U) fixture.runtime.tensors[5].address += 2U;
        if (fault == 7U) fixture.runtime.tensors[5].address = fixture.output[0].address;
        if (fault == 8U) fixture.runtime.tensors[5].address = fixture.temporary[1].address;
        if (fault == 9U) {
            other_resources.model_weights = &other_runtime;
            contexts[1].resources = &other_resources;
        }
        if (fault == 10U) fixture.runtime.tensors[5].allocation_generation = 0U;
        if (fault == 11U) fixture.manifest.tensors[5].byte_size -= 4U;
        if (fault == 12U) fixture.manifest.tensors[5].dimensions[1] = 47U;
        if (fault == 13U) fixture.runtime.tensors[6].address += 2U;
        try {
            { const HipLaunchScope scope(fixture.scope()); invalid([&] { fixture.preflight(contexts); }); }
            capture_rejection(fixture, contexts);
            fixture.owner.finish();
            require(read(fixture.scratch) == before_scratch && read(fixture.state) == before_state,
                "mixed typed metadata rejection changed scratch or persistent state");
        } catch (...) { restore(); throw; }
    }
    restore(); fixture.readonly();
}

void family_gdn_f32_gates() {
    Fixture fixture(true, false, true);
    Result finite_reference;
    for (unsigned flavor = 0; flavor < 5U; ++flavor) {
        fixture.prepare(flavor);
        const auto reference = fixture.run(false);
        same(reference, fixture.run(true)); same(reference, fixture.run(true));
        if (flavor < 3U) finite(reference);
        if (flavor == 0U) finite_reference = reference;
    }
    fixture.prepare(0U, true);
    const auto swapped = fixture.run(true);
    same(fixture.run(false), swapped); finite(swapped);
    require(swapped.convolution != finite_reference.convolution && swapped.matrix != finite_reference.matrix,
        "mixed F32 GDN causal-order witness is insensitive to swapped rows");
    rejections(fixture); mixed_gdn_rejections(fixture); wrong_gdn_entry(fixture);
    gdn_typed_input_stage(fixture);
    adversarial_f32_gates(fixture, [&] {
        gdn_typed_input_stage(fixture);
        for (const auto flavor : {0U,3U,4U}) {
            fixture.prepare(flavor);
            const auto reference = fixture.run(false);
            same(reference, fixture.run(true)); same(reference, fixture.run(true));
        }
    });
    for (const auto index : {0U,1U,8U}) {
        const auto& dims = fixture.manifest.tensors[index].dimensions;
        adversarial_q8_weight(fixture, index, static_cast<std::uint32_t>(dims[0]),
            static_cast<std::uint32_t>(dims[1]), [&] {
                fixture.prepare(4U); same(fixture.run(false), fixture.run(true));
                if (index != 8U) gdn_typed_input_stage(fixture);
            });
    }
}

// Native component evidence for the exact closed projection tiles used by the
// T runtime. This does not replace full-model T producer/continuation gates.
// One real QSA step (canonical layer-3 geometry, Q8 projections, budget 2048,
// 4-cell blocks), eight row views of one physical window arena.
struct QsaFixture {
    Owner owner;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch, state, kv, raw;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    SerialSessionStateLayout state_layout;
    SessionStateMemoryBinding state_memory;
    std::vector<unsigned char> host_state, initial_state, seed;
    ExecutionRuntimeResources resources;
    std::array<ExecutionDispatchContext, 8> rows;
    HipIndexedAttentionLauncherConfig config;
    std::string selected_id;
    ExecutionLauncher selected;

    QsaFixture(bool prefetch, bool wave8, std::uint64_t context_tokens) {
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [](const auto& value) { return value.op == SerialOp::indexed_attention; });
        require(found != execution.steps.end(), "canonical QSA operator missing");
        step = *found;
        const auto& step_io = io.steps[step.id];
        serial.alignment = 256U;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes});
            serial.total_bytes = offset + bytes;
        };
        allocation(step_io.reads[0], 2560U * 2U);
        allocation(step_io.writes[0], 2560U * 2U);
        allocation(step_io.temporaries[0], build_indexed_attention_workspace_layout(model).total_bytes);
        serial.total_bytes = (serial.total_bytes + 255U) / 256U * 256U;
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U, 8U);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e515341ULL, scratch.address, scratch.bytes, 43U};
        manifest.architecture = "qwen4_exp";
        const auto layer = std::to_string(*step.layer);
        add("blk." + layer + ".attn_q.weight", TensorRole::attention_projection, {2560, 12288}, q8(2560, 12288, 11));
        add("blk." + layer + ".attn_k.weight", TensorRole::attention_projection, {2560, 512}, q8(2560, 512, 12));
        add("blk." + layer + ".attn_v.weight", TensorRole::attention_projection, {2560, 512}, q8(2560, 512, 13));
        add("blk." + layer + ".attn_output.weight", TensorRole::attention_projection, {6144, 2560}, q8(6144, 2560, 14));
        add("blk." + layer + ".attn_q_norm.weight", TensorRole::normalization, {256}, f32(256U, 1.0F, 0.03F), 0U);
        add("blk." + layer + ".attn_k_norm.weight", TensorRole::normalization, {256}, f32(256U, 0.9F, 0.02F), 0U);
        add("blk." + layer + ".indexer.q_proj.weight", TensorRole::sparse_index, {2560, 512}, q8(2560, 512, 15));
        add("blk." + layer + ".indexer.k_proj.weight", TensorRole::sparse_index, {2560, 128}, q8(2560, 128, 16));
        add("blk." + layer + ".indexer.q_norm.weight", TensorRole::sparse_index, {128}, f32(128U, 1.1F, 0.02F), 0U);
        add("blk." + layer + ".indexer.k_norm.weight", TensorRole::sparse_index, {128}, f32(128U, 0.95F, 0.03F), 0U);
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e515342ULL;
        access.accesses.push_back({step.id, step.op, 0U, {0, 1, 2, 3}});
        access.accesses.push_back({step.id, step.op, 1U, {6, 7, 8, 9}});
        access.accesses.push_back({step.id, step.op, 2U, {4, 5}});
        SerialSessionStateLayoutRequest request; request.context_tokens = context_tokens;
        state_layout = build_serial_session_state_layout(model, request);
        state = owner.allocate(state_layout.device_total_bytes);
        host_state.resize(static_cast<std::size_t>(state_layout.host_total_bytes));
        state_memory = {state_layout.layout_identity, state.address, state.bytes, 47U,
            reinterpret_cast<std::uintptr_t>(host_state.data()), state_layout.host_total_bytes, 53U};
        const auto k = resolve_indexed_attention_layer_state(state_layout, state_memory, model, *step.layer, SessionStateComponent::attention_kv);
        const auto r = resolve_indexed_attention_layer_state(state_layout, state_memory, model, *step.layer, SessionStateComponent::sparse_index);
        kv = {k.address, k.bytes}; raw = {r.address, r.bytes};
        // Finite BF16 history in every QSA cell of the tested layer.
        initial_state.assign(static_cast<std::size_t>(state.bytes), 0U);
        for (const auto view : {kv, raw}) {
            std::vector<std::uint16_t> values(static_cast<std::size_t>(view.bytes / 2U));
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto mixed = (i * 2654435761ULL + (view.address == kv.address ? 17U : 91U)) >> 7U;
                values[i] = bf16(static_cast<float>(static_cast<int>(mixed % 2001U) - 1000) * 0.001F);
            }
            std::memcpy(initial_state.data() + (view.address - state.address), values.data(), values.size() * 2U);
        }
        resources = {&memory, &manifest, &runtime, &access};
        resources.session_state_layout = &state_layout; resources.session_state_memory = &state_memory;
        config.model = model;
        config.prefetch_q8_input = prefetch;
        config.wave8_q8_output = wave8;
        if (prefetch) {
            config.prefetch_eligible_layers = {*step.layer};
            config.prefetch_stats = std::make_shared<qsa_input_prefetch::RuntimeDispatchStats>();
        }
        ExecutionBindingRequest binding;
        binding.prefetch_q8_qsa_input = prefetch; binding.qsa_input_prefetch_layers = config.prefetch_eligible_layers;
        binding.wave8_q8_qsa_output = wave8;
        kernels::KernelRegistry registry;
        register_hip_indexed_attention_kernel_variant(model, binding, config, registry);
        require(registry.variants().size() == 1U, "QSA serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_indexed_attention_launcher(selected_id, &config);
        seed.assign(static_cast<std::size_t>(scratch.bytes), 0xa5U);
        for (unsigned row = 0; row < 8U; ++row) {
            auto& context = rows[row];
            context.execution_identity = memory.execution_identity;
            context.step_id = step.id; context.op = step.op; context.layer = step.layer;
            context.kernel_id = selected_id;
            context.reads = step_io.reads; context.writes = step_io.writes; context.temporaries = step_io.temporaries;
            context.arena = &window.row_view[row]; context.memory = &memory; context.resources = &resources;
            context.session_state_layout = &state_layout; context.session_state_memory = &state_memory;
            context.reads_state = step.reads_state; context.writes_state = step.writes_state;
            context.input_token_id = 101U + row;
            const auto input = resolve_dispatch_read(context, 0);
            std::vector<std::uint16_t> values(2560U);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + row * 13U) % 41U) - 20) * (0.011F + 0.003F * static_cast<float>(row)));
            std::memcpy(seed.data() + (input.address - scratch.address), values.data(), values.size() * 2U);
        }
    }
    void add(std::string name, TensorRole role, std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes,
             std::uint32_t type = 8U) {
        ManifestTensor tensor;
        tensor.name = std::move(name); tensor.role = role; tensor.layer = step.layer;
        tensor.ggml_type = type; tensor.dimensions = std::move(dimensions); tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes = tensor.absolute_offset + tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    std::span<const ExecutionDispatchContext> group(std::uint32_t count) const { return {rows.data(), count}; }
    void place(std::uint64_t position) { for (unsigned r = 0; r < 8U; ++r) rows[r].position = position + r; }
    void reset() { write(scratch, seed); write(state, initial_state); }
    void serial_rows(std::uint32_t count) { for (std::uint32_t r = 0; r < count; ++r) selected.function(rows[r], selected.user_data); }
    struct Payload { std::vector<unsigned char> arena, state; };
    Payload run(std::uint32_t count, std::uint64_t position, bool candidate) {
        reset(); place(position);
        {
            const HipLaunchScope active(scope());
            if (candidate) { indexed_attention_known_window_rows_preflight(group(count), config);
                indexed_attention_known_window_rows_launch(group(count), config); }
            else serial_rows(count);
        }
        owner.finish();
        return {read(scratch), read(state)};
    }
    // Cells [0, cells) of the tested layer's K/V and raw index state.
    std::vector<unsigned char> cells(const std::vector<unsigned char>& bytes, Span view, std::uint64_t row_bytes, std::uint64_t count) const {
        const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(view.address - state.address);
        return {begin, begin + static_cast<std::ptrdiff_t>(row_bytes * count)};
    }
};

void qsa_rows_family(bool prefetch, bool wave8) {
    QsaFixture f(prefetch, wave8, 2304U);
    unsigned compared = 0;
    // Positions cover p=0, partial-block tails, the full-visible/top-k boundary
    // (visible 2048..2053 straddles 512 complete blocks) and a pure top-k window.
    for (const std::uint32_t count : {2U, 4U, 8U})
        for (const std::uint64_t position : {0ULL, 1ULL, 3ULL, 6ULL, 2042ULL, 2045ULL, 2047ULL, 2049ULL, 2201ULL}) {
            const auto before = prefetch ? f.config.prefetch_stats->dispatched_projections.load() : 0U;
            const auto reference = f.run(count, position, false);
            const auto middle = prefetch ? f.config.prefetch_stats->dispatched_projections.load() : 0U;
            const auto candidate = f.run(count, position, true);
            if (prefetch) require(middle - before == 5U * count && f.config.prefetch_stats->dispatched_projections.load() - middle == 5U,
                "QSA rows prefetch submission count differs");
            require(candidate.arena == reference.arena, "QSA rows output/workspace bytes differ from ascending serial calls");
            require(candidate.state == reference.state, "QSA rows KV/raw-index state differs from ascending serial calls");
            std::vector<std::vector<unsigned char>> outputs;
            for (std::uint32_t r = 0; r < count; ++r) {
                const auto view = resolve_dispatch_write(f.rows[r], 0);
                const auto begin = candidate.arena.begin() + static_cast<std::ptrdiff_t>(view.address - f.scratch.address);
                outputs.emplace_back(begin, begin + 5120);
                for (std::size_t i = 0; i < 5120U; i += 2U)
                    require((outputs.back()[i + 1U] & 0x7fU) != 0x7fU || (outputs.back()[i] & 0x80U) == 0U, "QSA rows output is nonfinite");
                require(r == 0U || outputs[r] != outputs[r - 1U], "QSA fixture rows are insensitive to their inputs");
            }
            ++compared;
        }
    guards(f.scratch); guards(f.state);
    for (const auto& weight : f.weights) { require(read(weight.device) == weight.original, "QSA weights changed"); guards(weight.device); }

    // Tail snapshot/commit for every n_keep, the commit replayed from one captured graph.
    const auto journal = f.owner.allocate(qsa_index_journal_bytes(f.model));
    const auto plan = prepare_qsa_index_tail(f.model, f.state_layout, f.state_memory, {journal.address, journal.bytes});
    const auto words = f.owner.allocate(16U);
    auto* device_position = reinterpret_cast<std::uint64_t*>(words.address);
    auto* device_keep = reinterpret_cast<std::uint32_t*>(words.address + 8U);
    hipGraph_t graph{}; hipGraphExec_t commit{};
    check(hipStreamBeginCapture(f.owner.stream, hipStreamCaptureModeThreadLocal), "capture QSA tail commit");
    try { enqueue_qsa_index_tail_commit(plan, device_position, device_keep, f.owner.stream); }
    catch (...) { (void)hipStreamEndCapture(f.owner.stream, &graph); throw; }
    check(hipStreamEndCapture(f.owner.stream, &graph), "finish QSA tail commit capture");
    f.owner.graphs.push_back(graph);
    check(hipGraphInstantiate(&commit, graph, nullptr, nullptr, 0), "instantiate QSA tail commit");
    const auto state_contract = build_indexed_attention_state_layout(f.model);
    unsigned commits = 0;
    for (const std::uint32_t count : {2U, 4U, 8U})
        for (const std::uint64_t position : {5ULL, 2046ULL, 2203ULL}) {
            for (std::uint32_t keep = 0; keep <= count; ++keep) {
                const auto golden = keep ? f.run(keep, position, false).state : f.initial_state;
                const auto expected_cells = position + keep;
                f.reset(); f.place(position);
                check(hipMemcpy(device_position, &position, 8U, hipMemcpyHostToDevice), "QSA tail position");
                check(hipMemcpy(device_keep, &keep, 4U, hipMemcpyHostToDevice), "QSA tail n_keep");
                {
                    const HipLaunchScope active(f.scope());
                    enqueue_qsa_index_tail_snapshot(plan, device_position, f.owner.stream);
                    indexed_attention_known_window_rows_launch(f.group(count), f.config, {journal.address, journal.bytes});
                }
                // Clobber the tail block and every window cell before committing.
                const auto start = position - position % 4U;
                check(hipMemsetAsync(reinterpret_cast<void*>(f.raw.address + start * state_contract.raw_index_row_bytes), 0xff,
                    static_cast<std::size_t>((position + count - start) * state_contract.raw_index_row_bytes), f.owner.stream), "clobber QSA tail");
                check(hipGraphLaunch(commit, f.owner.stream), "replay QSA tail commit");
                f.owner.finish();
                const auto actual = read(f.state);
                require(f.cells(actual, f.raw, state_contract.raw_index_row_bytes, expected_cells) ==
                        f.cells(golden, f.raw, state_contract.raw_index_row_bytes, expected_cells),
                    "QSA tail commit raw index differs from serial accepted prefix");
                require(f.cells(actual, f.kv, state_contract.kv_row_bytes, expected_cells) ==
                        f.cells(golden, f.kv, state_contract.kv_row_bytes, expected_cells),
                    "QSA accepted KV prefix differs from serial");
                ++commits;
            }
        }
    check(hipGraphExecDestroy(commit), "destroy QSA tail commit");

    // Pure admission rejects before any enqueue.
    f.reset(); f.place(64U);
    const auto rejected = [&](auto&& mutate, const char* message) {
        auto bad = f.rows; mutate(bad);
        bool thrown = false;
        { const HipLaunchScope active(f.scope());
          try { indexed_attention_known_window_rows_launch(std::span<const ExecutionDispatchContext>(bad.data(), 4U), f.config); }
          catch (const std::invalid_argument&) { thrown = true; } }
        f.owner.finish();
        require(thrown && read(f.scratch) == f.seed && read(f.state) == f.initial_state, message);
    };
    rejected([](auto& rows) { rows[2].position += 1U; }, "QSA rows require consecutive positions");
    rejected([](auto& rows) { rows[1].arena = rows[0].arena; }, "QSA rows require disjoint row views");
    rejected([](auto& rows) { rows[3].layer = 7U; }, "QSA rows require one layer");
    const std::string foreign = f.selected_id + ".foreign";
    rejected([&](auto& rows) { rows[0].kernel_id = foreign; }, "QSA rows require the row's serial kernel configuration");
    {
        bool thrown = false;
        try { indexed_attention_known_window_rows_preflight(f.group(3U), f.config); } catch (const std::invalid_argument&) { thrown = true; }
        require(thrown, "QSA rows admit only two, four or eight rows");
    }
    std::cout << "QSA rows prefetch=" << prefetch << " wave8=" << wave8 << ": " << compared
              << " bit-exact windows, " << commits << " tail commits\n";

    if (!std::getenv("STRIX_QSA_ROWS_BENCH")) return;
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "bench event"); check(hipEventCreate(&end), "bench event");
    const auto time = [&](auto&& work) {
        std::vector<float> samples;
        for (unsigned i = 0; i < 28U; ++i) {
            check(hipEventRecord(begin, f.owner.stream), "bench begin");
            { const HipLaunchScope active(f.scope()); work(); }
            check(hipEventRecord(end, f.owner.stream), "bench end");
            check(hipEventSynchronize(end), "bench wait");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "bench elapsed");
            if (i >= 3U) samples.push_back(ms);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    for (const std::uint64_t position : {256ULL, 2201ULL})
        for (const std::uint32_t count : {2U, 4U, 8U}) {
            f.reset(); f.place(position);
            const auto serial_ms = time([&] { f.serial_rows(count); });
            const auto rows_ms = time([&] { indexed_attention_known_window_rows_launch(f.group(count), f.config); });
            const auto one_ms = time([&] { f.serial_rows(1U); });
            std::cout << "QSA bench p=" << position << " T=" << count << " serial(Tx1)=" << serial_ms << "ms rows=" << rows_ms
                      << "ms 1-row=" << one_ms << "ms speedup=" << serial_ms / rows_ms << "x rows/1-row=" << rows_ms / one_ms << "\n";
        }
    check(hipEventDestroy(begin), "bench event"); check(hipEventDestroy(end), "bench event");
}

// Whole QSA layer (synthetic weights, model shapes) at long context: the
// 1-row launcher at visible = context and T-row windows ending there.
void bench_qsa_long_context() {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "bench event"); check(hipEventCreate(&end), "bench event");
    for (const std::uint64_t context : {2049ULL, 4096ULL, 32768ULL, 131072ULL, 262144ULL}) {
        QsaFixture f(true, true, context);
        f.reset();
        const auto time = [&](auto&& work) {
            std::vector<float> samples;
            for (unsigned i = 0; i < 23U; ++i) {
                check(hipEventRecord(begin, f.owner.stream), "bench begin");
                { const HipLaunchScope active(f.scope()); work(); }
                check(hipEventRecord(end, f.owner.stream), "bench end");
                check(hipEventSynchronize(end), "bench wait");
                float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "bench elapsed");
                if (i >= 3U) samples.push_back(ms);
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2U];
        };
        f.place(context - 1U);
        const auto one_ms = time([&] { f.serial_rows(1U); });
        std::cout << "QSA layer context=" << context << " 1-row=" << one_ms << "ms";
        for (const std::uint32_t count : {2U, 4U, 8U}) {
            f.place(context - count);
            const auto rows_ms = time([&] { indexed_attention_known_window_rows_launch(f.group(count), f.config); });
            std::cout << " T" << count << "=" << rows_ms << "ms";
        }
        std::cout << '\n' << std::flush;
    }
    check(hipEventDestroy(begin), "bench event"); check(hipEventDestroy(end), "bench event");
}

void family_prefill_tiles(bool gdn,std::uint32_t count) {
    Fixture f(gdn,false,gdn,64U);
    const auto arena=build_known_prefill_arena(f.serial,256U,count);
    const auto storage=f.owner.allocate(arena.physical.total_bytes);
    const ExecutionMemoryBinding memory{0x5450524546494c31ULL,storage.address,storage.bytes,41U};
    auto resources=f.resources[0];resources.scratch=&memory;
    std::vector<ExecutionDispatchContext> rows(count);
    std::vector<ExecutionBufferAddress> input(count),output(count),temporary(count);
    std::vector<unsigned char> seed(static_cast<std::size_t>(storage.bytes),0xa5U);
    for(std::uint32_t r=0;r<count;++r) {
        rows[r]=f.rows[0];rows[r].execution_identity=memory.execution_identity;rows[r].arena=&arena.row_view[r];
        rows[r].memory=&memory;rows[r].resources=&resources;rows[r].position=r;rows[r].input_token_id=31U+r;
        input[r]=resolve_dispatch_read(rows[r],0);output[r]=resolve_dispatch_write(rows[r],0);temporary[r]=resolve_dispatch_temporary(rows[r],0);
        std::vector<std::uint16_t> values(2560U);
        for(std::size_t col=0;col<values.size();++col)
            values[col]=bf16(static_cast<float>(static_cast<int>((col*7U+r*13U)%37U)-18)*(.003F+static_cast<float>(r%5U)*.002F));
        std::memcpy(seed.data()+(input[r].address-storage.address),values.data(),values.size()*2U);
    }
    struct Payload {std::vector<unsigned char> arena,state;};
    const auto run=[&](bool candidate) {
        write(storage,seed);if(gdn) write(f.state,f.initial_state);
        {
            const HipLaunchScope active(f.scope());
            // Whole-T pure admission is complete before any row can mutate.
            for(std::uint32_t first=0;first<count;) {
                const auto width=candidate&&count-first>=8U?8U:1U;
                if(width==8U) {
                    const auto group=std::span<const ExecutionDispatchContext>(rows.data()+first,8U);
                    if(gdn) gdn_known_window_rows_preflight(group,f.gdn_config);
                    else shared_expert_known_window_rows_preflight(group,f.shared_config);
                } else {require(f.selected.preflight!=nullptr,"serial component lacks pure preflight");f.selected.preflight(rows[first],f.selected.user_data);}
                first+=width;
            }
            for(std::uint32_t first=0;first<count;) {
                const auto width=candidate&&count-first>=8U?8U:1U;
                if(width==8U) {
                    const auto group=std::span<const ExecutionDispatchContext>(rows.data()+first,8U);
                    if(gdn) gdn_known_window_rows_launch(group,f.gdn_config);
                    else shared_expert_known_window_rows_launch(group,f.shared_config);
                } else f.selected.function(rows[first],f.selected.user_data);
                first+=width;
            }
        }
        f.owner.finish();Payload result{read(storage),gdn?read(f.state):std::vector<unsigned char>{}};
        auto unchanged=seed;
        for(std::uint32_t r=0;r<count;++r) for(const auto view:{output[r],temporary[r]})
            std::copy_n(result.arena.begin()+static_cast<std::ptrdiff_t>(view.address-storage.address),
                static_cast<std::size_t>(view.bytes),unchanged.begin()+static_cast<std::ptrdiff_t>(view.address-storage.address));
        require(result.arena==unchanged,"T projection tile modified read-only input or internal arena canary");guards(storage);
        for(const auto& weight:f.weights) {require(read(weight.device)==weight.original,"T projection tile modified weights");guards(weight.device);}
        if(gdn) {
            auto expected=f.initial_state;
            for(const auto view:{f.matrix,f.convolution}) std::copy_n(result.state.begin()+static_cast<std::ptrdiff_t>(view.address-f.state.address),
                static_cast<std::size_t>(view.bytes),expected.begin()+static_cast<std::ptrdiff_t>(view.address-f.state.address));
            require(result.state==expected,"T causal tile modified another layer/state component");guards(f.state);
        }
        return result;
    };
    const auto reference=run(false);
    for(unsigned repeat=0;repeat<2U;++repeat) {
        const auto candidate=run(true);
        require(candidate.arena==reference.arena&&candidate.state==reference.state,
            "T tiled exact projections/full workspace/causal state differ from ascending original R1 calls");
    }
    // Late tile admission faults must be rejected before the first tile runs.
    write(storage,seed);if(gdn) write(f.state,f.initial_state);
    const auto state_before=gdn?read(f.state):std::vector<unsigned char>{};
    auto bad=rows;bad.back().resources=nullptr;bool rejected=false;
    {
        const HipLaunchScope active(f.scope());
        try {for(std::uint32_t first=0;first<count;) {
            const auto width=count-first>=8U?8U:1U;
            if(width==8U) {
                const auto group=std::span<const ExecutionDispatchContext>(bad.data()+first,8U);
                if(gdn) gdn_known_window_rows_preflight(group,f.gdn_config);else shared_expert_known_window_rows_preflight(group,f.shared_config);
            } else f.selected.preflight(bad[first],f.selected.user_data);
            first+=width;
        }}
        catch(const std::exception&) {rejected=true;}
        require(rejected,"late T row missing its resource owner reaches original admission before any tile enqueue");
    }
    f.owner.finish();require(read(storage)==seed&&(!gdn||read(f.state)==state_before),"late T admission failure changed arena or state");
}

void family(bool gdn, bool prefetch = false) {
    Fixture fixture(gdn, prefetch);
    Result finite_reference;
    for (unsigned flavor = 0; flavor < 4U; ++flavor) {
        fixture.prepare(flavor);
        const auto serial = fixture.run(false);
        const auto candidate = fixture.run(true);
        same(serial, candidate);
        same(candidate, fixture.run(true)); // reset, repeat, full-bit determinism
        if (flavor < 3U) finite(candidate);
        if (flavor == 0U) finite_reference = candidate;
    }
    if (gdn) {
        fixture.prepare(0U, true);
        const auto swapped = fixture.run(true);
        same(fixture.run(false), swapped);
        finite(swapped);
        require(swapped.convolution != finite_reference.convolution && swapped.matrix != finite_reference.matrix,
            "finite GDN fixture is insensitive to causal input order");
    }
    rejections(fixture);
    if (gdn) { gdn_projection_rows(fixture); gdn_typed_input_stage(fixture); wrong_gdn_entry(fixture); }
    else for (unsigned weight = 0; weight < 3U; ++weight) {
        const auto& dimensions = fixture.manifest.tensors[weight].dimensions;
        adversarial_q8_weight(fixture, weight, static_cast<std::uint32_t>(dimensions[0]),
            static_cast<std::uint32_t>(dimensions[1]), [&] {
                fixture.prepare(3U);
                const auto reference = fixture.run(false);
                same(reference, fixture.run(true));
            });
    }
}
// HC rows: one canonical attn/mlp/final mixer, T row views of the window arena,
// real HC shapes and formats (F32 norm, Q8 down/up, F32 injection).
struct HyperFixture {
    Owner owner;
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    Span scratch;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding runtime;
    ExecutionTensorAccessPlan access;
    std::vector<Weight> weights;
    ExecutionRuntimeResources resources;
    std::vector<ExecutionDispatchContext> rows;
    std::vector<unsigned char> seed;
    HipHyperMixLauncherConfig config;
    std::string selected_id;
    ExecutionLauncher selected;
    bool final_mix;

    HyperFixture(SerialOp op, bool splitk, bool lane, std::uint32_t count) : final_mix(op == SerialOp::final_hyper_mix) {
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [&](const auto& value) { return value.op == op && (final_mix || value.layer == 0U); });
        require(found != execution.steps.end(), "canonical HC mixer missing");
        step = *found;
        const auto& step_io = io.steps[step.id];
        require(step_io.reads.size() == 1U && step_io.writes.size() == (final_mix ? 1U : 2U) &&
            step_io.temporaries.size() == 1U, "HC fixture IO contract differs");
        serial.alignment = 256U;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes});
            serial.total_bytes = offset + bytes;
        };
        allocation(step_io.reads[0], 10240U * 2U);
        allocation(step_io.writes[0], 2560U * 2U);
        if (!final_mix) allocation(step_io.writes[1], 4U * 4U);
        allocation(step_io.temporaries[0], (10240U + 320U) * 4U);
        serial.total_bytes = (serial.total_bytes + 255U) / 256U * 256U;
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U, count);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x4b4e4f574e484301ULL, scratch.address, scratch.bytes, 19U};
        manifest.architecture = "qwen4_exp";
        const auto salt = static_cast<unsigned>(op) * 7U;
        add("hc_norm.weight", 0U, {10240}, f32(10240U, 1.0F, 0.04F));
        add("hc_down.weight", 8U, {10240, 320}, q8(10240, 320, 11U + salt));
        add("hc_up.weight", 8U, {320, 10240}, q8(320, 10240, 5U + salt));
        if (!final_mix) add("hc_inject.weight", 0U, {10240, 4}, f32(40960U, 0.0004F, 0.0002F));
        for (std::uint32_t slot = 0; slot < weights.size(); ++slot) access.accesses.push_back({step.id, step.op, slot, {slot}});
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1U;
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x4b4e4f574e484302ULL;
        resources = {&memory, &manifest, &runtime, &access};
        config.hidden_size = 2560U; config.hc_count = 4U; config.lowrank = 320U;
        config.rms_epsilon = model.rms_norm_epsilon;
        config.parallel = config.staged_projections = config.fused_injection = config.typed_projection = true;
        config.prefetch_rows = config.prefetch_q8_output = true;
        config.lane_chain_output = lane; config.splitk_projection = splitk;
        ExecutionBindingRequest binding; binding.lane_chain_hyper_output = lane;
        kernels::KernelRegistry registry;
        register_hip_hyper_mix_kernel_variant(model, binding, config, registry);
        require(registry.variants().size() == 1U, "HC serial selection is ambiguous");
        selected_id = registry.variants().front().id;
        selected = make_hip_hyper_mix_launcher(selected_id, &config);
        rows.resize(count);
        for (std::uint32_t r = 0; r < count; ++r) {
            auto& context = rows[r];
            context.execution_identity = memory.execution_identity;
            context.step_id = step.id; context.op = step.op; context.layer = step.layer;
            context.kernel_id = selected_id;
            context.reads = step_io.reads; context.writes = step_io.writes; context.temporaries = step_io.temporaries;
            context.arena = &window.row_view[r]; context.memory = &memory; context.resources = &resources;
            context.position = r; context.input_token_id = 17U + r;
        }
    }
    void add(std::string name, std::uint32_t type, std::vector<std::uint64_t> dimensions, std::vector<unsigned char> bytes) {
        ManifestTensor tensor;
        tensor.name = final_mix ? std::move(name) : "blk.0." + name; tensor.role = TensorRole::dense_projection;
        tensor.layer = step.layer; tensor.ggml_type = type; tensor.dimensions = std::move(dimensions);
        tensor.byte_size = bytes.size();
        tensor.absolute_offset = (manifest.total_tensor_bytes + 15U) / 16U * 16U;
        manifest.total_tensor_bytes = tensor.absolute_offset + tensor.byte_size;
        const auto device = owner.allocate(bytes.size());
        write(device, bytes);
        const auto index = static_cast<std::uint32_t>(weights.size());
        weights.push_back({device, std::move(bytes)});
        manifest.tensors.push_back(std::move(tensor));
        runtime.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 7U, 0U});
    }
    HipLaunchContext scope() const { HipLaunchContext result; result.stream = owner.stream; return result; }
    void prepare(unsigned flavor) {
        seed.assign(static_cast<std::size_t>(scratch.bytes), 0xa5);
        constexpr std::array<std::uint16_t, 10> edge{0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x4f00, 0xcf00, 0x3f80};
        for (std::uint32_t r = 0; r < rows.size(); ++r) {
            const auto source = flavor == 3U ? 0U : r;
            std::vector<std::uint16_t> values(10240U);
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (flavor == 1U) values[i] = edge[(i * 3U + source) % edge.size()];
                else if (flavor == 2U) values[i] = bf16((i % 2U ? -1.0F : 1.0F) * std::ldexp(1.0F, static_cast<int>((i + source * 5U) % 41U) - 20));
                else values[i] = bf16(static_cast<float>(static_cast<int>((i * 7U + source * 13U) % 37U) - 18) *
                    (0.003F + static_cast<float>(source % 5U) * 0.002F));
            }
            const auto input = resolve_dispatch_read(rows[r], 0);
            std::memcpy(seed.data() + (input.address - scratch.address), values.data(), values.size() * 2U);
        }
    }
    void serial_rows() const { for (const auto& row : rows) selected.function(row, selected.user_data); }
    void candidate_rows() const {
        hyper_mix_known_window_rows_preflight(rows, config);
        hyper_mix_known_window_rows_launch(rows, config);
    }
    std::vector<unsigned char> run(bool candidate) {
        write(scratch, seed);
        { const HipLaunchScope owned(scope()); if (candidate) candidate_rows(); else serial_rows(); }
        owner.finish();
        const auto result = read(scratch);
        guards(scratch);
        for (const auto& weight : weights) {
            require(read(weight.device) == weight.original, "HC rows changed weights");
            guards(weight.device);
        }
        auto unchanged = seed;
        for (const auto& row : rows) {
            std::vector<ExecutionBufferAddress> views{resolve_dispatch_write(row, 0), resolve_dispatch_temporary(row, 0)};
            if (!final_mix) views.push_back(resolve_dispatch_write(row, 1));
            for (const auto& view : views)
                std::copy_n(result.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address),
                    static_cast<std::size_t>(view.bytes), unchanged.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address));
        }
        require(result == unchanged, "HC rows modified a read-only input or interior canary");
        return result;
    }
    std::vector<unsigned char> output(const std::vector<unsigned char>& arena, std::uint32_t r) const {
        const auto view = resolve_dispatch_write(rows[r], 0);
        const auto begin = arena.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address);
        return {begin, begin + static_cast<std::ptrdiff_t>(view.bytes)};
    }
};

void family_hyper_rows(SerialOp op, bool splitk, bool lane, std::uint32_t count) {
    HyperFixture f(op, splitk, lane, count);
    for (unsigned flavor = 0; flavor < 4U; ++flavor) {
        f.prepare(flavor);
        const auto serial = f.run(false);
        const auto candidate = f.run(true);
        require(candidate == serial, "HC rows output/injection/workspace differ bitwise from ascending R1 calls");
        require(f.run(true) == candidate, "HC rows are not deterministic across repeats");
        if (flavor == 0U) {
            for (std::uint32_t r = 0; r < count; ++r) {
                const auto bytes = f.output(candidate, r);
                for (std::size_t offset = 0; offset < bytes.size(); offset += 2U) {
                    std::uint16_t bits{}; std::memcpy(&bits, bytes.data() + offset, 2U);
                    require((bits & 0x7f80U) != 0x7f80U, "finite HC fixture produced a nonfinite output");
                }
                if (r) require(bytes != f.output(candidate, 0U), "HC rows fixture is insensitive to the row input");
            }
        }
        if (flavor == 3U)
            for (std::uint32_t r = 1; r < count; ++r)
                require(f.output(candidate, r) == f.output(candidate, 0U), "identical HC rows produce different bytes");
    }
    // Pure admission: every rejection precedes any enqueue.
    f.prepare(0U); write(f.scratch, f.seed);
    const auto reject = [&](auto&& work, const char* message) {
        bool rejected = false;
        { const HipLaunchScope owned(f.scope());
          try { work(); } catch (const std::exception&) { rejected = true; } }
        f.owner.finish();
        require(rejected, message);
        require(read(f.scratch) == f.seed, "rejected HC rows admission changed the arena");
    };
    reject([&] { hyper_mix_known_window_rows_preflight(std::span<const ExecutionDispatchContext>(f.rows.data(), 1U), f.config); },
        "one-row HC rows entry was admitted");
    auto other = f.config; other.lane_chain_output = !lane; other.prefetch_q8_output = true;
    reject([&] { hyper_mix_known_window_rows_launch(f.rows, other); }, "HC rows accepted a different serial selection");
    auto bad = f.rows; bad.back().arena = bad.front().arena;
    reject([&] { hyper_mix_known_window_rows_launch(bad, f.config); }, "aliased HC row outputs were admitted");
    bad = f.rows; bad.back().kernel_id = std::nullopt;
    reject([&] { hyper_mix_known_window_rows_launch(bad, f.config); }, "HC row without serial id was admitted");
    bad = f.rows; bad.back().resources = nullptr;
    reject([&] { hyper_mix_known_window_rows_launch(bad, f.config); }, "late HC row without resources was admitted");
    auto fp64 = f.config; fp64.splitk_projection = true; fp64.splitk_projection_fp64 = true;
    reject([&] { hyper_mix_known_window_rows_preflight(f.rows, fp64); }, "FP64 HC oracle was admitted as a window entry");
}

// Median of warm hipEvent timings; MALL flushed before each sample when cold.
void bench_hyper_rows(bool splitk, bool lane, std::uint32_t count, bool cold) {
    HyperFixture f(SerialOp::attn_hyper_mix, splitk, lane, count);
    f.prepare(0U); write(f.scratch, f.seed);
    const auto flush = cold ? f.owner.allocate(96ULL << 20U) : Span{};
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "create HC bench event"); check(hipEventCreate(&stop), "create HC bench event");
    const auto measure = [&](bool candidate) {
        std::vector<float> samples;
        const HipLaunchScope owned(f.scope());
        for (unsigned i = 0; i < 28U; ++i) {
            if (cold) check(hipMemsetAsync(reinterpret_cast<void*>(flush.address), static_cast<int>(i), flush.bytes, f.owner.stream), "flush MALL");
            check(hipEventRecord(start, f.owner.stream), "record HC bench start");
            if (candidate) hyper_mix_known_window_rows_launch(f.rows, f.config); else f.serial_rows();
            check(hipEventRecord(stop, f.owner.stream), "record HC bench stop");
            check(hipEventSynchronize(stop), "complete HC bench sample");
            float ms = 0.0F; check(hipEventElapsedTime(&ms, start, stop), "HC bench elapsed");
            if (i >= 3U) samples.push_back(ms * 1000.0F);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    const auto serial = measure(false), rows = measure(true);
    f.owner.finish();
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    std::cout << "hc-rows-bench projection=" << (splitk ? "splitk-f32" : "batched-v8") << " output=" << (lane ? "lane-chain" : "q8-batched")
        << " T=" << count << (cold ? " cold" : " warm") << " serial_us=" << serial << " rows_us=" << rows
        << " speedup=" << serial / rows << '\n';
}
} // namespace

int main() try {
    if (std::getenv("STRIX_QSA_LONG_BENCH")) { bench_qsa_long_context(); return 0; }
    family(false, false);
    family(false, true);
    family(true);
    family_gdn_f32_gates();
    for(auto rows:{16U,17U}) {family_prefill_tiles(false,rows);family_prefill_tiles(true,rows);}
    std::uint32_t hc_cases = 0;
    for (auto op : {SerialOp::attn_hyper_mix, SerialOp::mlp_hyper_mix, SerialOp::final_hyper_mix})
        for (bool splitk : {false, true}) for (bool lane : {false, true}) for (std::uint32_t rows : {2U, 4U, 8U}) {
            family_hyper_rows(op, splitk, lane, rows); ++hc_cases;
        }
    std::cout << "known-window HC rows bit-exact cases passed: " << hc_cases << '\n';
    if (std::getenv("STRIX_HC_ROWS_BENCH"))
        for (bool cold : {false, true}) for (bool splitk : {false, true}) for (bool lane : {false, true})
            for (std::uint32_t rows : {2U, 4U, 8U}) bench_hyper_rows(splitk, lane, rows, cold);
    qsa_rows_family(true, true);
    qsa_rows_family(false, false);
    std::cout << "known-window shared/GDN/QSA component fixtures passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "known-window component fixture failed: " << error.what() << '\n';
    return 1;
}
