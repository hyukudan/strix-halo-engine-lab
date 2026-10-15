#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "virtual_q8_oracle.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <array>
#include <limits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

static_assert(sizeof(Q8Block) == 34);

const strix::ExecutionStep& lm_head_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::lm_head) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no LM-head step");
}

std::vector<Q8Block> make_weights(
    std::uint32_t rows,
    std::uint32_t width) {
    const auto blocks_per_row =
        width / 32U;
    std::vector<Q8Block> blocks(
        static_cast<std::size_t>(rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        for (std::uint32_t block_index = 0;
             block_index < blocks_per_row;
             ++block_index) {
            auto& block =
                blocks[
                    static_cast<std::size_t>(row) *
                        blocks_per_row +
                    block_index];
            block.scale =
                __float2half(
                    0.0005F *
                    static_cast<float>(
                        (row + 1U) * (block_index % 7U + 1U)));
            for (std::uint32_t offset = 0;
                 offset < 32U;
                 ++offset) {
                block.values[offset] =
                    static_cast<std::int8_t>(
                        static_cast<int>(
                            (row * 3U +
                             block_index * 5U +
                             offset) %
                            17U) -
                        8);
            }
        }
    }
    return blocks;
}

float q8_value(
    const std::vector<Q8Block>& weights,
    std::uint32_t row,
    std::uint32_t width,
    std::uint32_t feature) {
    const auto blocks_per_row =
        width / 32U;
    const auto& block =
        weights[
            static_cast<std::size_t>(row) *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

}  // namespace

void run_case(std::uint32_t requested_hidden, bool bf16, unsigned pattern = 0,
              std::uint32_t vocab = 19) {
    auto model =
        strix::ModelContract::qwen38_flash_next();
    if (requested_hidden > model.hidden_size) {
        model.hidden_size = requested_hidden;
        model.ple_embed_dim = requested_hidden;
    }
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        lm_head_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;
    request.activation_element_bytes = 4;
    request.logit_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc LM-head arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x4c4d484541440001ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto hidden_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto logits_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    const auto hidden_size =
        requested_hidden;
    // Default 19 includes a partial final block; explicit 1/4/7 are below 8.

    std::vector<float> hidden(
        hidden_size);
    for (std::size_t index = 0;
         index < hidden.size();
         ++index) {
        hidden[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 29U) -
                14) /
            31.0F;
    }
    if (pattern == 1U) {
        std::fill(hidden.begin(), hidden.end(), 0.0F);
        hidden[0] = 1e20F; hidden[4] = -1e20F; hidden[128] = 3.14F;
    }
    if (pattern == 2U) std::fill(hidden.begin(), hidden.end(), -1.0F);
    if (pattern == 4U) std::fill(hidden.begin(), hidden.end(), std::ldexp(1.0F, -130));
    std::vector<std::uint16_t> packed_hidden(hidden.size());
    if (bf16) for (std::size_t index = 0; index < hidden.size(); ++index) {
        auto bits = std::bit_cast<std::uint32_t>(hidden[index]);
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        packed_hidden[index] = static_cast<std::uint16_t>(bits >> 16U);
        hidden[index] = std::bit_cast<float>(static_cast<std::uint32_t>(packed_hidden[index]) << 16U);
    }
    if (bf16) {
        std::vector<std::uint16_t> guard(static_cast<std::size_t>(model.hidden_size), 0x7fc0U);
        check(hipMemcpy(reinterpret_cast<void*>(hidden_address.address), guard.data(),
            guard.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice), "poison BF16 input tail");
    } else {
        std::vector<std::uint32_t> guard(static_cast<std::size_t>(model.hidden_size), 0x7fc00000U);
        check(hipMemcpy(reinterpret_cast<void*>(hidden_address.address), guard.data(),
            guard.size() * sizeof(std::uint32_t), hipMemcpyHostToDevice), "poison FP32 input tail");
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                hidden_address.address),
            bf16 ? static_cast<const void*>(packed_hidden.data()) : hidden.data(),
            hidden.size() *
                (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyHostToDevice),
        "copy LM-head hidden input");

    auto weights = make_weights(vocab, hidden_size);
    if (pattern == 1U) for (auto& block : weights) {
        block.scale = __float2half(1.0F);
        std::fill(std::begin(block.values), std::end(block.values), std::int8_t{1});
    }
    if (pattern == 2U) for (auto& block : weights) {
        block.scale = __float2half(1.0F);
        std::fill(std::begin(block.values), std::end(block.values), std::int8_t{0});
    }
    if (pattern == 3U) {
        constexpr std::array codes{-128, -127, -1, 0, 1, 127};
        const std::array scales{std::ldexp(1.0F, -24), -1.0F, -0.0F, 65504.0F};
        for (std::size_t block_index = 0; block_index < weights.size(); ++block_index) {
            auto& block = weights[block_index];
            block.scale = __float2half(scales[block_index % scales.size()]);
            for (std::size_t offset = 0; offset < 32; ++offset)
                block.values[offset] = static_cast<std::int8_t>(codes[(block_index + offset) % codes.size()]);
        }
    }
    const auto exact_weight_bytes = weights.size() * sizeof(Q8Block);
    auto guarded_weights = weights;
    Q8Block guard_block{};
    guard_block.scale = __float2half(std::numeric_limits<float>::quiet_NaN());
    guarded_weights.push_back(guard_block);
    void* device_weights = nullptr;
    check(
        hipMalloc(
            &device_weights,
            guarded_weights.size() *
                sizeof(Q8Block)),
        "hipMalloc LM-head weights");
    check(
        hipMemcpy(
            device_weights,
            guarded_weights.data(),
            guarded_weights.size() *
                sizeof(Q8Block),
            hipMemcpyHostToDevice),
        "copy LM-head weights");

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    strix::ManifestTensor tensor;
    tensor.name = "output.weight";
    tensor.dimensions = {
        hidden_size,
        vocab,
    };
    tensor.ggml_type = 8;
    tensor.role =
        strix::TensorRole::output_head;
    tensor.byte_size = exact_weight_bytes;
    manifest.total_tensor_bytes =
        tensor.byte_size;
    manifest.tensors.push_back(
        tensor);
    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    runtime.tensors.push_back(
        {
            0,
            strix::RuntimeTensorStorage::resident,
            reinterpret_cast<std::uintptr_t>(
                device_weights),
            tensor.byte_size,
            1,
            0,
        });

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x4c4d484541440002ULL;
    access.accesses.push_back(
        {
            step.id,
            strix::SerialOp::lm_head,
            0,
            {0},
        });

    strix::ExecutionRuntimeResources resources;
    resources.scratch =
        &memory;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;

    strix::ExecutionDispatchContext context;
    context.step_id = step.id;
    context.op =
        strix::SerialOp::lm_head;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.resources =
        &resources;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipLmHeadLauncherConfig config;
    config.hidden_size =
        hidden_size;
    config.vocab_size =
        vocab;
    config.input =
        bf16 ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;

    auto launcher =
        strix::make_hip_lm_head_launcher(
            "test.lm_head",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "LM-head synchronize");

    std::vector<float> actual(vocab);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<const void*>(
                logits_address.address),
            actual.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy LM-head logits");

    for (std::uint32_t row = 0;
         row < vocab;
         ++row) {
        double expected = 0.0;
        for (std::uint32_t feature = 0;
             feature < hidden_size;
             ++feature) {
            expected +=
                static_cast<double>(
                    hidden[feature]) *
                static_cast<double>(
                    q8_value(
                        weights,
                        row,
                        hidden_size,
                        feature));
        }
        const auto expected_float =
            static_cast<float>(expected);
        const auto error =
            std::abs(
                actual[row] -
                expected_float);
        const auto scale =
            std::max(
                1.0F,
                std::abs(
                    expected_float));
        if (pattern == 0U && error >
            2e-5F * scale) {
            expect(
                false,
                "HIP LM-head Q8 logits match CPU reference");
            break;
        }
    }

    if (pattern == 1U) for (const auto value : actual)
        expect(std::bit_cast<std::uint32_t>(value) == 0U,
            "device legacy tree cancellation gives positive zero; premature local tree would differ");
    std::vector<float> decoded(hidden_size);
    bool denormal_oracle_diff = false;
    for (std::uint32_t row = 0; row < vocab; ++row) {
        for (std::uint32_t feature = 0; feature < hidden_size; ++feature)
            decoded[feature] = q8_value(weights, row, hidden_size, feature);
        const auto oracle = virtual_q8_test::legacy_oracle(hidden, decoded);
        const bool exact = std::bit_cast<std::uint32_t>(oracle) == std::bit_cast<std::uint32_t>(actual[row]);
        if (pattern == 4U) denormal_oracle_diff = denormal_oracle_diff || !exact;
        else expect(exact, "legacy device logits match fused CPU256-leaf oracle bitwise");
    }
    if (pattern == 2U) for (const auto value : actual)
        expect(std::bit_cast<std::uint32_t>(value) == 0U, "negative-zero products accumulate to positive zero");
    if (pattern == 4U) std::cout << "DENORMAL_ORACLE input=" << (bf16 ? "bf16" : "fp32")
        << " differs_from_cpu=" << denormal_oracle_diff << " candidate must still match legacy exactly\n";
    auto virtual_config = config;
    virtual_config.virtual_q8_wave = true;
    auto candidate = strix::make_hip_lm_head_launcher("test.lm_head.virtual", &virtual_config);
    expect(candidate.implementation_identity != launcher.implementation_identity,
        "virtual Q8 head has a distinct implementation identity");
    std::vector<float> observed(vocab);
    const auto verify = [&] {
        check(hipMemcpy(observed.data(), reinterpret_cast<void*>(logits_address.address),
            observed.size() * sizeof(float), hipMemcpyDeviceToHost), "read virtual Q8 logits");
        expect(std::memcmp(actual.data(), observed.data(), actual.size() * sizeof(float)) == 0,
            "virtual256 Q8 head preserves every legacy logit bit");
    };
    check(hipMemset(reinterpret_cast<void*>(logits_address.address), 0xff, vocab * sizeof(float)), "poison candidate destination");
    candidate.function(context, candidate.user_data);
    check(hipDeviceSynchronize(), "drain virtual Q8 head");
    verify();
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create virtual head stream");
    strix::HipLaunchContext scope;
    scope.stream = stream;
    {
        strix::HipLaunchScope owned(scope);
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin virtual head operator capture");
        candidate.function(context, candidate.user_data);
        check(hipStreamEndCapture(stream, &graph), "finish virtual head operator capture");
    }
    check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate virtual head capture");
    const auto kernel_params = [&](hipGraph_t captured) {
        std::size_t count = 0;
        check(hipGraphGetNodes(captured, nullptr, &count), "count LM-head graph nodes");
        if (count != 1) throw std::runtime_error("LM-head capture must contain exactly one node");
        hipGraphNode_t node = nullptr;
        check(hipGraphGetNodes(captured, &node, &count), "read LM-head graph node");
        hipGraphNodeType type{};
        check(hipGraphNodeGetType(node, &type), "read LM-head node type");
        if (type != hipGraphNodeTypeKernel) throw std::runtime_error("LM-head graph node must be a kernel");
        hipKernelNodeParams params{};
        check(hipGraphKernelNodeGetParams(node, &params), "read LM-head kernel parameters");
        return params;
    };
    const auto virtual_params = kernel_params(graph);
    expect(virtual_params.blockDim.x == 32U && virtual_params.blockDim.y == 1U &&
        virtual_params.blockDim.z == 1U && virtual_params.gridDim.x == vocab &&
        virtual_params.gridDim.y == 1U && virtual_params.gridDim.z == 1U &&
        virtual_params.sharedMemBytes == 0U, "capture proves candidate launches32 threads and zero dynamic LDS");
    hipGraph_t legacy_graph = nullptr;
    {
        strix::HipLaunchScope owned(scope);
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin legacy head capture");
        launcher.function(context, launcher.user_data);
        check(hipStreamEndCapture(stream, &legacy_graph), "finish legacy head capture");
    }
    const auto legacy_params = kernel_params(legacy_graph);
    expect(legacy_params.blockDim.x == 256U && legacy_params.func != virtual_params.func,
        "legacy capture has256 threads and a distinct native function");
    check(hipGraphDestroy(legacy_graph), "destroy legacy head capture");
    for (unsigned replay = 0; replay < 3; ++replay) {
        check(hipMemsetAsync(reinterpret_cast<void*>(logits_address.address), 0xff, vocab * sizeof(float), stream), "stream-ordered capture poison");
        check(hipGraphLaunch(executable, stream), "replay virtual head operator");
        check(hipStreamSynchronize(stream), "drain virtual head replay");
        verify();
    }
    check(hipGraphExecDestroy(executable), "destroy virtual head executable");
    check(hipGraphDestroy(graph), "destroy virtual head graph");
    if (bf16) for (const bool wave8 : {false, true}) {
        auto organized = virtual_config;
        organized.prefetch_q8_wave = true;
        organized.wave8_q8_wave = wave8;
        const std::string id = "hip.lm_head.bf16.h" + std::to_string(hidden_size) +
            ".v" + std::to_string(vocab) + ".r1.c1" +
            (wave8 ? ".prefetch8-wave8-v1" : ".prefetch8-load-v1") + ".virtual256.q8";
        const auto grouped = strix::make_hip_lm_head_launcher(id, &organized);
        expect(grouped.implementation_identity != candidate.implementation_identity,
            "prefetch organization has a distinct implementation identity");
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            constexpr std::uint32_t canary = 0x55aabbccU;
            check(hipMemcpy(reinterpret_cast<void*>(logits_address.address + vocab * sizeof(float)),
                &canary, sizeof(canary), hipMemcpyHostToDevice), "set prefetch tail canary");
            check(hipMemset(reinterpret_cast<void*>(logits_address.address), 0xff, vocab * sizeof(float)), "poison prefetch head output");
            { strix::HipLaunchScope owned(scope); grouped.function(context, grouped.user_data); }
            check(hipStreamSynchronize(stream), "drain prefetch head");
            verify(); // Full-output bitwise comparison with unchanged original.
            std::uint32_t observed_canary = 0;
            check(hipMemcpy(&observed_canary, reinterpret_cast<void*>(logits_address.address + vocab * sizeof(float)),
                sizeof(observed_canary), hipMemcpyDeviceToHost), "read prefetch tail canary");
            expect(observed_canary == canary, "partial final wave8 block preserves destination tail");
        }
        hipGraph_t rejected_graph = nullptr;
        bool capture_rejected = false;
        {
            strix::HipLaunchScope owned(scope);
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin rejected prefetch head capture");
            try { grouped.function(context, grouped.user_data); }
            catch (const std::invalid_argument&) { capture_rejected = true; }
            check(hipStreamEndCapture(stream, &rejected_graph), "end rejected prefetch head capture");
        }
        std::size_t rejected_nodes = 0;
        check(hipGraphGetNodes(rejected_graph, nullptr, &rejected_nodes), "count rejected head nodes");
        expect(capture_rejected && rejected_nodes == 0, "prefetch head rejects real stream capture before any enqueue");
        check(hipGraphDestroy(rejected_graph), "destroy rejected head graph");
        const auto saved_type = manifest.tensors.front().ggml_type;
        manifest.tensors.front().ggml_type = 20U; // Fits Q8 extent; reaches the kind gate.
        bool rejected_format = false;
        std::string format_error;
        check(hipMemset(reinterpret_cast<void*>(logits_address.address), 0xff, vocab * sizeof(float)), "poison non-Q8 prefetch output");
        try { grouped.function(context, grouped.user_data); }
        catch (const std::invalid_argument& error) { rejected_format = true; format_error = error.what(); }
        manifest.tensors.front().ggml_type = saved_type;
        check(hipDeviceSynchronize(), "drain rejected prefetch head");
        std::vector<std::uint32_t> untouched(vocab);
        check(hipMemcpy(untouched.data(), reinterpret_cast<void*>(logits_address.address), vocab * sizeof(float), hipMemcpyDeviceToHost), "read rejected prefetch head output");
        expect(rejected_format && format_error == "prefetch LM head supports BF16/Q8_0 only" &&
            std::all_of(untouched.begin(), untouched.end(), [](auto word) { return word == 0xffffffffU; }),
            "non-Q8 prefetch call rejects and preserves all output poison");
    }
    check(hipStreamDestroy(stream), "destroy virtual head stream");
    auto bad_config = virtual_config;
    bad_config.hidden_size = 0;
    bool rejected = false;
    try { (void)strix::make_hip_lm_head_launcher("test.invalid.virtual", &bad_config); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "virtual head rejects zero geometry");
    const auto poison_output = [&] {
        check(hipMemset(reinterpret_cast<void*>(logits_address.address), 0xff, vocab * sizeof(float)), "poison rejected destination");
        check(hipDeviceSynchronize(), "drain rejected destination poison");
    };
    const auto verify_poison = [&] {
        check(hipDeviceSynchronize(), "drain any erroneously launched rejected work");
        std::vector<std::uint32_t> poisoned(vocab);
        check(hipMemcpy(poisoned.data(), reinterpret_cast<void*>(logits_address.address),
            poisoned.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "read rejected destination");
        expect(std::all_of(poisoned.begin(), poisoned.end(), [](auto word) { return word == 0xffffffffU; }),
            "rejected call preserves every poisoned destination word");
    };
    const auto saved_type = manifest.tensors.front().ggml_type;
    for (const auto kind : {99U, 20U, 23U}) {
        if (kind == 23U && hidden_size % 256U != 0U) continue;
        manifest.tensors.front().ggml_type = kind;
        poison_output();
        rejected = false;
        bool reached_q8_gate = false;
        try { candidate.function(context, candidate.user_data); }
        catch (const std::invalid_argument& error) {
            rejected = true;
            reached_q8_gate = std::string(error.what()).find("supports Q8_0 weights only") != std::string::npos;
        }
        expect(rejected && (kind == 99U || reached_q8_gate), "supported non-Q8 format reaches virtual Q8 gate");
        verify_poison();
    }
    manifest.tensors.front().ggml_type = saved_type;
    auto missing_context = context;
    missing_context.writes = {};
    poison_output();
    rejected = false;
    try { candidate.function(missing_context, candidate.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "virtual Q8 head rejects missing output before launch");
    verify_poison();

    check(
        hipFree(device_weights),
        "hipFree LM-head weights");
    check(
        hipFree(arena_pointer),
        "hipFree LM-head arena");

 }

void test_reordering_negative_control() {
    std::array<float, 256> leaves{};
    leaves[0] = 1e20F; leaves[4] = -1e20F; leaves[128] = 3.14F;
    auto original = leaves;
    for (std::size_t stride = 128; stride; stride >>= 1U)
        for (std::size_t lane = 0; lane < stride; ++lane)
            original[lane] += original[lane + stride];
    std::array<float, 32> prematurely_local{};
    for (std::size_t lane = 0; lane < 32; ++lane) {
        std::array<float, 8> local{};
        for (std::size_t j = 0; j < 8; ++j) local[j] = leaves[8 * lane + j];
        for (std::size_t stride = 4; stride; stride >>= 1U)
            for (std::size_t j = 0; j < stride; ++j) local[j] += local[j + stride];
        prematurely_local[lane] = local[0];
    }
    for (std::size_t stride = 16; stride; stride >>= 1U)
        for (std::size_t lane = 0; lane < stride; ++lane)
            prematurely_local[lane] += prematurely_local[lane + stride];
    expect(std::bit_cast<std::uint32_t>(original[0]) != std::bit_cast<std::uint32_t>(prematurely_local[0]),
        "independent negative control detects premature local reduction");
}

int main() try {
    test_reordering_negative_control();
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::ExecutionBindingRequest request;
    strix::HipLmHeadLauncherConfig default_config;
    default_config.hidden_size = static_cast<std::uint32_t>(model.hidden_size);
    default_config.vocab_size = static_cast<std::uint32_t>(model.vocab_size);
    auto virtual_config = default_config;
    virtual_config.virtual_q8_wave = true;
    strix::kernels::KernelRegistry baseline_registry, virtual_registry;
    strix::register_hip_lm_head_kernel_variant(model, request, default_config, baseline_registry);
    strix::register_hip_lm_head_kernel_variant(model, request, virtual_config, virtual_registry);
    const auto baseline = baseline_registry.variants().front();
    const auto candidate = virtual_registry.variants().front();
    expect(candidate.id != baseline.id && candidate.compiler_identity != baseline.compiler_identity,
        "virtual Q8 registry distinguishes kernel and compiler identity");
    expect(baseline.resources.workgroup_threads == 256U && baseline.resources.lds_bytes == 1024U &&
           candidate.resources.workgroup_threads == 32U && candidate.resources.lds_bytes == 0U &&
           candidate.resources.scratch_bytes == 0U,
        "virtual Q8 registry preserves baseline resources and admits no new workspace");
    for (const auto width : {32U, 96U, 256U, 288U, 640U, 2560U}) {
        run_case(width, false);
        run_case(width, true);
    }
    run_case(256U, false, 1U);
    run_case(256U, true, 1U);
    for (const auto pattern : {2U, 3U, 4U}) {
        run_case(640U, false, pattern);
        run_case(640U, true, pattern);
    }
    for (const auto vocab : {1U, 4U, 7U}) {
        run_case(32U, true, 3U, vocab);
        run_case(10240U, true, 0U, vocab);
    }
    if (failures == 0) {
        std::cout
            << "HIP LM-head launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP LM-head launcher tests: "
        << error.what() << '\n';
    return 2;
}
