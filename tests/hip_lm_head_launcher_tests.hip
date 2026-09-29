#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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
                        row + 1U));
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

int main() try {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
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
        static_cast<std::uint32_t>(
            model.hidden_size);
    constexpr std::uint32_t vocab = 16;

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
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                hidden_address.address),
            hidden.data(),
            hidden.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy LM-head hidden input");

    const auto weights =
        make_weights(
            vocab,
            hidden_size);
    void* device_weights = nullptr;
    check(
        hipMalloc(
            &device_weights,
            weights.size() *
                sizeof(Q8Block)),
        "hipMalloc LM-head weights");
    check(
        hipMemcpy(
            device_weights,
            weights.data(),
            weights.size() *
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
    tensor.byte_size =
        weights.size() *
        sizeof(Q8Block);
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
        strix::HipActivationStorage::fp32;

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
        if (error >
            2e-5F * scale) {
            expect(
                false,
                "HIP LM-head Q8 logits match CPU reference");
            break;
        }
    }

    check(
        hipFree(device_weights),
        "hipFree LM-head weights");
    check(
        hipFree(arena_pointer),
        "hipFree LM-head arena");

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
