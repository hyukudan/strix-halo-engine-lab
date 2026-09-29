#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
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

void expect(
    bool condition,
    const char* message) {
    if (!condition) {
        std::cerr
            << "FAIL: "
            << message
            << '\n';
        ++failures;
    }
}

void check(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

static_assert(
    sizeof(IQ4NLBlock) == 18);

constexpr std::int8_t codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

const strix::ExecutionStep&
first_expert_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step :
         plan.steps) {
        if (step.op ==
            strix::SerialOp::moe_experts) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no routed-expert step");
}

std::vector<IQ4NLBlock>
make_weights(
    std::uint32_t experts,
    std::uint32_t output_width,
    std::uint32_t input_width,
    std::uint32_t salt) {
    const auto blocks_per_row =
        input_width / 32U;
    const auto rows =
        static_cast<std::uint64_t>(
            experts) *
        output_width;

    std::vector<IQ4NLBlock> result(
        static_cast<std::size_t>(
            rows) *
        blocks_per_row);

    for (std::size_t block_index = 0;
         block_index < result.size();
         ++block_index) {
        auto& block =
            result[block_index];
        block.d =
            __float2half(
                0.00002F *
                static_cast<float>(
                    (block_index +
                     salt) %
                        7U +
                    1U));
        for (std::uint32_t byte = 0;
             byte < 16U;
             ++byte) {
            const auto low =
                static_cast<std::uint8_t>(
                    (block_index +
                     byte * 3U +
                     salt) %
                    16U);
            const auto high =
                static_cast<std::uint8_t>(
                    (block_index * 5U +
                     byte +
                     salt) %
                    16U);
            block.qs[byte] =
                static_cast<std::uint8_t>(
                    low |
                    static_cast<std::uint8_t>(
                        high << 4U));
        }
    }
    return result;
}

float weight_value(
    const std::vector<IQ4NLBlock>& weights,
    std::uint32_t expert,
    std::uint32_t output,
    std::uint32_t output_width,
    std::uint32_t input_width,
    std::uint32_t feature) {
    const auto blocks_per_row =
        input_width / 32U;
    const auto row =
        static_cast<std::uint64_t>(
            expert) *
            output_width +
        output;
    const auto& block =
        weights[
            row *
                blocks_per_row +
            feature / 32U];

    const auto within =
        feature % 32U;
    const auto packed =
        block.qs[
            within % 16U];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    return
        __half2float(block.d) *
        static_cast<float>(
            codebook[code]);
}

float silu(float value) {
    return value /
           (1.0F +
            std::exp(-value));
}

}  // namespace

int main() try {
    constexpr std::uint32_t hidden = 64;
    constexpr std::uint32_t intermediate = 32;
    constexpr std::uint32_t experts = 4;
    constexpr std::uint32_t top_k = 2;

    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_expert_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.activation_element_bytes = 4;
    buffer_request.mlp_intermediate_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            buffer_request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc routed-expert arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x4d4f455445535401ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto hidden_input =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto route_indices =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[1]);
    const auto route_weights =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[2]);
    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    std::vector<float> hidden_values(
        hidden);
    for (std::size_t index = 0;
         index < hidden_values.size();
         ++index) {
        hidden_values[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 19U) -
                9) /
            23.0F;
    }

    const std::vector<std::uint32_t>
        selected{1U, 3U};
    const std::vector<float>
        selected_weights{0.7F, 0.3F};

    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                hidden_input.address),
            hidden_values.data(),
            hidden_values.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy routed-expert hidden");
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                route_indices.address),
            selected.data(),
            selected.size() *
                sizeof(std::uint32_t),
            hipMemcpyHostToDevice),
        "copy routed-expert indices");
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                route_weights.address),
            selected_weights.data(),
            selected_weights.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy routed-expert weights");

    const auto gate =
        make_weights(
            experts,
            intermediate,
            hidden,
            3U);
    const auto up =
        make_weights(
            experts,
            intermediate,
            hidden,
            7U);
    const auto down =
        make_weights(
            experts,
            hidden,
            intermediate,
            11U);

    std::vector<float> expected(
        hidden,
        0.0F);
    for (std::uint32_t slot = 0;
         slot < top_k;
         ++slot) {
        const auto expert =
            selected[slot];
        std::vector<float> temp(
            intermediate);

        for (std::uint32_t inner = 0;
             inner < intermediate;
             ++inner) {
            double gate_sum = 0.0;
            double up_sum = 0.0;
            for (std::uint32_t feature = 0;
                 feature < hidden;
                 ++feature) {
                gate_sum +=
                    static_cast<double>(
                        hidden_values[
                            feature]) *
                    weight_value(
                        gate,
                        expert,
                        inner,
                        intermediate,
                        hidden,
                        feature);
                up_sum +=
                    static_cast<double>(
                        hidden_values[
                            feature]) *
                    weight_value(
                        up,
                        expert,
                        inner,
                        intermediate,
                        hidden,
                        feature);
            }
            temp[inner] =
                silu(
                    static_cast<float>(
                        gate_sum)) *
                static_cast<float>(
                    up_sum);
        }

        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            double value = 0.0;
            for (std::uint32_t inner = 0;
                 inner < intermediate;
                 ++inner) {
                value +=
                    static_cast<double>(
                        temp[inner]) *
                    weight_value(
                        down,
                        expert,
                        feature,
                        hidden,
                        intermediate,
                        inner);
            }
            expected[feature] +=
                static_cast<float>(
                    value) *
                selected_weights[slot];
        }
    }

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    auto add_tensor =
        [&](std::string name,
            std::vector<std::uint64_t> dims,
            std::uint64_t bytes) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.ggml_type = 20;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size =
                bytes;
            tensor.role =
                strix::TensorRole::
                    moe_expert;
            tensor.layer =
                step.layer;
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_tensor(
        "blk.0.ffn_gate_exps.weight",
        {hidden, intermediate, experts},
        gate.size() *
            sizeof(IQ4NLBlock));
    add_tensor(
        "blk.0.ffn_up_exps.weight",
        {hidden, intermediate, experts},
        up.size() *
            sizeof(IQ4NLBlock));
    add_tensor(
        "blk.0.ffn_down_exps.weight",
        {intermediate, hidden, experts},
        down.size() *
            sizeof(IQ4NLBlock));

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        3,
        nullptr);
    const auto upload =
        [&](std::size_t index,
            const void* source,
            std::size_t bytes) {
            check(
                hipMalloc(
                    &device_weights[index],
                    bytes),
                "hipMalloc routed-expert weights");
            check(
                hipMemcpy(
                    device_weights[index],
                    source,
                    bytes,
                    hipMemcpyHostToDevice),
                "copy routed-expert weights");
        };
    upload(
        0,
        gate.data(),
        gate.size() *
            sizeof(IQ4NLBlock));
    upload(
        1,
        up.data(),
        up.size() *
            sizeof(IQ4NLBlock));
    upload(
        2,
        down.data(),
        down.size() *
            sizeof(IQ4NLBlock));

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    for (std::size_t index = 0;
         index < 3;
         ++index) {
        runtime.tensors.push_back(
            {
                static_cast<std::uint32_t>(
                    index),
                strix::RuntimeTensorStorage::
                    resident,
                reinterpret_cast<std::uintptr_t>(
                    device_weights[index]),
                manifest.tensors[index].
                    byte_size,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x4d4f455445535402ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0, 1, 2},
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
    context.step_id =
        step.id;
    context.op =
        step.op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries =
        step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.resources =
        &resources;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipMoeExpertsLauncherConfig config;
    config.hidden_size =
        hidden;
    config.intermediate_size =
        intermediate;
    config.expert_count =
        experts;
    config.top_k =
        top_k;
    config.activation =
        strix::HipActivationStorage::fp32;

    auto launcher =
        strix::make_hip_moe_experts_launcher(
            "test.moe_experts",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "routed-expert synchronize");

    std::vector<float> actual(
        hidden);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy routed-expert output");

    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    actual[index] -
                    expected[index]));
    }
    expect(
        maximum_error <= 3e-4F,
        "IQ4_NL routed experts match CPU reference");

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree routed-expert weight");
    }
    check(
        hipFree(arena_pointer),
        "hipFree routed-expert arena");

    if (failures == 0) {
        std::cout
            << "HIP routed-expert launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP routed-expert launcher tests: "
        << error.what() << '\n';
    return 2;
}
