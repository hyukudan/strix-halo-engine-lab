#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
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

const strix::ExecutionStep& first_shared_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::shared_expert) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no shared-expert step");
}

std::vector<Q8Block> make_q8(
    std::uint32_t rows,
    std::uint32_t width,
    float scale_base) {
    const auto blocks_per_row =
        width / 32U;
    std::vector<Q8Block> result(
        static_cast<std::size_t>(rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        for (std::uint32_t block_index = 0;
             block_index < blocks_per_row;
             ++block_index) {
            auto& block =
                result[
                    static_cast<std::size_t>(row) *
                        blocks_per_row +
                    block_index];
            block.scale =
                __float2half(
                    scale_base *
                    static_cast<float>(
                        row % 7U + 1U));
            for (std::uint32_t offset = 0;
                 offset < 32U;
                 ++offset) {
                block.values[offset] =
                    static_cast<std::int8_t>(
                        static_cast<int>(
                            (row * 3U +
                             block_index * 5U +
                             offset) %
                            11U) -
                        5);
            }
        }
    }
    return result;
}

float q8_value(
    const std::vector<Q8Block>& blocks,
    std::uint32_t row,
    std::uint32_t width,
    std::uint32_t feature) {
    const auto blocks_per_row =
        width / 32U;
    const auto& block =
        blocks[
            static_cast<std::size_t>(row) *
                blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

float silu(float x) {
    return x / (1.0F + std::exp(-x));
}

float sigmoid(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}

}  // namespace

int main() try {
    constexpr std::uint32_t hidden = 64;
    constexpr std::uint32_t intermediate = 32;

    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_shared_step(plan);
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
        "hipMalloc shared-expert arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x5348455854455354ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto input =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    std::vector<float> hidden_values(hidden);
    for (std::size_t i = 0;
         i < hidden_values.size();
         ++i) {
        hidden_values[i] =
            static_cast<float>(
                static_cast<int>(i % 17U) -
                8) /
            19.0F;
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input.address),
            hidden_values.data(),
            hidden_values.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy shared-expert input");

    const auto gate =
        make_q8(
            intermediate,
            hidden,
            0.0007F);
    std::vector<float> up(
        static_cast<std::size_t>(
            intermediate) *
        hidden);
    for (std::size_t i = 0;
         i < up.size();
         ++i) {
        up[i] =
            static_cast<float>(
                static_cast<int>(i % 13U) -
                6) *
            0.0009F;
    }
    const auto down =
        make_q8(
            hidden,
            intermediate,
            0.0006F);
    std::vector<float> shared_gate(
        hidden);
    for (std::size_t i = 0;
         i < shared_gate.size();
         ++i) {
        shared_gate[i] =
            static_cast<float>(
                static_cast<int>(i % 9U) -
                4) *
            0.0011F;
    }

    std::vector<float> expected_intermediate(
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
                    hidden_values[feature]) *
                q8_value(
                    gate,
                    inner,
                    hidden,
                    feature);
            up_sum +=
                static_cast<double>(
                    hidden_values[feature]) *
                static_cast<double>(
                    up[
                        static_cast<std::size_t>(
                            inner) *
                            hidden +
                        feature]);
        }
        expected_intermediate[inner] =
            silu(
                static_cast<float>(
                    gate_sum)) *
            static_cast<float>(
                up_sum);
    }

    double shared_gate_sum = 0.0;
    for (std::uint32_t feature = 0;
         feature < hidden;
         ++feature) {
        shared_gate_sum +=
            static_cast<double>(
                hidden_values[feature]) *
            static_cast<double>(
                shared_gate[feature]);
    }
    const float expected_scalar =
        sigmoid(
            static_cast<float>(
                shared_gate_sum));

    std::vector<float> expected(hidden);
    for (std::uint32_t feature = 0;
         feature < hidden;
         ++feature) {
        double value = 0.0;
        for (std::uint32_t inner = 0;
             inner < intermediate;
             ++inner) {
            value +=
                static_cast<double>(
                    expected_intermediate[
                        inner]) *
                q8_value(
                    down,
                    feature,
                    intermediate,
                    inner);
        }
        expected[feature] =
            static_cast<float>(
                value) *
            expected_scalar;
    }

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    auto add_tensor =
        [&](std::string name,
            std::uint32_t type,
            std::vector<std::uint64_t> dims,
            std::uint64_t bytes) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.ggml_type =
                type;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size =
                bytes;
            tensor.role =
                strix::TensorRole::
                    dense_projection;
            tensor.layer =
                step.layer;
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_tensor(
        "blk.0.shared_expert.gate_proj.weight",
        8,
        {hidden, intermediate},
        gate.size() * sizeof(Q8Block));
    add_tensor(
        "blk.0.shared_expert.up_proj.weight",
        0,
        {hidden, intermediate},
        up.size() * sizeof(float));
    add_tensor(
        "blk.0.shared_expert.down_proj.weight",
        8,
        {intermediate, hidden},
        down.size() * sizeof(Q8Block));
    add_tensor(
        "blk.0.shared_expert_gate.weight",
        0,
        {hidden, 1},
        shared_gate.size() * sizeof(float));

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    const auto upload =
        [&](std::size_t index,
            const void* source,
            std::size_t bytes) {
            check(
                hipMalloc(
                    &device_weights[index],
                    bytes),
                "hipMalloc shared-expert weight");
            check(
                hipMemcpy(
                    device_weights[index],
                    source,
                    bytes,
                    hipMemcpyHostToDevice),
                "copy shared-expert weight");
        };
    upload(
        0,
        gate.data(),
        gate.size() *
            sizeof(Q8Block));
    upload(
        1,
        up.data(),
        up.size() *
            sizeof(float));
    upload(
        2,
        down.data(),
        down.size() *
            sizeof(Q8Block));
    upload(
        3,
        shared_gate.data(),
        shared_gate.size() *
            sizeof(float));

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    for (std::size_t index = 0;
         index <
            manifest.tensors.size();
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
        0x5348455854455355ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0, 1, 2, 3},
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

    strix::HipSharedExpertLauncherConfig config;
    config.hidden_size =
        hidden;
    config.intermediate_size =
        intermediate;
    config.activation =
        strix::HipActivationStorage::fp32;

    auto launcher =
        strix::make_hip_shared_expert_launcher(
            "test.shared_expert",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "shared-expert synchronize");

    std::vector<float> actual(hidden);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy shared-expert output");

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
        maximum_error <= 2e-4F,
        "mixed-format shared expert matches CPU reference");

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree shared-expert weight");
    }
    check(
        hipFree(arena_pointer),
        "hipFree shared-expert arena");

    if (failures == 0) {
        std::cout
            << "HIP shared-expert launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP shared-expert launcher tests: "
        << error.what() << '\n';
    return 2;
}
