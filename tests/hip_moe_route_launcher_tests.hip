#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
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

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        bits >> 16U);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            value) << 16U;
    float output = 0.0F;
    std::memcpy(
        &output,
        &bits,
        sizeof(output));
    return output;
}

const strix::ExecutionStep& first_route_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::moe_route) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no MoE route step");
}

const strix::ExecutionStepIo& io_for(
    const strix::SerialExecutionIoPlan& io,
    std::uint32_t step_id) {
    if (step_id >= io.steps.size()) {
        throw std::out_of_range(
            "step id exceeds IO plan");
    }
    return io.steps[step_id];
}

std::vector<float> make_hidden(
    std::size_t count) {
    std::vector<float> values(
        count,
        0.001F);
    for (std::size_t index = 0;
         index < count;
         ++index) {
        values[index] +=
            static_cast<float>(
                index % 7U) *
            0.00001F;
    }
    return values;
}

std::vector<float> make_f32_weights(
    std::uint32_t experts,
    std::uint32_t hidden) {
    std::vector<float> weights(
        static_cast<std::size_t>(
            experts) *
        hidden);
    for (std::uint32_t expert = 0;
         expert < experts;
         ++expert) {
        const float value =
            0.001F *
            static_cast<float>(
                expert + 1U);
        std::fill(
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        expert) *
                    hidden),
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        expert + 1U) *
                    hidden),
            value);
    }
    return weights;
}

std::vector<Q8Block> make_q8_weights(
    std::uint32_t experts,
    std::uint32_t hidden) {
    if (hidden % 32U != 0) {
        throw std::invalid_argument(
            "Q8 fixture hidden width must be divisible by 32");
    }
    const auto blocks_per_row =
        hidden / 32U;
    std::vector<Q8Block> weights(
        static_cast<std::size_t>(
            experts) *
        blocks_per_row);

    for (std::uint32_t expert = 0;
         expert < experts;
         ++expert) {
        const float scale =
            0.001F *
            static_cast<float>(
                expert + 1U);
        for (std::uint32_t block = 0;
             block < blocks_per_row;
             ++block) {
            auto& value =
                weights[
                    static_cast<std::size_t>(
                        expert) *
                        blocks_per_row +
                    block];
            value.scale =
                __float2half(scale);
            std::fill(
                std::begin(value.values),
                std::end(value.values),
                std::int8_t{1});
        }
    }
    return weights;
}

template <typename WeightReader>
void cpu_route_reference(
    const std::vector<float>& hidden,
    std::uint32_t rows,
    std::uint32_t hidden_size,
    std::uint32_t experts,
    std::uint32_t top_k,
    WeightReader&& weight,
    std::vector<std::uint32_t>& indices,
    std::vector<float>& selected_weights) {
    indices.resize(
        static_cast<std::size_t>(
            rows) *
        top_k);
    selected_weights.resize(
        static_cast<std::size_t>(
            rows) *
        top_k);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        std::vector<float> probabilities(
            experts);
        float maximum =
            -std::numeric_limits<float>::infinity();

        for (std::uint32_t expert = 0;
             expert < experts;
             ++expert) {
            float sum = 0.0F;
            for (std::uint32_t feature = 0;
                 feature < hidden_size;
                 ++feature) {
                sum +=
                    hidden[
                        static_cast<std::size_t>(
                            row) *
                            hidden_size +
                        feature] *
                    weight(
                        expert,
                        feature);
            }
            probabilities[expert] = sum;
            maximum =
                std::max(
                    maximum,
                    sum);
        }

        float denominator = 0.0F;
        for (auto& value :
             probabilities) {
            value =
                std::exp(
                    value - maximum);
            denominator += value;
        }
        for (auto& value :
             probabilities) {
            value /= denominator;
        }

        std::vector<std::uint32_t> order(
            experts);
        std::iota(
            order.begin(),
            order.end(),
            0U);
        std::partial_sort(
            order.begin(),
            order.begin() +
                static_cast<std::ptrdiff_t>(
                    top_k),
            order.end(),
            [&](std::uint32_t left,
                std::uint32_t right) {
                if (probabilities[left] !=
                    probabilities[right]) {
                    return
                        probabilities[left] >
                        probabilities[right];
                }
                return left < right;
            });

        float chosen_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            chosen_sum +=
                probabilities[
                    order[slot]];
        }

        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            const auto output =
                static_cast<std::size_t>(
                    row) *
                    top_k +
                slot;
            indices[output] =
                order[slot];
            selected_weights[output] =
                probabilities[
                    order[slot]] /
                chosen_sum;
        }
    }
}

void run_case(
    std::uint32_t ggml_type,
    strix::HipActivationStorage activation) {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    const std::uint32_t rows = 1;
    const auto hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    const auto experts =
        static_cast<std::uint32_t>(
            model.num_experts);
    const auto top_k =
        static_cast<std::uint32_t>(
            model.num_experts_per_token);

    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_route_step(plan);
    const auto& step_io =
        io_for(io, step.id);

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = rows;
    buffer_request.activation_element_bytes =
        static_cast<std::uint32_t>(
            activation);
    buffer_request.control_element_bytes = 4;
    buffer_request.index_element_bytes = 4;
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
        "hipMalloc route arena");

    const auto hidden_address =
        strix::resolve_execution_buffer(
            0x1010ULL,
            arena,
            strix::ExecutionMemoryBinding{
                0x1010ULL,
                reinterpret_cast<
                    std::uintptr_t>(
                    arena_pointer),
                arena.total_bytes,
                1,
            },
            step_io.reads[0]);

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x1010ULL;
    memory.arena_base =
        reinterpret_cast<
            std::uintptr_t>(
                arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto indices_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);
    const auto weights_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[1]);

    const auto hidden_float =
        make_hidden(
            static_cast<std::size_t>(
                rows) *
            hidden_size);

    if (activation ==
        strix::HipActivationStorage::fp32) {
        check(
            hipMemcpy(
                reinterpret_cast<void*>(
                    hidden_address.address),
                hidden_float.data(),
                hidden_float.size() *
                    sizeof(float),
                hipMemcpyHostToDevice),
            "copy FP32 hidden");
    } else {
        std::vector<std::uint16_t> packed(
            hidden_float.size());
        std::transform(
            hidden_float.begin(),
            hidden_float.end(),
            packed.begin(),
            float_to_bf16);
        check(
            hipMemcpy(
                reinterpret_cast<void*>(
                    hidden_address.address),
                packed.data(),
                packed.size() *
                    sizeof(std::uint16_t),
                hipMemcpyHostToDevice),
            "copy BF16 hidden");
    }

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    strix::ManifestTensor router;
    router.name =
        "blk.0.ffn_gate_inp.weight";
    router.dimensions = {
        hidden_size,
        experts,
    };
    router.ggml_type =
        ggml_type;
    router.role =
        strix::TensorRole::moe_router;
    router.layer = 0;

    std::vector<float> f32_weights;
    std::vector<Q8Block> q8_weights;
    void* device_weights = nullptr;

    if (ggml_type == 0) {
        f32_weights =
            make_f32_weights(
                experts,
                hidden_size);
        router.byte_size =
            f32_weights.size() *
            sizeof(float);
        check(
            hipMalloc(
                &device_weights,
                static_cast<std::size_t>(
                    router.byte_size)),
            "hipMalloc F32 router weights");
        check(
            hipMemcpy(
                device_weights,
                f32_weights.data(),
                static_cast<std::size_t>(
                    router.byte_size),
                hipMemcpyHostToDevice),
            "copy F32 router weights");
    } else if (ggml_type == 8) {
        q8_weights =
            make_q8_weights(
                experts,
                hidden_size);
        router.byte_size =
            q8_weights.size() *
            sizeof(Q8Block);
        check(
            hipMalloc(
                &device_weights,
                static_cast<std::size_t>(
                    router.byte_size)),
            "hipMalloc Q8 router weights");
        check(
            hipMemcpy(
                device_weights,
                q8_weights.data(),
                static_cast<std::size_t>(
                    router.byte_size),
                hipMemcpyHostToDevice),
            "copy Q8 router weights");
    } else {
        throw std::logic_error(
            "unsupported route test type");
    }

    manifest.total_tensor_bytes =
        router.byte_size;
    manifest.tensors.push_back(
        router);
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
            reinterpret_cast<
                std::uintptr_t>(
                    device_weights),
            router.byte_size,
            1,
            0,
        });

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x2020ULL;
    access.accesses.push_back(
        {
            step.id,
            strix::SerialOp::moe_route,
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
    context.op = step.op;
    context.layer = step.layer;
    context.reads = step_io.reads;
    context.writes = step_io.writes;
    context.arena = &arena;
    context.memory = &memory;
    context.resources =
        &resources;
    context.logical_rows = rows;
    context.request_concurrency = 1;

    strix::HipMoeRouteLauncherConfig config;
    config.hidden_size =
        hidden_size;
    config.expert_count =
        experts;
    config.top_k =
        top_k;
    config.activation =
        activation;
    config.normalize_selected = true;

    auto launcher =
        strix::make_hip_moe_route_launcher(
            "test.moe_route",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "MoE route synchronize");

    std::vector<std::uint32_t> actual_indices(
        static_cast<std::size_t>(
            rows) *
        top_k);
    std::vector<float> actual_weights(
        actual_indices.size());

    check(
        hipMemcpy(
            actual_indices.data(),
            reinterpret_cast<
                const void*>(
                    indices_address.address),
            actual_indices.size() *
                sizeof(std::uint32_t),
            hipMemcpyDeviceToHost),
        "copy route indices");
    check(
        hipMemcpy(
            actual_weights.data(),
            reinterpret_cast<
                const void*>(
                    weights_address.address),
            actual_weights.size() *
                sizeof(float),
            hipMemcpyDeviceToHost),
        "copy route weights");

    std::vector<float> reference_hidden =
        hidden_float;
    if (activation ==
        strix::HipActivationStorage::bf16) {
        for (auto& value :
             reference_hidden) {
            value =
                bf16_to_float(
                    float_to_bf16(
                        value));
        }
    }

    std::vector<std::uint32_t> expected_indices;
    std::vector<float> expected_weights;

    if (ggml_type == 0) {
        cpu_route_reference(
            reference_hidden,
            rows,
            hidden_size,
            experts,
            top_k,
            [&](std::uint32_t expert,
                std::uint32_t feature) {
                return f32_weights[
                    static_cast<
                        std::size_t>(
                        expert) *
                        hidden_size +
                    feature];
            },
            expected_indices,
            expected_weights);
    } else {
        const auto blocks_per_row =
            hidden_size / 32U;
        cpu_route_reference(
            reference_hidden,
            rows,
            hidden_size,
            experts,
            top_k,
            [&](std::uint32_t expert,
                std::uint32_t feature) {
                const auto& block =
                    q8_weights[
                        static_cast<
                            std::size_t>(
                            expert) *
                            blocks_per_row +
                        feature / 32U];
                return
                    __half2float(
                        block.scale) *
                    static_cast<float>(
                        block.values[
                            feature % 32U]);
            },
            expected_indices,
            expected_weights);
    }

    expect(
        actual_indices ==
            expected_indices,
        "HIP MoE route selected indices match CPU reference");

    float max_error = 0.0F;
    for (std::size_t index = 0;
         index <
            actual_weights.size();
         ++index) {
        max_error =
            std::max(
                max_error,
                std::abs(
                    actual_weights[index] -
                    expected_weights[index]));
    }
    expect(
        max_error <= 1e-5F,
        "HIP MoE route normalized weights match CPU reference");

    check(
        hipFree(device_weights),
        "hipFree router weights");
    check(
        hipFree(arena_pointer),
        "hipFree route arena");
}

void test_catalog_and_bundle() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    strix::ExecutionBindingRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    strix::HipMoeRouteLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    config.expert_count =
        static_cast<std::uint32_t>(
            model.num_experts);
    config.top_k =
        static_cast<std::uint32_t>(
            model.num_experts_per_token);
    config.activation =
        strix::HipActivationStorage::bf16;

    strix::kernels::KernelRegistry kernels;
    strix::register_hip_moe_route_kernel_variant(
        model,
        request,
        config,
        kernels);

    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto bindings =
        strix::bind_serial_execution_plan(
            model,
            execution,
            kernels,
            request);

    std::uint32_t route_steps = 0;
    for (std::size_t index = 0;
         index <
            execution.steps.size();
         ++index) {
        if (execution.steps[index].op !=
            strix::SerialOp::moe_route) {
            continue;
        }
        ++route_steps;
        expect(
            bindings.bindings[index].
                kernel_id.has_value(),
            "HIP MoE route catalog binds every route layer");
    }
    expect(
        route_steps ==
            model.hidden_layers &&
        bindings.bound_steps ==
            route_steps,
        "HIP MoE route catalog binds exactly all route steps");

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x3030ULL;
    compiled.execution =
        execution;
    compiled.bindings =
        bindings;

    strix::ExecutionLauncherRegistry launchers;
    strix::HipMoeRouteLauncherBundle bundle(
        config);
    bundle.register_for_compiled_execution(
        compiled,
        launchers);

    const auto route_it =
        std::find_if(
            bindings.bindings.begin(),
            bindings.bindings.end(),
            [](const auto& binding) {
                return binding.op ==
                    strix::SerialOp::moe_route;
            });
    expect(
        route_it !=
            bindings.bindings.end() &&
        route_it->kernel_id.has_value() &&
        launchers.find_kernel_launcher(
            *route_it->kernel_id) !=
            nullptr,
        "HIP MoE route launcher bundle materializes selected route id");
}

}  // namespace

int main() try {
    test_catalog_and_bundle();
    run_case(
        0,
        strix::HipActivationStorage::fp32);
    run_case(
        8,
        strix::HipActivationStorage::bf16);

    if (failures == 0) {
        std::cout
            << "HIP MoE route launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP MoE route launcher tests: "
        << error.what() << '\n';
    return 2;
}
