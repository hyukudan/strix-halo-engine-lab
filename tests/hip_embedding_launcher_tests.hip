#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

constexpr std::array<std::int8_t, 16> codebook{
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

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

void set_iq4_local_scale(
    IQ4XSBlock& block,
    std::uint32_t subblock,
    std::uint8_t value) {
    const auto low =
        static_cast<std::uint8_t>(
            value & 0x0fU);
    const auto high =
        static_cast<std::uint16_t>(
            (value >> 4U) & 0x03U);

    auto& low_byte =
        block.scales_l[
            subblock / 2U];
    if ((subblock % 2U) == 0U) {
        low_byte =
            static_cast<std::uint8_t>(
                (low_byte & 0xf0U) |
                low);
    } else {
        low_byte =
            static_cast<std::uint8_t>(
                (low_byte & 0x0fU) |
                static_cast<std::uint8_t>(
                    low << 4U));
    }

    const auto shift =
        2U * subblock;
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h &
            static_cast<std::uint16_t>(
                ~(0x03U << shift)));
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h |
            static_cast<std::uint16_t>(
                high << shift));
}

const strix::ExecutionStep& embedding_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::token_embedding) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no embedding step");
}

std::vector<Q8Block> make_q8(
    std::uint32_t rows,
    std::uint32_t width) {
    const auto blocks_per_row =
        width / 32U;
    std::vector<Q8Block> blocks(
        static_cast<std::size_t>(
            rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        const float scale =
            0.001F *
            static_cast<float>(
                row + 1U);
        for (std::uint32_t block_index = 0;
             block_index < blocks_per_row;
             ++block_index) {
            auto& block =
                blocks[
                    static_cast<std::size_t>(
                        row) *
                        blocks_per_row +
                    block_index];
            block.scale =
                __float2half(scale);
            for (std::uint32_t offset = 0;
                 offset < 32U;
                 ++offset) {
                block.values[offset] =
                    static_cast<std::int8_t>(
                        static_cast<int>(
                            (block_index * 3U +
                             offset) %
                            15U) -
                        7);
            }
        }
    }
    return blocks;
}

std::vector<IQ4XSBlock> make_iq4(
    std::uint32_t rows,
    std::uint32_t width) {
    const auto blocks_per_row =
        width / 256U;
    std::vector<IQ4XSBlock> blocks(
        static_cast<std::size_t>(
            rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        const float scale =
            0.0005F *
            static_cast<float>(
                row + 1U);
        for (std::uint32_t block_index = 0;
             block_index < blocks_per_row;
             ++block_index) {
            auto& block =
                blocks[
                    static_cast<std::size_t>(
                        row) *
                        blocks_per_row +
                    block_index];
            block.d =
                __float2half(scale);
            block.scales_h = 0;
            std::fill(
                std::begin(block.scales_l),
                std::end(block.scales_l),
                std::uint8_t{0});
            for (std::uint32_t subblock = 0;
                 subblock < 8U;
                 ++subblock) {
                set_iq4_local_scale(
                    block,
                    subblock,
                    33U);
                for (std::uint32_t byte = 0;
                     byte < 16U;
                     ++byte) {
                    const auto code =
                        static_cast<std::uint8_t>(
                            8U +
                            ((block_index +
                              subblock +
                              byte) %
                             4U));
                    block.qs[
                        subblock * 16U +
                        byte] =
                        static_cast<std::uint8_t>(
                            code |
                            static_cast<std::uint8_t>(
                                code << 4U));
                }
            }
        }
    }
    return blocks;
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
            static_cast<std::size_t>(
                row) *
                blocks_per_row +
            feature / 32U];
    return
        __half2float(
            block.scale) *
        static_cast<float>(
            block.values[
                feature % 32U]);
}

std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto packed =
        block.scales_l[
            subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (packed >>
             (4U *
              (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return
        static_cast<std::uint8_t>(
            low |
            static_cast<std::uint8_t>(
                high << 4U));
}

float iq4_value(
    const std::vector<IQ4XSBlock>& blocks,
    std::uint32_t row,
    std::uint32_t width,
    std::uint32_t feature) {
    const auto blocks_per_row =
        width / 256U;
    const auto& block =
        blocks[
            static_cast<std::size_t>(
                row) *
                blocks_per_row +
            feature / 256U];
    const auto within =
        feature % 256U;
    const auto subblock =
        within / 32U;
    const auto suboffset =
        within % 32U;
    const auto packed =
        block.qs[
            subblock * 16U +
            suboffset % 16U];
    const auto code =
        suboffset < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    const int local =
        static_cast<int>(
            iq4_local_scale(
                block,
                subblock)) -
        32;

    return
        __half2float(block.d) *
        static_cast<float>(local) *
        static_cast<float>(
            codebook[code]);
}

template <typename Packed>
void upload_weights(
    const std::vector<Packed>& values,
    void** device) {
    check(
        hipMalloc(
            device,
            values.size() *
                sizeof(Packed)),
        "hipMalloc embedding weights");
    check(
        hipMemcpy(
            *device,
            values.data(),
            values.size() *
                sizeof(Packed),
            hipMemcpyHostToDevice),
        "copy embedding weights");
}

template <typename Expected>
void run_case(
    std::uint32_t ggml_type,
    strix::HipActivationStorage output_storage,
    Expected&& expected_value) {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();
    const auto width =
        static_cast<std::uint32_t>(
            model.hidden_size);
    constexpr std::uint32_t vocab = 16;
    constexpr std::uint32_t token = 7;

    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        embedding_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;
    request.activation_element_bytes =
        static_cast<std::uint32_t>(
            output_storage);
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
        "hipMalloc embedding arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x11110000ULL +
        ggml_type;
    memory.arena_base =
        reinterpret_cast<
            std::uintptr_t>(
                arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    strix::ManifestTensor tensor;
    tensor.name =
        "token_embd.weight";
    tensor.dimensions = {
        width,
        vocab,
    };
    tensor.ggml_type =
        ggml_type;
    tensor.role =
        strix::TensorRole::embedding;

    void* device_weights = nullptr;
    std::uint64_t weight_bytes = 0;

    if (ggml_type == 8) {
        const auto weights =
            make_q8(
                vocab,
                width);
        weight_bytes =
            weights.size() *
            sizeof(Q8Block);
        upload_weights(
            weights,
            &device_weights);
    } else if (ggml_type == 23) {
        const auto weights =
            make_iq4(
                vocab,
                width);
        weight_bytes =
            weights.size() *
            sizeof(IQ4XSBlock);
        upload_weights(
            weights,
            &device_weights);
    } else {
        throw std::logic_error(
            "unsupported embedding test type");
    }

    tensor.byte_size =
        weight_bytes;
    manifest.total_tensor_bytes =
        weight_bytes;
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
            reinterpret_cast<
                std::uintptr_t>(
                    device_weights),
            weight_bytes,
            1,
            0,
        });

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x22220000ULL +
        ggml_type;
    access.accesses.push_back(
        {
            step.id,
            strix::SerialOp::token_embedding,
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
    context.step_id =
        step.id;
    context.op =
        step.op;
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
    context.input_token_id =
        token;

    strix::HipEmbeddingLauncherConfig config;
    config.hidden_size =
        width;
    config.output =
        output_storage;

    auto launcher =
        strix::make_hip_embedding_launcher(
            "test.embedding",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "embedding synchronize");

    if (output_storage ==
        strix::HipActivationStorage::fp32) {
        std::vector<float> actual(
            width);
        check(
            hipMemcpy(
                actual.data(),
                reinterpret_cast<
                    const void*>(
                        output.address),
                actual.size() *
                    sizeof(float),
                hipMemcpyDeviceToHost),
            "copy FP32 embedding output");

        for (std::uint32_t feature = 0;
             feature < width;
             ++feature) {
            const float expected =
                expected_value(
                    token,
                    feature);
            if (std::abs(
                    actual[feature] -
                    expected) >
                1e-6F) {
                expect(false,
                       "FP32 embedding output matches packed row reference");
                break;
            }
        }
    } else {
        std::vector<std::uint16_t> actual(
            width);
        check(
            hipMemcpy(
                actual.data(),
                reinterpret_cast<
                    const void*>(
                        output.address),
                actual.size() *
                    sizeof(std::uint16_t),
                hipMemcpyDeviceToHost),
            "copy BF16 embedding output");

        for (std::uint32_t feature = 0;
             feature < width;
             ++feature) {
            const auto expected =
                float_to_bf16(
                    expected_value(
                        token,
                        feature));
            if (actual[feature] !=
                expected) {
                expect(false,
                       "BF16 embedding output matches packed row reference");
                break;
            }
        }
    }

    auto invalid = context;
    invalid.input_token_id =
        vocab;
    bool token_rejected = false;
    try {
        launcher.function(
            invalid,
            launcher.user_data);
    } catch (const std::out_of_range&) {
        token_rejected = true;
    }
    expect(
        token_rejected,
        "embedding launcher rejects token outside manifest vocabulary");

    check(
        hipFree(device_weights),
        "hipFree embedding weights");
    check(
        hipFree(arena_pointer),
        "hipFree embedding arena");
}

void test_runtime_cases() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();
    const auto width =
        static_cast<std::uint32_t>(
            model.hidden_size);

    const auto q8 =
        make_q8(16, width);
    run_case(
        8,
        strix::HipActivationStorage::fp32,
        [&](std::uint32_t row,
            std::uint32_t feature) {
            return q8_value(
                q8,
                row,
                width,
                feature);
        });

    const auto iq4 =
        make_iq4(16, width);
    run_case(
        23,
        strix::HipActivationStorage::bf16,
        [&](std::uint32_t row,
            std::uint32_t feature) {
            return iq4_value(
                iq4,
                row,
                width,
                feature);
        });
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

    strix::HipEmbeddingLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    config.output =
        strix::HipActivationStorage::bf16;

    strix::kernels::KernelRegistry registry;
    strix::register_hip_embedding_kernel_variant(
        model,
        request,
        config,
        registry);

    expect(
        registry.variants().size() == 1 &&
        !registry.variants()[0].graph_safe,
        "embedding v1 registers one non-graph-safe experimental variant");

    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto bindings =
        strix::bind_serial_execution_plan(
            model,
            execution,
            registry,
            request);

    expect(
        bindings.bound_steps == 1 &&
        bindings.bindings.front().
            kernel_id.has_value(),
        "embedding catalog binds only token embedding");

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x3333ULL;
    compiled.execution =
        execution;
    compiled.bindings =
        bindings;

    strix::ExecutionLauncherRegistry launchers;
    strix::HipEmbeddingLauncherBundle bundle(
        config);
    bundle.register_for_compiled_execution(
        compiled,
        launchers);

    expect(
        launchers.find_kernel_launcher(
            *bindings.bindings.front().
                kernel_id) != nullptr,
        "embedding launcher bundle materializes selected kernel id");

    auto rows_two = request;
    rows_two.logical_rows = 2;
    strix::kernels::KernelRegistry unused;
    bool rows_rejected = false;
    try {
        strix::register_hip_embedding_kernel_variant(
            model,
            rows_two,
            config,
            unused);
    } catch (const std::invalid_argument&) {
        rows_rejected = true;
    }
    expect(
        rows_rejected,
        "embedding v1 refuses multi-row catalog registration");
}

}  // namespace

int main() try {
    test_catalog_and_bundle();
    test_runtime_cases();

    if (failures == 0) {
        std::cout
            << "HIP embedding launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP embedding launcher tests: "
        << error.what() << '\n';
    return 2;
}
