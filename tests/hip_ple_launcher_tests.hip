#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

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

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits +=
        0x7fffU +
        ((bits >> 16U) & 1U);
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

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

static_assert(sizeof(Q8Block) == 34);

class FixtureLookup final
    : public strix::PleLookupBackend {
public:
    explicit FixtureLookup(
        std::vector<std::uint16_t> gathered)
        : gathered_(std::move(gathered)) {}

    std::uint64_t implementation_identity()
        const noexcept override {
        return 0x504c455445535401ULL;
    }

    strix::PleLookupHandle start(
        const strix::PleLookupRequest& request) override {
        if (active_) {
            throw std::logic_error(
                "fixture lookup already active");
        }
        expect(
            request.row_ids.size() == 16 &&
                request.row_bytes == 320,
            "PLE launcher requests canonical 16x320-byte rows");
        active_ = true;
        return 1;
    }

    void finish(
        strix::PleLookupHandle handle,
        std::span<std::byte> destination) override {
        if (!active_ || handle != 1) {
            throw std::invalid_argument(
                "fixture lookup handle mismatch");
        }
        const auto bytes =
            gathered_.size() *
            sizeof(std::uint16_t);
        if (destination.size() < bytes) {
            throw std::invalid_argument(
                "fixture lookup destination undersized");
        }
        std::memcpy(
            destination.data(),
            gathered_.data(),
            bytes);
        active_ = false;
    }

    void cancel(
        strix::PleLookupHandle handle) noexcept override {
        if (handle == 1) {
            active_ = false;
        }
    }

private:
    std::vector<std::uint16_t> gathered_;
    bool active_ = false;
};

const strix::ExecutionStep& ple_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::ple) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no PLE step");
}

std::vector<Q8Block> make_q8(
    std::uint32_t rows,
    std::uint32_t width,
    bool zero_values) {
    const auto blocks_per_row =
        width / 32U;
    std::vector<Q8Block> blocks(
        static_cast<std::size_t>(rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        const float scale =
            zero_values
                ? 0.001F
                : 0.0001F *
                      static_cast<float>(
                          row % 7U + 1U);
        for (std::uint32_t block = 0;
             block < blocks_per_row;
             ++block) {
            auto& value =
                blocks[
                    static_cast<std::size_t>(row) *
                        blocks_per_row +
                    block];
            value.scale =
                __float2half(scale);
            std::fill(
                std::begin(value.values),
                std::end(value.values),
                static_cast<std::int8_t>(
                    zero_values ? 0 : 1));
        }
    }
    return blocks;
}

template <typename T>
void upload(
    const std::vector<T>& values,
    void** device) {
    check(
        hipMalloc(
            device,
            values.size() *
                sizeof(T)),
        "hipMalloc PLE test weight");
    check(
        hipMemcpy(
            *device,
            values.data(),
            values.size() *
                sizeof(T),
            hipMemcpyHostToDevice),
        "copy PLE test weight");
}

}  // namespace

int main() try {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto hash_contract =
        strix::build_ple_hash_contract(
            model);

    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        ple_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;
    request.activation_element_bytes = 2;
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
        "hipMalloc PLE arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x504c455445535410ULL;
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

    const auto expanded =
        static_cast<std::size_t>(
            model.hidden_size *
            model.hc_count);
    std::vector<std::uint16_t> input_values(
        expanded);
    for (std::size_t index = 0;
         index < input_values.size();
         ++index) {
        const float value =
            static_cast<float>(
                static_cast<int>(
                    index % 19U) -
                9) /
            32.0F;
        input_values[index] =
            float_to_bf16(value);
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input.address),
            input_values.data(),
            input_values.size() *
                sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy PLE hyper input");

    std::vector<std::uint16_t> gathered(
        static_cast<std::size_t>(
            model.ple_embed_dim),
        float_to_bf16(0.01F));
    FixtureLookup lookup_backend(
        gathered);

    const auto key_weights =
        make_q8(
            static_cast<std::uint32_t>(
                expanded),
            static_cast<std::uint32_t>(
                model.ple_embed_dim),
            true);
    const auto value_weights =
        make_q8(
            static_cast<std::uint32_t>(
                model.hidden_size),
            static_cast<std::uint32_t>(
                model.ple_embed_dim),
            false);

    std::vector<float> norm_key(
        expanded, 0.0F);
    std::vector<float> norm_query(
        expanded, 0.0F);
    std::vector<float> norm_conv(
        expanded, 0.0F);
    std::vector<float> conv(
        static_cast<std::size_t>(
            model.ple_conv_kernel_size) *
            expanded,
        0.0F);

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";

    auto add_tensor =
        [&](std::string name,
            strix::TensorRole role,
            std::uint32_t type,
            std::vector<std::uint64_t> dims,
            std::uint64_t bytes,
            bool layered) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.role = role;
            tensor.ggml_type = type;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size = bytes;
            if (layered) {
                tensor.layer =
                    step.layer;
            }
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    const auto synthetic_table_bytes =
        hash_contract.minimum_table_rows *
        hash_contract.row_bytes;
    add_tensor(
        "per_layer_token_embd.weight",
        strix::TensorRole::ngram_table,
        30,
        {
            model.ple_head_dim(),
            hash_contract.minimum_table_rows,
        },
        synthetic_table_bytes,
        false);
    add_tensor(
        "blk.1.ple_key.weight",
        strix::TensorRole::ple_parameter,
        8,
        {
            model.ple_embed_dim,
            expanded,
        },
        key_weights.size() *
            sizeof(Q8Block),
        true);
    add_tensor(
        "blk.1.ple_value.weight",
        strix::TensorRole::ple_parameter,
        8,
        {
            model.ple_embed_dim,
            model.hidden_size,
        },
        value_weights.size() *
            sizeof(Q8Block),
        true);
    add_tensor(
        "blk.1.ple_norm_key.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.hidden_size,
            model.hc_count,
        },
        norm_key.size() *
            sizeof(float),
        true);
    add_tensor(
        "blk.1.ple_norm_query.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.hidden_size,
            model.hc_count,
        },
        norm_query.size() *
            sizeof(float),
        true);
    add_tensor(
        "blk.1.ple_norm_conv.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.hidden_size,
            model.hc_count,
        },
        norm_conv.size() *
            sizeof(float),
        true);
    add_tensor(
        "blk.1.ple_conv1d.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.ple_conv_kernel_size,
            expanded,
        },
        conv.size() *
            sizeof(float),
        true);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    upload(
        key_weights,
        &device_weights[1]);
    upload(
        value_weights,
        &device_weights[2]);
    upload(
        norm_key,
        &device_weights[3]);
    upload(
        norm_query,
        &device_weights[4]);
    upload(
        norm_conv,
        &device_weights[5]);
    upload(
        conv,
        &device_weights[6]);

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    runtime.tensors.reserve(
        manifest.tensors.size());

    runtime.tensors.push_back(
        {
            0,
            strix::RuntimeTensorStorage::file_mapped,
            static_cast<std::uintptr_t>(
                0x100000ULL),
            synthetic_table_bytes,
            1,
            0,
        });
    for (std::uint32_t index = 1;
         index <
            manifest.tensors.size();
         ++index) {
        runtime.tensors.push_back(
            {
                index,
                strix::RuntimeTensorStorage::resident,
                reinterpret_cast<std::uintptr_t>(
                    device_weights[index]),
                manifest.tensors[index].byte_size,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x504c455445535411ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0},
        });
    access.accesses.push_back(
        {
            step.id,
            step.op,
            1,
            {1, 2, 3, 4, 5, 6},
        });

    const auto conv_bytes =
        strix::ple_conv_state_bytes_per_session(
            model);
    void* conv_state_pointer = nullptr;
    check(
        hipMalloc(
            &conv_state_pointer,
            static_cast<std::size_t>(
                conv_bytes)),
        "hipMalloc PLE conv state");
    check(
        hipMemset(
            conv_state_pointer,
            0,
            static_cast<std::size_t>(
                conv_bytes)),
        "clear PLE conv state");

    std::array<std::uint32_t, 2>
        history{
            123U,
            456U,
        };

    strix::SerialSessionStateLayout state_layout;
    state_layout.context_tokens = 4;
    state_layout.device_alignment = 256;
    state_layout.host_alignment = 64;
    state_layout.device_total_bytes =
        conv_bytes;
    state_layout.host_total_bytes =
        history.size() *
        sizeof(std::uint32_t);
    state_layout.layout_identity =
        0x504c455445535412ULL;
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::ple_convolution,
            strix::PersistentState::convolution_history,
            strix::SessionStateStorage::device,
            0,
            conv_bytes,
            false,
        });
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::ngram_history,
            strix::PersistentState::ngram_history,
            strix::SessionStateStorage::host,
            0,
            history.size() *
                sizeof(std::uint32_t),
            false,
        });

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        reinterpret_cast<std::uintptr_t>(
            conv_state_pointer);
    state_memory.device_bytes =
        conv_bytes;
    state_memory.device_allocation_generation = 1;
    state_memory.host_base =
        reinterpret_cast<std::uintptr_t>(
            history.data());
    state_memory.host_bytes =
        history.size() *
        sizeof(std::uint32_t);
    state_memory.host_allocation_generation = 1;

    strix::ExecutionRuntimeResources resources;
    resources.scratch = &memory;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &state_memory;
    resources.ple_lookup =
        &lookup_backend;

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
    context.session_state_layout =
        &state_layout;
    context.session_state_memory =
        &state_memory;
    context.resources =
        &resources;
    context.reads_state =
        step.reads_state;
    context.writes_state =
        step.writes_state;
    context.logical_rows = 1;
    context.request_concurrency = 1;
    context.input_token_id = 5;
    context.position = 0;

    strix::HipPleLauncherConfig config;
    config.model = model;
    config.activation =
        strix::HipActivationStorage::bf16;

    auto launcher =
        strix::make_hip_ple_launcher(
            "test.ple",
            &config);
    launcher.function(
        context,
        launcher.user_data);

    std::vector<std::uint16_t> actual(
        expanded);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual.size() *
                sizeof(std::uint16_t),
            hipMemcpyDeviceToHost),
        "copy PLE output");

    double gathered_sum = 0.0;
    for (const auto value :
         gathered) {
        gathered_sum +=
            bf16_to_float(value);
    }

    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        const auto feature =
            static_cast<std::uint32_t>(
                index %
                model.hidden_size);
        const auto block =
            feature / 32U;
        const auto& q8 =
            value_weights[
                static_cast<std::size_t>(
                    feature) *
                    (model.ple_embed_dim / 32U) +
                block];
        static_cast<void>(q8);
        const float scale =
            __half2float(
                value_weights[
                    static_cast<std::size_t>(
                        feature) *
                        (model.ple_embed_dim / 32U)].
                    scale);
        const float projected =
            static_cast<float>(
                gathered_sum) *
            scale;
        const float expected =
            bf16_to_float(
                input_values[index]) +
            0.5F *
                projected;
        const float actual_value =
            bf16_to_float(
                actual[index]);
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    actual_value -
                    expected));
    }

    expect(
        maximum_error <= 0.02F,
        "PLE zero-key fixture matches residual plus half-gated value");
    expect(
        history[0] ==
            model.ple_eos_token_id &&
        history[1] == 5,
        "PLE position-zero commit writes EOS-padded next n-gram history");

    std::vector<std::uint16_t> conv_state(
        static_cast<std::size_t>(
            conv_bytes /
            sizeof(std::uint16_t)));
    check(
        hipMemcpy(
            conv_state.data(),
            conv_state_pointer,
            static_cast<std::size_t>(
                conv_bytes),
            hipMemcpyDeviceToHost),
        "copy PLE conv state");
    const auto history_len =
        static_cast<std::size_t>(
            (model.ple_conv_kernel_size -
             1ULL) *
            model.ngram_size);
    expect(
        conv_state[
            history_len - 1U] != 0,
        "PLE convolution state commits normalized current token");

    for (std::size_t index = 1;
         index < device_weights.size();
         ++index) {
        check(
            hipFree(
                device_weights[index]),
            "hipFree PLE weight");
    }
    check(
        hipFree(
            conv_state_pointer),
        "hipFree PLE conv state");
    check(
        hipFree(
            arena_pointer),
        "hipFree PLE arena");

    if (failures == 0) {
        std::cout
            << "HIP PLE launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP PLE launcher tests: "
        << error.what() << '\n';
    return 2;
}
