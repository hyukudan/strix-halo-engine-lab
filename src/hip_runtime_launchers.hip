#include "strix/hip_runtime_launchers.hpp"

#include "strix/dispatch_resources.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

constexpr std::uint64_t hyper_expand_tag =
    0x485950455850414eULL;
constexpr std::uint64_t inject_tag =
    0x494e4a4543540001ULL;
constexpr std::uint64_t moe_combine_tag =
    0x4d4f45434f4d4201ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "HIP elementwise launcher size overflow");
    }
    return left * right;
}

std::uint64_t activation_bytes(
    const HipElementwiseLauncherConfig& config) {
    switch (config.activation) {
    case HipActivationStorage::bf16:
        return 2;
    case HipActivationStorage::fp32:
        return 4;
    }
    throw std::invalid_argument(
        "unsupported HIP activation storage");
}

const HipElementwiseLauncherConfig&
require_config(void* user_data) {
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP elementwise launcher requires configuration");
    }
    const auto& config =
        *static_cast<HipElementwiseLauncherConfig*>(
            user_data);
    if (config.hidden_size == 0 ||
        config.hc_count == 0) {
        throw std::invalid_argument(
            "HIP elementwise launcher geometry must be non-zero");
    }
    static_cast<void>(
        activation_bytes(config));
    return config;
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

__device__ float load_activation(
    const float* value) {
    return *value;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return __uint_as_float(
        static_cast<std::uint32_t>(*value) << 16U);
}

__device__ void store_activation(
    float* destination,
    float value) {
    *destination = value;
}

__device__ void store_activation(
    std::uint16_t* destination,
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    *destination =
        static_cast<std::uint16_t>(
            rounded >> 16U);
}

template <typename Storage>
__global__ void hyper_expand_kernel(
    const Storage* input,
    Storage* output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t rows) {
    const auto index =
        static_cast<std::uint64_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    const auto expanded_width =
        static_cast<std::uint64_t>(
            hidden_size) *
        hc_count;
    const auto total =
        expanded_width * rows;
    if (index >= total) {
        return;
    }

    const auto row =
        index / expanded_width;
    const auto within =
        index % expanded_width;
    const auto feature =
        within %
        static_cast<std::uint64_t>(
            hidden_size);

    const auto source =
        row *
            static_cast<std::uint64_t>(
                hidden_size) +
        feature;

    store_activation(
        &output[index],
        load_activation(
            &input[source]));
}

template <typename Storage>
__global__ void inject_kernel(
    const Storage* hyper_input,
    const Storage* hidden,
    const float* injection_weights,
    Storage* output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    std::uint32_t rows) {
    const auto index =
        static_cast<std::uint64_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    const auto expanded_width =
        static_cast<std::uint64_t>(
            hidden_size) *
        hc_count;
    const auto total =
        expanded_width * rows;
    if (index >= total) {
        return;
    }

    const auto row =
        index / expanded_width;
    const auto within =
        index % expanded_width;
    const auto stream =
        within /
        static_cast<std::uint64_t>(
            hidden_size);
    const auto feature =
        within %
        static_cast<std::uint64_t>(
            hidden_size);

    const auto hidden_index =
        row *
            static_cast<std::uint64_t>(
                hidden_size) +
        feature;
    const auto weight_index =
        row *
            static_cast<std::uint64_t>(
                hc_count) +
        stream;

    const float result =
        load_activation(
            &hyper_input[index]) +
        load_activation(
            &hidden[hidden_index]) *
        injection_weights[weight_index];

    store_activation(
        &output[index],
        result);
}

template <typename Storage>
__global__ void moe_combine_kernel(
    const Storage* shared,
    const Storage* routed,
    Storage* output,
    std::uint64_t elements) {
    const auto index =
        static_cast<std::uint64_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index >= elements) {
        return;
    }
    store_activation(
        &output[index],
        load_activation(&shared[index]) +
        load_activation(&routed[index]));
}

template <typename Kernel>
void launch_1d(
    std::uint64_t elements,
    Kernel&& launch) {
    constexpr std::uint32_t threads = 256;
    if (elements == 0) {
        throw std::invalid_argument(
            "HIP elementwise launch requires non-zero elements");
    }
    const auto blocks64 =
        (elements + threads - 1ULL) /
        threads;
    if (blocks64 >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "HIP elementwise grid exceeds uint32 block count");
    }
    launch(
        static_cast<std::uint32_t>(
            blocks64),
        threads);
    check_hip(
        hipGetLastError(),
        "HIP elementwise kernel launch");
}

void validate_buffer(
    const ExecutionBufferAddress& buffer,
    std::uint64_t required,
    const char* name) {
    if (buffer.address == 0 ||
        buffer.bytes < required) {
        throw std::invalid_argument(
            std::string(name) +
            " buffer is missing or undersized");
    }
}

template <typename Storage>
void launch_hyper_expand_typed(
    const ExecutionDispatchContext& context,
    const HipElementwiseLauncherConfig& config,
    const ExecutionBufferAddress& input,
    const ExecutionBufferAddress& output) {
    const auto elements =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            config.hc_count);
    launch_1d(
        elements,
        [&](std::uint32_t blocks,
            std::uint32_t threads) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    hyper_expand_kernel<Storage>),
                dim3(blocks),
                dim3(threads),
                0,
                0,
                reinterpret_cast<const Storage*>(
                    input.address),
                reinterpret_cast<Storage*>(
                    output.address),
                config.hidden_size,
                config.hc_count,
                context.logical_rows);
        });
}

void launch_hyper_expand(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::hyper_expand) {
        throw std::invalid_argument(
            "HIP hyper-expand launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP hyper-expand expects one read and one write buffer");
    }

    const auto& config =
        require_config(user_data);
    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto element_bytes =
        activation_bytes(config);

    const auto input_required =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            element_bytes);
    const auto output_required =
        checked_mul(
            checked_mul(
                checked_mul(
                    context.logical_rows,
                    config.hidden_size),
                config.hc_count),
            element_bytes);
    validate_buffer(
        input,
        input_required,
        "hyper-expand input");
    validate_buffer(
        output,
        output_required,
        "hyper-expand output");

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_hyper_expand_typed<
            std::uint16_t>(
                context,
                config,
                input,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_hyper_expand_typed<float>(
            context,
            config,
            input,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable activation storage");
}

template <typename Storage>
void launch_inject_typed(
    const ExecutionDispatchContext& context,
    const HipElementwiseLauncherConfig& config,
    const ExecutionBufferAddress& hyper,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& weights,
    const ExecutionBufferAddress& output) {
    const auto elements =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            config.hc_count);
    launch_1d(
        elements,
        [&](std::uint32_t blocks,
            std::uint32_t threads) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    inject_kernel<Storage>),
                dim3(blocks),
                dim3(threads),
                0,
                0,
                reinterpret_cast<const Storage*>(
                    hyper.address),
                reinterpret_cast<const Storage*>(
                    hidden.address),
                reinterpret_cast<const float*>(
                    weights.address),
                reinterpret_cast<Storage*>(
                    output.address),
                config.hidden_size,
                config.hc_count,
                context.logical_rows);
        });
}

void launch_inject(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op != SerialOp::attn_inject &&
        context.op != SerialOp::mlp_inject) {
        throw std::invalid_argument(
            "HIP inject launcher attached to wrong operator");
    }
    if (context.reads.size() != 3 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP inject expects three reads and one write");
    }

    const auto& config =
        require_config(user_data);
    const auto hyper =
        resolve_dispatch_read(
            context, 0);
    const auto hidden =
        resolve_dispatch_read(
            context, 1);
    const auto weights =
        resolve_dispatch_read(
            context, 2);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto element_bytes =
        activation_bytes(config);

    const auto expanded_elements =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            config.hc_count);
    const auto expanded_bytes =
        checked_mul(
            expanded_elements,
            element_bytes);
    const auto hidden_bytes =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            element_bytes);
    const auto weight_bytes =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hc_count),
            sizeof(float));

    validate_buffer(
        hyper,
        expanded_bytes,
        "inject hyper input");
    validate_buffer(
        hidden,
        hidden_bytes,
        "inject hidden input");
    validate_buffer(
        weights,
        weight_bytes,
        "inject weight input");
    validate_buffer(
        output,
        expanded_bytes,
        "inject output");

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_inject_typed<
            std::uint16_t>(
                context,
                config,
                hyper,
                hidden,
                weights,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_inject_typed<float>(
            context,
            config,
            hyper,
            hidden,
            weights,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable activation storage");
}

template <typename Storage>
void launch_moe_combine_typed(
    const ExecutionDispatchContext& context,
    const HipElementwiseLauncherConfig& config,
    const ExecutionBufferAddress& shared,
    const ExecutionBufferAddress& routed,
    const ExecutionBufferAddress& output) {
    const auto elements =
        checked_mul(
            context.logical_rows,
            config.hidden_size);
    launch_1d(
        elements,
        [&](std::uint32_t blocks,
            std::uint32_t threads) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    moe_combine_kernel<Storage>),
                dim3(blocks),
                dim3(threads),
                0,
                0,
                reinterpret_cast<const Storage*>(
                    shared.address),
                reinterpret_cast<const Storage*>(
                    routed.address),
                reinterpret_cast<Storage*>(
                    output.address),
                elements);
        });
}

void launch_moe_combine(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::moe_combine) {
        throw std::invalid_argument(
            "HIP MoE-combine launcher attached to wrong operator");
    }
    if (context.reads.size() != 2 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP MoE-combine expects two reads and one write");
    }

    const auto& config =
        require_config(user_data);
    const auto shared =
        resolve_dispatch_read(
            context, 0);
    const auto routed =
        resolve_dispatch_read(
            context, 1);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto required =
        checked_mul(
            checked_mul(
                context.logical_rows,
                config.hidden_size),
            activation_bytes(config));

    validate_buffer(
        shared,
        required,
        "MoE-combine shared input");
    validate_buffer(
        routed,
        required,
        "MoE-combine routed input");
    validate_buffer(
        output,
        required,
        "MoE-combine output");

    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_moe_combine_typed<
            std::uint16_t>(
                context,
                config,
                shared,
                routed,
                output);
        return;
    case HipActivationStorage::fp32:
        launch_moe_combine_typed<float>(
            context,
            config,
            shared,
            routed,
            output);
        return;
    }
    throw std::logic_error(
        "unreachable activation storage");
}

ExecutionLauncher make_launcher(
    std::string kernel_id,
    std::uint64_t tag,
    ExecutionLaunchFunction function,
    HipElementwiseLauncherConfig* config) {
    if (kernel_id.empty()) {
        throw std::invalid_argument(
            "HIP launcher kernel id must not be empty");
    }
    if (config == nullptr) {
        throw std::invalid_argument(
            "HIP launcher configuration must not be null");
    }
    const auto identity =
        hip_elementwise_launcher_identity(
            tag, *config);
    return ExecutionLauncher{
        std::move(kernel_id),
        identity,
        function,
        config,
    };
}

}  // namespace


namespace {

const char* activation_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

kernels::KernelVariant make_elementwise_variant(
    std::string id,
    kernels::OperatorFamily family,
    const kernels::ShapeClass& shape,
    const HipElementwiseLauncherConfig& config,
    kernels::QualificationStatus qualification) {
    kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision =
        activation_name(config.activation);
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision =
        activation_name(config.activation);
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification = qualification;
    variant.graph_safe = true;
    variant.input_layout = "row-major";
    variant.output_layout = "row-major";
    variant.compiler_identity =
        "hip-gfx1151-elementwise-v1";
    variant.measured_seconds = 0.0;
    return variant;
}

std::string elementwise_kernel_id(
    std::string_view operation,
    const ExecutionBindingRequest& request,
    const HipElementwiseLauncherConfig& config) {
    return
        "hip." +
        std::string(operation) +
        "." +
        activation_name(config.activation) +
        ".h" +
        std::to_string(config.hidden_size) +
        ".hc" +
        std::to_string(config.hc_count) +
        ".r" +
        std::to_string(request.logical_rows) +
        ".c" +
        std::to_string(request.request_concurrency);
}

}  // namespace

void register_hip_elementwise_kernel_variants(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipElementwiseLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "HIP elementwise variants require non-zero rows and concurrency");
    }
    if (config.hidden_size != model.hidden_size ||
        config.hc_count != model.hc_count) {
        throw std::invalid_argument(
            "HIP elementwise variant geometry does not match model contract");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP elementwise catalog may only register experimental or oracle-matched variants");
    }
    static_cast<void>(
        activation_bytes(config));

    const auto hyper_shape =
        execution_shape_for(
            model,
            SerialOp::hyper_expand,
            request);
    const auto inject_shape =
        execution_shape_for(
            model,
            SerialOp::attn_inject,
            request);
    const auto combine_shape =
        execution_shape_for(
            model,
            SerialOp::moe_combine,
            request);

    registry.add(
        make_elementwise_variant(
            elementwise_kernel_id(
                "hyper_expand",
                request,
                config),
            kernels::OperatorFamily::hyper_expand,
            hyper_shape,
            config,
            qualification));

    registry.add(
        make_elementwise_variant(
            elementwise_kernel_id(
                "inject",
                request,
                config),
            kernels::OperatorFamily::elementwise_inject,
            inject_shape,
            config,
            qualification));

    registry.add(
        make_elementwise_variant(
            elementwise_kernel_id(
                "moe_combine",
                request,
                config),
            kernels::OperatorFamily::moe_combine,
            combine_shape,
            config,
            qualification));
}

std::uint64_t
hip_elementwise_launcher_identity(
    std::uint64_t operation_tag,
    const HipElementwiseLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, operation_tag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.hc_count);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    return hash;
}

ExecutionLauncher
make_hip_hyper_expand_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config) {
    return make_launcher(
        std::move(kernel_id),
        hyper_expand_tag,
        &launch_hyper_expand,
        config);
}

ExecutionLauncher
make_hip_inject_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config) {
    return make_launcher(
        std::move(kernel_id),
        inject_tag,
        &launch_inject,
        config);
}

ExecutionLauncher
make_hip_moe_combine_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config) {
    return make_launcher(
        std::move(kernel_id),
        moe_combine_tag,
        &launch_moe_combine,
        config);
}

}  // namespace strix


namespace strix {
namespace {

enum class ElementwiseLauncherKind {
    hyper_expand,
    inject,
    moe_combine,
};

std::optional<ElementwiseLauncherKind>
elementwise_launcher_kind(
    SerialOp op) noexcept {
    switch (op) {
    case SerialOp::hyper_expand:
        return ElementwiseLauncherKind::hyper_expand;
    case SerialOp::attn_inject:
    case SerialOp::mlp_inject:
        return ElementwiseLauncherKind::inject;
    case SerialOp::moe_combine:
        return ElementwiseLauncherKind::moe_combine;
    default:
        return std::nullopt;
    }
}

}  // namespace

void HipElementwiseLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP launcher bundle requires a valid compiled execution");
    }

    struct Registered {
        std::string kernel_id;
        ElementwiseLauncherKind kind;
    };
    std::vector<Registered> registered;

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        const auto kind =
            elementwise_launcher_kind(
                step.op);
        if (!kind.has_value()) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP elementwise step is not bound to a kernel id");
        }

        const auto existing =
            std::find_if(
                registered.begin(),
                registered.end(),
                [&](const Registered& value) {
                    return value.kernel_id ==
                           *binding.kernel_id;
                });

        if (existing != registered.end()) {
            if (existing->kind != *kind) {
                throw std::invalid_argument(
                    "one kernel id is bound to incompatible HIP elementwise semantics");
            }
            continue;
        }

        ExecutionLauncher launcher;
        switch (*kind) {
        case ElementwiseLauncherKind::hyper_expand:
            launcher =
                make_hip_hyper_expand_launcher(
                    *binding.kernel_id,
                    &config_);
            break;
        case ElementwiseLauncherKind::inject:
            launcher =
                make_hip_inject_launcher(
                    *binding.kernel_id,
                    &config_);
            break;
        case ElementwiseLauncherKind::moe_combine:
            launcher =
                make_hip_moe_combine_launcher(
                    *binding.kernel_id,
                    &config_);
            break;
        }

        registry.add_kernel_launcher(
            std::move(launcher));
        registered.push_back(
            Registered{
                *binding.kernel_id,
                *kind,
            });
    }
}

}  // namespace strix
