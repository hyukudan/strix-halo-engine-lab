#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
#include "strix/hip_gdn_lm_wave8_diagnostic.hpp"
#endif
#include "strix/hip_launch_context.hpp"
#include "strix/hip_lm_head_launcher.hpp"

#include "strix/dispatch_resources.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x4c4d484541440001ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4NLBlock {
    __half d;
    std::uint8_t qs[16];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4NLBlock) == 18);
static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __constant__
std::int8_t iq4_codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

enum class WeightKind {
    f32,
    f16,
    bf16,
    q8_0,
    iq4_nl,
    iq4_xs,
};

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

void validate_config(
    const HipLmHeadLauncherConfig& config) {
    if (config.hidden_size == 0 ||
        config.vocab_size == 0) {
        throw std::invalid_argument(
            "HIP LM-head geometry must be non-zero");
    }
    if (config.input !=
            HipActivationStorage::bf16 &&
        config.input !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP LM-head input storage is unsupported");
    }
}

const char* storage_name(
    HipActivationStorage storage) noexcept {
    switch (storage) {
    case HipActivationStorage::bf16:
        return "bf16";
    case HipActivationStorage::fp32:
        return "fp32";
    }
    return "unknown";
}

WeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 20: return WeightKind::iq4_nl;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP LM head supports F32, F16, BF16, Q8_0, IQ4_NL and IQ4_XS");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ float load_activation(
    const float* value) {
    return *value;
}

__device__ float load_activation(
    const std::uint16_t* value) {
    return bf16_to_float(*value);
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U * (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return static_cast<std::uint8_t>(
        low |
        static_cast<std::uint8_t>(
            high << 4U));
}

template <WeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const float*>(weights);
    return values[
        static_cast<std::uint64_t>(row) *
            row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[
            static_cast<std::uint64_t>(row) *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[
            static_cast<std::uint64_t>(row) *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_nl>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4NLBlock*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
                blocks_per_row +
            feature / 32U];
    const auto within =
        feature % 32U;
    const auto packed =
        block.qs[within % 16U];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    return __half2float(block.d) *
           static_cast<float>(
               iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint32_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(row) *
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

    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(
               iq4_codebook[code]);
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t row_width,
    std::uint64_t rows) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                row_width) *
            rows * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                row_width) *
            rows * 2ULL;
    case WeightKind::q8_0:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 LM-head row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            rows *
            sizeof(Q8Block);
    case WeightKind::iq4_nl:
        if (row_width % 32U != 0) {
            throw std::invalid_argument(
                "IQ4_NL LM-head row width must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 32U) *
            rows *
            sizeof(IQ4NLBlock);
    case WeightKind::iq4_xs:
        if (row_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS LM-head row width must be divisible by 256");
        }
        return
            static_cast<std::uint64_t>(
                row_width / 256U) *
            rows *
            sizeof(IQ4XSBlock);
    }
    throw std::logic_error(
        "unreachable LM-head weight kind");
}

template <
    typename Input,
    WeightKind Kind>
__global__ void lm_head_kernel(
    const Input* hidden,
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t vocab_size,
    float* logits) {
    const auto vocab_row =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (vocab_row >= vocab_size) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < hidden_size;
         feature += kThreads) {
        sum +=
            load_activation(
                &hidden[feature]) *
            load_weight<Kind>(
                weights,
                hidden_size,
                vocab_row,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        logits[vocab_row] =
            scratch[0];
    }
}

// Emulate all 256 original leaves and their tree inside one 32-lane wave.
// Local register reduction MUST follow the five cross-lane stages.
template <typename Input>
__global__ __launch_bounds__(32) void virtual_q8_lm_head_kernel(
    const Input* hidden, const void* weights, std::uint32_t hidden_size,
    std::uint32_t vocab_size, float* logits) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    if (row >= vocab_size) return;
    float partial[8] = {};
    for (std::uint64_t chunk = 0; chunk < hidden_size; chunk += kThreads) {
#pragma unroll
        for (std::uint32_t local = 0; local < 8; ++local) {
            const auto feature = chunk + lane * 8ULL + local;
            if (feature < hidden_size)
                partial[local] += load_activation(&hidden[feature]) *
                    load_weight<WeightKind::q8_0>(weights, hidden_size, row,
                        static_cast<std::uint32_t>(feature));
        }
    }
    for (std::uint32_t distance = 16; distance > 0; distance >>= 1U) {
#pragma unroll
        for (std::uint32_t local = 0; local < 8; ++local) {
            const float partner = __shfl_down(partial[local], distance, 32);
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4];
        partial[1] += partial[5];
        partial[2] += partial[6];
        partial[3] += partial[7];
        partial[0] += partial[2];
        partial[1] += partial[3];
        partial[0] += partial[1];
        logits[row] = partial[0];
    }
}

#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
// Project-owned GDN prefetch8 load body, preserving the LM coefficient/leaf DAG.
template <typename Input, unsigned Waves>
__global__ __launch_bounds__(32U * Waves)
void prefetch8_wave_q8_lm_head_diagnostic_kernel(
    const Input* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * Waves + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        // The prelaunch Q8 width%32 gate proves all eight leaves are valid
        // together and remain inside one block. No tail load is speculative.
        if (base < input_width) {
            const auto feature = static_cast<std::uint32_t>(base);
            const auto& block = blocks[feature / 32U];
            Input bits[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                bits[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += load_activation(&bits[local]) *
                    (scale * static_cast<float>(codes[local]));
        }
    }
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto partner = __shfl_down(partial[local], distance, 32);
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = partial[0];
    }
}
#endif

template <
    typename Input,
    WeightKind Kind>
void launch_typed(
    const RuntimeTensorBinding& weights,
    const HipLmHeadLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& logits) {
    if (config.virtual_q8_wave) {
        if constexpr (Kind != WeightKind::q8_0) {
            throw std::invalid_argument("virtual-wave LM head supports Q8_0 weights only");
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(virtual_q8_lm_head_kernel<Input>),
                dim3(config.vocab_size), dim3(32), 0, hip_execution_stream(),
                reinterpret_cast<const Input*>(hidden.address),
                reinterpret_cast<const void*>(weights.address), config.hidden_size,
                config.vocab_size, reinterpret_cast<float*>(logits.address));
            check_hip(hipGetLastError(), "HIP virtual-wave Q8 LM-head launch");
            return;
        }
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            lm_head_kernel<
                Input,
                Kind>),
        dim3(config.vocab_size),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<const Input*>(
            hidden.address),
        reinterpret_cast<const void*>(
            weights.address),
        config.hidden_size,
        config.vocab_size,
        reinterpret_cast<float*>(
            logits.address));
    check_hip(
        hipGetLastError(),
        "HIP LM-head launch");
}

template <typename Input>
void launch_by_weight(
    WeightKind kind,
    const RuntimeTensorBinding& weights,
    const HipLmHeadLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const ExecutionBufferAddress& logits) {
    switch (kind) {
    case WeightKind::f32:
        launch_typed<Input, WeightKind::f32>(
            weights, config, hidden, logits);
        return;
    case WeightKind::f16:
        launch_typed<Input, WeightKind::f16>(
            weights, config, hidden, logits);
        return;
    case WeightKind::bf16:
        launch_typed<Input, WeightKind::bf16>(
            weights, config, hidden, logits);
        return;
    case WeightKind::q8_0:
        launch_typed<Input, WeightKind::q8_0>(
            weights, config, hidden, logits);
        return;
    case WeightKind::iq4_nl:
        launch_typed<Input, WeightKind::iq4_nl>(
            weights, config, hidden, logits);
        return;
    case WeightKind::iq4_xs:
        launch_typed<Input, WeightKind::iq4_xs>(
            weights, config, hidden, logits);
        return;
    }
    throw std::logic_error(
        "unreachable LM-head weight kind");
}

void launch_lm_head(
    const ExecutionDispatchContext& context,
    void* user_data) {
    if (context.op !=
        SerialOp::lm_head) {
        throw std::invalid_argument(
            "HIP LM-head launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1) {
        throw std::invalid_argument(
            "HIP LM-head expects one read and one write");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP LM-head v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP LM-head requires configuration");
    }

    const auto& config =
        *static_cast<
            HipLmHeadLauncherConfig*>(
                user_data);
    validate_config(config);

    const auto hidden =
        resolve_dispatch_read(
            context, 0);
    const auto logits =
        resolve_dispatch_write(
            context, 0);
    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP LM-head requires exactly one output tensor");
    }
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP LM-head requires model manifest resources");
    }

    const auto tensor_index =
        access.tensor_indices.front();
    if (tensor_index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP LM-head tensor index exceeds manifest");
    }

    const auto& manifest_tensor =
        context.resources->
            model_manifest->
            tensors[tensor_index];
    const auto& runtime_tensor =
        resolve_dispatch_tensor(
            context,
            tensor_index);

    if (manifest_tensor.role !=
            TensorRole::output_head ||
        manifest_tensor.dimensions.size() < 2 ||
        manifest_tensor.dimensions[0] !=
            config.hidden_size ||
        manifest_tensor.dimensions[1] !=
            config.vocab_size) {
        throw std::invalid_argument(
            "HIP LM-head tensor geometry does not match contract");
    }
    if (!runtime_tensor_gpu_visible(runtime_tensor.storage) ||
        runtime_tensor.address == 0) {
        throw std::invalid_argument(
            "HIP LM-head requires resident or repacked GPU-visible weights");
    }

    const auto kind =
        weight_kind(
            manifest_tensor.ggml_type);
    const auto required =
        required_weight_bytes(
            kind,
            config.hidden_size,
            config.vocab_size);
    if (runtime_tensor.bytes < required) {
        throw std::invalid_argument(
            "HIP LM-head runtime weight binding is undersized");
    }

    const auto input_required =
        static_cast<std::uint64_t>(
            config.hidden_size) *
        static_cast<std::uint32_t>(
            config.input);
    const auto output_required =
        static_cast<std::uint64_t>(
            config.vocab_size) *
        sizeof(float);
    if (hidden.bytes < input_required ||
        logits.bytes < output_required) {
        throw std::invalid_argument(
            "HIP LM-head scratch buffers are undersized");
    }

    switch (config.input) {
    case HipActivationStorage::bf16:
        launch_by_weight<
            std::uint16_t>(
                kind,
                runtime_tensor,
                config,
                hidden,
                logits);
        return;
    case HipActivationStorage::fp32:
        launch_by_weight<float>(
            kind,
            runtime_tensor,
            config,
            hidden,
            logits);
        return;
    }
    throw std::logic_error(
        "unreachable LM-head input storage");
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config) {
    return
        "hip.lm_head." +
        std::string(
            storage_name(
                config.input)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".v" +
        std::to_string(
            config.vocab_size) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.virtual_q8_wave ? ".virtual256.q8" : "");
}

std::uint64_t implementation_identity(
    const HipLmHeadLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.vocab_size);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.input));
    if (config.virtual_q8_wave) hash_u64(hash, 0x5649525451323536ULL);
    return hash;
}

}  // namespace

void register_hip_lm_head_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.hidden_size !=
            model.hidden_size ||
        config.vocab_size !=
            model.vocab_size) {
        throw std::invalid_argument(
            "HIP LM-head config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP LM-head v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP LM-head may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::lm_head;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::lm_head,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        config.virtual_q8_wave ? 32U : kThreads;
    variant.resources.lds_bytes =
        config.virtual_q8_wave ? 0U : kThreads * sizeof(float);
    variant.numerics.input_precision =
        storage_name(
            config.input);
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "fp32";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = true;
    variant.input_layout =
        "row-major";
    variant.output_layout =
        "fp32-logits";
    variant.compiler_identity =
        config.virtual_q8_wave ? "hip-gfx1151-lm-head-v2-virtual256-q8" : "hip-gfx1151-lm-head-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_lm_head_launcher(
    std::string kernel_id_value,
    HipLmHeadLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP LM-head launcher requires id and config");
    }
    validate_config(*config);
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_lm_head,
        config,
    };
}

void HipLmHeadLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP LM-head bundle requires valid compiled execution");
    }

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::lm_head) {
            continue;
        }
        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP LM-head step has no kernel binding");
        }
        ExecutionBindingRequest request;
        request.logical_rows = compiled.logical_rows;
        request.request_concurrency = compiled.request_concurrency;
        if (*binding.kernel_id != kernel_id(request, config_)) {
            throw std::invalid_argument(
                "default HIP LM-head cannot execute another numerical kernel/profile");
        }
        registry.add_kernel_launcher(
            make_hip_lm_head_launcher(
                *binding.kernel_id,
                &config_));
        return;
    }
    throw std::invalid_argument(
        "compiled execution has no LM-head step");
}


#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
namespace gdn_lm_wave8_diagnostic::detail {
const void* lm_symbol(Kind kind, Arm arm) {
    if (kind != Kind::lm_bf16) throw std::invalid_argument("invalid LM diagnostic kind");
    switch (arm) {
    case Arm::reference:
        return reinterpret_cast<const void*>(virtual_q8_lm_head_kernel<std::uint16_t>);
    case Arm::prefetch1:
        return reinterpret_cast<const void*>(prefetch8_wave_q8_lm_head_diagnostic_kernel<std::uint16_t, 1>);
    case Arm::wave8:
        return reinterpret_cast<const void*>(prefetch8_wave_q8_lm_head_diagnostic_kernel<std::uint16_t, 8>);
    }
    throw std::invalid_argument("invalid LM diagnostic arm");
}
void enqueue_lm(Shape shape, Span input, Span weights, Span output, Kind kind, Arm arm, hipStream_t stream) {
    const auto launch = validate(shape, input, weights, output, kind, arm);
    if (kind != Kind::lm_bf16) throw std::invalid_argument("invalid LM diagnostic kind");
    check_hip(hipPeekAtLastError(), "LM wave8 diagnostic stale error");
    if (arm == Arm::reference) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(virtual_q8_lm_head_kernel<std::uint16_t>),
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    } else if (arm == Arm::prefetch1) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_diagnostic_kernel<std::uint16_t, 1>),
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prefetch8_wave_q8_lm_head_diagnostic_kernel<std::uint16_t, 8>),
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    }
    check_hip(hipGetLastError(), "LM wave8 diagnostic launch");
}
} // namespace gdn_lm_wave8_diagnostic::detail
#endif

}  // namespace strix
