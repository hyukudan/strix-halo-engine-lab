#ifdef STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC
#include "strix/gdn_recurrent_layout_diagnostic.hpp"
#endif
#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
#include "strix/hip_gdn_lm_wave8_diagnostic.hpp"
#endif
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_gdn_islands.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/q8_projection_manifest.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/memory_plan.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/graph_operator_control.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kProjectionThreads = 256;
constexpr std::uint32_t kRecurrentThreads = 128;
constexpr std::uint32_t kValueTile = 32;
constexpr std::uint64_t kLauncherTag =
    0x47444e4c41554e32ULL;
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

enum class ProjectionMode { legacy256, virtual_stream, prefetch8, prefetch8_wave8 };

enum class SmallWeightKind : std::uint32_t {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct GdnTensors {
    TensorPair qkv;
    TensorPair z;
    TensorPair beta;
    TensorPair alpha;
    TensorPair conv;
    TensorPair dt;
    TensorPair a;
    TensorPair norm;
    TensorPair out;
};

struct Workspace {
    float* qkv = nullptr;
    float* z = nullptr;
    float* beta = nullptr;
    float* gate = nullptr;
    float* core = nullptr;
    std::uint64_t elements = 0;
};

__global__ void validate_gdn_graph_position(const std::uint64_t* position,
    std::uint64_t context_tokens, std::uint32_t* status) {
    if (*position >= context_tokens) atomicOr(status, 2U);
}
__global__ void validate_gdn_graph_input(const std::uint16_t* input, std::uint32_t hidden,
    std::uint32_t* status, std::uint32_t failure) {
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden; index += gridDim.x * blockDim.x) {
        const auto bits = static_cast<std::uint32_t>(input[index]) << 16U;
        if ((bits & 0x7f800000U) == 0x7f800000U) atomicOr(status, failure);
    }
}
__global__ void validate_gdn_graph_values(const float* values, std::uint64_t count, std::uint32_t* status) {
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridDim.x * blockDim.x) {
        if (!isfinite(values[i])) atomicOr(status, 8U);
    }
}

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

void hash_float(
    std::uint64_t& hash,
    float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(
        sizeof(bits) == sizeof(value));
    std::memcpy(
        &bits,
        &value,
        sizeof(bits));
    hash_u64(hash, bits);
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

std::uint32_t checked_u32(
    std::uint64_t value,
    const char* name) {
    if (value == 0 ||
        value >
            std::numeric_limits<
                std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) +
            " does not fit uint32");
    }
    return static_cast<std::uint32_t>(
        value);
}

void validate_config(
    const HipGdnLauncherConfig& config) {
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.prefetch_q8_input_projections && !config.virtual_q8_input_projections)
        throw std::invalid_argument("HIP GDN prefetch Q8 input requires virtual Q8 input");
    if (config.wave8_q8_input_projections) {
        q8_projection_organization::validate_modes(config.virtual_q8_input_projections,
            config.prefetch_q8_input_projections, true,
            config.activation == HipActivationStorage::bf16, config.model.hidden_size,
            config.model.gdn_value_heads * config.model.gdn_value_head_dim);
        if (config.enable_device_graph_inputs)
            throw std::invalid_argument("wave8 GDN input supports eager execution only");
    }
    if (config.wave8_q8_output_projection) {
        if (!config.virtual_q8_output_projection)
            throw std::invalid_argument("wave8 GDN output requires virtual Q8 output");
        if (config.enable_device_graph_inputs)
            throw std::invalid_argument("wave8 GDN output supports eager execution only");
        if (config.model.gdn_value_heads > 10240U / config.model.gdn_value_head_dim)
            throw std::invalid_argument("wave8 GDN output width outside admitted bounds");
        q8_projection_organization::validate_geometry(
            config.model.gdn_value_heads * config.model.gdn_value_head_dim, config.model.hidden_size);
    }
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP GDN v1 targets canonical BF16 activations only");
    }
    if (!config.model.gdn_qk_l2norm ||
        !config.model.gdn_conv_silu ||
        !config.model.gdn_output_gate_sigmoid) {
        throw std::invalid_argument(
            "HIP GDN v1 requires pinned Qwen4Exp GDN semantics");
    }
    if (config.model.gdn_key_head_dim !=
            kRecurrentThreads ||
        config.model.gdn_value_head_dim !=
            kRecurrentThreads ||
        config.model.gdn_key_head_dim !=
            config.model.gdn_value_head_dim ||
        config.model.gdn_value_heads %
                config.model.gdn_key_heads !=
            0 ||
        config.model.gdn_value_head_dim %
                kValueTile !=
            0) {
        throw std::invalid_argument(
            "HIP GDN v1 requires 128-wide equal/tileable key and value heads");
    }
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
            "HIP GDN projection supports F32, F16, BF16, Q8_0, IQ4_NL and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP GDN small tensors support F32, F16 and BF16");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ std::uint16_t float_to_bf16(
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        rounded >> 16U);
}

__device__ float round_bf16(
    float value) {
    return bf16_to_float(
        float_to_bf16(value));
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto low_byte =
        block.scales_l[subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (low_byte >>
             (4U *
              (subblock % 2U))) &
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
    std::uint64_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(
        weights)[
        row *
            row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return __half2float(
        static_cast<const __half*>(
            weights)[
            row *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return bf16_to_float(
        static_cast<
            const std::uint16_t*>(
                weights)[
            row *
                row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_nl>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4NLBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
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
    return __half2float(block.d) *
           static_cast<float>(
               iq4_codebook[code]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row *
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

__device__ float load_small(
    const void* values,
    SmallWeightKind kind,
    std::uint64_t index) {
    switch (kind) {
    case SmallWeightKind::f32:
        return static_cast<
            const float*>(values)[index];
    case SmallWeightKind::f16:
        return __half2float(
            static_cast<
                const __half*>(
                    values)[index]);
    case SmallWeightKind::bf16:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    values)[index]);
    }
    return 0.0F;
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input_width,
    std::uint64_t output_rows) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 2ULL;
    case WeightKind::q8_0:
    case WeightKind::iq4_nl:
        if (input_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0/IQ4_NL GDN projection width must divide 32");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 32U) *
            output_rows *
            (kind == WeightKind::q8_0
                 ? sizeof(Q8Block)
                 : sizeof(IQ4NLBlock));
    case WeightKind::iq4_xs:
        if (input_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS GDN projection width must divide 256");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    }
    throw std::logic_error(
        "unreachable GDN weight kind");
}

std::uint64_t small_weight_bytes(
    SmallWeightKind kind,
    std::uint64_t elements) {
    switch (kind) {
    case SmallWeightKind::f32:
        return elements * 4ULL;
    case SmallWeightKind::f16:
    case SmallWeightKind::bf16:
        return elements * 2ULL;
    }
    throw std::logic_error(
        "unreachable GDN small weight kind");
}


template <WeightKind Kind>
__global__ void project_bf16_to_fp32_kernel(
    const std::uint16_t* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kProjectionThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kProjectionThreads) {
        sum +=
            bf16_to_float(
                input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kProjectionThreads / 2U;
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
        output[out] =
            round_bf16(
                scratch[0]);
    }
}

// Eight ascending virtual leaves per physical lane preserve the original
// 256-leaf tree, including its operand order and BF16-in-FP32 epilogue.
__global__ __launch_bounds__(32)
void project_virtual_q8_bf16_to_fp32_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    float partial[8]{};
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto feature = chunk + 8U * lane + local;
            if (feature < input_width)
                partial[local] += bf16_to_float(input[feature]) *
                    load_weight<WeightKind::q8_0>(weights, input_width, row,
                        static_cast<std::uint32_t>(feature));
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
        output[row] = round_bf16(partial[0]);
    }
}

// Distinct opt-in load organization; existing virtual and legacy kernels
// remain unchanged. Actual ISA, not these arrays, qualifies load batching.
__global__ __launch_bounds__(32)
void project_prefetch8_q8_bf16_to_fp32_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
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
            std::uint16_t bits[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                bits[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += bf16_to_float(bits[local]) *
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
        output[row] = round_bf16(partial[0]);
    }
}
// Explicit geometry overlay of the incumbent prefetch8 body above.
__global__ __launch_bounds__(256)
void project_prefetch8_wave8_q8_bf16_to_fp32_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
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
            std::uint16_t bits[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                bits[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += bf16_to_float(bits[local]) *
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
        output[row] = round_bf16(partial[0]);
    }
}


template <WeightKind Kind>
__global__ void project_fp32_to_bf16_kernel(
    const float* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kProjectionThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kProjectionThreads) {
        sum +=
            input[feature] *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kProjectionThreads / 2U;
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
        output[out] =
            float_to_bf16(
                scratch[0]);
    }
}

// One wave simulates the same256 original leaves: virtual t=8*lane+j.
// Ascending per-leaf FMA chains and the128..1 tree are unchanged.
__global__ __launch_bounds__(32)
void project_virtual_q8_fp32_to_bf16_kernel(
    const float* input, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, std::uint16_t* output) {
    const auto row=static_cast<std::uint32_t>(blockIdx.x);
    if(row>=output_width) return;
    const auto lane=static_cast<std::uint32_t>(threadIdx.x);
    float partial[8]{};
    for(std::uint64_t chunk=0;chunk<input_width;chunk+=256U) {
#pragma unroll
        for(unsigned local=0;local<8;++local) {
            const auto feature=chunk+8U*lane+local;
            if(feature<input_width)
                partial[local]+=input[feature]*load_weight<WeightKind::q8_0>(
                    weights,input_width,row,static_cast<std::uint32_t>(feature));
        }
    }
#pragma unroll
    for(unsigned distance=16;distance>0;distance/=2) {
#pragma unroll
        for(unsigned local=0;local<8;++local) {
            const auto partner=__shfl_down(partial[local],distance,32);
            if(lane<distance) partial[local]+=partner;
        }
    }
    if(lane==0) {
        partial[0]+=partial[4];partial[1]+=partial[5];
        partial[2]+=partial[6];partial[3]+=partial[7];
        partial[0]+=partial[2];partial[1]+=partial[3];partial[0]+=partial[1];
        output[row]=float_to_bf16(partial[0]);
    }
}

// Actual incumbent virtual-output ISA uses lower/self as src0 at every tree
// site. Keep this new-only helper distinct from the input projection trees.
__device__ __forceinline__ float gdn_output_virtual_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}

// One complete wave owns one output. Eight ascending virtual leaves reproduce
// the incumbent256-leaf DAG and BF16 epilogue; waves share no mutable storage.
__global__ __launch_bounds__(256)
void project_prefetch8_wave8_q8_fp32_to_bf16_kernel(
    const float* input, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, std::uint16_t* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint64_t chunk = 0; chunk < input_width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        if (base < input_width) {
            const auto feature = static_cast<std::uint32_t>(base);
            const auto& block = blocks[feature / 32U];
            float values[8];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                values[local] = input[feature + local];
                codes[local] = block.values[feature % 32U + local];
            }
            const float scale = __half2float(block.scale);
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += values[local] * (scale * static_cast<float>(codes[local]));
        }
    }
    // The shuffled value is read before this stage overwrites dead upper lanes.
    // Lane0's lineage therefore preserves the original128..8 stages exactly.
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local)
            partial[local] = gdn_output_virtual_tree_add(partial[local], __shfl_down(partial[local], distance, 32));
    }
    if (lane == 0) {
        partial[0] = gdn_output_virtual_tree_add(partial[0], partial[4]);
        partial[1] = gdn_output_virtual_tree_add(partial[1], partial[5]);
        partial[2] = gdn_output_virtual_tree_add(partial[2], partial[6]);
        partial[3] = gdn_output_virtual_tree_add(partial[3], partial[7]);
        partial[0] = gdn_output_virtual_tree_add(partial[0], partial[2]);
        partial[1] = gdn_output_virtual_tree_add(partial[1], partial[3]);
        partial[0] = gdn_output_virtual_tree_add(partial[0], partial[1]);
        output[row] = float_to_bf16(partial[0]);
    }
}


// New-only rows2 arithmetic. The selected wave8 input ISA binds coefficient
// to src0 and BF16 activation to src1; its FP32 output counterpart reverses
// those two operands. Both selected trees bind lower/self to src0.
__device__ __forceinline__ float gdn_known_window_q8_coefficient(
    unsigned scale_bits, float code) {
    float result;
    asm volatile("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]"
                 : "=v"(result) : "v"(scale_bits), "v"(code));
    return result;
}
__device__ __forceinline__ float gdn_known_window_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}
template <bool Bf16Input>
__device__ __forceinline__ void gdn_known_window_q8_rows2_dot(
    const void* input0, const void* input1, const void* weights,
    std::uint32_t width, std::uint32_t out, float (&result)[2]) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[2][8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) +
        static_cast<std::uint64_t>(out) * (width / 32U);
    for (std::uint64_t chunk = 0; chunk < width; chunk += 256U) {
        const auto base = chunk + 8U * lane;
        if (base < width) {
            const auto& block = blocks[base / 32U];
            std::int8_t codes[8];
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local)
                codes[local] = block.values[base % 32U + local];
            const auto scale_bits = static_cast<unsigned>(__half_as_ushort(block.scale));
            float values[2][8];
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                if constexpr (Bf16Input) {
                    values[0][local] = bf16_to_float(static_cast<const std::uint16_t*>(input0)[base + local]);
                    values[1][local] = bf16_to_float(static_cast<const std::uint16_t*>(input1)[base + local]);
                } else {
                    values[0][local] = static_cast<const float*>(input0)[base + local];
                    values[1][local] = static_cast<const float*>(input1)[base + local];
                }
            }
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                // One decoded coefficient is reused by two independent leaf chains.
                const float coefficient = gdn_known_window_q8_coefficient(scale_bits,
                    static_cast<float>(codes[local]));
                if constexpr (Bf16Input) {
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[0][local])
                                 : "v"(coefficient), "v"(values[0][local]));
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[1][local])
                                 : "v"(coefficient), "v"(values[1][local]));
                } else {
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[0][local])
                                 : "v"(values[0][local]), "v"(coefficient));
                    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(partial[1][local])
                                 : "v"(values[1][local]), "v"(coefficient));
                }
            }
        }
    }
#pragma unroll
    for (unsigned distance = 16U; distance; distance /= 2U) {
#pragma unroll
        for (unsigned row = 0; row < 2U; ++row) {
#pragma unroll
            for (unsigned local = 0; local < 8U; ++local) {
                const float partner = __shfl_down(partial[row][local], distance, 32);
                if constexpr (Bf16Input) {
                    if (lane < distance)
                        partial[row][local] = gdn_known_window_tree_add(partial[row][local], partner);
                } else partial[row][local] = gdn_known_window_tree_add(partial[row][local], partner);
            }
        }
    }
    if (lane == 0U) {
#pragma unroll
        for (unsigned row = 0; row < 2U; ++row) {
            auto* values = partial[row];
            values[0] = gdn_known_window_tree_add(values[0], values[4]);
            values[1] = gdn_known_window_tree_add(values[1], values[5]);
            values[2] = gdn_known_window_tree_add(values[2], values[6]);
            values[3] = gdn_known_window_tree_add(values[3], values[7]);
            values[0] = gdn_known_window_tree_add(values[0], values[2]);
            values[1] = gdn_known_window_tree_add(values[1], values[3]);
            result[row] = gdn_known_window_tree_add(values[0], values[1]);
        }
    }
}
__global__ __launch_bounds__(256)
void project_wave8_q8_bf16_to_fp32_rows2_kernel(
    const std::uint16_t* input0, const std::uint16_t* input1, const void* weights,
    std::uint32_t width, std::uint32_t outputs, float* output0, float* output1) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (out >= outputs) return; // complete-wave guard, no block barriers
    float result[2]{};
    gdn_known_window_q8_rows2_dot<true>(input0, input1, weights, width, out, result);
    if ((threadIdx.x % 32U) == 0U) {
        output0[out] = round_bf16(result[0]); output1[out] = round_bf16(result[1]);
    }
}
__global__ __launch_bounds__(256)
void project_wave8_q8_fp32_to_bf16_rows2_kernel(
    const float* input0, const float* input1, const void* weights,
    std::uint32_t width, std::uint32_t outputs, std::uint16_t* output0, std::uint16_t* output1) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (out >= outputs) return;
    float result[2]{};
    gdn_known_window_q8_rows2_dot<false>(input0, input1, weights, width, out, result);
    if ((threadIdx.x % 32U) == 0U) {
        output0[out] = float_to_bf16(result[0]); output1[out] = float_to_bf16(result[1]);
    }
}

__global__ void conv_update_kernel(
    float* qkv,
    std::uint16_t* conv_state,
    const void* conv_weights,
    SmallWeightKind weight_kind_value,
    std::uint32_t channels,
    std::uint32_t kernel_size, std::uint32_t* graph_status) {
    if (graph_status != nullptr && *graph_status != 0) return;
    const auto channel =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (channel >= channels) {
        return;
    }

    auto* state =
        conv_state +
        static_cast<std::uint64_t>(
            channel) *
            kernel_size;

    for (std::uint32_t index = 0;
         index + 1U < kernel_size;
         ++index) {
        state[index] =
            state[index + 1U];
    }
    state[kernel_size - 1U] =
        float_to_bf16(
            qkv[channel]);

    float sum = 0.0F;
    for (std::uint32_t index = 0;
         index < kernel_size;
         ++index) {
        sum +=
            bf16_to_float(
                state[index]) *
            load_small(
                conv_weights,
                weight_kind_value,
                static_cast<std::uint64_t>(
                    channel) *
                    kernel_size +
                index);
    }
    const float silu =
        sum /
        (1.0F +
         expf(-sum));
    qkv[channel] =
        round_bf16(
            silu);
}

__device__ float softplus(
    float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return expf(value);
    }
    return log1pf(
        expf(value));
}

__global__ void transform_beta_gate_kernel(
    float* beta,
    float* gate,
    const void* dt,
    SmallWeightKind dt_kind,
    const void* a,
    SmallWeightKind a_kind,
    std::uint32_t heads) {
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (head >= heads) {
        return;
    }

    beta[head] =
        round_bf16(
            1.0F /
            (1.0F +
             expf(-beta[head])));

    const float biased =
        gate[head] +
        load_small(
            dt,
            dt_kind,
            head);
    gate[head] =
        load_small(
            a,
            a_kind,
            head) *
        softplus(
            biased);
}

__global__ void normalize_qk_kernel(
    float* qkv,
    std::uint32_t key_heads,
    std::uint32_t head_dim,
    float epsilon, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const bool is_key =
        block >= key_heads;
    const auto head =
        is_key
            ? block - key_heads
            : block;
    if (head >= key_heads ||
        threadIdx.x >= head_dim) {
        return;
    }

    __shared__ float scratch[kRecurrentThreads];

    const auto base =
        (is_key
             ? static_cast<std::uint64_t>(
                   key_heads) *
                   head_dim
             : 0ULL) +
        static_cast<std::uint64_t>(
            head) *
            head_dim;

    const float value =
        qkv[
            base +
            threadIdx.x];
    scratch[threadIdx.x] =
        value * value;
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
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

    const float inverse =
        rsqrtf(
            scratch[0] +
            epsilon);
    if (graph_status != nullptr && (!isfinite(scratch[0]) || !isfinite(inverse))) {
        if (threadIdx.x == 0) atomicOr(graph_status, 8U);
        return;
    }
    const float q_scale =
        is_key
            ? 1.0F
            : rsqrtf(
                  static_cast<float>(
                      head_dim));
    qkv[
        base +
        threadIdx.x] =
        value *
        inverse *
        q_scale;
}

__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_step_kernel(
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto tile_count =
        head_dim /
        kValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kRecurrentThreads *
        kValueTile];
    // The first reduction's broadcast must remain immutable while every wave
    // reads it. Reusing reduction[column] for the next tree lets lane zero
    // overwrite a delta before another wave has consumed it.
    __shared__ float deltas[kValueTile];

    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        retained[column] =
            state[state_index] *
            expf(
                gate[value_head]);
        reduction[
            static_cast<std::size_t>(
                lane) *
                kValueTile +
            column] =
            retained[column] *
            qkv[
                k_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            deltas[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[column]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto delta =
            deltas[column];
        retained[column] +=
            qkv[
                k_base +
                lane] *
            delta;
        reduction[
            static_cast<std::size_t>(
                lane) *
                kValueTile +
            column] =
            retained[column] *
            qkv[
                q_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[
                    static_cast<std::size_t>(
                        lane) *
                        kValueTile +
                    column] +=
                    reduction[
                        static_cast<std::size_t>(
                            lane + stride) *
                            kValueTile +
                        column];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[column];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        state[state_index] =
            retained[column];
    }
}

#ifdef STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC
// Only physical LDS addresses differ from the incumbent recurrence. Logical
// columns, arithmetic expressions, barriers and separate delta lifetime remain.
template <gdn_recurrent_layout_diagnostic::Layout Layout>
__device__ __forceinline__ constexpr std::size_t recurrent_layout_index(
    std::uint32_t lane, std::uint32_t column) {
    if constexpr (Layout == gdn_recurrent_layout_diagnostic::Layout::transposed)
        return static_cast<std::size_t>(column) * kRecurrentThreads + lane;
    else
        return static_cast<std::size_t>(lane) *
            (Layout == gdn_recurrent_layout_diagnostic::Layout::stride33 ? 33U : kValueTile) + column;
}

template <gdn_recurrent_layout_diagnostic::Layout Layout>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_layout_diagnostic_kernel(
    const float* qkv,
    const float* beta,
    const float* gate,
    float* state,
    float* core,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto tile_count =
        head_dim /
        kValueTile;
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto value_head =
        block /
        tile_count;
    const auto tile =
        block %
        tile_count;
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    if (value_head >= value_heads ||
        lane >= head_dim) {
        return;
    }

    const auto key_head =
        value_head %
        key_heads;
    const auto value_base =
        tile *
        kValueTile;

    const auto q_base =
        static_cast<std::uint64_t>(
            key_head) *
        head_dim;
    const auto k_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim +
        q_base;
    const auto v_base =
        static_cast<std::uint64_t>(
            key_heads) *
            head_dim *
            2ULL +
        static_cast<std::uint64_t>(
            value_head) *
            head_dim;

    __shared__ float reduction[
        kRecurrentThreads *
        (Layout == gdn_recurrent_layout_diagnostic::Layout::stride33 ? 33U : kValueTile)];
    // The first reduction's broadcast must remain immutable while every wave
    // reads it. Reusing reduction[column] for the next tree lets lane zero
    // overwrite a delta before another wave has consumed it.
    __shared__ float deltas[kValueTile];

    float retained[kValueTile];
#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        retained[column] =
            state[state_index] *
            expf(
                gate[value_head]);
        reduction[recurrent_layout_index<Layout>(lane, column)] =
            retained[column] *
            qkv[
                k_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[recurrent_layout_index<Layout>(lane, column)] +=
                    reduction[recurrent_layout_index<Layout>(lane + stride, column)];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            const auto value_index =
                value_base +
                column;
            deltas[column] =
                (qkv[
                     v_base +
                     value_index] -
                 reduction[recurrent_layout_index<Layout>(0U, column)]) *
                beta[value_head];
        }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto delta =
            deltas[column];
        retained[column] +=
            qkv[
                k_base +
                lane] *
            delta;
        reduction[recurrent_layout_index<Layout>(lane, column)] =
            retained[column] *
            qkv[
                q_base +
                lane];
    }
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
#pragma unroll
            for (std::uint32_t column = 0;
                 column < kValueTile;
                 ++column) {
                reduction[recurrent_layout_index<Layout>(lane, column)] +=
                    reduction[recurrent_layout_index<Layout>(lane + stride, column)];
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
#pragma unroll
        for (std::uint32_t column = 0;
             column < kValueTile;
             ++column) {
            core[
                static_cast<std::uint64_t>(
                    value_head) *
                    head_dim +
                value_base +
                column] =
                reduction[recurrent_layout_index<Layout>(0U, column)];
        }
    }

#pragma unroll
    for (std::uint32_t column = 0;
         column < kValueTile;
         ++column) {
        const auto value_index =
            value_base +
            column;
        const auto state_index =
            (static_cast<std::uint64_t>(
                 value_head) *
                 head_dim +
             lane) *
                head_dim +
            value_index;
        state[state_index] =
            retained[column];
    }
}

#endif

__global__ void gated_rms_norm_kernel(
    float* core,
    const float* z,
    const void* norm_weights,
    SmallWeightKind norm_kind,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float epsilon, std::uint32_t* graph_status) {
    __shared__ unsigned graph_reject;
    if (graph_status != nullptr) {
        if (threadIdx.x == 0) graph_reject = atomicAdd(graph_status, 0U) != 0;
        __syncthreads();
        if (graph_reject) return;
    }
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (head >= heads ||
        threadIdx.x >= head_dim) {
        return;
    }

    __shared__ float scratch[kRecurrentThreads];
    const auto index =
        static_cast<std::uint64_t>(
            head) *
            head_dim +
        threadIdx.x;
    const float value =
        core[index];
    scratch[threadIdx.x] =
        value * value;
    __syncthreads();

    for (std::uint32_t stride =
             kRecurrentThreads / 2U;
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

    const float inv =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (graph_status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv))) {
        if (threadIdx.x == 0) atomicOr(graph_status, 8U);
        return;
    }
    const float gate_value =
        1.0F /
        (1.0F +
         expf(-z[index]));
    core[index] =
        value *
        inv *
        load_small(
            norm_weights,
            norm_kind,
            threadIdx.x) *
        gate_value;
}

struct GdnKernelNode {
    const void* function;
    dim3 grid, block;
    std::size_t shared_bytes;
};
using GdnKernelLedger = std::vector<GdnKernelNode>;
struct GdnEmitTarget {
    hipStream_t stream;
    std::uint32_t* graph_status;
    const std::uint64_t* graph_position;
    std::uint64_t context_tokens;
    GdnKernelLedger* ledger;
};
void record_gdn_kernel(GdnKernelLedger* ledger, const void* function, dim3 grid, dim3 block) {
    if (ledger) ledger->push_back({function, grid, block, 0});
}

template <WeightKind Kind>
void launch_projection(
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output, hipStream_t stream, GdnKernelLedger* ledger) {
    record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_bf16_to_fp32_kernel<Kind>),
        dim3(output_width), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_bf16_to_fp32_kernel<
                Kind>),
        dim3(output_width),
        dim3(kProjectionThreads),
        0,
        stream,
        input,
        reinterpret_cast<const void*>(
            weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP GDN input projection launch");
}

void launch_projection_by_kind(
    WeightKind kind,
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output, ProjectionMode mode, hipStream_t stream, GdnKernelLedger* ledger) {
    switch (kind) {
    case WeightKind::f32:
        launch_projection<WeightKind::f32>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::f16:
        launch_projection<WeightKind::f16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::bf16:
        launch_projection<WeightKind::bf16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::q8_0:
        if (mode == ProjectionMode::prefetch8_wave8) {
            record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_prefetch8_wave8_q8_bf16_to_fp32_kernel), dim3((output_width + 7U) / 8U), dim3(256));
            hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel,
                dim3((output_width + 7U) / 8U), dim3(256), 0, stream,
                input, reinterpret_cast<const void*>(weights.address),
                input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP GDN prefetch8 wave8 Q8 input projection launch");
            return;
        }
        if (mode == ProjectionMode::prefetch8) {
            record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_prefetch8_q8_bf16_to_fp32_kernel), dim3(output_width), dim3(32));
            hipLaunchKernelGGL(project_prefetch8_q8_bf16_to_fp32_kernel,
                dim3(output_width), dim3(32), 0, stream,
                input, reinterpret_cast<const void*>(weights.address),
                input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP GDN prefetch8 Q8 input projection launch");
            return;
        }
        if (mode == ProjectionMode::virtual_stream) {
            record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_virtual_q8_bf16_to_fp32_kernel), dim3(output_width), dim3(32));
            hipLaunchKernelGGL(project_virtual_q8_bf16_to_fp32_kernel,
                dim3(output_width), dim3(32), 0, stream,
                input, reinterpret_cast<const void*>(weights.address),
                input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP GDN virtual Q8 input projection launch");
            return;
        }
        launch_projection<WeightKind::q8_0>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_nl:
        launch_projection<WeightKind::iq4_nl>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_xs:
        launch_projection<WeightKind::iq4_xs>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    }
}

template <WeightKind Kind>
void launch_output_projection(
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output, hipStream_t stream, GdnKernelLedger* ledger) {
    record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_fp32_to_bf16_kernel<Kind>),
        dim3(output_width), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_fp32_to_bf16_kernel<
                Kind>),
        dim3(output_width),
        dim3(kProjectionThreads),
        0,
        stream,
        input,
        reinterpret_cast<const void*>(
            weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP GDN output projection launch");
}

void launch_output_by_kind(
    WeightKind kind,
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint16_t* output, bool virtual_q8, bool wave8, hipStream_t stream, GdnKernelLedger* ledger) {
    if (wave8) {
        if (!virtual_q8 || kind != WeightKind::q8_0)
            throw std::invalid_argument("wave8 GDN output requires virtual Q8_0 weights");
        q8_projection_organization::validate_fp32_bf16_views(input_width, output_width,
            {reinterpret_cast<std::uintptr_t>(input), static_cast<std::uint64_t>(input_width) * sizeof(float)},
            {weights.address, weights.bytes},
            {reinterpret_cast<std::uintptr_t>(output), static_cast<std::uint64_t>(output_width) * sizeof(std::uint16_t)});
        record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_prefetch8_wave8_q8_fp32_to_bf16_kernel), dim3((output_width + 7U) / 8U), dim3(256));
        hipLaunchKernelGGL(project_prefetch8_wave8_q8_fp32_to_bf16_kernel, dim3((output_width + 7U) / 8U), dim3(256), 0,
            stream, input, reinterpret_cast<const void*>(weights.address), input_width, output_width, output);
        check_hip(hipGetLastError(), "HIP GDN prefetch8 wave8 Q8 output projection launch");
        return;
    }
    if(virtual_q8) {
        if(kind!=WeightKind::q8_0 || input_width%32U!=0U)
            throw std::invalid_argument("virtual Q8 GDN output requires aligned Q8_0 weights");
        record_gdn_kernel(ledger, reinterpret_cast<const void*>(project_virtual_q8_fp32_to_bf16_kernel), dim3(output_width), dim3(32));
        hipLaunchKernelGGL(project_virtual_q8_fp32_to_bf16_kernel,dim3(output_width),dim3(32),0,
            stream,input,reinterpret_cast<const void*>(weights.address),input_width,output_width,output);
        check_hip(hipGetLastError(),"HIP GDN virtual Q8 output projection launch");return;
    }
    switch (kind) {
    case WeightKind::f32:
        launch_output_projection<WeightKind::f32>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::f16:
        launch_output_projection<WeightKind::f16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::bf16:
        launch_output_projection<WeightKind::bf16>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::q8_0:
        launch_output_projection<WeightKind::q8_0>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_nl:
        launch_output_projection<WeightKind::iq4_nl>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    case WeightKind::iq4_xs:
        launch_output_projection<WeightKind::iq4_xs>(
            input, weights, input_width, output_width, output, stream, ledger);
        return;
    }
}

bool contains(
    std::string_view text,
    std::string_view needle) {
    return text.find(needle) !=
        std::string_view::npos;
}

TensorPair tensor_pair(
    const ExecutionDispatchContext& context,
    std::uint32_t index) {
    if (context.resources == nullptr ||
        context.resources->model_manifest ==
            nullptr) {
        throw std::invalid_argument(
            "HIP GDN requires model manifest resources");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP GDN tensor index exceeds manifest");
    }
    return TensorPair{
        &context.resources->
            model_manifest->
            tensors[index],
        &resolve_dispatch_tensor(
            context,
            index),
    };
}

void assign_unique(
    TensorPair& destination,
    std::uint32_t& count,
    TensorPair candidate) {
    destination =
        candidate;
    ++count;
}

GdnTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const HipGdnLauncherConfig& config) {
    const auto& recurrent =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& attention =
        resolve_dispatch_tensor_access(
            context, 1);

    GdnTensors result;
    std::uint32_t qkv_count = 0;
    std::uint32_t z_count = 0;
    std::uint32_t beta_count = 0;
    std::uint32_t alpha_count = 0;
    std::uint32_t conv_count = 0;
    std::uint32_t dt_count = 0;
    std::uint32_t a_count = 0;
    std::uint32_t norm_count = 0;
    std::uint32_t out_count = 0;

    for (const auto index :
         attention.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::attention_projection) {
            continue;
        }
        if (contains(
                tensor.name,
                "attn_qkv")) {
            assign_unique(
                result.qkv,
                qkv_count,
                pair);
        } else if (contains(
                       tensor.name,
                       "attn_gate")) {
            assign_unique(
                result.z,
                z_count,
                pair);
        }
    }

    for (const auto index :
         recurrent.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::recurrent_state) {
            continue;
        }
        const auto name =
            std::string_view(
                tensor.name);
        if (contains(
                name,
                "ssm_conv1d")) {
            assign_unique(
                result.conv,
                conv_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_beta")) {
            assign_unique(
                result.beta,
                beta_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_alpha")) {
            assign_unique(
                result.alpha,
                alpha_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_dt")) {
            assign_unique(
                result.dt,
                dt_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_norm")) {
            assign_unique(
                result.norm,
                norm_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_out")) {
            assign_unique(
                result.out,
                out_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_a") &&
                   !contains(
                       name,
                       "ssm_alpha")) {
            assign_unique(
                result.a,
                a_count,
                pair);
        } else if (contains(
                       name,
                       "ssm_in")) {
            throw std::invalid_argument(
                "HIP GDN v1 does not support legacy fused ssm_in export");
        }
    }

    if (qkv_count != 1 ||
        z_count != 1 ||
        beta_count != 1 ||
        alpha_count != 1 ||
        conv_count != 1 ||
        dt_count != 1 ||
        a_count != 1 ||
        norm_count != 1 ||
        out_count != 1) {
        throw std::invalid_argument(
            "HIP GDN tensor family is missing or ambiguous");
    }

    const auto hidden =
        config.model.hidden_size;
    const auto key_width =
        config.model.gdn_key_heads *
        config.model.gdn_key_head_dim;
    const auto value_width =
        config.model.gdn_value_heads *
        config.model.gdn_value_head_dim;
    const auto conv_width =
        key_width * 2ULL +
        value_width;

    const auto expect_matrix =
        [&](const TensorPair& pair,
            std::uint64_t input,
            std::uint64_t output,
            const char* name) {
            if (pair.manifest->
                    dimensions.size() < 2 ||
                pair.manifest->
                    dimensions[0] !=
                    input ||
                pair.manifest->
                    dimensions[1] !=
                    output) {
                throw std::invalid_argument(
                    std::string(name) +
                    " GDN tensor geometry mismatch");
            }
        };
    const auto expect_vector =
        [&](const TensorPair& pair,
            std::uint64_t elements,
            const char* name) {
            if (pair.manifest->
                    dimensions.empty() ||
                pair.manifest->
                    dimensions[0] !=
                    elements) {
                throw std::invalid_argument(
                    std::string(name) +
                    " GDN vector geometry mismatch");
            }
        };

    expect_matrix(
        result.qkv,
        hidden,
        conv_width,
        "qkv");
    expect_matrix(
        result.z,
        hidden,
        value_width,
        "z");
    expect_matrix(
        result.beta,
        hidden,
        config.model.gdn_value_heads,
        "beta");
    expect_matrix(
        result.alpha,
        hidden,
        config.model.gdn_value_heads,
        "alpha");
    expect_matrix(
        result.conv,
        config.model.gdn_conv_kernel,
        conv_width,
        "conv");
    expect_vector(
        result.dt,
        config.model.gdn_value_heads,
        "dt");
    expect_vector(
        result.a,
        config.model.gdn_value_heads,
        "A");
    expect_vector(
        result.norm,
        config.model.gdn_value_head_dim,
        "norm");
    expect_matrix(
        result.out,
        value_width,
        hidden,
        "out");

    return result;
}

void validate_projection_tensor(
    const TensorPair& pair,
    std::uint32_t input_width,
    std::uint64_t output_rows,
    const char* name) {
    if (pair.runtime == nullptr ||
        pair.manifest == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN projection is not GPU-visible");
    }
    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->
                    ggml_type),
            input_width,
            output_rows);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN projection binding is undersized");
    }
}

void validate_small_tensor(
    const TensorPair& pair,
    std::uint64_t elements,
    const char* name) {
    if (pair.runtime == nullptr ||
        pair.manifest == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN tensor is not GPU-visible");
    }
    const auto required =
        small_weight_bytes(
            small_weight_kind(
                pair.manifest->
                    ggml_type),
            elements);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " GDN tensor binding is undersized");
    }
}

Workspace workspace_for(
    const ExecutionBufferAddress& temporary,
    const ModelContract& model) {
    const auto key_width =
        model.gdn_key_heads *
        model.gdn_key_head_dim;
    const auto value_width =
        model.gdn_value_heads *
        model.gdn_value_head_dim;
    const auto conv_width =
        key_width * 2ULL +
        value_width;
    const auto elements =
        conv_width +
        value_width +
        model.gdn_value_heads * 2ULL +
        value_width;

    if (temporary.address == 0 ||
        temporary.bytes <
            elements *
                sizeof(float)) {
        throw std::invalid_argument(
            "HIP GDN temporary workspace is undersized");
    }

    auto* base =
        reinterpret_cast<float*>(
            temporary.address);
    Workspace result;
    result.qkv = base;
    result.z =
        result.qkv +
        conv_width;
    result.beta =
        result.z +
        value_width;
    result.gate =
        result.beta +
        model.gdn_value_heads;
    result.core =
        result.gate +
        model.gdn_value_heads;
    result.elements =
        elements;
    return result;
}

struct PreparedGdnLaunch {
    ExecutionBufferAddress input, output, temporary;
    Workspace workspace;
    GdnTensors tensors;
    SessionStateAddress matrix_state, conv_state;
    std::uint32_t hidden, key_heads, value_heads, head_dim, value_width, conv_width, kernel;
    ProjectionMode projection_mode;
    bool graph_capture;
};

PreparedGdnLaunch prepare_gdn_launch(
    const ExecutionDispatchContext& context, const HipGdnLauncherConfig* config_ptr) {
    if (context.op !=
        SerialOp::gated_delta_net) {
        throw std::invalid_argument(
            "HIP GDN launcher attached to wrong operator");
    }
    if (!context.layer.has_value() ||
        context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP GDN expects layer id, one read/write and one temporary");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP GDN v1 supports only C1 one-row decode");
    }
    if (config_ptr == nullptr) {
        throw std::invalid_argument(
            "HIP GDN requires configuration");
    }

    const auto& config = *config_ptr;
    validate_config(config);

    if (config.wave8_q8_input_projections || config.wave8_q8_output_projection) {
        if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
            throw std::invalid_argument("wave8 GDN projections support eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query wave8 GDN capture");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("wave8 GDN projections cannot be captured");
    }

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);
    const auto workspace =
        workspace_for(
            temporary,
            config.model);
    const auto tensors =
        identify_tensors(
            context,
            config);

    const auto hidden =
        checked_u32(
            config.model.hidden_size,
            "hidden_size");
    const auto key_heads =
        checked_u32(
            config.model.gdn_key_heads,
            "gdn_key_heads");
    const auto value_heads =
        checked_u32(
            config.model.gdn_value_heads,
            "gdn_value_heads");
    const auto head_dim =
        checked_u32(
            config.model.gdn_key_head_dim,
            "gdn_head_dim");
    const auto key_width =
        checked_u32(
            config.model.gdn_key_heads *
                config.model.gdn_key_head_dim,
            "gdn_key_width");
    const auto value_width =
        checked_u32(
            config.model.gdn_value_heads *
                config.model.gdn_value_head_dim,
            "gdn_value_width");
    const auto conv_width =
        checked_u32(
            static_cast<std::uint64_t>(
                key_width) *
                2ULL +
            value_width,
            "gdn_conv_width");
    const auto kernel =
        checked_u32(
            config.model.gdn_conv_kernel,
            "gdn_conv_kernel");

    const auto hidden_bytes =
        static_cast<std::uint64_t>(
            hidden) *
        sizeof(std::uint16_t);
    if (input.bytes <
            hidden_bytes ||
        output.bytes <
            hidden_bytes) {
        throw std::invalid_argument(
            "HIP GDN hidden buffers are undersized");
    }

    validate_projection_tensor(
        tensors.qkv,
        hidden,
        conv_width,
        "qkv");
    validate_projection_tensor(
        tensors.z,
        hidden,
        value_width,
        "z");
    validate_projection_tensor(
        tensors.beta,
        hidden,
        value_heads,
        "beta");
    validate_projection_tensor(
        tensors.alpha,
        hidden,
        value_heads,
        "alpha");
    // Keep the eight-feature dereference proof local to the opt-in path,
    // independently of required_weight_bytes' existing alignment check.
    if (config.prefetch_q8_input_projections && hidden % 32U != 0U &&
        (tensors.qkv.manifest->ggml_type == 8U || tensors.z.manifest->ggml_type == 8U ||
         tensors.beta.manifest->ggml_type == 8U || tensors.alpha.manifest->ggml_type == 8U))
        throw std::invalid_argument("HIP GDN prefetch8 Q8 input width must align to32");
    if (config.wave8_q8_input_projections) {
        if (tensors.qkv.manifest->ggml_type != 8U || tensors.z.manifest->ggml_type != 8U)
            throw std::invalid_argument("wave8 GDN input requires Q8_0 qkv and z weights");
        const auto validate = [&](const TensorPair& tensor, std::uint32_t rows, float* target) {
            if (tensor.manifest->ggml_type != 8U) return;
            q8_projection_organization::validate_views(hidden, rows,
                {input.address, input.bytes}, {tensor.runtime->address, tensor.runtime->bytes},
                {reinterpret_cast<std::uintptr_t>(target), static_cast<std::uint64_t>(rows) * sizeof(float)});
        };
        validate(tensors.qkv, conv_width, workspace.qkv);
        validate(tensors.z, value_width, workspace.z);
        validate(tensors.beta, value_heads, workspace.beta);
        validate(tensors.alpha, value_heads, workspace.gate);
    }
    const auto projection_mode = config.wave8_q8_input_projections ? ProjectionMode::prefetch8_wave8
        : config.prefetch_q8_input_projections ? ProjectionMode::prefetch8
        : config.virtual_q8_input_projections ? ProjectionMode::virtual_stream : ProjectionMode::legacy256;

    validate_projection_tensor(
        tensors.out,
        value_width,
        hidden,
        "out");
    if(config.virtual_q8_output_projection &&
        (tensors.out.manifest->ggml_type!=8U || value_width%32U!=0U))
        throw std::invalid_argument("virtual Q8 GDN output requires aligned Q8_0 weights");
    if (config.wave8_q8_output_projection) {
        const auto core_offset = q8_projection_organization::gdn_output_core_offset_bytes(config.model);
        const auto core_address = reinterpret_cast<std::uintptr_t>(workspace.core);
        const auto core_bytes = static_cast<std::uint64_t>(value_width) * sizeof(float);
        if (core_address < temporary.address || core_address - temporary.address != core_offset ||
            core_offset > temporary.bytes || core_bytes > temporary.bytes - core_offset)
            throw std::invalid_argument("wave8 GDN output core workspace mapping differs");
        q8_projection_organization::validate_fp32_bf16_views(value_width, hidden,
            {core_address, core_bytes}, {tensors.out.runtime->address, tensors.out.runtime->bytes},
            {output.address, output.bytes});
    }
    validate_small_tensor(
        tensors.conv,
        static_cast<std::uint64_t>(
            conv_width) *
            kernel,
        "conv");
    validate_small_tensor(
        tensors.dt,
        value_heads,
        "dt");
    validate_small_tensor(
        tensors.a,
        value_heads,
        "A");
    validate_small_tensor(
        tensors.norm,
        head_dim,
        "norm");

    const auto matrix_state =
        resolve_dispatch_gdn_layer_state(
            context,
            config.model,
            SessionStateComponent::
                recurrent_matrix);
    const auto conv_state =
        resolve_dispatch_gdn_layer_state(
            context,
            config.model,
            SessionStateComponent::
                gdn_convolution);
    if (matrix_state.storage !=
            SessionStateStorage::device ||
        conv_state.storage !=
            SessionStateStorage::device ||
        matrix_state.bytes <
            gdn_matrix_state_bytes_per_layer(
                config.model) ||
        conv_state.bytes <
            gdn_conv_state_bytes_per_layer(
                config.model)) {
        throw std::invalid_argument(
            "HIP GDN persistent state bindings are invalid");
    }

    if (const auto* graph = current_hip_launch_context(); graph && graph->graph_capture) {
        if (!config.enable_device_graph_inputs) {
            throw std::invalid_argument("GDN device graph inputs are not explicitly enabled");
        }
        const auto* layout = context.resources && context.resources->session_state_layout
            ? context.resources->session_state_layout : context.session_state_layout;
        if (!layout || graph->context_tokens != layout->context_tokens ||
            graph->context_tokens > config.model.max_position_embeddings) {
            throw std::invalid_argument("GDN graph context exceeds the model position limit");
        }
        GraphOperatorControlRequest request;
        request.execution_identity = context.execution_identity;
        request.context_tokens = graph->context_tokens;
        request.arguments = graph->arguments;
        request.status = graph->status;
        for (const auto* view : {&input, &output}) {
            request.resources.push_back({view->address, view->bytes, view->allocation_generation, 2});
        }
        request.resources.push_back({temporary.address, temporary.bytes, temporary.allocation_generation, 4});
        request.resources.push_back({matrix_state.address, matrix_state.bytes, matrix_state.allocation_generation, 4});
        request.resources.push_back({conv_state.address, conv_state.bytes, conv_state.allocation_generation, 2});
        for (const auto* tensor : {&tensors.qkv, &tensors.z, &tensors.beta, &tensors.alpha,
                                  &tensors.conv, &tensors.dt, &tensors.a, &tensors.norm, &tensors.out}) {
            request.resources.push_back({tensor->runtime->address, tensor->runtime->bytes,
                tensor->runtime->allocation_generation, tensor->manifest->ggml_type == 0 ? 4U : 2U});
        }
        (void)prepare_graph_operator_control(request);
    }
    return {input, output, temporary, workspace, tensors, matrix_state, conv_state,
            hidden, key_heads, value_heads, head_dim, value_width, conv_width, kernel,
            projection_mode, current_hip_launch_context() && current_hip_launch_context()->graph_capture};
}

void preflight_gdn(const ExecutionDispatchContext& context, void* user_data) {
    static_cast<void>(prepare_gdn_launch(context, static_cast<const HipGdnLauncherConfig*>(user_data)));
}

void emit_gdn_sequence(const PreparedGdnLaunch& prepared, const HipGdnLauncherConfig& config,
    const GdnEmitTarget& target) {
    const auto& input = prepared.input; const auto& output = prepared.output;
    const auto& workspace = prepared.workspace; const auto& tensors = prepared.tensors;
    const auto& matrix_state = prepared.matrix_state; const auto& conv_state = prepared.conv_state;
    const auto hidden = prepared.hidden, key_heads = prepared.key_heads, value_heads = prepared.value_heads;
    const auto head_dim = prepared.head_dim, value_width = prepared.value_width, conv_width = prepared.conv_width;
    const auto kernel = prepared.kernel;
    const auto projection_mode = prepared.projection_mode;
    const auto* hidden_input = reinterpret_cast<const std::uint16_t*>(input.address);
    if (prepared.graph_capture) {
        hipLaunchKernelGGL(validate_gdn_graph_position, dim3(1), dim3(1), 0, target.stream,
            target.graph_position, target.context_tokens, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN device position validation");
        hipLaunchKernelGGL(validate_gdn_graph_input, dim3(32), dim3(kProjectionThreads), 0, target.stream,
            hidden_input, hidden, target.graph_status, 4U);
        check_hip(hipGetLastError(), "HIP GDN finite input validation");
    }

    launch_projection_by_kind(
        weight_kind(
            tensors.qkv.manifest->
                ggml_type),
        hidden_input,
        *tensors.qkv.runtime,
        hidden,
        conv_width,
        workspace.qkv, projection_mode, target.stream, target.ledger);
    launch_projection_by_kind(
        weight_kind(
            tensors.z.manifest->
                ggml_type),
        hidden_input,
        *tensors.z.runtime,
        hidden,
        value_width,
        workspace.z, projection_mode, target.stream, target.ledger);
    launch_projection_by_kind(
        weight_kind(
            tensors.beta.manifest->
                ggml_type),
        hidden_input,
        *tensors.beta.runtime,
        hidden,
        value_heads,
        workspace.beta, projection_mode, target.stream, target.ledger);
    launch_projection_by_kind(
        weight_kind(
            tensors.alpha.manifest->
                ggml_type),
        hidden_input,
        *tensors.alpha.runtime,
        hidden,
        value_heads,
        workspace.gate, projection_mode, target.stream, target.ledger);

    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, workspace.qkv,
            static_cast<std::uint64_t>(conv_width) + value_width + value_heads * 2ULL, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN finite projection validation");
    }
    const auto conv_blocks =
        (conv_width +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(conv_update_kernel),
        dim3(conv_blocks), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        conv_update_kernel,
        dim3(conv_blocks),
        dim3(kProjectionThreads),
        0,
        target.stream,
        workspace.qkv,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<const void*>(
            tensors.conv.runtime->
                address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        conv_width,
        kernel, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN causal convolution launch");

    const auto parameter_blocks =
        (value_heads +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(transform_beta_gate_kernel),
        dim3(parameter_blocks), dim3(kProjectionThreads));
    hipLaunchKernelGGL(
        transform_beta_gate_kernel,
        dim3(parameter_blocks),
        dim3(kProjectionThreads),
        0,
        target.stream,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<const void*>(
            tensors.dt.runtime->
                address),
        small_weight_kind(
            tensors.dt.manifest->
                ggml_type),
        reinterpret_cast<const void*>(
            tensors.a.runtime->
                address),
        small_weight_kind(
            tensors.a.manifest->
                ggml_type),
        value_heads);
    check_hip(
        hipGetLastError(),
        "HIP GDN beta/gate transform launch");

    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(normalize_qk_kernel),
        dim3(key_heads * 2U), dim3(kRecurrentThreads));
    hipLaunchKernelGGL(
        normalize_qk_kernel,
        dim3(key_heads * 2U),
        dim3(kRecurrentThreads),
        0,
        target.stream,
        workspace.qkv,
        key_heads,
        head_dim,
        config.model.
            gdn_qk_l2_epsilon, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN q/k normalization launch");

    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, workspace.qkv,
            static_cast<std::uint64_t>(conv_width) + value_width + value_heads * 2ULL, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN finite recurrent input validation");
    }
    const auto tiles =
        head_dim /
        kValueTile;
    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(recurrent_step_kernel),
        dim3(value_heads * tiles), dim3(kRecurrentThreads));
    hipLaunchKernelGGL(
        recurrent_step_kernel,
        dim3(value_heads * tiles),
        dim3(kRecurrentThreads),
        0,
        target.stream,
        workspace.qkv,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<float*>(
            matrix_state.address),
        workspace.core,
        key_heads,
        value_heads,
        head_dim, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN recurrent update launch");
    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, workspace.core, value_width, target.graph_status);
        check_hip(hipGetLastError(), "HIP GDN finite recurrent output validation");
    }

    record_gdn_kernel(target.ledger, reinterpret_cast<const void*>(gated_rms_norm_kernel),
        dim3(value_heads), dim3(kRecurrentThreads));
    hipLaunchKernelGGL(
        gated_rms_norm_kernel,
        dim3(value_heads),
        dim3(kRecurrentThreads),
        0,
        target.stream,
        workspace.core,
        workspace.z,
        reinterpret_cast<const void*>(
            tensors.norm.runtime->
                address),
        small_weight_kind(
            tensors.norm.manifest->
                ggml_type),
        value_heads,
        head_dim,
        config.model.
            rms_norm_epsilon, target.graph_status);
    check_hip(
        hipGetLastError(),
        "HIP GDN gated RMSNorm launch");

    launch_output_by_kind(
        weight_kind(
            tensors.out.manifest->
                ggml_type),
        workspace.core,
        *tensors.out.runtime,
        value_width,
        hidden,
        reinterpret_cast<
            std::uint16_t*>(
                output.address),config.virtual_q8_output_projection,config.wave8_q8_output_projection, target.stream, target.ledger);
    if (target.graph_status != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_input, dim3(32), dim3(kProjectionThreads), 0,
            target.stream, reinterpret_cast<const std::uint16_t*>(output.address), hidden,
            target.graph_status, 8U);
        check_hip(hipGetLastError(), "HIP GDN finite final output validation");
    }
}

void launch_gdn(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_gdn_launch(context, static_cast<const HipGdnLauncherConfig*>(user_data));
    const auto* graph = current_hip_launch_context();
    emit_gdn_sequence(prepared, *static_cast<const HipGdnLauncherConfig*>(user_data),
        {hip_execution_stream(), hip_graph_status(), hip_graph_position(),
         graph ? graph->context_tokens : 0, nullptr});
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config) {
    return
        "hip.gdn.bf16.h" +
        std::to_string(
            config.model.hidden_size) +
        ".kh" +
        std::to_string(
            config.model.gdn_key_heads) +
        ".vh" +
        std::to_string(
            config.model.gdn_value_heads) +
        ".d" +
        std::to_string(
            config.model.gdn_key_head_dim) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (request.device_graph_inputs ? ".device" : ".scalar") +
        (config.wave8_q8_input_projections ? std::string(q8_projection_organization::gdn_wave8_marker) : "") +
        (config.wave8_q8_output_projection ? std::string(q8_projection_organization::gdn_output_wave8_marker) : "") +
        (config.prefetch_q8_input_projections ? ".prefetch8" : "") +
        (config.virtual_q8_input_projections ? ".virtual256.q8-input" : "") +
        (config.virtual_q8_output_projection ? ".virtual256.q8-output" : "");
}

std::uint64_t implementation_identity(
    const HipGdnLauncherConfig& config) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.enable_device_graph_inputs);
    hash_u64(hash, config.model.max_position_embeddings);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.gdn_key_heads);
    hash_u64(
        hash,
        config.model.gdn_key_head_dim);
    hash_u64(
        hash,
        config.model.gdn_value_heads);
    hash_u64(
        hash,
        config.model.gdn_value_head_dim);
    hash_u64(
        hash,
        config.model.gdn_conv_kernel);
    hash_float(
        hash,
        config.model.rms_norm_epsilon);
    hash_float(
        hash,
        config.model.gdn_qk_l2_epsilon);
    hash_u64(
        hash,
        config.model.gdn_output_gate_sigmoid
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_qk_l2norm
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_conv_silu
            ? 1ULL
            : 0ULL);
    if (config.virtual_q8_input_projections) hash_u64(hash, 0x47444e5138563235ULL);
    if (config.virtual_q8_output_projection) hash_u64(hash, 0x47444e514f555431ULL);
    if (config.prefetch_q8_input_projections) hash_u64(hash, 0x47444e5138504631ULL);
    if (config.wave8_q8_input_projections) hash_u64(hash, 0x47444e5138574156ULL);
    if (config.wave8_q8_output_projection) hash_u64(hash, 0x47444e514f573831ULL);
    return hash;
}


// R2 consumes the same complete pure host admission as the R1 factory.
using GdnKnownWindowRow = PreparedGdnLaunch;

void launch_gdn_known_window_core(const GdnKnownWindowRow& row,
    const HipGdnLauncherConfig& config) {
    const auto& workspace = row.workspace; const auto& tensors = row.tensors;
    const auto& matrix_state = row.matrix_state; const auto& conv_state = row.conv_state;
    const auto key_heads = row.key_heads, value_heads = row.value_heads, head_dim = row.head_dim;
    const auto value_width = row.value_width, conv_width = row.conv_width, kernel = row.kernel;
    const auto conv_blocks =
        (conv_width +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    hipLaunchKernelGGL(
        conv_update_kernel,
        dim3(conv_blocks),
        dim3(kProjectionThreads),
        0,
        hip_execution_stream(),
        workspace.qkv,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<const void*>(
            tensors.conv.runtime->
                address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        conv_width,
        kernel, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN causal convolution launch");

    const auto parameter_blocks =
        (value_heads +
         kProjectionThreads - 1U) /
        kProjectionThreads;
    hipLaunchKernelGGL(
        transform_beta_gate_kernel,
        dim3(parameter_blocks),
        dim3(kProjectionThreads),
        0,
        hip_execution_stream(),
        workspace.beta,
        workspace.gate,
        reinterpret_cast<const void*>(
            tensors.dt.runtime->
                address),
        small_weight_kind(
            tensors.dt.manifest->
                ggml_type),
        reinterpret_cast<const void*>(
            tensors.a.runtime->
                address),
        small_weight_kind(
            tensors.a.manifest->
                ggml_type),
        value_heads);
    check_hip(
        hipGetLastError(),
        "HIP GDN beta/gate transform launch");

    hipLaunchKernelGGL(
        normalize_qk_kernel,
        dim3(key_heads * 2U),
        dim3(kRecurrentThreads),
        0,
        hip_execution_stream(),
        workspace.qkv,
        key_heads,
        head_dim,
        config.model.
            gdn_qk_l2_epsilon, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN q/k normalization launch");

    if (hip_graph_status() != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            hip_execution_stream(), workspace.qkv,
            static_cast<std::uint64_t>(conv_width) + value_width + value_heads * 2ULL, hip_graph_status());
        check_hip(hipGetLastError(), "HIP GDN finite recurrent input validation");
    }
    const auto tiles =
        head_dim /
        kValueTile;
    hipLaunchKernelGGL(
        recurrent_step_kernel,
        dim3(value_heads * tiles),
        dim3(kRecurrentThreads),
        0,
        hip_execution_stream(),
        workspace.qkv,
        workspace.beta,
        workspace.gate,
        reinterpret_cast<float*>(
            matrix_state.address),
        workspace.core,
        key_heads,
        value_heads,
        head_dim, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN recurrent update launch");
    if (hip_graph_status() != nullptr) {
        hipLaunchKernelGGL(validate_gdn_graph_values, dim3(32), dim3(kProjectionThreads), 0,
            hip_execution_stream(), workspace.core, value_width, hip_graph_status());
        check_hip(hipGetLastError(), "HIP GDN finite recurrent output validation");
    }

    hipLaunchKernelGGL(
        gated_rms_norm_kernel,
        dim3(value_heads),
        dim3(kRecurrentThreads),
        0,
        hip_execution_stream(),
        workspace.core,
        workspace.z,
        reinterpret_cast<const void*>(
            tensors.norm.runtime->
                address),
        small_weight_kind(
            tensors.norm.manifest->
                ggml_type),
        value_heads,
        head_dim,
        config.model.
            rms_norm_epsilon, hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP GDN gated RMSNorm launch");

}


struct GdnKnownWindowRange { std::uintptr_t address; std::uint64_t bytes; };
void validate_gdn_known_window_range(GdnKnownWindowRange view, std::uint64_t alignment) {
    if (!view.address || view.address % alignment || !view.bytes ||
        view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
        throw std::invalid_argument("GDN known-window owner alignment or extent is invalid");
}
void require_gdn_known_window_disjoint(GdnKnownWindowRange a, GdnKnownWindowRange b) {
    if (a.address < b.address + b.bytes && b.address < a.address + a.bytes)
        throw std::invalid_argument("GDN known-window writable or row views alias");
}
std::array<GdnKnownWindowRow, 2> prepare_gdn_known_window_rows(
    std::span<const ExecutionDispatchContext, 2> rows, const HipGdnLauncherConfig& config) {
    validate_config(config);
    const auto& model = config.model;
    if (!config.virtual_q8_input_projections || !config.prefetch_q8_input_projections ||
        !config.wave8_q8_input_projections || !config.virtual_q8_output_projection ||
        !config.wave8_q8_output_projection || config.enable_device_graph_inputs ||
        model.hidden_size != 2560U || model.hidden_layers != 48U || model.full_attention_interval != 4U ||
        model.gdn_key_heads != 16U || model.gdn_value_heads != 48U ||
        model.gdn_key_head_dim != 128U || model.gdn_value_head_dim != 128U || model.gdn_conv_kernel != 4U)
        throw std::invalid_argument("GDN known-window R2 requires canonical wave8 Q8 input and output");
    const auto* scope = current_hip_launch_context();
    if (!scope || scope->graph_capture || !hip_execution_stream() || hip_graph_status() || hip_graph_position())
        throw std::invalid_argument("GDN known-window R2 requires one owned eager stream");
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture), "query GDN known-window R2 capture");
    if (capture != hipStreamCaptureStatusNone)
        throw std::invalid_argument("GDN known-window R2 cannot be captured");
    ExecutionBindingRequest selected;
    const auto expected_kernel = kernel_id(selected, config);
    const auto states = PersistentState::recurrent_matrix | PersistentState::convolution_history;
    for (const auto& context : rows) {
        if (!context.execution_identity || !context.arena || !context.resources || !context.resources->scratch ||
            (context.memory && context.memory != context.resources->scratch) ||
            context.prepared_tensor_views || context.prepared_arena_views ||
            !context.resources->session_state_layout || !context.resources->session_state_memory ||
            (context.session_state_layout && context.session_state_layout != context.resources->session_state_layout) ||
            (context.session_state_memory && context.session_state_memory != context.resources->session_state_memory))
            throw std::invalid_argument("GDN known-window R2 requires its current window resource bindings");
        validate_execution_memory_binding(context.execution_identity, *context.arena, *context.resources->scratch);
        if (!context.kernel_id || *context.kernel_id != expected_kernel ||
            context.reads_state != states || context.writes_state != states)
            throw std::invalid_argument("GDN known-window R2 selected serial variant or state flags differ");
        if (context.position >= context.resources->session_state_layout->context_tokens ||
            context.position >= model.max_position_embeddings)
            throw std::invalid_argument("GDN known-window R2 position exceeds capacity");
        // Validate whole supplied scratch views before workspace pointer arithmetic.
        if (context.reads.size() != 1U || context.writes.size() != 1U || context.temporaries.size() != 1U)
            throw std::invalid_argument("GDN known-window R2 requires exact row IO");
        for (const auto view : {resolve_dispatch_read(context, 0), resolve_dispatch_write(context, 0),
                               resolve_dispatch_temporary(context, 0)}) {
            validate_gdn_known_window_range({view.address, view.bytes}, 16U);
            const auto& owner = *context.resources->scratch;
            if (view.address < owner.arena_base || view.address - owner.arena_base > owner.arena_bytes ||
                view.bytes > owner.arena_bytes - (view.address - owner.arena_base))
                throw std::invalid_argument("GDN known-window row view exceeds its physical owner");
        }
    }
    const auto& a = rows[0]; const auto& b = rows[1];
    if (a.execution_identity != b.execution_identity || a.step_id != b.step_id || a.op != b.op || a.layer != b.layer ||
        a.reads[0] != b.reads[0] || a.writes[0] != b.writes[0] || a.temporaries[0] != b.temporaries[0] ||
        a.resources->scratch != b.resources->scratch ||
        a.resources->model_manifest != b.resources->model_manifest ||
        a.resources->model_weights != b.resources->model_weights || a.resources->tensor_access != b.resources->tensor_access ||
        a.resources->session_state_layout != b.resources->session_state_layout ||
        a.resources->session_state_memory != b.resources->session_state_memory ||
        a.position == std::numeric_limits<std::uint64_t>::max() || b.position != a.position + 1U)
        throw std::invalid_argument("GDN known-window R2 rows do not share one ordered execution");
    // Every incumbent row-level deep check completes before any kernel submission.
    std::array<GdnKnownWindowRow, 2> result{
        prepare_gdn_launch(a, &config), prepare_gdn_launch(b, &config)};
    const auto same_state = [](const SessionStateAddress& x, const SessionStateAddress& y) {
        return x.component == y.component && x.storage == y.storage && x.address == y.address &&
            x.bytes == y.bytes && x.allocation_generation == y.allocation_generation && x.provisional == y.provisional;
    };
    if (!same_state(result[0].matrix_state, result[1].matrix_state) ||
        !same_state(result[0].conv_state, result[1].conv_state))
        throw std::invalid_argument("GDN known-window R2 causal states differ between rows");
    const std::array<TensorPair GdnTensors::*, 9> members{
        &GdnTensors::qkv, &GdnTensors::z, &GdnTensors::beta, &GdnTensors::alpha, &GdnTensors::conv,
        &GdnTensors::dt, &GdnTensors::a, &GdnTensors::norm, &GdnTensors::out};
    std::array<GdnKnownWindowRange, 11> readonly{};
    readonly[0] = {result[0].input.address, result[0].input.bytes};
    readonly[1] = {result[1].input.address, result[1].input.bytes};
    require_gdn_known_window_disjoint(readonly[0], readonly[1]);
    for (std::size_t i = 0; i < members.size(); ++i) {
        const auto& x = result[0].tensors.*members[i]; const auto& y = result[1].tensors.*members[i];
        if (x.manifest != y.manifest || x.runtime != y.runtime)
            throw std::invalid_argument("GDN known-window R2 weights differ between rows");
        readonly[i + 2U] = {x.runtime->address, x.runtime->bytes};
        validate_gdn_known_window_range(readonly[i + 2U], x.manifest->ggml_type == 0U ? 4U : 2U);
    }
    for (const auto member : {&GdnTensors::qkv, &GdnTensors::z, &GdnTensors::beta, &GdnTensors::alpha, &GdnTensors::out})
        if ((result[0].tensors.*member).manifest->ggml_type != 8U)
            throw std::invalid_argument("GDN known-window R2 requires all five Q8_0 projections");
    const std::array<GdnKnownWindowRange, 6> writable{{
        {result[0].output.address, result[0].output.bytes}, {result[1].output.address, result[1].output.bytes},
        {result[0].temporary.address, result[0].temporary.bytes}, {result[1].temporary.address, result[1].temporary.bytes},
        {result[0].matrix_state.address, result[0].matrix_state.bytes},
        {result[0].conv_state.address, result[0].conv_state.bytes}}};
    for (std::size_t i = 0; i < writable.size(); ++i) {
        validate_gdn_known_window_range(writable[i], i == 5U ? 2U : 4U);
        for (const auto input : readonly) require_gdn_known_window_disjoint(writable[i], input);
        for (std::size_t j = 0; j < i; ++j) require_gdn_known_window_disjoint(writable[i], writable[j]);
    }
    check_hip(hipPeekAtLastError(), "GDN known-window R2 pre-enqueue stale error");
    return result;
}

}  // namespace


void gdn_known_window_r2_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config) {
    static_cast<void>(prepare_gdn_known_window_rows(rows, config));
}
void gdn_known_window_r2_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config) {
    // Repeat current-view validation at this public entry; preflight isn't a permit.
    const auto prepared = prepare_gdn_known_window_rows(rows, config);
    const auto& a = prepared[0]; const auto& b = prepared[1];
    const auto* input0 = reinterpret_cast<const std::uint16_t*>(a.input.address);
    const auto* input1 = reinterpret_cast<const std::uint16_t*>(b.input.address);
    const auto project = [&](const TensorPair& tensor, std::uint32_t outputs, float* out0, float* out1) {
        hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_rows2_kernel,
            dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, hip_execution_stream(),
            input0, input1, reinterpret_cast<const void*>(tensor.runtime->address),
            a.hidden, outputs, out0, out1);
        check_hip(hipGetLastError(), "GDN known-window R2 input projection launch");
    };
    project(a.tensors.qkv, a.conv_width, a.workspace.qkv, b.workspace.qkv);
    project(a.tensors.z, a.value_width, a.workspace.z, b.workspace.z);
    project(a.tensors.beta, a.value_heads, a.workspace.beta, b.workspace.beta);
    project(a.tensors.alpha, a.value_heads, a.workspace.gate, b.workspace.gate);
    // Causality is unchanged: the same stream finishes row0's state update
    // before row1 reads that matrix/convolution state. Workspace rows are disjoint.
    launch_gdn_known_window_core(a, config);
    launch_gdn_known_window_core(b, config);
    hipLaunchKernelGGL(project_wave8_q8_fp32_to_bf16_rows2_kernel,
        dim3((a.hidden + 7U) / 8U), dim3(kProjectionThreads), 0, hip_execution_stream(),
        a.workspace.core, b.workspace.core, reinterpret_cast<const void*>(a.tensors.out.runtime->address),
        a.value_width, a.hidden, reinterpret_cast<std::uint16_t*>(a.output.address),
        reinterpret_cast<std::uint16_t*>(b.output.address));
    check_hip(hipGetLastError(), "GDN known-window R2 output projection launch");
}


#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
void gdn_known_window_r2_projection_probe(GdnKnownWindowProjectionDirection direction,
    GdnKnownWindowProjectionMode mode, std::span<const GdnKnownWindowProjectionView, 2> input,
    GdnKnownWindowProjectionView weights, std::span<const GdnKnownWindowProjectionView, 2> output,
    std::uint32_t width, std::uint32_t outputs, void* stream) {
    if ((direction != GdnKnownWindowProjectionDirection::bf16_to_fp32 &&
         direction != GdnKnownWindowProjectionDirection::fp32_to_bf16) ||
        (mode != GdnKnownWindowProjectionMode::incumbent_row_loop && mode != GdnKnownWindowProjectionMode::rows2) ||
        !stream || static_cast<hipStream_t>(stream) == hipStreamPerThread || static_cast<hipStream_t>(stream) == hipStreamLegacy)
        throw std::invalid_argument("GDN known-window projection diagnostic mode or stream is invalid");
    const bool bf16 = direction == GdnKnownWindowProjectionDirection::bf16_to_fp32;
    for (unsigned row = 0; row < 2U; ++row) {
        const q8_projection_organization::View x{input[row].address, input[row].bytes};
        const q8_projection_organization::View w{weights.address, weights.bytes};
        const q8_projection_organization::View y{output[row].address, output[row].bytes};
        if (bf16) q8_projection_organization::validate_views(width, outputs, x, w, y);
        else q8_projection_organization::validate_fp32_bf16_views(width, outputs, x, w, y);
    }
    for (unsigned row = 0; row < 2U; ++row) {
        const GdnKnownWindowRange write{output[row].address, output[row].bytes};
        for (const auto read : input) require_gdn_known_window_disjoint(write, {read.address, read.bytes});
        require_gdn_known_window_disjoint(write, {weights.address, weights.bytes});
    }
    require_gdn_known_window_disjoint({output[0].address, output[0].bytes}, {output[1].address, output[1].bytes});
    const auto native_stream = static_cast<hipStream_t>(stream);
    const auto* packed = reinterpret_cast<const void*>(weights.address);
    check_hip(hipPeekAtLastError(), "GDN known-window diagnostic pre-enqueue stale error");
    if (mode == GdnKnownWindowProjectionMode::incumbent_row_loop) {
        for (unsigned row = 0; row < 2U; ++row) {
            if (bf16) {
                hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel,
                    dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
                    reinterpret_cast<const std::uint16_t*>(input[row].address), packed, width, outputs,
                    reinterpret_cast<float*>(output[row].address));
            } else {
                hipLaunchKernelGGL(project_prefetch8_wave8_q8_fp32_to_bf16_kernel,
                    dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
                    reinterpret_cast<const float*>(input[row].address), packed, width, outputs,
                    reinterpret_cast<std::uint16_t*>(output[row].address));
            }
            check_hip(hipGetLastError(), "GDN known-window diagnostic incumbent projection");
        }
    } else if (bf16) {
        hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_rows2_kernel,
            dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
            reinterpret_cast<const std::uint16_t*>(input[0].address), reinterpret_cast<const std::uint16_t*>(input[1].address),
            packed, width, outputs, reinterpret_cast<float*>(output[0].address), reinterpret_cast<float*>(output[1].address));
        check_hip(hipGetLastError(), "GDN known-window diagnostic R2 input projection");
    } else {
        hipLaunchKernelGGL(project_wave8_q8_fp32_to_bf16_rows2_kernel,
            dim3((outputs + 7U) / 8U), dim3(kProjectionThreads), 0, native_stream,
            reinterpret_cast<const float*>(input[0].address), reinterpret_cast<const float*>(input[1].address),
            packed, width, outputs, reinterpret_cast<std::uint16_t*>(output[0].address), reinterpret_cast<std::uint16_t*>(output[1].address));
        check_hip(hipGetLastError(), "GDN known-window diagnostic R2 output projection");
    }
}
#endif

void register_hip_gdn_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    validate_config(config);
    if (request.wave8_q8_gdn_input != config.wave8_q8_input_projections)
        throw std::invalid_argument("GDN wave8 request/config mismatch");
    if (request.wave8_q8_gdn_output != config.wave8_q8_output_projection)
        throw std::invalid_argument("GDN wave8 output request/config mismatch");
    if ((config.wave8_q8_input_projections || config.wave8_q8_output_projection) && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("wave8 GDN input registers eager variants only");
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.gdn_key_heads !=
            model.gdn_key_heads ||
        config.model.gdn_key_head_dim !=
            model.gdn_key_head_dim ||
        config.model.gdn_value_heads !=
            model.gdn_value_heads ||
        config.model.gdn_value_head_dim !=
            model.gdn_value_head_dim ||
        config.model.gdn_conv_kernel !=
            model.gdn_conv_kernel ||
        config.model.rms_norm_epsilon !=
            model.rms_norm_epsilon ||
        config.model.gdn_qk_l2_epsilon !=
            model.gdn_qk_l2_epsilon ||
        config.model.gdn_output_gate_sigmoid !=
            model.gdn_output_gate_sigmoid ||
        config.model.gdn_qk_l2norm !=
            model.gdn_qk_l2norm ||
        config.model.gdn_conv_silu !=
            model.gdn_conv_silu) {
        throw std::invalid_argument(
            "HIP GDN config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP GDN v1 registers only C1 one-row variants");
    }
    if (request.device_graph_inputs && (!request.require_graph_safe || !config.enable_device_graph_inputs)) {
        throw std::invalid_argument("GDN device graph variant requires explicit graph input and graph-safe mode");
    }
    if (qualification ==
            kernels::QualificationStatus::
                target_qualified ||
        qualification ==
            kernels::QualificationStatus::
                production ||
        qualification ==
            kernels::QualificationStatus::
                rejected) {
        throw std::invalid_argument(
            "unmeasured HIP GDN may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            recurrent;
    variant.phase =
        kernels::ExecutionPhase::
            decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::gated_delta_net,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kProjectionThreads;
    variant.resources.lds_bytes =
        // Conservative 256-byte-aligned bound for both FP32 arrays and the
        // graph rejection broadcast, including possible compiler padding.
        ((kRecurrentThreads * kValueTile * sizeof(float) +
          kValueTile * sizeof(float) + sizeof(std::uint32_t) + 255U) / 256U) * 256U;
    variant.resources.scratch_bytes =
        build_serial_execution_buffer_plan(
            model,
            ExecutionBufferRequest{
                1,
                2,
                4,
                4,
                4,
                4,
            }).gdn_workspace_bytes;
    variant.numerics.input_precision =
        "bf16";
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = request.device_graph_inputs && config.enable_device_graph_inputs;
    variant.input_layout =
        "row-major";
    variant.output_layout =
        "row-major";
    variant.compiler_identity = config.prefetch_q8_input_projections ? "hip-gfx1151-gdn-v4-prefetch8-virtual256-q8-input"
        : config.virtual_q8_input_projections ? "hip-gfx1151-gdn-v3-virtual256-q8-input" : "hip-gfx1151-gdn-v2";
    if(config.virtual_q8_output_projection) variant.compiler_identity += "-virtual256-q8-output";
    if(config.wave8_q8_input_projections) variant.compiler_identity += ":wave8-input-v1";
    if(config.wave8_q8_output_projection) variant.compiler_identity += ":wave8-q8-output-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

namespace {
// The owner is immutable; these are full descriptor values, never hash keys.
struct GdnIslandView {
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0, generation = 0;
    bool operator==(const GdnIslandView&) const = default;
};
struct GdnIslandTensor {
    const RuntimeTensorBinding* binding = nullptr;
    std::size_t slot = 0;
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0, generation = 0;
    RuntimeTensorStorage storage = RuntimeTensorStorage::resident;
    std::uint32_t ggml_type = 0;
    std::uint32_t tensor_index = 0, owner_node = 0;
    bool operator==(const GdnIslandTensor&) const = default;
};
struct GdnIslandDescriptor {
    std::uint32_t step = 0, layer = 0;
    std::array<GdnIslandView, 3> buffers;
    std::array<std::uintptr_t, 5> workspace;
    std::array<GdnIslandTensor, 9> tensors;
    std::array<GdnIslandView, 2> state;
    std::array<SessionStateStorage, 2> state_storage;
    std::array<std::uint32_t, 7> dimensions;
    ProjectionMode projection_mode = ProjectionMode::legacy256;
    bool graph_capture = false;
    bool operator==(const GdnIslandDescriptor&) const = default;
};
std::array<TensorPair, 9> island_tensors(const GdnTensors& tensors) {
    return {tensors.qkv, tensors.z, tensors.beta, tensors.alpha, tensors.conv,
        tensors.dt, tensors.a, tensors.norm, tensors.out};
}
GdnIslandDescriptor island_descriptor(const ExecutionDispatchContext& context, const PreparedGdnLaunch& p,
    const ModelRuntimeBinding& actual, const GdnIslandDescriptor* expected = nullptr) {
    GdnIslandDescriptor d;
    d.step = context.step_id; d.layer = context.layer.value();
    d.buffers = {{{p.input.address, p.input.bytes, p.input.allocation_generation},
        {p.output.address, p.output.bytes, p.output.allocation_generation},
        {p.temporary.address, p.temporary.bytes, p.temporary.allocation_generation}}};
    d.workspace = {reinterpret_cast<std::uintptr_t>(p.workspace.qkv), reinterpret_cast<std::uintptr_t>(p.workspace.z),
        reinterpret_cast<std::uintptr_t>(p.workspace.beta), reinterpret_cast<std::uintptr_t>(p.workspace.gate),
        reinterpret_cast<std::uintptr_t>(p.workspace.core)};
    const auto tensors = island_tensors(p.tensors);
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        std::size_t slot = 0;
        if (expected) {
            slot = expected->tensors[i].slot;
            // Check the current slot address without dereferencing the recorded
            // pointer: vector replacement invalidates admission, not lifetime.
            if (slot >= actual.tensors.size() || &actual.tensors[slot] != expected->tensors[i].binding)
                throw std::invalid_argument("GDN island canonical weight owner slot changed");
        } else {
            // Setup only: the unprepared context must resolve the real owner.
            const auto found = std::find_if(actual.tensors.begin(), actual.tensors.end(),
                [&](const RuntimeTensorBinding& tensor) { return &tensor == t.runtime; });
            if (found == actual.tensors.end())
                throw std::invalid_argument("GDN island setup tensor lacks an actual owner slot");
            slot = static_cast<std::size_t>(found - actual.tensors.begin());
        }
        const auto& canonical = actual.tensors[slot];
        const auto& resolved = *t.runtime;
        // PreparedTensorViews owns a deep metadata copy. Normalize that pointer
        // only after every resolved runtime field matches the retained owner.
        if (resolved.tensor_index != canonical.tensor_index || resolved.storage != canonical.storage ||
            resolved.address != canonical.address || resolved.bytes != canonical.bytes ||
            resolved.allocation_generation != canonical.allocation_generation || resolved.owner_node != canonical.owner_node)
            throw std::invalid_argument("GDN island resolved weight differs from its actual canonical owner slot");
        d.tensors[i] = {&canonical, slot, resolved.address, resolved.bytes,
            resolved.allocation_generation, resolved.storage, t.manifest->ggml_type,
            resolved.tensor_index, resolved.owner_node};
    }
    d.state = {{{p.matrix_state.address, p.matrix_state.bytes, p.matrix_state.allocation_generation},
        {p.conv_state.address, p.conv_state.bytes, p.conv_state.allocation_generation}}};
    d.state_storage = {p.matrix_state.storage, p.conv_state.storage};
    d.dimensions = {p.hidden, p.key_heads, p.value_heads, p.head_dim, p.value_width, p.conv_width, p.kernel};
    d.projection_mode = p.projection_mode; d.graph_capture = p.graph_capture;
    return d;
}

bool same_island_config(const HipGdnLauncherConfig& a, const HipGdnLauncherConfig& b) noexcept {
    return a.model.vocab_size == b.model.vocab_size &&
        a.model.hidden_size == b.model.hidden_size &&
        a.model.max_position_embeddings == b.model.max_position_embeddings &&
        a.model.hidden_layers == b.model.hidden_layers &&
        a.model.full_attention_interval == b.model.full_attention_interval &&
        a.model.ple_layer_1based == b.model.ple_layer_1based &&
        a.model.ple_embed_dim == b.model.ple_embed_dim &&
        a.model.ple_conv_kernel_size == b.model.ple_conv_kernel_size &&
        a.model.ngram_size == b.model.ngram_size &&
        a.model.heads_per_ngram == b.model.heads_per_ngram &&
        a.model.ngram_vocab_size_base == b.model.ngram_vocab_size_base &&
        a.model.ngram_vocab_divisor == b.model.ngram_vocab_divisor &&
        a.model.ngram_hash_seed == b.model.ngram_hash_seed &&
        a.model.ple_eos_token_id == b.model.ple_eos_token_id &&
        a.model.hc_count == b.model.hc_count &&
        a.model.hc_lowrank == b.model.hc_lowrank &&
        a.model.full_attention_layers == b.model.full_attention_layers &&
        a.model.gdn_layers == b.model.gdn_layers &&
        a.model.attention_heads == b.model.attention_heads &&
        a.model.kv_heads == b.model.kv_heads &&
        a.model.head_dim == b.model.head_dim &&
        a.model.mtp_full_attention_layers == b.model.mtp_full_attention_layers &&
        a.model.mtp_use_dedicated_embeddings == b.model.mtp_use_dedicated_embeddings &&
        a.model.mtp_hidden_state_source_layer_explicit == b.model.mtp_hidden_state_source_layer_explicit &&
        a.model.gdn_key_heads == b.model.gdn_key_heads &&
        a.model.gdn_key_head_dim == b.model.gdn_key_head_dim &&
        a.model.gdn_value_heads == b.model.gdn_value_heads &&
        a.model.gdn_value_head_dim == b.model.gdn_value_head_dim &&
        a.model.gdn_conv_kernel == b.model.gdn_conv_kernel &&
        a.model.indexer_n_heads == b.model.indexer_n_heads &&
        a.model.indexer_kv_heads == b.model.indexer_kv_heads &&
        a.model.indexer_head_dim == b.model.indexer_head_dim &&
        a.model.indexer_budget == b.model.indexer_budget &&
        a.model.indexer_compress_ratio == b.model.indexer_compress_ratio &&
        std::bit_cast<std::uint32_t>(a.model.partial_rotary_factor) == std::bit_cast<std::uint32_t>(b.model.partial_rotary_factor) &&
        std::bit_cast<std::uint32_t>(a.model.rope_theta) == std::bit_cast<std::uint32_t>(b.model.rope_theta) &&
        a.model.moe_intermediate_size == b.model.moe_intermediate_size &&
        a.model.shared_expert_intermediate_size == b.model.shared_expert_intermediate_size &&
        a.model.num_experts == b.model.num_experts &&
        a.model.num_experts_per_token == b.model.num_experts_per_token &&
        std::bit_cast<std::uint32_t>(a.model.rms_norm_epsilon) == std::bit_cast<std::uint32_t>(b.model.rms_norm_epsilon) &&
        a.model.norm_topk_probability == b.model.norm_topk_probability &&
        a.model.gdn_output_gate_sigmoid == b.model.gdn_output_gate_sigmoid &&
        a.model.gdn_qk_l2norm == b.model.gdn_qk_l2norm &&
        std::bit_cast<std::uint32_t>(a.model.gdn_qk_l2_epsilon) == std::bit_cast<std::uint32_t>(b.model.gdn_qk_l2_epsilon) &&
        a.model.gdn_conv_silu == b.model.gdn_conv_silu &&
        a.model.bytes_bf16 == b.model.bytes_bf16 &&
        a.model.bytes_fp32 == b.model.bytes_fp32 &&
        a.activation == b.activation &&
        a.enable_device_graph_inputs == b.enable_device_graph_inputs &&
        a.virtual_q8_input_projections == b.virtual_q8_input_projections &&
        a.prefetch_q8_input_projections == b.prefetch_q8_input_projections &&
        a.wave8_q8_input_projections == b.wave8_q8_input_projections &&
        a.virtual_q8_output_projection == b.virtual_q8_output_projection &&
        a.wave8_q8_output_projection == b.wave8_q8_output_projection;
}

std::uint64_t island_organization_identity(const HipGdnLauncherConfig& config) noexcept {
    auto identity = implementation_identity(config);
    hash_u64(identity, 0x47444e49534c3031ULL); // GDNISL01: organization only.
    return identity;
}
bool same_island_memory(const ExecutionMemoryBinding& a, const ExecutionMemoryBinding& b) noexcept {
    return a.execution_identity == b.execution_identity && a.arena_base == b.arena_base &&
        a.arena_bytes == b.arena_bytes && a.allocation_generation == b.allocation_generation;
}
void require_island_d1_scope(const HipLaunchContext& scope, std::uint64_t identity) {
    const auto& record = scope.deferred_moe_status;
    if (scope.graph_capture || !scope.stream || scope.stream == hipStreamLegacy || scope.stream == hipStreamPerThread ||
        record.execution_identity != identity || !record.allocation_generation || !record.arena_base ||
        record.arena_base % alignof(std::uint32_t) || record.arena_bytes != 8 ||
        record.arena_base > std::numeric_limits<std::uintptr_t>::max() - 8U)
        throw std::invalid_argument("GDN islands require an owned eager D1 stream and bounded eight-byte record");
}
void validate_island_graph(hipGraph_t graph, const GdnKernelLedger& ledger) {
    if (ledger.size() != 10) throw std::runtime_error("GDN island emitter must record ten kernel nodes");
    std::size_t count = 0;
    check_hip(hipGraphGetNodes(graph, nullptr, &count), "GDN island node count");
    if (count != ledger.size()) throw std::runtime_error("GDN island must capture exactly ten nodes");
    std::vector<hipGraphNode_t> nodes(count);
    check_hip(hipGraphGetNodes(graph, nodes.data(), &count), "GDN island node inventory");
    if (count != ledger.size()) throw std::runtime_error("GDN island node inventory changed");
    std::vector<bool> used(count, false);
    hipGraphNode_t previous = nullptr;
    for (const auto& expected : ledger) {
        std::size_t selected = count;
        for (std::size_t i = 0; i < count; ++i) {
            if (used[i]) continue;
            std::size_t dependencies = 0;
            check_hip(hipGraphNodeGetDependencies(nodes[i], nullptr, &dependencies), "GDN island dependencies");
            if (dependencies != (previous ? 1U : 0U)) continue;
            if (previous) {
                hipGraphNode_t dependency = nullptr;
                check_hip(hipGraphNodeGetDependencies(nodes[i], &dependency, &dependencies), "GDN island dependency edge");
                if (dependencies != 1 || dependency != previous) continue;
            }
            if (selected != count) throw std::runtime_error("GDN island graph is not one linear chain");
            selected = i;
        }
        if (selected == count) throw std::runtime_error("GDN island graph has a missing linear dependency");
        hipGraphNodeType type{};
        check_hip(hipGraphNodeGetType(nodes[selected], &type), "GDN island node type");
        if (type != hipGraphNodeTypeKernel) throw std::runtime_error("GDN island captured a non-kernel node");
        hipKernelNodeParams parameters{};
        check_hip(hipGraphKernelNodeGetParams(nodes[selected], &parameters), "GDN island kernel parameters");
        if (parameters.func != expected.function || parameters.gridDim.x != expected.grid.x ||
            parameters.gridDim.y != expected.grid.y || parameters.gridDim.z != expected.grid.z ||
            parameters.blockDim.x != expected.block.x || parameters.blockDim.y != expected.block.y ||
            parameters.blockDim.z != expected.block.z || parameters.sharedMemBytes != expected.shared_bytes)
            throw std::runtime_error("GDN island kernel node differs from the emitter ledger");
        used[selected] = true; previous = nodes[selected];
    }
}
} // namespace

struct HipGdnIslands::Impl {
    struct Island {
        GdnIslandDescriptor descriptor;
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
    };
    // Retain actual owners until every graph is destroyed after quiescence.
    std::shared_ptr<const void> resources_owner;
    std::shared_ptr<const HipModelResidency> weights;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    const CompiledSerialExecution compiled;
    const HipGdnLauncherConfig config;
    const ExecutionRuntimeResources* resources;
    const ExecutionRuntimeResources resource_bindings;
    const HipLaunchContext scope;
    const HipGdnIslandBudget budget;
    std::vector<std::unique_ptr<Island>> steps;
    HipGdnIslandReceipt observed;
    hipStream_t setup_stream = nullptr;
    bool capturing = false, failed = false;
    hipGraph_t abandoned_graph = nullptr;
    ModelContentIdentity model_content;
    std::uint64_t manifest_identity = 0;
    std::uint64_t weight_generation = 0;

    Impl(const CompiledSerialExecution& c, const HipGdnLauncherConfig& cfg,
        const ExecutionRuntimeResources& r, const HipLaunchContext& s,
        std::shared_ptr<const HipModelResidency> w, std::shared_ptr<const void> owner,
        std::shared_ptr<const HipSetupQuiescenceGuard> g, HipGdnIslandBudget b)
        : resources_owner(std::move(owner)), weights(std::move(w)), guard(std::move(g)),
          compiled(c), config(cfg), resources(&r), resource_bindings(r), scope(s), budget(b) {}

    void finish_capture() noexcept {
        if (!capturing) return;
        hipGraph_t graph = nullptr;
        const auto error = hipStreamEndCapture(setup_stream, &graph);
        if (error == hipSuccess || error == hipErrorStreamCaptureInvalidated) capturing = false;
        if (graph && hipGraphDestroy(graph) != hipSuccess) abandoned_graph = graph;
    }
    bool destroy() noexcept {
        if (capturing) return false;
        bool okay = true;
        for (auto& island : steps) if (island) {
            if (island->executable) {
                if (hipGraphExecDestroy(island->executable) == hipSuccess) island->executable = nullptr;
                else okay = false;
            }
            if (island->graph) {
                if (hipGraphDestroy(island->graph) == hipSuccess) island->graph = nullptr;
                else okay = false;
            }
        }
        if (abandoned_graph) {
            if (hipGraphDestroy(abandoned_graph) == hipSuccess) abandoned_graph = nullptr;
            else okay = false;
        }
        if (setup_stream) {
            if (hipStreamDestroy(setup_stream) == hipSuccess) setup_stream = nullptr;
            else okay = false;
        }
        return okay;
    }
    void device_range(GdnIslandView range) const {
        if (!range.address || !range.bytes || !range.generation ||
            range.bytes > std::numeric_limits<std::uintptr_t>::max() - range.address)
            throw std::invalid_argument("GDN island has an invalid native resource range");
        hipPointerAttribute_t attributes{};
        check_hip(hipPointerGetAttributes(&attributes, reinterpret_cast<const void*>(range.address)), "GDN island device identity");
        if (attributes.type != hipMemoryTypeDevice || attributes.isManaged || attributes.device != guard->device_index())
            throw std::invalid_argument("GDN island scratch/state requires original-device allocations");
        hipDeviceptr_t allocation{}; std::size_t bytes = 0;
        check_hip(hipMemGetAddressRange(&allocation, &bytes, reinterpret_cast<hipDeviceptr_t>(range.address)),
            "GDN island native extent");
        const auto base = reinterpret_cast<std::uintptr_t>(allocation);
        if (range.address < base || range.address - base > bytes || range.bytes > bytes - (range.address - base))
            throw std::invalid_argument("GDN island resource exceeds its native allocation");
    }
    void build() {
        if (current_hip_launch_context()) throw std::invalid_argument("GDN island setup requires no active launch scope");
        validate_config(config);
        const bool legacy = !config.virtual_q8_input_projections && !config.prefetch_q8_input_projections &&
            !config.virtual_q8_output_projection;
        const bool profile = config.virtual_q8_input_projections && config.prefetch_q8_input_projections &&
            config.virtual_q8_output_projection;
        if (config.enable_device_graph_inputs || config.wave8_q8_input_projections || config.wave8_q8_output_projection ||
            (!legacy && !profile) ||
            compiled.device_graph_inputs || compiled.logical_rows != 1 || compiled.request_concurrency != 1 ||
            !compiled.execution_identity || !compiled.kernel_plan_identity ||
            compiled.execution.steps.size() != compiled.io.steps.size() ||
            compiled.execution.steps.size() != compiled.bindings.bindings.size())
            throw std::invalid_argument("GDN islands require a complete eager R1/C1 non-wave8 source");
        if (!resources_owner || !weights || !guard || &weights->binding() != resources->model_weights || !weights->generation())
            throw std::invalid_argument("GDN islands require the actual retained D1 and weight owners");
        require_island_d1_scope(scope, compiled.execution_identity);
        validate_execution_runtime_resources(compiled.execution_identity, compiled.arena, *resources, 0);
        model_content = weights->binding().model_content_identity;
        manifest_identity = weights->binding().manifest_layout_identity;
        weight_generation = weights->generation();
        std::uint32_t count = 0;
        std::vector<std::uint32_t> layers;
        ExecutionBindingRequest request;
        for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
            const auto& step = compiled.execution.steps[i];
            const auto& binding = compiled.bindings.bindings[i];
            const auto& io = compiled.io.steps[i];
            if (step.id != i || binding.step_id != step.id || binding.op != step.op || io.step_id != step.id || io.op != step.op)
                throw std::invalid_argument("GDN island source step/binding/IO must be dense and consistent");
            if (step.op != SerialOp::gated_delta_net) continue;
            if (!step.layer || binding.host_or_runtime_step || !binding.kernel_id ||
                *binding.kernel_id != kernel_id(request, config) ||
                std::find(layers.begin(), layers.end(), *step.layer) != layers.end())
                throw std::invalid_argument("GDN island source requires one exact selected kernel per GDN layer");
            layers.push_back(*step.layer);
            if (count == std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("GDN island count overflow");
            ++count;
        }
        if (!count || count > budget.max_islands || count > budget.max_kernel_nodes / 10U)
            throw std::invalid_argument("GDN island count exceeds explicit setup admission");
        steps.resize(compiled.execution.steps.size());
        observed.organization_identity = island_organization_identity(config);
        observed.islands = count; observed.kernel_nodes = count * 10U;
        const auto& scratch = *resources->scratch;
        const auto& state = *resources->session_state_memory;
        device_range({scratch.arena_base, scratch.arena_bytes, scratch.allocation_generation});
        device_range({state.device_base, state.device_bytes, state.device_allocation_generation});
        device_range({scope.deferred_moe_status.arena_base, 8, scope.deferred_moe_status.allocation_generation});
        const std::array<GdnIslandView, 3> writable_owners{{
            {scratch.arena_base, scratch.arena_bytes, scratch.allocation_generation},
            {state.device_base, state.device_bytes, state.device_allocation_generation},
            {scope.deferred_moe_status.arena_base, 8, scope.deferred_moe_status.allocation_generation}}};
        const auto overlaps = [](GdnIslandView a, GdnIslandView b) {
            return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
        };
        for (std::size_t i = 0; i < writable_owners.size(); ++i)
            for (std::size_t j = i + 1; j < writable_owners.size(); ++j)
                if (overlaps(writable_owners[i], writable_owners[j]))
                    throw std::invalid_argument("GDN island scratch, state and D1 record owners must be disjoint");
        for (const auto& tensor : weights->binding().tensors) {
            if (!runtime_tensor_gpu_visible(tensor.storage)) continue;
            if (!tensor.address || !tensor.bytes || tensor.bytes > std::numeric_limits<std::uintptr_t>::max() - tensor.address)
                throw std::invalid_argument("GDN island weight extent is invalid");
            for (const auto& owner : writable_owners)
                if (overlaps(owner, {tensor.address, tensor.bytes, tensor.allocation_generation}))
                    throw std::invalid_argument("GDN island writable owner aliases weights");
        }
        check_hip(hipStreamCreateWithFlags(&setup_stream, hipStreamNonBlocking), "GDN island setup stream create");
        std::size_t before = 0, after = 0, total = 0;
        const bool before_observed = hipMemGetInfo(&before, &total) == hipSuccess;
        try {
            for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
                const auto& step = compiled.execution.steps[i];
                if (step.op != SerialOp::gated_delta_net) continue;
                const auto& io = compiled.io.steps[i];
                ExecutionDispatchContext context;
                context.execution_identity = compiled.execution_identity;
                context.step_id = step.id; context.op = step.op; context.layer = step.layer;
                context.logical_rows = 1; context.request_concurrency = 1;
                context.reads = io.reads; context.writes = io.writes; context.temporaries = io.temporaries;
                context.arena = &compiled.arena; context.memory = resources->scratch; context.resources = resources;
                context.session_state_layout = resources->session_state_layout;
                context.session_state_memory = resources->session_state_memory;
                context.reads_state = step.reads_state; context.writes_state = step.writes_state;
                context.kernel_id = *compiled.bindings.bindings[i].kernel_id;
                const auto prepared = prepare_gdn_launch(context, &config);
                auto island = std::make_unique<Island>();
                island->descriptor = island_descriptor(context, prepared, weights->binding());
                for (const auto& buffer : island->descriptor.buffers) device_range(buffer);
                for (const auto& view : island->descriptor.state) device_range(view);
                for (const auto& tensor : island->descriptor.tensors) {
                    if ((tensor.storage != RuntimeTensorStorage::resident && tensor.storage != RuntimeTensorStorage::repacked_cache &&
                         tensor.storage != RuntimeTensorStorage::registered_file_mapped) ||
                        tensor.generation != weights->generation() || !weights->covers_device_range(tensor.address, tensor.bytes))
                        throw std::invalid_argument("GDN island tensor is outside its actual retained residency");
                }
                steps[i] = std::move(island);
                GdnKernelLedger ledger; ledger.reserve(10);
                check_hip(hipStreamBeginCapture(setup_stream, hipStreamCaptureModeThreadLocal), "GDN island capture begin");
                capturing = true;
                hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
                check_hip(hipStreamIsCapturing(setup_stream, &capture), "GDN island capture state");
                if (capture != hipStreamCaptureStatusActive || current_hip_launch_context())
                    throw std::runtime_error("GDN island capture requires the explicit setup target without a launch scope");
                emit_gdn_sequence(prepared, config, {setup_stream, nullptr, nullptr, 0, &ledger});
                const auto ended = hipStreamEndCapture(setup_stream, &steps[i]->graph);
                if (ended == hipSuccess || ended == hipErrorStreamCaptureInvalidated) capturing = false;
                check_hip(ended, "GDN island capture end");
                validate_island_graph(steps[i]->graph, ledger);
                check_hip(hipGraphInstantiate(&steps[i]->executable, steps[i]->graph, nullptr, nullptr, 0), "GDN island instantiate");
            }
        } catch (...) { finish_capture(); throw; }
        check_hip(hipStreamDestroy(setup_stream), "GDN island setup stream destroy"); setup_stream = nullptr;
        const bool after_observed = hipMemGetInfo(&after, &total) == hipSuccess;
        if (before_observed && after_observed && before <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) &&
            after <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            observed.memory_observed = true;
            observed.observed_setup_free_delta_bytes = static_cast<std::int64_t>(before) - static_cast<std::int64_t>(after);
            if (budget.observed_setup_delta_reject_bytes && observed.observed_setup_free_delta_bytes > 0 &&
                static_cast<std::uint64_t>(observed.observed_setup_free_delta_bytes) > budget.observed_setup_delta_reject_bytes)
                throw std::invalid_argument("GDN island observed setup memory delta exceeds explicit rejection threshold");
        }
    }
    Island& admit(const ExecutionDispatchContext& context, const PreparedGdnLaunch& prepared) {
        if (failed) throw std::logic_error("GDN island owner is failed and cannot be reused");
        const auto* active = current_hip_launch_context();
        if (!active || active->graph_capture || active->stream != scope.stream ||
            !same_island_memory(active->deferred_moe_status, scope.deferred_moe_status) ||
            context.execution_identity != compiled.execution_identity || context.resources != resources ||
            resources->scratch != resource_bindings.scratch || resources->model_manifest != resource_bindings.model_manifest ||
            resources->tensor_access != resource_bindings.tensor_access ||
            resources->session_state_layout != resource_bindings.session_state_layout ||
            resources->session_state_memory != resource_bindings.session_state_memory ||
            resources->model_weights != &weights->binding() || weights->binding().model_content_identity != model_content ||
            weights->binding().manifest_layout_identity != manifest_identity || weights->generation() != weight_generation ||
            context.step_id >= steps.size() || !steps[context.step_id] ||
            !(island_descriptor(context, prepared, weights->binding(), &steps[context.step_id]->descriptor) ==
                steps[context.step_id]->descriptor))
            throw std::invalid_argument("GDN island replay scope, owner or complete descriptor changed");
        return *steps[context.step_id];
    }
    void registration(const CompiledSerialExecution& source, const HipGdnLauncherConfig& cfg) const {
        if (failed || source.execution_identity != compiled.execution_identity ||
            source.kernel_plan_identity != compiled.kernel_plan_identity || source.device_graph_inputs ||
            source.logical_rows != 1 || source.request_concurrency != 1 || !same_island_config(config, cfg) ||
            source.execution.steps.size() != compiled.execution.steps.size() ||
            source.bindings.bindings.size() != compiled.bindings.bindings.size())
            throw std::invalid_argument("GDN island registration source/config differs from its fixed owner");
        std::size_t count = 0;
        for (std::size_t i = 0; i < source.execution.steps.size(); ++i) {
            const auto& step = source.execution.steps[i];
            if (step.op != SerialOp::gated_delta_net) {
                if (steps[i]) throw std::invalid_argument("GDN island registration lost a GDN step");
                continue;
            }
            if (!steps[i] || step.id != i || !step.layer || *step.layer != steps[i]->descriptor.layer)
                throw std::invalid_argument("GDN island registration step/layer coverage changed");
            const auto& binding = source.bindings.bindings[i];
            const auto& original = compiled.bindings.bindings[i];
            if (binding.step_id != step.id || binding.op != step.op || binding.host_or_runtime_step ||
                binding.kernel_id != original.kernel_id)
                throw std::invalid_argument("GDN island registration selected numerical kernel changed");
            ++count;
        }
        if (count != observed.islands) throw std::invalid_argument("GDN island registration has incomplete coverage");
    }
};

HipGdnIslands::HipGdnIslands(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
std::shared_ptr<HipGdnIslands> HipGdnIslands::create(const CompiledSerialExecution& compiled,
    const HipGdnLauncherConfig& config, const ExecutionRuntimeResources& resources, const HipLaunchContext& scope,
    std::shared_ptr<const HipModelResidency> weights, std::shared_ptr<const void> owner,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard, HipGdnIslandBudget budget) {
    if (!weights || !owner || !guard) throw std::invalid_argument("GDN island setup requires strong actual owners");
    auto impl = std::make_unique<Impl>(compiled, config, resources, scope, std::move(weights), std::move(owner), guard, budget);
    try { guard->run([&] { impl->build(); }); }
    catch (...) {
        impl->finish_capture(); guard->recover_or_fail_stop();
        if (!impl->destroy()) (void)impl.release();
        throw;
    }
    return std::shared_ptr<HipGdnIslands>(new HipGdnIslands(std::move(impl)));
}
HipGdnIslands::~HipGdnIslands() {
    if (impl_) {
        impl_->guard->recover_or_fail_stop();
        if (!impl_->destroy()) (void)impl_.release();
    }
}
HipGdnIslandReceipt HipGdnIslands::receipt() const noexcept { return impl_->observed; }
void HipGdnIslands::preflight_gdn_island(const ExecutionDispatchContext& context, void* opaque) {
    if (!opaque) throw std::invalid_argument("GDN island launcher lacks its owner");
    auto& owner = *static_cast<HipGdnIslands*>(opaque);
    try {
        const auto prepared = prepare_gdn_launch(context, &owner.impl_->config);
        (void)owner.impl_->admit(context, prepared);
    } catch (...) { owner.impl_->failed = true; throw; }
}
void HipGdnIslands::launch_gdn_island(const ExecutionDispatchContext& context, void* opaque) {
    if (!opaque) throw std::invalid_argument("GDN island launcher lacks its owner");
    auto& owner = *static_cast<HipGdnIslands*>(opaque);
    try {
        const auto prepared = prepare_gdn_launch(context, &owner.impl_->config);
        auto& island = owner.impl_->admit(context, prepared);
        if (owner.impl_->observed.replays == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("GDN island replay counter overflow");
        check_hip(hipGraphLaunch(island.executable, owner.impl_->scope.stream), "GDN island replay");
        ++owner.impl_->observed.replays;
    } catch (...) { owner.impl_->failed = true; throw; }
}


ExecutionLauncher
make_hip_gdn_launcher(
    std::string kernel_id_value,
    HipGdnLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP GDN launcher requires id and config");
    }
    validate_config(*config);
    if (config->wave8_q8_input_projections || config->wave8_q8_output_projection ||
        kernel_id_value.find(q8_projection_organization::gdn_wave8_marker) != std::string::npos ||
        kernel_id_value.find(q8_projection_organization::gdn_output_wave8_marker) != std::string::npos) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("wave8 GDN launcher requires its exact eager C1 kernel id");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_gdn,
        config,
        &preflight_gdn,
    };
}

void HipGdnLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry, HipGdnIslands* islands) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP GDN bundle requires valid compiled execution");
    }

    if (islands) islands->impl_->registration(compiled, config_);
    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::gated_delta_net) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP GDN step has no kernel binding");
        }

        std::string_view selected_id=*binding.kernel_id;
        constexpr std::string_view output_suffix=".virtual256.q8-output";
        if(selected_id.ends_with(output_suffix)!=config_.virtual_q8_output_projection)
            throw std::invalid_argument("HIP GDN selected kernel disagrees with virtual Q8 output configuration");
        if(config_.virtual_q8_output_projection) selected_id.remove_suffix(output_suffix.size());
        if (selected_id.ends_with(".virtual256.q8-input") !=
            config_.virtual_q8_input_projections)
            throw std::invalid_argument("HIP GDN selected kernel disagrees with virtual Q8 configuration");
        if (selected_id.ends_with(".prefetch8.virtual256.q8-input") != config_.prefetch_q8_input_projections)
            throw std::invalid_argument("HIP GDN selected kernel disagrees with prefetch Q8 configuration");
        if (config_.wave8_q8_input_projections || config_.wave8_q8_output_projection ||
            selected_id.find(q8_projection_organization::gdn_wave8_marker) != std::string::npos ||
            selected_id.find(q8_projection_organization::gdn_output_wave8_marker) != std::string::npos) {
            ExecutionBindingRequest request;
            request.logical_rows = compiled.logical_rows;
            request.request_concurrency = compiled.request_concurrency;
            if (*binding.kernel_id != kernel_id(request, config_))
                throw std::invalid_argument("HIP GDN wave8 selected kernel disagrees with exact configuration");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            if (islands) {
                registry.add_kernel_launcher({registered_id, island_organization_identity(config_),
                    &HipGdnIslands::launch_gdn_island, islands, &HipGdnIslands::preflight_gdn_island});
            } else {
                registry.add_kernel_launcher(
                    make_hip_gdn_launcher(registered_id, &config_));
            }
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP GDN bundle requires one selected kernel id");
        }
    }
}


#ifdef STRIX_BUILD_GDN_LM_WAVE8_DIAGNOSTIC
namespace gdn_lm_wave8_diagnostic::detail {
const void* gdn_symbol(Arm arm) {
    switch (arm) {
    case Arm::reference: case Arm::prefetch1:
        return reinterpret_cast<const void*>(project_prefetch8_q8_bf16_to_fp32_kernel);
    case Arm::wave8:
        return reinterpret_cast<const void*>(project_prefetch8_wave8_q8_bf16_to_fp32_kernel);
    }
    throw std::invalid_argument("invalid GDN diagnostic arm");
}
void enqueue_gdn(Shape shape, Span input, Span weights, Span output, Arm arm, hipStream_t stream) {
    const auto launch = validate(shape, input, weights, output, Kind::gdn_bf16, arm);
    check_hip(hipPeekAtLastError(), "GDN wave8 diagnostic stale error");
    if (arm == Arm::wave8) {
        hipLaunchKernelGGL(project_prefetch8_wave8_q8_bf16_to_fp32_kernel,
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    } else {
        hipLaunchKernelGGL(project_prefetch8_q8_bf16_to_fp32_kernel,
            dim3(launch.grid_x), dim3(launch.block_x), 0, stream,
            reinterpret_cast<const std::uint16_t*>(input.address),
            reinterpret_cast<const void*>(weights.address), shape.width, shape.outputs,
            reinterpret_cast<float*>(output.address));
    }
    check_hip(hipGetLastError(), "GDN wave8 diagnostic launch");
}
} // namespace gdn_lm_wave8_diagnostic::detail
#endif

#ifdef STRIX_BUILD_GDN_RECURRENT_LAYOUT_DIAGNOSTIC
namespace gdn_recurrent_layout_diagnostic {
const void* kernel_symbol(Layout layout) {
    switch (layout) {
    case Layout::reference:
        return reinterpret_cast<const void*>(recurrent_step_kernel);
    case Layout::stride32:
        return reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<Layout::stride32>);
    case Layout::stride33:
        return reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<Layout::stride33>);
    case Layout::transposed:
        return reinterpret_cast<const void*>(recurrent_layout_diagnostic_kernel<Layout::transposed>);
    }
    throw std::invalid_argument("invalid GDN recurrent diagnostic layout");
}

void enqueue(Shape shape, const Bindings& bindings, Layout layout, hipStream_t stream) {
    const auto launch = validate(shape, bindings, layout);
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("GDN recurrent diagnostic requires an explicitly owned stream");
    check_hip(hipPeekAtLastError(), "GDN recurrent diagnostic stale error");
    const auto* qkv = reinterpret_cast<const float*>(bindings.qkv.address);
    const auto* beta = reinterpret_cast<const float*>(bindings.beta.address);
    const auto* gate = reinterpret_cast<const float*>(bindings.gate.address);
    auto* state = reinterpret_cast<float*>(bindings.state.address);
    auto* core = reinterpret_cast<float*>(bindings.core.address);
    auto* status = reinterpret_cast<std::uint32_t*>(bindings.graph_status.address);
#define STRIX_RECURRENT_LAYOUT_LAUNCH(kernel) \
    hipLaunchKernelGGL(kernel, dim3(launch.grid_x), dim3(launch.block_x), 0, stream, \
        qkv, beta, gate, state, core, shape.key_heads, shape.value_heads, shape.head_dim, status)
    switch (layout) {
    case Layout::reference: STRIX_RECURRENT_LAYOUT_LAUNCH(recurrent_step_kernel); break;
    case Layout::stride32:
        STRIX_RECURRENT_LAYOUT_LAUNCH(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<Layout::stride32>)); break;
    case Layout::stride33:
        STRIX_RECURRENT_LAYOUT_LAUNCH(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<Layout::stride33>)); break;
    case Layout::transposed:
        STRIX_RECURRENT_LAYOUT_LAUNCH(HIP_KERNEL_NAME(recurrent_layout_diagnostic_kernel<Layout::transposed>)); break;
    }
#undef STRIX_RECURRENT_LAYOUT_LAUNCH
    check_hip(hipGetLastError(), "GDN recurrent diagnostic launch");
}
} // namespace gdn_recurrent_layout_diagnostic
#endif

}  // namespace strix
