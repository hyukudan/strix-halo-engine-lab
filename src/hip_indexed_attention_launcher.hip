#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_mtp_attention.hpp"
#include "strix/hip_mtp_native_validation.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/graph_operator_control.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/hip_qsa_selection_diagnostic.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/hip_qsa_gqa_reuse_diagnostic.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x5141534154544e32ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

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
    iq4_xs,
};

enum class SmallWeightKind {
    f32,
    f16,
    bf16,
};

struct GraphPositionKernelArguments {
    const std::uint64_t* position = nullptr;
    std::uint64_t context_tokens = 0;
    std::uint32_t* status = nullptr;
};

struct GraphSelectionArguments {
    GraphPositionKernelArguments position;
    std::uint32_t ratio = 0;
    std::uint32_t block_topk = 0;
};

void check_hip(hipError_t status, const char* operation);

GraphPositionKernelArguments graph_position_arguments() {
    const auto* scope = current_hip_launch_context();
    if (scope == nullptr || !scope->graph_capture) return {};
    return {hip_graph_position(), scope->context_tokens, hip_graph_status()};
}

__device__ bool graph_visible(GraphSelectionArguments dynamic, std::uint32_t& visible) {
    if (dynamic.position.position == nullptr) return true;
    const auto position = *dynamic.position.position;
    // Other blocks can publish arithmetic failure concurrently. Broadcast once
    // per block so a status transition cannot split threads around barriers.
    __shared__ std::uint32_t reject;
    if (threadIdx.x == 0) reject = atomicAdd(dynamic.position.status, 0U) != 0 || position >= dynamic.position.context_tokens;
    __syncthreads();
    if (reject) return false;
    visible = static_cast<std::uint32_t>(position + 1ULL);
    return true;
}

__device__ bool graph_selected_count(GraphSelectionArguments dynamic, std::uint32_t& count) {
    if (dynamic.position.position == nullptr) return true;
    std::uint32_t visible = 0;
    if (!graph_visible(dynamic, visible)) return false;
    const auto blocks = visible / dynamic.ratio;
    count = min(blocks, dynamic.block_topk) * dynamic.ratio + visible % dynamic.ratio;
    return true;
}

__global__ void validate_indexed_graph_control(GraphPositionKernelArguments dynamic,
    const std::uint16_t* input, std::uint32_t hidden) {
    if (atomicAdd(dynamic.status, 0U) != 0) return;
    if (*dynamic.position >= dynamic.context_tokens) atomicOr(dynamic.status, 2U);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        if ((input[i] & 0x7f80U) == 0x7f80U) atomicOr(dynamic.status, 4U);
    }
}

__global__ void validate_indexed_graph_values(const float* values, std::uint32_t count,
    std::uint32_t* status) {
    if (atomicAdd(status, 0U) != 0) return;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (!isfinite(values[i])) atomicOr(status, 8U);
    }
}

void validate_graph_values(const float* values, std::uint32_t count) {
    if (hip_graph_status() == nullptr) return;
    hipLaunchKernelGGL(validate_indexed_graph_values, dim3(1), dim3(kThreads), 0,
        hip_execution_stream(), values, count, hip_graph_status());
    check_hip(hipGetLastError(), "HIP indexed-attention finite intermediate validation");
}

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct IndexedTensors {
    TensorPair q;
    TensorPair k;
    TensorPair v;
    TensorPair out;
    TensorPair q_norm;
    TensorPair k_norm;
    TensorPair index_q;
    TensorPair index_k;
    TensorPair index_q_norm;
    TensorPair index_k_norm;
};

struct WorkspacePointers {
    float* index_query = nullptr;
    float* current_index_key = nullptr;
    float* block_scores = nullptr;
    std::uint32_t* selected_indices = nullptr;
    float* q_gate = nullptr;
    float* current_k = nullptr;
    float* current_v = nullptr;
    float* attention_scores = nullptr;
    float* attention_output = nullptr;
    qsa_selection_diagnostic::Item* selection_ping = nullptr;
    qsa_selection_diagnostic::Item* selection_pong = nullptr;
    std::uint32_t* selection_status = nullptr;
    std::uint64_t* attention_position = nullptr;
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

void hash_float(
    std::uint64_t& hash,
    float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
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
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) +
            " does not fit uint32");
    }
    return static_cast<std::uint32_t>(value);
}

void validate_config(
    const HipIndexedAttentionLauncherConfig& config) {
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    qsa_input_prefetch::validate_layers(config.model,config.prefetch_q8_input,config.prefetch_eligible_layers);
    (void)qsa_gqa_organization_marker(config.qsa_gqa_heads_per_block);
    if (config.qsa_gqa_heads_per_block != 1 && (config.enable_device_graph_inputs || config.model.head_dim != 256 ||
        config.model.attention_heads > 64 || config.model.hidden_layers > 48 ||
        config.model.indexer_budget != 2048 || config.model.indexer_compress_ratio != 4))
        throw std::invalid_argument("QSA GQA organization requires eager head256/budget2048/compression4");
    if (config.qsa_hierarchical_selection && (config.enable_device_graph_inputs ||
        config.model.indexer_budget != 2048 || config.model.indexer_compress_ratio != 4 ||
        config.model.hidden_layers > 48))
        throw std::invalid_argument("hierarchical QSA requires eager budget2048/compression4 and at most48 layers");
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 targets canonical BF16 activations only");
    }
    if (config.model.attention_heads %
            config.model.kv_heads !=
        0) {
        throw std::invalid_argument(
            "attention heads must divide evenly across KV heads");
    }
    if (config.model.head_dim != kThreads ||
        config.model.indexer_head_dim >
            kThreads ||
        config.model.rotary_dim() >
            config.model.indexer_head_dim ||
        config.model.rotary_dim() >
            config.model.head_dim) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 requires pinned/tileable head geometry");
    }
    if (config.wave8_q8_output) {
        if (config.enable_device_graph_inputs)
            throw std::invalid_argument("wave8 QSA output supports eager execution only");
        if (config.model.attention_heads > 10240U / config.model.head_dim)
            throw std::invalid_argument("wave8 QSA output width exceeds admitted bounds");
        q8_projection_organization::validate_geometry(
            config.model.attention_heads * config.model.head_dim, config.model.hidden_size);
    }
}

WeightKind weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 23: return WeightKind::iq4_xs;
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP indexed-attention projection supports F32, F16, BF16, Q8_0 and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP indexed-attention norm supports F32, F16 and BF16");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(bits) << 16U);
}

__device__ std::uint16_t float_to_bf16(
    float value) {
    const auto bits = __float_as_uint(value);
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
    std::uint64_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(weights)[
        row * row_width + feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(weights);
    return __half2float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(weights);
    return bf16_to_float(
        values[row * row_width + feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row * blocks_per_row +
            feature / 32U];
    return __half2float(block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row * blocks_per_row +
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

template <SmallWeightKind Kind>
__device__ float load_small_weight(
    const void* weights,
    std::uint32_t index);

template <>
__device__ float load_small_weight<
    SmallWeightKind::f32>(
    const void* weights,
    std::uint32_t index) {
    return static_cast<const float*>(
        weights)[index];
}

template <>
__device__ float load_small_weight<
    SmallWeightKind::f16>(
    const void* weights,
    std::uint32_t index) {
    return __half2float(
        static_cast<const __half*>(
            weights)[index]);
}

template <>
__device__ float load_small_weight<
    SmallWeightKind::bf16>(
    const void* weights,
    std::uint32_t index) {
    return bf16_to_float(
        static_cast<
            const std::uint16_t*>(
                weights)[index]);
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input,
    std::uint64_t output) {
    switch (kind) {
    case WeightKind::f32:
        return static_cast<std::uint64_t>(
                   input) *
               output * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return static_cast<std::uint64_t>(
                   input) *
               output * 2ULL;
    case WeightKind::q8_0:
        if (input % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 indexed-attention input width must divide 32");
        }
        return static_cast<std::uint64_t>(
                   input / 32U) *
               output *
               sizeof(Q8Block);
    case WeightKind::iq4_xs:
        if (input % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS indexed-attention input width must divide 256");
        }
        return static_cast<std::uint64_t>(
                   input / 256U) *
               output *
               sizeof(IQ4XSBlock);
    }
    throw std::logic_error(
        "unreachable indexed-attention weight kind");
}

std::uint64_t required_small_weight_bytes(
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
        "unreachable indexed-attention norm kind");
}

template <WeightKind Kind>
__global__ void linear_to_float_kernel(
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

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            load_activation(
                &input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                out,
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
        output[out] =
            round_bf16(
                scratch[0]);
    }
}

// Preserve the actual legacy LDS-tree operand priority: upper/partner src0,
// lower/self src1. Ordinary commutative lowering can select another NaN payload.
// Only the two new QSA input trees call this helper; old kernels remain literal.
__device__ __forceinline__ float qsa_input_legacy_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}

// Eight ascending virtual leaves per physical lane preserve the original
// 256-leaf tree, including its operand order and BF16-in-FP32 epilogue.
__global__ __launch_bounds__(32)
void linear_virtual256_q8_qsa_input_kernel(
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
            if (lane < distance) partial[local] = qsa_input_legacy_tree_add(partial[local], partner);
        }
    }
    if (lane == 0) {
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[4]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[5]);
        partial[2] = qsa_input_legacy_tree_add(partial[2], partial[6]); partial[3] = qsa_input_legacy_tree_add(partial[3], partial[7]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[2]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[3]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[1]);
        output[row] = round_bf16(partial[0]);
    }
}

// Distinct opt-in load organization; existing virtual and legacy kernels
// remain unchanged. Actual ISA, not these arrays, qualifies load batching.
__global__ __launch_bounds__(32)
void linear_prefetch8_q8_qsa_input_kernel(
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
            if (lane < distance) partial[local] = qsa_input_legacy_tree_add(partial[local], partner);
        }
    }
    if (lane == 0) {
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[4]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[5]);
        partial[2] = qsa_input_legacy_tree_add(partial[2], partial[6]); partial[3] = qsa_input_legacy_tree_add(partial[3], partial[7]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[2]); partial[1] = qsa_input_legacy_tree_add(partial[1], partial[3]);
        partial[0] = qsa_input_legacy_tree_add(partial[0], partial[1]);
        output[row] = round_bf16(partial[0]);
    }
}

template <WeightKind Kind>
__global__ void linear_float_to_bf16_kernel(
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

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
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
        output[out] =
            float_to_bf16(
                scratch[0]);
    }
}

// Actual original Q8 output LDS tree uses upper/partner src0 and lower/self
// src1 at every 128..1 stage. Keep this output-specific evidence and helper
// separate from the input projection tree and leave the old body unchanged.
__device__ __forceinline__ float qsa_output_legacy_tree_add(float lower, float upper) {
    float result;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(upper), "v"(lower));
    return result;
}

__global__ __launch_bounds__(256)
void linear_prefetch8_wave8_q8_float_to_bf16_kernel(
    const float* input, const void* weights, std::uint32_t input_width,
    std::uint32_t output_width, std::uint16_t* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U +
        static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return; // Whole inactive wave; no block barriers.
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
    // The shuffle reads partners before upper lanes are overwritten. Lane0's
    // 8 independent leaves preserve the old 128,64,32,16,8 stages exactly.
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local)
            partial[local] = qsa_output_legacy_tree_add(
                partial[local], __shfl_down(partial[local], distance, 32));
    }
    if (lane == 0) {
        partial[0] = qsa_output_legacy_tree_add(partial[0], partial[4]);
        partial[1] = qsa_output_legacy_tree_add(partial[1], partial[5]);
        partial[2] = qsa_output_legacy_tree_add(partial[2], partial[6]);
        partial[3] = qsa_output_legacy_tree_add(partial[3], partial[7]);
        partial[0] = qsa_output_legacy_tree_add(partial[0], partial[2]);
        partial[1] = qsa_output_legacy_tree_add(partial[1], partial[3]);
        partial[0] = qsa_output_legacy_tree_add(partial[0], partial[1]);
        output[row] = float_to_bf16(partial[0]);
    }
}

template <SmallWeightKind Kind>
__global__ void norm_rope_kernel(
    float* values,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const void* norm_weights,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    // Another head can report arithmetic failure while this block starts.
    // Broadcast a single atomic observation before any block-wide barriers;
    // per-thread early exits could otherwise diverge around __syncthreads.
    __shared__ unsigned graph_reject;
    if (dynamic.status != nullptr) {
        if (threadIdx.x == 0) graph_reject =
            atomicAdd(dynamic.status, 0U) != 0 || position >= dynamic.context_tokens ? 1U : 0U;
        __syncthreads();
        if (graph_reject != 0) return;
    }
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (head >= head_count ||
        feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* source = shared;
    auto* reduction =
        shared + head_dim;

    const auto index =
        static_cast<std::uint64_t>(
            head) *
            head_stride +
        feature;
    const float input =
        values[index];
    source[feature] = input;
    reduction[feature] =
        input * input;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            reduction[feature] +=
                reduction[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float mean = reduction[0] / static_cast<float>(head_dim) + epsilon;
    const float inv_rms = rsqrtf(mean);
    if (dynamic.status != nullptr && (!isfinite(reduction[0]) || !isfinite(mean) ||
        mean <= 0 || !isfinite(inv_rms))) {
        // The shared reduction and scalar are identical for every thread in
        // this head, so this return is uniform before the remaining barrier.
        if (feature == 0) atomicOr(dynamic.status, 8U);
        return;
    }
    source[feature] =
        round_bf16(
            input *
            inv_rms *
            load_small_weight<Kind>(
                 norm_weights,
                 feature));
    if (dynamic.status != nullptr && !isfinite(source[feature])) atomicOr(dynamic.status, 8U);
    __syncthreads();

    float result =
        source[feature];
    if (feature < rotary_dim) {
        const auto half =
            rotary_dim / 2U;
        const auto pair =
            feature % half;
        const float exponent =
            2.0F *
            static_cast<float>(pair) /
            static_cast<float>(
                rotary_dim);
        const float angle =
            static_cast<float>(
                position) *
            powf(
                theta,
                -exponent);
        const float cosine =
            cosf(angle);
        const float sine =
            sinf(angle);

        if (feature < half) {
            result =
                source[feature] *
                    cosine -
                source[
                    feature +
                    half] *
                    sine;
        } else {
            result =
                source[feature] *
                    cosine +
                source[
                    feature -
                    half] *
                    sine;
        }
    }
    const float rounded = round_bf16(result);
    if (dynamic.status != nullptr && (!isfinite(result) || !isfinite(rounded))) {
        atomicOr(dynamic.status, 8U);
        return;
    }
    values[index] = rounded;
}

__global__ void initialize_graph_attention_status(
    std::uint32_t* status, std::uint64_t scalar_position, GraphPositionKernelArguments dynamic) {
    const auto position = dynamic.position == nullptr ? scalar_position : *dynamic.position;
    *status = position >= dynamic.context_tokens ? 2U : 0U;
}

template <SmallWeightKind Kind>
__global__ void validate_graph_attention_sources(
    const float* query, const float* key, const float* value,
    const void* query_norm, const void* key_norm, std::uint64_t query_elements,
    std::uint64_t kv_elements, std::uint32_t norm_elements, std::uint32_t query_stride, std::uint32_t* status) {
    const auto maximum = query_elements > kv_elements ? query_elements : kv_elements;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < maximum || i < norm_elements; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        bool invalid = false;
        if (i < query_elements && i % query_stride < norm_elements) invalid = !isfinite(query[i]);
        if (i < kv_elements) invalid = invalid || !isfinite(key[i]) || !isfinite(value[i]) || !isfinite(round_bf16(value[i]));
        if (i < norm_elements) invalid = invalid || !isfinite(load_small_weight<Kind>(query_norm, static_cast<std::uint32_t>(i))) ||
            !isfinite(load_small_weight<Kind>(key_norm, static_cast<std::uint32_t>(i)));
        if (invalid) atomicOr(status, 4U);
    }
}

__global__ void store_raw_index_key_kernel(
    const float* current,
    std::uint16_t* state,
    std::uint32_t width,
    std::uint64_t position, GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    if (dynamic.status != nullptr && (*dynamic.status != 0 || position >= dynamic.context_tokens)) return;
    const auto feature =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (feature < width) {
        state[
            position *
                static_cast<std::uint64_t>(
                    width) +
            feature] =
            float_to_bf16(
                current[feature]);
    }
}

__global__ void store_kv_kernel(
    const float* current_k,
    const float* current_v,
    std::uint16_t* state,
    std::uint32_t kv_width,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    if (dynamic.status != nullptr && (*dynamic.status != 0 || position >= dynamic.context_tokens)) return;
    const auto feature =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (feature >= kv_width) {
        return;
    }
    const auto row =
        position *
        static_cast<std::uint64_t>(
            kv_width) *
        2ULL;
    state[row + feature] =
        float_to_bf16(
            current_k[feature]);
    state[
        row +
        kv_width +
        feature] =
        float_to_bf16(
            current_v[feature]);
}

template <SmallWeightKind Kind>
__global__ void block_score_kernel(
    const float* index_query,
    const std::uint16_t* raw_keys,
    std::uint32_t index_heads,
    std::uint32_t head_dim,
    std::uint32_t compress_ratio,
    const void* key_norm,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    float* block_scores, GraphSelectionArguments dynamic) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (dynamic.position.position != nullptr) {
        std::uint32_t visible = 0;
        if (!graph_visible(dynamic, visible) || visible / compress_ratio <= dynamic.block_topk ||
            block >= visible / compress_ratio) return;
    }
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch =
        shared + head_dim;

    float mean = 0.0F;
    const auto token_start =
        static_cast<std::uint64_t>(
            block) *
        compress_ratio;
    for (std::uint32_t offset = 0;
         offset < compress_ratio;
         ++offset) {
        mean +=
            bf16_to_float(
                raw_keys[
                    (token_start +
                     offset) *
                        head_dim +
                    feature]);
    }
    mean /=
        static_cast<float>(
            compress_ratio);

    // Upstream casts the FP32 mean back to raw-key dtype before k_norm.
    mean =
        round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] =
        mean * mean;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float inv_rms =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (dynamic.position.status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv_rms))) {
        if (feature == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    pooled[feature] =
        round_bf16(
            mean *
            inv_rms *
            load_small_weight<Kind>(
                 key_norm,
                 feature));
    __syncthreads();

    float rotated =
        pooled[feature];
    if (feature < rotary_dim) {
        const auto half =
            rotary_dim / 2U;
        const auto pair =
            feature % half;
        const float exponent =
            2.0F *
            static_cast<float>(pair) /
            static_cast<float>(
                rotary_dim);
        const float angle =
            static_cast<float>(
                token_start) *
            powf(
                theta,
                -exponent);
        const float cosine =
            cosf(angle);
        const float sine =
            sinf(angle);
        if (feature < half) {
            rotated =
                pooled[feature] *
                    cosine -
                pooled[
                    feature +
                    half] *
                    sine;
        } else {
            rotated =
                pooled[feature] *
                    cosine +
                pooled[
                    feature -
                    half] *
                    sine;
        }
    }
    // Every lane must finish reading its RoPE partner before any lane writes
    // the rotated value back into the shared pooled vector.
    __syncthreads();
    pooled[feature] =
        round_bf16(rotated);
    if (dynamic.position.status != nullptr && !isfinite(pooled[feature])) atomicOr(dynamic.position.status, 8U);
    __syncthreads();

    __shared__ float total;
    if (feature == 0) {
        total = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t head = 0;
         head < index_heads;
         ++head) {
        scratch[feature] =
            index_query[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim +
                feature] *
            pooled[feature];
        __syncthreads();

        for (std::uint32_t stride =
                 head_dim / 2U;
             stride > 0;
             stride >>= 1U) {
            if (feature < stride) {
                scratch[feature] +=
                    scratch[
                        feature +
                        stride];
            }
            __syncthreads();
        }
        if (feature == 0) {
            if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
            total +=
                fmaxf(
                    scratch[0],
                    0.0F);
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(total)) atomicOr(dynamic.position.status, 8U);
        block_scores[block] =
            total /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}

__global__ void fill_visible_indices_kernel(
    std::uint32_t* selected,
    std::uint32_t visible) {
    const auto index =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (index < visible) {
        selected[index] =
            index;
    }
}

// Device-owned control storage avoids a borrowed host async-copy lifetime.
__global__ void prepare_qsa_gqa_control(std::uint64_t* position, std::uint32_t* status,
    std::uint64_t scalar_position, bool preserve_selection_error) {
    if (threadIdx.x == 0) {
        *position = scalar_position;
        if (!preserve_selection_error) *status = 0;
    }
}
__global__ void publish_qsa_attention_status(const std::uint32_t* local, DeferredMoeFirstError* first_error) {
    if (threadIdx.x == 0) merge_deferred_moe_first_error(*first_error, *local, DeferredMoeDomain::qsa_attention);
}

// Local status belongs to this QSA workspace, not to the D1 domain word.
// Preserve the complete first D1 error and make indices safe even on rejection;
// the D1 completion fence rejects the token before host publication.
__global__ void publish_qsa_selection_status(const std::uint32_t* local,
    DeferredMoeFirstError* first_error, std::uint32_t* selected, std::uint32_t count) {
    const auto bits = *local;
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bits && index < count) selected[index] = 0;
    if (index == 0) merge_deferred_moe_first_error(*first_error, bits, DeferredMoeDomain::qsa_selection);
}

__global__ void select_blocks_kernel(
    float* block_scores,
    std::uint32_t num_blocks,
    std::uint32_t select_blocks,
    std::uint32_t compress_ratio,
    std::uint32_t visible_tokens,
    std::uint32_t* selected, GraphSelectionArguments dynamic) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }

    if (dynamic.position.position != nullptr) {
        if (!graph_visible(dynamic, visible_tokens)) return;
        num_blocks = visible_tokens / compress_ratio;
        if (num_blocks <= dynamic.block_topk) {
            for (std::uint32_t token = 0; token < visible_tokens; ++token) selected[token] = token;
            return;
        }
        select_blocks = dynamic.block_topk;
    }

    for (std::uint32_t slot = 0;
         slot < select_blocks;
         ++slot) {
        float best =
            -INFINITY;
        std::uint32_t best_block = 0;
        for (std::uint32_t block = 0;
             block < num_blocks;
             ++block) {
            const float value =
                block_scores[block];
            if (value > best ||
                (value == best &&
                 block < best_block)) {
                best = value;
                best_block = block;
            }
        }
        selected[slot] =
            best_block;
        block_scores[best_block] =
            -INFINITY;
    }

    // The reference applies a mask over the full KV sequence. Sort selected
    // blocks by position so the sparse gather preserves chronological order.
    for (std::uint32_t i = 1;
         i < select_blocks;
         ++i) {
        const auto value =
            selected[i];
        std::uint32_t j = i;
        while (j > 0 &&
               selected[j - 1] >
                   value) {
            selected[j] =
                selected[j - 1];
            --j;
        }
        selected[j] =
            value;
    }

    // Expand backwards in place from block ids to token ids.
    for (std::uint32_t reverse = select_blocks;
         reverse > 0;
         --reverse) {
        const auto block =
            selected[
                reverse - 1U];
        const auto base =
            (reverse - 1U) *
            compress_ratio;
        for (std::uint32_t offset = 0;
             offset < compress_ratio;
             ++offset) {
            selected[
                base +
                offset] =
                block *
                    compress_ratio +
                offset;
        }
    }

    const auto complete_tokens =
        num_blocks *
        compress_ratio;
    const auto selected_complete =
        select_blocks *
        compress_ratio;
    for (std::uint32_t token =
             complete_tokens;
         token < visible_tokens;
         ++token) {
        selected[
            selected_complete +
            token -
            complete_tokens] =
            token;
    }
}

__global__ void attention_score_kernel(
    const float* q_gate,
    const std::uint16_t* kv_state,
    const std::uint32_t* selected,
    std::uint32_t selected_count,
    std::uint32_t maximum_selected,
    std::uint32_t attention_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float* scores, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto flat =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto head =
        flat /
        (dynamic.position.position == nullptr ? selected_count : maximum_selected);
    const auto selected_slot =
        flat %
        (dynamic.position.position == nullptr ? selected_count : maximum_selected);
    if (head >= attention_heads || selected_slot >= selected_count) {
        return;
    }

    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    __shared__ float scratch[kThreads];

    const auto token =
        selected[selected_slot];
    const auto kv_group =
        attention_heads /
        kv_heads;
    const auto kv_head =
        head /
        kv_group;
    const auto kv_width =
        kv_heads *
        head_dim;
    const auto row_base =
        static_cast<std::uint64_t>(
            token) *
        kv_width *
        2ULL;

    float product = 0.0F;
    if (feature < head_dim) {
        product =
            q_gate[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim *
                    2ULL +
                feature] *
            bf16_to_float(
                kv_state[
                    row_base +
                    static_cast<std::uint64_t>(
                        kv_head) *
                        head_dim +
                    feature]);
    }
    scratch[feature] =
        product;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
        scores[
            static_cast<std::uint64_t>(
                head) *
                maximum_selected +
            selected_slot] =
            scratch[0] /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}

__global__ void attention_softmax_kernel(
    float* scores,
    std::uint32_t selected_count,
    std::uint32_t maximum_selected, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    __shared__ float scratch[kThreads];

    float local_max =
        -INFINITY;
    for (std::uint32_t index = lane;
         index < selected_count;
         index += kThreads) {
        local_max =
            fmaxf(
                local_max,
                scores[
                    static_cast<std::uint64_t>(
                        head) *
                        maximum_selected +
                    index]);
    }
    scratch[lane] =
        local_max;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane +
                        stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    if (dynamic.position.status != nullptr && !isfinite(maximum)) {
        if (lane == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    // Preserve the reduced maximum in each lane before reusing scratch for
    // the sum reduction (including waves with no active selected element).
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t index = lane;
         index < selected_count;
         index += kThreads) {
        local_sum +=
            expf(
                scores[
                    static_cast<std::uint64_t>(
                        head) *
                        maximum_selected +
                    index] -
                maximum);
    }
    scratch[lane] =
        local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[
                    lane +
                    stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];
    if (dynamic.position.status != nullptr && (!isfinite(denominator) || denominator <= 0.0F)) {
        if (lane == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }

    for (std::uint32_t index = lane;
         index < selected_count;
         index += kThreads) {
        const auto location =
            static_cast<std::uint64_t>(
                head) *
                maximum_selected +
            index;
        // Eager attention casts softmax back to query dtype.
        scores[location] =
            round_bf16(
                expf(
                    scores[location] -
                    maximum) /
                denominator);
    }
}

__global__ void attention_output_kernel(
    const float* q_gate,
    const std::uint16_t* kv_state,
    const std::uint32_t* selected,
    const float* scores,
    std::uint32_t selected_count,
    std::uint32_t maximum_selected,
    std::uint32_t attention_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float* output, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (head >= attention_heads ||
        feature >= head_dim) {
        return;
    }

    const auto kv_group =
        attention_heads /
        kv_heads;
    const auto kv_head =
        head /
        kv_group;
    const auto kv_width =
        kv_heads *
        head_dim;

    float sum = 0.0F;
    for (std::uint32_t slot = 0;
         slot < selected_count;
         ++slot) {
        const auto token =
            selected[slot];
        const auto row_base =
            static_cast<std::uint64_t>(
                token) *
            kv_width *
            2ULL;
        const auto value =
            bf16_to_float(
                kv_state[
                    row_base +
                    kv_width +
                    static_cast<std::uint64_t>(
                        kv_head) *
                        head_dim +
                    feature]);
        sum +=
            scores[
                static_cast<std::uint64_t>(
                    head) *
                    maximum_selected +
                slot] *
            value;
    }

    // Preserve the model's BF16 attention output and BF16 sigmoid gate before
    // the final output projection.
    const float attn =
        round_bf16(sum);
    const float gate =
        round_bf16(
            1.0F /
            (1.0F +
             expf(
                 -q_gate[
                     static_cast<std::uint64_t>(
                         head) *
                         head_dim *
                         2ULL +
                     head_dim +
                     feature])));
    output[
        static_cast<std::uint64_t>(
            head) *
            head_dim +
        feature] =
        round_bf16(
            attn * gate);
    if (dynamic.position.status != nullptr && !isfinite(round_bf16(attn * gate))) atomicOr(dynamic.position.status, 8U);
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
            "HIP indexed attention requires model manifest");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "indexed-attention tensor index exceeds manifest");
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
    const TensorPair& value) {
    destination = value;
    ++count;
}

IndexedTensors identify_tensors(
    const ExecutionDispatchContext& context) {
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared attention roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& r=cached->indexed_roles(context.execution_identity,context.step_id);
        return {tensor_pair(context,r.q),tensor_pair(context,r.k),tensor_pair(context,r.v),tensor_pair(context,r.out),
            tensor_pair(context,r.q_norm),tensor_pair(context,r.k_norm),tensor_pair(context,r.index_q),tensor_pair(context,r.index_k),
            tensor_pair(context,r.index_q_norm),tensor_pair(context,r.index_k_norm)};
    }

    const auto& attention =
        resolve_dispatch_tensor_access(
            context, 0);
    const auto& indexer =
        resolve_dispatch_tensor_access(
            context, 1);
    const auto& norms =
        resolve_dispatch_tensor_access(
            context, 2);

    IndexedTensors result;
    std::uint32_t q = 0;
    std::uint32_t k = 0;
    std::uint32_t v = 0;
    std::uint32_t out = 0;
    std::uint32_t qn = 0;
    std::uint32_t kn = 0;
    std::uint32_t iq = 0;
    std::uint32_t ik = 0;
    std::uint32_t iqn = 0;
    std::uint32_t ikn = 0;

    for (const auto index :
         attention.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto name =
            std::string_view(
                pair.manifest->name);
        if (contains(
                name,
                "attn_output") ||
            contains(
                name,
                "o_proj")) {
            assign_unique(
                result.out,
                out,
                pair);
        } else if (contains(
                       name,
                       "attn_q") ||
                   contains(
                       name,
                       "q_proj")) {
            assign_unique(
                result.q,
                q,
                pair);
        } else if (contains(
                       name,
                       "attn_k") ||
                   contains(
                       name,
                       "k_proj")) {
            assign_unique(
                result.k,
                k,
                pair);
        } else if (contains(
                       name,
                       "attn_v") ||
                   contains(
                       name,
                       "v_proj")) {
            assign_unique(
                result.v,
                v,
                pair);
        }
    }

    for (const auto index :
         indexer.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto name =
            std::string_view(
                pair.manifest->name);
        if (contains(
                name,
                "indexer.q_proj") ||
            contains(
                name,
                "indexer_q_proj")) {
            assign_unique(
                result.index_q,
                iq,
                pair);
        } else if (contains(
                       name,
                       "indexer.k_proj") ||
                   contains(
                       name,
                       "indexer_k_proj")) {
            assign_unique(
                result.index_k,
                ik,
                pair);
        } else if (contains(
                       name,
                       "indexer.q_norm") ||
                   contains(
                       name,
                       "indexer_q_norm")) {
            assign_unique(
                result.index_q_norm,
                iqn,
                pair);
        } else if (contains(
                       name,
                       "indexer.k_norm") ||
                   contains(
                       name,
                       "indexer_k_norm")) {
            assign_unique(
                result.index_k_norm,
                ikn,
                pair);
        }
    }

    for (const auto index :
         norms.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto name =
            std::string_view(
                pair.manifest->name);
        if (contains(
                name,
                "attn_q_norm")) {
            assign_unique(
                result.q_norm,
                qn,
                pair);
        } else if (contains(
                       name,
                       "attn_k_norm")) {
            assign_unique(
                result.k_norm,
                kn,
                pair);
        }
    }

    if (q != 1 || k != 1 ||
        v != 1 || out != 1 ||
        qn != 1 || kn != 1 ||
        iq != 1 || ik != 1 ||
        iqn != 1 || ikn != 1) {
        throw std::invalid_argument(
            "indexed-attention tensor family is missing or ambiguous");
    }
    return result;
}

void validate_runtime_tensor(
    const TensorPair& pair,
    std::uint64_t required,
    const char* name) {
    if (pair.manifest == nullptr ||
        pair.runtime == nullptr) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor is missing");
    }
    if (!runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0 ||
        pair.runtime->bytes < required) {
        throw std::invalid_argument(
            std::string(name) +
            " runtime tensor is not GPU-visible or is undersized");
    }
}

void validate_matrix(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint32_t output,
    const char* name) {
    if (pair.manifest->dimensions.size() < 2 ||
        pair.manifest->dimensions[0] != input ||
        pair.manifest->dimensions[1] != output) {
        throw std::invalid_argument(
            std::string(name) +
            " tensor geometry does not match contract");
    }
    const auto kind =
        weight_kind(
            pair.manifest->ggml_type);
    validate_runtime_tensor(
        pair,
        required_weight_bytes(
            kind,
            input,
            output),
        name);
}

void validate_norm(
    const TensorPair& pair,
    std::uint32_t width,
    const char* name) {
    if (pair.manifest->dimensions.empty() ||
        pair.manifest->dimensions[0] !=
            width) {
        throw std::invalid_argument(
            std::string(name) +
            " norm geometry does not match contract");
    }
    const auto kind =
        small_weight_kind(
            pair.manifest->ggml_type);
    validate_runtime_tensor(
        pair,
        required_small_weight_bytes(
            kind,
            width),
        name);
}

template <WeightKind Kind>
void launch_linear_to_float_typed(
    const ExecutionBufferAddress& input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            linear_to_float_kernel<Kind>),
        dim3(output_width),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<
            const std::uint16_t*>(
                input.address),
        reinterpret_cast<
            const void*>(
                weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention projection launch");
}

void launch_linear_to_float(
    const TensorPair& tensor,
    const ExecutionBufferAddress& input,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output, bool prefetch = false, qsa_input_prefetch::RuntimeDispatchStats* stats = nullptr) {
    if(prefetch && qsa_input_prefetch::eligible(*tensor.manifest)){
        if(tensor.runtime->address%16U!=0 || tensor.manifest->absolute_offset%16U!=0 || tensor.runtime->bytes<qsa_input_prefetch::required_weight_bytes(input_width,output_width))throw std::invalid_argument("QSA prefetch launch weight alignment/extent invalid");
        hipLaunchKernelGGL(linear_prefetch8_q8_qsa_input_kernel,dim3(output_width),dim3(32),0,hip_execution_stream(),reinterpret_cast<const std::uint16_t*>(input.address),reinterpret_cast<const void*>(tensor.runtime->address),input_width,output_width,output);
        check_hip(hipGetLastError(),"HIP indexed-attention prefetch8 Q8 input projection launch");
        if(stats)stats->dispatched_projections.fetch_add(1,std::memory_order_relaxed);
        return;
    }
    switch (weight_kind(
                tensor.manifest->
                    ggml_type)) {
    case WeightKind::f32:
        launch_linear_to_float_typed<
            WeightKind::f32>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::f16:
        launch_linear_to_float_typed<
            WeightKind::f16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::bf16:
        launch_linear_to_float_typed<
            WeightKind::bf16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::q8_0:
        launch_linear_to_float_typed<
            WeightKind::q8_0>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::iq4_xs:
        launch_linear_to_float_typed<
            WeightKind::iq4_xs>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    }
    throw std::logic_error(
        "unreachable indexed projection kind");
}

template <WeightKind Kind>
void launch_linear_float_to_bf16_typed(
    const float* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    const ExecutionBufferAddress& output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            linear_float_to_bf16_kernel<Kind>),
        dim3(output_width),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        input,
        reinterpret_cast<
            const void*>(
                weights.address),
        input_width,
        output_width,
        reinterpret_cast<
            std::uint16_t*>(
                output.address));
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention output projection launch");
}

void launch_linear_float_to_bf16(
    const TensorPair& tensor,
    const float* input,
    std::uint32_t input_width,
    std::uint32_t output_width,
    const ExecutionBufferAddress& output, bool wave8 = false) {
    if (wave8) {
        if (tensor.manifest->ggml_type != 8U || input_width % 32U)
            throw std::invalid_argument("wave8 QSA output requires aligned Q8_0 weights");
        q8_projection_organization::validate_geometry(input_width, output_width);
        hipLaunchKernelGGL(linear_prefetch8_wave8_q8_float_to_bf16_kernel,
            dim3((output_width + 7U) / 8U), dim3(256), 0, hip_execution_stream(), input,
            reinterpret_cast<const void*>(tensor.runtime->address), input_width, output_width,
            reinterpret_cast<std::uint16_t*>(output.address));
        check_hip(hipGetLastError(), "HIP indexed-attention wave8 Q8 output projection launch");
        return;
    }
    switch (weight_kind(
                tensor.manifest->
                    ggml_type)) {
    case WeightKind::f32:
        launch_linear_float_to_bf16_typed<
            WeightKind::f32>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::f16:
        launch_linear_float_to_bf16_typed<
            WeightKind::f16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::bf16:
        launch_linear_float_to_bf16_typed<
            WeightKind::bf16>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::q8_0:
        launch_linear_float_to_bf16_typed<
            WeightKind::q8_0>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    case WeightKind::iq4_xs:
        launch_linear_float_to_bf16_typed<
            WeightKind::iq4_xs>(
                input, *tensor.runtime,
                input_width, output_width,
                output);
        return;
    }
    throw std::logic_error(
        "unreachable indexed output projection kind");
}

template <SmallWeightKind Kind>
void launch_norm_rope_typed(
    float* values,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const RuntimeTensorBinding& norm,
    const ModelContract& model,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic = {}, hipStream_t stream = hipStreamLegacy) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            norm_rope_kernel<Kind>),
        dim3(heads),
        dim3(head_dim),
        2ULL *
            head_dim *
            sizeof(float),
        stream,
        values,
        heads,
        head_dim,
        head_stride,
        reinterpret_cast<
            const void*>(
                norm.address),
        model.rms_norm_epsilon,
        checked_u32(
            model.rotary_dim(),
            "rotary_dim"),
        model.rope_theta,
        position, dynamic);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention norm/rope launch");
}

void launch_norm_rope(
    float* values,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const TensorPair& norm,
    const ModelContract& model,
    std::uint64_t position) {
    switch (small_weight_kind(
                norm.manifest->
                    ggml_type)) {
    case SmallWeightKind::f32:
        launch_norm_rope_typed<
            SmallWeightKind::f32>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position, graph_position_arguments(), hip_execution_stream());
        return;
    case SmallWeightKind::f16:
        launch_norm_rope_typed<
            SmallWeightKind::f16>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position, graph_position_arguments(), hip_execution_stream());
        return;
    case SmallWeightKind::bf16:
        launch_norm_rope_typed<
            SmallWeightKind::bf16>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position, graph_position_arguments(), hip_execution_stream());
        return;
    }
    throw std::logic_error(
        "unreachable indexed norm kind");
}

template <SmallWeightKind Kind>
void launch_block_scores_typed(
    const WorkspacePointers& workspace,
    const SessionStateAddress& raw_state,
    const TensorPair& k_norm,
    const ModelContract& model,
    std::uint32_t complete_blocks, GraphSelectionArguments dynamic) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            block_score_kernel<Kind>),
        dim3(complete_blocks),
        dim3(
            checked_u32(
                model.indexer_head_dim,
                "indexer_head_dim")),
        2ULL *
            model.indexer_head_dim *
            sizeof(float),
        hip_execution_stream(),
        workspace.index_query,
        reinterpret_cast<
            const std::uint16_t*>(
                raw_state.address),
        checked_u32(
            model.indexer_n_heads,
            "indexer_n_heads"),
        checked_u32(
            model.indexer_head_dim,
            "indexer_head_dim"),
        checked_u32(
            model.indexer_compress_ratio,
            "indexer_compress_ratio"),
        reinterpret_cast<
            const void*>(
                k_norm.runtime->address),
        model.rms_norm_epsilon,
        checked_u32(
            model.rotary_dim(),
            "rotary_dim"),
        model.rope_theta,
        workspace.block_scores, dynamic);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention block-score launch");
}

void launch_block_scores(
    const WorkspacePointers& workspace,
    const SessionStateAddress& raw_state,
    const TensorPair& k_norm,
    const ModelContract& model,
    std::uint32_t complete_blocks, GraphSelectionArguments dynamic = {}) {
    switch (small_weight_kind(
                k_norm.manifest->
                    ggml_type)) {
    case SmallWeightKind::f32:
        launch_block_scores_typed<
            SmallWeightKind::f32>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks, dynamic);
        return;
    case SmallWeightKind::f16:
        launch_block_scores_typed<
            SmallWeightKind::f16>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks, dynamic);
        return;
    case SmallWeightKind::bf16:
        launch_block_scores_typed<
            SmallWeightKind::bf16>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks, dynamic);
        return;
    }
    throw std::logic_error(
        "unreachable indexed k-norm kind");
}

const SerialSessionStateLayout&
state_layout(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->
            session_state_layout !=
            nullptr) {
        return *context.resources->
            session_state_layout;
    }
    if (context.session_state_layout !=
        nullptr) {
        return *context.session_state_layout;
    }
    throw std::invalid_argument(
        "HIP indexed attention requires session-state layout");
}

const SessionStateMemoryBinding&
state_memory(
    const ExecutionDispatchContext& context) {
    if (context.resources != nullptr &&
        context.resources->
            session_state_memory !=
            nullptr) {
        return *context.resources->
            session_state_memory;
    }
    if (context.session_state_memory !=
        nullptr) {
        return *context.session_state_memory;
    }
    throw std::invalid_argument(
        "HIP indexed attention requires session-state memory");
}

WorkspacePointers workspace_pointers(
    const ExecutionBufferAddress& temporary,
    const IndexedAttentionWorkspaceLayout& layout) {
    const auto base =
        temporary.address;
    return WorkspacePointers{
        reinterpret_cast<float*>(
            base +
            layout.index_query_offset),
        reinterpret_cast<float*>(
            base +
            layout.current_index_key_offset),
        reinterpret_cast<float*>(
            base +
            layout.block_scores_offset),
        reinterpret_cast<std::uint32_t*>(
            base +
            layout.selected_indices_offset),
        reinterpret_cast<float*>(
            base +
            layout.q_gate_offset),
        reinterpret_cast<float*>(
            base +
            layout.current_k_offset),
        reinterpret_cast<float*>(
            base +
            layout.current_v_offset),
        reinterpret_cast<float*>(
            base +
            layout.attention_scores_offset),
        reinterpret_cast<float*>(
            base +
            layout.attention_output_offset),
        layout.selection_ping_bytes ? reinterpret_cast<qsa_selection_diagnostic::Item*>(base + layout.selection_ping_offset) : nullptr,
        layout.selection_pong_bytes ? reinterpret_cast<qsa_selection_diagnostic::Item*>(base + layout.selection_pong_offset) : nullptr,
        layout.selection_status_bytes ? reinterpret_cast<std::uint32_t*>(base + layout.selection_status_offset) : nullptr,
        layout.attention_position_bytes ? reinterpret_cast<std::uint64_t*>(base + layout.attention_position_offset) : nullptr,
    };
}

// Complete metadata-only preparation. GPU input values are never read here.
struct PreparedIndexedLaunch {
    ExecutionBufferAddress input, output, temporary;
    IndexedAttentionWorkspaceLayout workspace_layout;
    WorkspacePointers workspace;
    const SerialSessionStateLayout* session_layout;
    GraphPositionKernelArguments dynamic_position;
    bool device_graph;
    SessionStateAddress kv_state, raw_state;
    IndexedTensors tensors;
    std::uint32_t hidden, attention_heads, kv_heads, head_dim, attention_width, kv_width;
    std::uint32_t q_gate_width, index_q_width, index_dim;
};
struct PreparedIndexedSelection {
    std::uint32_t visible, ratio, block_topk, complete_blocks, selected_count;
    GraphSelectionArguments dynamic_selection;
};

PreparedIndexedLaunch prepare_indexed_attention_launch(
    const ExecutionDispatchContext& context, const HipIndexedAttentionLauncherConfig* config_ptr) {
    if (context.op !=
        SerialOp::indexed_attention) {
        throw std::invalid_argument(
            "HIP indexed-attention launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1 ||
        !context.layer.has_value()) {
        throw std::invalid_argument(
            "HIP indexed attention requires one read/write/temp and a layer");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 supports only C1 one-row execution");
    }
    if (config_ptr == nullptr) {
        throw std::invalid_argument(
            "HIP indexed attention requires configuration");
    }

    const auto& config = *config_ptr;
    validate_config(config);
    if (config.wave8_q8_output) {
        const auto* scope = current_hip_launch_context();
        if (scope && scope->graph_capture)
            throw std::invalid_argument("wave8 QSA output supports eager execution only");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(hip_execution_stream(), &capture),
            "HIP wave8 QSA output capture query");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("wave8 QSA output supports eager execution only");
    }
    if (config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1) {
        const auto* scope = current_hip_launch_context();
        if (!scope || scope->graph_capture || !scope->stream || scope->stream == hipStreamLegacy ||
            scope->stream == hipStreamPerThread || !hip_deferred_moe_status() ||
            scope->deferred_moe_status.execution_identity != context.execution_identity ||
            scope->deferred_moe_status.arena_bytes != 8 || !scope->deferred_moe_status.allocation_generation ||
            scope->deferred_moe_status.arena_base % alignof(DeferredMoeFirstError) ||
            scope->deferred_moe_status.arena_base > UINTPTR_MAX - 8U)
            throw std::invalid_argument("hierarchical QSA requires its current owned eager D1 scope");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(scope->stream, &capture), "HIP hierarchical QSA capture query");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("hierarchical QSA supports eager execution only");
    }
    const auto& model =
        config.model;

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);

    const auto workspace_layout =
        build_indexed_attention_workspace_layout(
            model, 256, config.qsa_hierarchical_selection, config.qsa_gqa_heads_per_block);
    if (temporary.bytes <
        workspace_layout.total_bytes) {
        throw std::invalid_argument(
            "indexed-attention temporary is undersized");
    }
    const auto workspace =
        workspace_pointers(
            temporary,
            workspace_layout);

    const auto& session_layout =
        state_layout(context);
    const auto dynamic_position = graph_position_arguments();
    const bool device_graph = dynamic_position.position != nullptr;
    if (!device_graph && context.position >=
        session_layout.context_tokens) {
        throw std::out_of_range(
            "indexed-attention position exceeds session capacity");
    }
    const auto& session_memory =
        state_memory(context);
    const auto kv_state =
        resolve_indexed_attention_layer_state(
            session_layout,
            session_memory,
            model,
            *context.layer,
            SessionStateComponent::
                attention_kv);
    const auto raw_state =
        resolve_indexed_attention_layer_state(
            session_layout,
            session_memory,
            model,
            *context.layer,
            SessionStateComponent::
                sparse_index);
    if (kv_state.storage !=
            SessionStateStorage::device ||
        raw_state.storage !=
            SessionStateStorage::device) {
        throw std::invalid_argument(
            "indexed-attention KV/indexer state must be device resident");
    }

    const auto state_contract =
        build_indexed_attention_state_layout(
            model);
    if (kv_state.bytes <
            session_layout.context_tokens *
                state_contract.kv_row_bytes ||
        raw_state.bytes <
            session_layout.context_tokens *
                state_contract.raw_index_row_bytes) {
        throw std::invalid_argument(
            "indexed-attention state slices are undersized");
    }

    const auto tensors =
        identify_tensors(context);

    const auto hidden =
        checked_u32(
            model.hidden_size,
            "hidden_size");
    const auto attention_heads =
        checked_u32(
            model.attention_heads,
            "attention_heads");
    const auto kv_heads =
        checked_u32(
            model.kv_heads,
            "kv_heads");
    const auto head_dim =
        checked_u32(
            model.head_dim,
            "head_dim");
    const auto attention_width =
        checked_u32(
            model.attention_heads *
                model.head_dim,
            "attention_width");
    const auto kv_width =
        checked_u32(
            model.kv_heads *
                model.head_dim,
            "kv_width");
    const auto q_gate_width =
        checked_u32(
            model.attention_heads *
                model.head_dim *
                2ULL,
            "q_gate_width");
    const auto index_q_width =
        checked_u32(
            model.indexer_n_heads *
                model.indexer_head_dim,
            "index_q_width");
    const auto index_dim =
        checked_u32(
            model.indexer_head_dim,
            "indexer_head_dim");

    validate_matrix(
        tensors.q,
        hidden,
        q_gate_width,
        "attention q/gate");
    validate_matrix(
        tensors.k,
        hidden,
        kv_width,
        "attention k");
    validate_matrix(
        tensors.v,
        hidden,
        kv_width,
        "attention v");
    validate_matrix(
        tensors.out,
        attention_width,
        hidden,
        "attention output");
    validate_matrix(
        tensors.index_q,
        hidden,
        index_q_width,
        "indexer q");
    validate_matrix(
        tensors.index_k,
        hidden,
        index_dim,
        "indexer k");
    validate_norm(
        tensors.q_norm,
        head_dim,
        "attention q norm");
    validate_norm(
        tensors.k_norm,
        head_dim,
        "attention k norm");
    validate_norm(
        tensors.index_q_norm,
        index_dim,
        "indexer q norm");
    validate_norm(
        tensors.index_k_norm,
        index_dim,
        "indexer k norm");

    const auto input_required =
        static_cast<std::uint64_t>(
            hidden) *
        model.bytes_bf16;
    if (input.bytes <
            input_required ||
        output.bytes <
            input_required) {
        throw std::invalid_argument(
            "indexed-attention hidden buffers are undersized");
    }

    if (config.wave8_q8_output) {
        if (tensors.out.manifest->ggml_type != 8U)
            throw std::invalid_argument("wave8 QSA output requires aligned Q8_0 weights");
        q8_projection_organization::validate_fp32_bf16_views(attention_width, hidden,
            {reinterpret_cast<std::uintptr_t>(workspace.attention_output),
             static_cast<std::uint64_t>(attention_width) * sizeof(float)},
            {tensors.out.runtime->address, tensors.out.runtime->bytes},
            {output.address, output.bytes});
    }
    if(config.prefetch_q8_input){
        if(!qsa_input_prefetch::selected(true,config.prefetch_eligible_layers,context.layer))throw std::invalid_argument("QSA prefetch launch layer differs from admitted plan");
        if(input.address%2U!=0 || output.address%2U!=0 || temporary.address%4U!=0 || kv_state.address%2U!=0 || raw_state.address%2U!=0)throw std::invalid_argument("QSA prefetch owner alignment invalid");
        unsigned eligible=0;
        for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.index_q,&tensors.index_k})if(qsa_input_prefetch::eligible(*tensor->manifest)){
            ++eligible;const auto bytes=qsa_input_prefetch::required_weight_bytes(tensor->manifest->dimensions[0],tensor->manifest->dimensions[1]);
            if(tensor->manifest->absolute_offset%16U!=0 || tensor->runtime->address%16U!=0 || tensor->manifest->byte_size<bytes || tensor->runtime->bytes<bytes)throw std::invalid_argument("QSA prefetch weight alignment/extent invalid");
        }
        if(eligible==0)throw std::invalid_argument("QSA prefetch step has no eligible projection");
        struct Range{std::uintptr_t address;std::uint64_t bytes;};
        std::array<Range,11> reads{};reads[0]={input.address,input.bytes};unsigned read_index=1;
        for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.out,&tensors.q_norm,&tensors.k_norm,&tensors.index_q,&tensors.index_k,&tensors.index_q_norm,&tensors.index_k_norm})reads[read_index++]={tensor->runtime->address,tensor->runtime->bytes};
        const std::array<Range,4> writes{{{temporary.address,temporary.bytes},{output.address,output.bytes},{kv_state.address,kv_state.bytes},{raw_state.address,raw_state.bytes}}};
        auto valid=[](Range view){return view.address!=0 && view.bytes!=0 && view.bytes<=UINTPTR_MAX-view.address;};
        auto overlap=[](Range x,Range y){return x.address<y.address+y.bytes && y.address<x.address+x.bytes;};
        for(auto view:reads)if(!valid(view))throw std::invalid_argument("QSA prefetch invalid read owner");
        for(unsigned i=0;i<writes.size();++i){if(!valid(writes[i]))throw std::invalid_argument("QSA prefetch invalid writable owner");for(auto view:reads)if(overlap(writes[i],view))throw std::invalid_argument("QSA prefetch writable/read owner overlap");for(unsigned j=0;j<i;++j)if(overlap(writes[i],writes[j]))throw std::invalid_argument("QSA prefetch writable owner overlap");}
    }

    if (config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1) {
        if ((config.qsa_hierarchical_selection && session_layout.context_tokens > qsa_hierarchical_selection_max_visible) ||
            session_layout.context_tokens > 262144 || temporary.address % 8U ||
            !context.layer || *context.layer >= 48 || device_graph)
            throw std::invalid_argument("hierarchical QSA context, layer or workspace alignment is outside admission");
        const auto& record = current_hip_launch_context()->deferred_moe_status;
        struct Range { std::uintptr_t address; std::uint64_t bytes; };
        const auto valid = [](Range x) { return x.address && x.bytes && x.bytes <= UINTPTR_MAX - x.address; };
        const auto overlaps = [](Range x, Range y) { return x.address < y.address + y.bytes && y.address < x.address + x.bytes; };
        const Range status{record.arena_base, record.arena_bytes};
        std::array<Range, 11> reads{};
        reads[0] = {input.address, input.bytes};
        std::size_t read_index = 1;
        for (const auto* tensor : {&tensors.q, &tensors.k, &tensors.v, &tensors.out, &tensors.q_norm,
            &tensors.k_norm, &tensors.index_q, &tensors.index_k, &tensors.index_q_norm, &tensors.index_k_norm})
            reads[read_index++] = {tensor->runtime->address, tensor->runtime->bytes};
        const std::array<Range, 4> writes{{{temporary.address, temporary.bytes}, {output.address, output.bytes},
            {kv_state.address, kv_state.bytes}, {raw_state.address, raw_state.bytes}}};
        for (const auto read : reads)
            if (!valid(read) || overlaps(status, read)) throw std::invalid_argument("hierarchical QSA invalid or status-aliasing read owner");
        for (std::size_t i = 0; i < writes.size(); ++i) {
            if (!valid(writes[i]) || overlaps(status, writes[i]))
                throw std::invalid_argument("hierarchical QSA invalid or status-aliasing write owner");
            for (const auto read : reads) if (overlaps(writes[i], read))
                throw std::invalid_argument("hierarchical QSA writable/read owner overlap");
            for (std::size_t j = 0; j < i; ++j) if (overlaps(writes[i], writes[j]))
                throw std::invalid_argument("hierarchical QSA writable owner overlap");
        }
        hipPointerAttribute_t attributes{};
        check_hip(hipPointerGetAttributes(&attributes, reinterpret_cast<const void*>(record.arena_base)),
            "HIP hierarchical QSA status device");
        int device = -1;
        check_hip(hipGetDevice(&device), "HIP hierarchical QSA current device");
        if (attributes.type != hipMemoryTypeDevice || attributes.isManaged || attributes.device != device)
            throw std::invalid_argument("hierarchical QSA D1 record is not on the current device");
        hipDeviceptr_t allocation{}; std::size_t bytes = 0;
        check_hip(hipMemGetAddressRange(&allocation, &bytes, reinterpret_cast<hipDeviceptr_t>(record.arena_base)),
            "HIP hierarchical QSA status extent");
        const auto base = reinterpret_cast<std::uintptr_t>(allocation);
        if (record.arena_base < base || record.arena_base - base > bytes || 8U > bytes - (record.arena_base - base))
            throw std::invalid_argument("hierarchical QSA D1 record exceeds physical owner");
    }
    if (device_graph) {
        const auto& scope = *current_hip_launch_context();
        if (!config.enable_device_graph_inputs || scope.context_tokens != session_layout.context_tokens ||
            scope.context_tokens > model.max_position_embeddings ||
            scope.context_tokens > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("indexed-attention device graph context/configuration is invalid");
        }
        GraphOperatorControlRequest request;
        request.execution_identity = context.execution_identity;
        request.context_tokens = scope.context_tokens;
        request.arguments = scope.arguments;
        request.status = scope.status;
        for (const auto* view : {&input, &output}) request.resources.push_back(
            {view->address, view->bytes, view->allocation_generation, 2});
        request.resources.push_back({temporary.address, temporary.bytes, temporary.allocation_generation, 4});
        for (const auto* state : {&kv_state, &raw_state}) request.resources.push_back(
            {state->address, state->bytes, state->allocation_generation, 2});
        for (const auto* tensor : {&tensors.q, &tensors.k, &tensors.v, &tensors.out,
            &tensors.q_norm, &tensors.k_norm, &tensors.index_q, &tensors.index_k,
            &tensors.index_q_norm, &tensors.index_k_norm}) request.resources.push_back(
            {tensor->runtime->address, tensor->runtime->bytes, tensor->runtime->allocation_generation,
             tensor->manifest->ggml_type == 0 ? 4U : 2U});
        (void)prepare_graph_operator_control(request);
    }
    if (config.qsa_hierarchical_selection || config.qsa_gqa_heads_per_block != 1)
        check_hip(hipPeekAtLastError(), "HIP hierarchical QSA preexisting error");
    return {input, output, temporary, workspace_layout, workspace, &session_layout,
            dynamic_position, device_graph, kv_state, raw_state, tensors, hidden,
            attention_heads, kv_heads, head_dim, attention_width, kv_width,
            q_gate_width, index_q_width, index_dim};
}

PreparedIndexedSelection prepare_indexed_attention_selection(const ExecutionDispatchContext& context,
    const PreparedIndexedLaunch& prepared, const ModelContract& model) {
    const auto& session_layout = *prepared.session_layout;
    const auto& workspace_layout = prepared.workspace_layout;
    const auto dynamic_position = prepared.dynamic_position;
    const auto device_graph = prepared.device_graph;
    const auto visible64 =
        device_graph ? session_layout.context_tokens : context.position + 1ULL;
    if (visible64 >
        std::numeric_limits<
            std::uint32_t>::max()) {
        throw std::overflow_error(
            "indexed-attention visible token count exceeds uint32");
    }
    const auto visible =
        static_cast<std::uint32_t>(
            visible64);
    const auto ratio =
        checked_u32(
            model.indexer_compress_ratio,
            "indexer_compress_ratio");
    const auto budget =
        checked_u32(
            model.indexer_budget,
            "indexer_budget");
    const auto complete_blocks =
        visible / ratio;
    const auto block_topk =
        budget / ratio;
    const auto selected_blocks =
        std::min(
            complete_blocks,
            block_topk);
    const auto tail =
        visible -
        complete_blocks *
            ratio;
    const auto selected_count = device_graph ? checked_u32(workspace_layout.maximum_selected_tokens, "maximum_selected_tokens") :
        selected_blocks *
            ratio +
        tail;

    if (selected_count == 0 ||
        selected_count >
            workspace_layout.
                maximum_selected_tokens) {
        throw std::logic_error(
            "indexed-attention selected-token count exceeds workspace");
    }

    const GraphSelectionArguments dynamic_selection{dynamic_position, ratio, block_topk};
    return {visible, ratio, block_topk, complete_blocks, selected_count, dynamic_selection};
}

std::uint64_t prepare_indexed_attention_score_grid(std::uint32_t attention_heads,
    std::uint32_t selected_count) {
    const auto score_blocks64 =
        static_cast<std::uint64_t>(
            attention_heads) *
        selected_count;
    if (score_blocks64 >
        std::numeric_limits<
            std::uint32_t>::max()) {
        throw std::overflow_error(
            "indexed-attention score grid exceeds uint32");
    }
    return score_blocks64;
}

void preflight_indexed_attention(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_indexed_attention_launch(context,
        static_cast<const HipIndexedAttentionLauncherConfig*>(user_data));
    const auto& model = static_cast<const HipIndexedAttentionLauncherConfig*>(user_data)->model;
    // These guards otherwise occur in nested launch helpers after preceding
    // projections. Preflight evaluates metadata only; the normal launch keeps
    // each original check at its original phase.
    static_cast<void>(checked_u32(model.indexer_n_heads, "indexer_n_heads"));
    static_cast<void>(checked_u32(model.rotary_dim(), "rotary_dim"));
    const auto selection = prepare_indexed_attention_selection(context, prepared, model);
    static_cast<void>(prepare_indexed_attention_score_grid(prepared.attention_heads, selection.selected_count));
    static_cast<void>(checked_u32(prepared.workspace_layout.maximum_selected_tokens, "maximum_selected_tokens"));
}

void launch_indexed_attention(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_indexed_attention_launch(context,
        static_cast<const HipIndexedAttentionLauncherConfig*>(user_data));
    const auto& config = *static_cast<const HipIndexedAttentionLauncherConfig*>(user_data);
    const auto& model = config.model;
    const auto& input = prepared.input; const auto& output = prepared.output;
    const auto& workspace_layout = prepared.workspace_layout; const auto& workspace = prepared.workspace;
    const auto dynamic_position = prepared.dynamic_position;
    const auto device_graph = prepared.device_graph;
    const auto& kv_state = prepared.kv_state; const auto& raw_state = prepared.raw_state;
    const auto& tensors = prepared.tensors;
    const auto hidden = prepared.hidden, attention_heads = prepared.attention_heads, kv_heads = prepared.kv_heads;
    const auto head_dim = prepared.head_dim, attention_width = prepared.attention_width, kv_width = prepared.kv_width;
    const auto q_gate_width = prepared.q_gate_width, index_q_width = prepared.index_q_width, index_dim = prepared.index_dim;
    if (device_graph) {
        hipLaunchKernelGGL(validate_indexed_graph_control, dim3(1), dim3(kThreads), 0,
            hip_execution_stream(), dynamic_position,
            reinterpret_cast<const std::uint16_t*>(input.address), hidden);
        check_hip(hipGetLastError(), "HIP indexed-attention device control validation");
    }

    // Indexer projections. Outputs are rounded to BF16 values represented in
    // FP32 workspace so subsequent FP32 normalization starts from model dtype.
    launch_linear_to_float(
        tensors.index_q,
        input,
        hidden,
        index_q_width,
        workspace.index_query, config.prefetch_q8_input, config.prefetch_stats.get());
    launch_linear_to_float(
        tensors.index_k,
        input,
        hidden,
        index_dim,
        workspace.current_index_key, config.prefetch_q8_input, config.prefetch_stats.get());

    launch_norm_rope(
        workspace.index_query,
        checked_u32(
            model.indexer_n_heads,
            "indexer_n_heads"),
        index_dim,
        index_dim,
        tensors.index_q_norm,
        model,
        context.position);

    const auto raw_blocks =
        (index_dim +
         kThreads - 1U) /
        kThreads;
    validate_graph_values(workspace.current_index_key, index_dim);
    hipLaunchKernelGGL(
        store_raw_index_key_kernel,
        dim3(raw_blocks),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.current_index_key,
        reinterpret_cast<
            std::uint16_t*>(
                raw_state.address),
        index_dim,
        context.position, dynamic_position);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention raw-index state update");

    const auto selection = prepare_indexed_attention_selection(context, prepared, model);
    const auto visible = selection.visible, ratio = selection.ratio, block_topk = selection.block_topk;
    const auto complete_blocks = selection.complete_blocks, selected_count = selection.selected_count;
    const auto dynamic_selection = selection.dynamic_selection;
    if (!device_graph && complete_blocks <=
        block_topk) {
        const auto blocks =
            (visible +
             kThreads - 1U) /
            kThreads;
        hipLaunchKernelGGL(
            fill_visible_indices_kernel,
            dim3(blocks),
            dim3(kThreads),
            0, hip_execution_stream(),
            workspace.selected_indices,
            visible);
        check_hip(
            hipGetLastError(),
            "HIP indexed-attention full-visible selection");
    } else {
        launch_block_scores(
            workspace,
            raw_state,
            tensors.index_k_norm,
            model,
            std::max(complete_blocks, 1U), dynamic_selection);
        if (config.qsa_hierarchical_selection) {
            using namespace qsa_selection_diagnostic;
            const Shape shape{complete_blocks, visible - complete_blocks * ratio};
            const auto items = scratch_items(shape);
            check_hip(hipMemsetAsync(workspace.selection_status, 0, sizeof(std::uint32_t), hip_execution_stream()),
                "HIP hierarchical QSA local status reset");
            enqueue(shape, {workspace.block_scores, complete_blocks}, {workspace.selected_indices, selected_count},
                {workspace.selection_ping, items}, {workspace.selection_pong, items},
                {workspace.selection_status, 1}, hip_execution_stream());
            hipLaunchKernelGGL(publish_qsa_selection_status, dim3((selected_count + kThreads - 1U) / kThreads),
                dim3(kThreads), 0, hip_execution_stream(), workspace.selection_status,
                reinterpret_cast<DeferredMoeFirstError*>(hip_deferred_moe_status()), workspace.selected_indices, selected_count);
            check_hip(hipGetLastError(), "HIP hierarchical QSA deferred status publication");
            if (config.selection_stats) {
                config.selection_stats->submissions.fetch_add(1, std::memory_order_relaxed);
                config.selection_stats->layer_submissions[*context.layer].fetch_add(1, std::memory_order_relaxed);
            }
        } else {
        hipLaunchKernelGGL(
            select_blocks_kernel,
            dim3(1),
            dim3(1),
            0, hip_execution_stream(),
            workspace.block_scores,
            complete_blocks,
            block_topk,
            ratio,
            visible,
            workspace.selected_indices, dynamic_selection);
        check_hip(
            hipGetLastError(),
            "HIP indexed-attention top-k block selection");
        }
    }

    // Main Q/K/V projections.
    launch_linear_to_float(
        tensors.q,
        input,
        hidden,
        q_gate_width,
        workspace.q_gate, config.prefetch_q8_input, config.prefetch_stats.get());
    launch_linear_to_float(
        tensors.k,
        input,
        hidden,
        kv_width,
        workspace.current_k, config.prefetch_q8_input, config.prefetch_stats.get());
    launch_linear_to_float(
        tensors.v,
        input,
        hidden,
        kv_width,
        workspace.current_v, config.prefetch_q8_input, config.prefetch_stats.get());

    launch_norm_rope(
        workspace.q_gate,
        attention_heads,
        head_dim,
        head_dim * 2U,
        tensors.q_norm,
        model,
        context.position);
    launch_norm_rope(
        workspace.current_k,
        kv_heads,
        head_dim,
        head_dim,
        tensors.k_norm,
        model,
        context.position);

    const auto kv_blocks =
        (kv_width +
         kThreads - 1U) /
        kThreads;
    validate_graph_values(workspace.current_v, kv_width);
    hipLaunchKernelGGL(
        store_kv_kernel,
        dim3(kv_blocks),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.current_k,
        workspace.current_v,
        reinterpret_cast<
            std::uint16_t*>(
                kv_state.address),
        kv_width,
        context.position, dynamic_position);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention KV state update");

    if (config.qsa_gqa_heads_per_block != 1) {
        hipLaunchKernelGGL(prepare_qsa_gqa_control, dim3(1), dim3(1), 0, hip_execution_stream(),
            workspace.attention_position, workspace.selection_status, context.position,
            config.qsa_hierarchical_selection && complete_blocks > block_topk);
        check_hip(hipGetLastError(), "HIP QSA grouped attention control");
        using namespace qsa_gqa_reuse_diagnostic;
        const Shape shape{attention_heads, kv_heads, head_dim,
            checked_u32(prepared.session_layout->context_tokens, "QSA GQA context")};
        const auto maximum = maximum_selected_tokens(shape);
        const Bindings views{
            {reinterpret_cast<std::uintptr_t>(workspace.q_gate), static_cast<std::uint64_t>(q_gate_width) * sizeof(float)},
            {kv_state.address, static_cast<std::uint64_t>(shape.context_tokens) * kv_width * 2U * sizeof(std::uint16_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.selected_indices), static_cast<std::uint64_t>(maximum) * sizeof(std::uint32_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.attention_position), sizeof(std::uint64_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.selection_status), sizeof(std::uint32_t)},
            {reinterpret_cast<std::uintptr_t>(workspace.attention_scores), static_cast<std::uint64_t>(attention_heads) * maximum * sizeof(float)},
            {reinterpret_cast<std::uintptr_t>(workspace.attention_output), static_cast<std::uint64_t>(attention_width) * sizeof(float)}};
        enqueue(shape, views, config.qsa_gqa_heads_per_block == 2 ? Mode::heads2 : Mode::heads4, hip_execution_stream());
        hipLaunchKernelGGL(publish_qsa_attention_status, dim3(1), dim3(1), 0, hip_execution_stream(),
            workspace.selection_status, reinterpret_cast<DeferredMoeFirstError*>(hip_deferred_moe_status()));
        check_hip(hipGetLastError(), "HIP QSA grouped attention deferred publication");
        if (config.gqa_stats) {
            config.gqa_stats->submissions.fetch_add(1, std::memory_order_relaxed);
            config.gqa_stats->layer_submissions[*context.layer].fetch_add(1, std::memory_order_relaxed);
        }
    } else {
    const auto score_blocks64 = prepare_indexed_attention_score_grid(attention_heads, selected_count);
    hipLaunchKernelGGL(
        attention_score_kernel,
        dim3(
            static_cast<std::uint32_t>(
                score_blocks64)),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.q_gate,
        reinterpret_cast<
            const std::uint16_t*>(
                kv_state.address),
        workspace.selected_indices,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"),
        attention_heads,
        kv_heads,
        head_dim,
        workspace.attention_scores, dynamic_selection);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention score launch");

    hipLaunchKernelGGL(
        attention_softmax_kernel,
        dim3(attention_heads),
        dim3(kThreads),
        0, hip_execution_stream(),
        workspace.attention_scores,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"), dynamic_selection);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention softmax launch");

    hipLaunchKernelGGL(
        attention_output_kernel,
        dim3(attention_heads),
        dim3(head_dim),
        0, hip_execution_stream(),
        workspace.q_gate,
        reinterpret_cast<
            const std::uint16_t*>(
                kv_state.address),
        workspace.selected_indices,
        workspace.attention_scores,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"),
        attention_heads,
        kv_heads,
        head_dim,
        workspace.attention_output, dynamic_selection);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention value/gate launch");

    }

    launch_linear_float_to_bf16(
        tensors.out,
        workspace.attention_output,
        attention_width,
        hidden,
        output, config.wave8_q8_output);
    if (device_graph) {
        hipLaunchKernelGGL(validate_indexed_graph_control, dim3(1), dim3(kThreads), 0,
            hip_execution_stream(), dynamic_position,
            reinterpret_cast<const std::uint16_t*>(output.address), hidden);
        check_hip(hipGetLastError(), "HIP indexed-attention finite output validation");
    }
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config) {
    return
        "hip.indexed_attention.bf16.h" +
        std::to_string(
            config.model.hidden_size) +
        ".q" +
        std::to_string(
            config.model.attention_heads) +
        "x" +
        std::to_string(
            config.model.head_dim) +
        ".kv" +
        std::to_string(
            config.model.kv_heads) +
        ".idx" +
        std::to_string(
            config.model.indexer_n_heads) +
        "x" +
        std::to_string(
            config.model.indexer_head_dim) +
        ".b" +
        std::to_string(
            config.model.indexer_budget) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (request.device_graph_inputs ? ".device" : ".scalar") +
        std::string(qsa_gqa_organization_marker(config.qsa_gqa_heads_per_block)) +
        (config.qsa_hierarchical_selection ? std::string(qsa_hierarchical_selection_marker) : "") +
        (config.wave8_q8_output ? std::string(q8_projection_organization::qsa_output_wave8_marker) : "") +
        (config.prefetch_q8_input ? std::string(qsa_input_prefetch::kernel_suffix) : "");
}

std::uint64_t implementation_identity(
    const HipIndexedAttentionLauncherConfig& config) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.attention_heads);
    hash_u64(
        hash,
        config.model.kv_heads);
    hash_u64(
        hash,
        config.model.head_dim);
    hash_u64(
        hash,
        config.model.indexer_n_heads);
    hash_u64(
        hash,
        config.model.indexer_kv_heads);
    hash_u64(
        hash,
        config.model.indexer_head_dim);
    hash_u64(
        hash,
        config.model.indexer_budget);
    hash_u64(
        hash,
        config.model.indexer_compress_ratio);
    hash_u64(
        hash,
        config.model.rotary_dim());
    hash_float(
        hash,
        config.model.rope_theta);
    hash_float(
        hash,
        config.model.rms_norm_epsilon);
    hash_u64(hash, config.enable_device_graph_inputs);
    hash_u64(hash, config.model.max_position_embeddings);
    if(config.prefetch_q8_input){hash_u64(hash,0x5153415052465631ULL);for(auto layer:config.prefetch_eligible_layers)hash_u64(hash,layer);}
    if(config.wave8_q8_output) hash_u64(hash, 0x5153414f57385631ULL);
    if(config.qsa_hierarchical_selection) hash_u64(hash, 0x515341544f503531ULL);
    if(config.qsa_gqa_heads_per_block != 1) {
        hash_u64(hash, 0x5153414751417631ULL); hash_u64(hash, config.qsa_gqa_heads_per_block);
    }
    return hash;
}

}  // namespace

void launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode mode,HipQsaInputProjectionView input,HipQsaInputProjectionView weights,HipQsaInputProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream){
    const auto required=qsa_input_prefetch::required_weight_bytes(width,outputs);
    const auto valid=[](HipQsaInputProjectionView v,std::uint64_t n,unsigned align){return v.address!=0 && v.address%align==0 && v.bytes==n && v.bytes<=UINTPTR_MAX-v.address;};
    if((mode!=HipQsaInputProjectionMode::legacy256 && mode!=HipQsaInputProjectionMode::virtual256 && mode!=HipQsaInputProjectionMode::prefetch8) || !valid(input,static_cast<std::uint64_t>(width)*2U,16) || !valid(weights,required,16) || !valid(output,static_cast<std::uint64_t>(outputs)*4U,16))throw std::invalid_argument("QSA input projection probe exact views invalid");
    const auto overlap=[](HipQsaInputProjectionView x,HipQsaInputProjectionView y){return x.address<y.address+y.bytes && y.address<x.address+x.bytes;};
    if(overlap(output,input)||overlap(output,weights))throw std::invalid_argument("QSA input projection probe writable alias");
    check_hip(hipPeekAtLastError(),"QSA projection probe stale error");
    auto hip_stream=static_cast<hipStream_t>(stream);auto* in=reinterpret_cast<const std::uint16_t*>(input.address);auto* w=reinterpret_cast<const void*>(weights.address);auto* out=reinterpret_cast<float*>(output.address);
    if(mode==HipQsaInputProjectionMode::legacy256)hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_kernel<WeightKind::q8_0>),dim3(outputs),dim3(256),0,hip_stream,in,w,width,outputs,out);
    else if(mode==HipQsaInputProjectionMode::virtual256)hipLaunchKernelGGL(linear_virtual256_q8_qsa_input_kernel,dim3(outputs),dim3(32),0,hip_stream,in,w,width,outputs,out);
    else hipLaunchKernelGGL(linear_prefetch8_q8_qsa_input_kernel,dim3(outputs),dim3(32),0,hip_stream,in,w,width,outputs,out);
    check_hip(hipGetLastError(),"QSA input projection probe launch");
}

#ifdef STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC
void launch_hip_qsa_output_projection_probe(HipQsaOutputProjectionMode mode,
    HipQsaInputProjectionView input, HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output, std::uint32_t width, std::uint32_t outputs, void* stream) {
    if ((mode != HipQsaOutputProjectionMode::legacy256 && mode != HipQsaOutputProjectionMode::wave8) ||
        !stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("QSA output projection diagnostic requires mode and owned stream");
    q8_projection_organization::validate_fp32_bf16_views(width, outputs,
        {input.address, input.bytes}, {weights.address, weights.bytes}, {output.address, output.bytes});
    auto* in = reinterpret_cast<const float*>(input.address);
    auto* out = reinterpret_cast<std::uint16_t*>(output.address);
    const auto* weight = reinterpret_cast<const void*>(weights.address);
    const auto owned_stream = static_cast<hipStream_t>(stream);
    if (mode == HipQsaOutputProjectionMode::legacy256)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_float_to_bf16_kernel<WeightKind::q8_0>),
            dim3(outputs), dim3(256), 0, owned_stream, in, weight, width, outputs, out);
    else
        hipLaunchKernelGGL(linear_prefetch8_wave8_q8_float_to_bf16_kernel,
            dim3((outputs + 7U) / 8U), dim3(256), 0, owned_stream, in, weight, width, outputs, out);
    check_hip(hipGetLastError(), "HIP QSA output projection diagnostic launch");
}
#endif

void register_hip_indexed_attention_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    validate_config(config);
    if (request.qsa_gqa_heads_per_block != config.qsa_gqa_heads_per_block)
        throw std::invalid_argument("QSA GQA request/config mismatch");
    if (config.qsa_gqa_heads_per_block != 1 && (!request.deferred_moe_status || request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("QSA GQA organization registers eager D1 variants only");
    if (request.qsa_hierarchical_selection != config.qsa_hierarchical_selection)
        throw std::invalid_argument("hierarchical QSA request/config mismatch");
    if (config.qsa_hierarchical_selection && (!request.deferred_moe_status || request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("hierarchical QSA registers eager D1 variants only");
    if (request.wave8_q8_qsa_output != config.wave8_q8_output)
        throw std::invalid_argument("wave8 QSA output request/config mismatch");
    if (config.wave8_q8_output && (request.device_graph_inputs || request.require_graph_safe))
        throw std::invalid_argument("wave8 QSA output registers eager variants only");
    if(config.prefetch_q8_input!=request.prefetch_q8_qsa_input || config.prefetch_eligible_layers!=request.qsa_input_prefetch_layers)throw std::invalid_argument("indexed QSA prefetch config differs from binding request");
    qsa_input_prefetch::validate_layers(model,request.prefetch_q8_qsa_input,request.qsa_input_prefetch_layers);

    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.attention_heads !=
            model.attention_heads ||
        config.model.kv_heads !=
            model.kv_heads ||
        config.model.head_dim !=
            model.head_dim ||
        config.model.indexer_n_heads !=
            model.indexer_n_heads ||
        config.model.indexer_head_dim !=
            model.indexer_head_dim ||
        config.model.indexer_budget !=
            model.indexer_budget ||
        config.model.indexer_compress_ratio !=
            model.indexer_compress_ratio) {
        throw std::invalid_argument(
            "HIP indexed-attention config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 registers only C1 one-row variants");
    }

    if (request.device_graph_inputs && (!request.require_graph_safe || !config.enable_device_graph_inputs)) {
        throw std::invalid_argument("indexed device graph inputs require explicit graph configuration");
    }

    // QSA top-k ties remain source-undefined. Keep v1 experimental until a
    // canonical tie contract or tie-free checkpoint fixtures qualify it.
    if (qualification !=
        kernels::QualificationStatus::
            experimental) {
        throw std::invalid_argument(
            "HIP indexed-attention v1 remains experimental while QSA tie semantics are unresolved");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            indexed_attention;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::indexed_attention,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        2U *
        kThreads *
        sizeof(float);
    variant.resources.scratch_bytes =
        build_indexed_attention_workspace_layout(
            model, 256, config.qsa_hierarchical_selection, config.qsa_gqa_heads_per_block).total_bytes;
    if (config.qsa_hierarchical_selection) variant.resources.lds_bytes = 8192;
    if (config.qsa_gqa_heads_per_block != 1) variant.resources.lds_bytes =
        std::max(variant.resources.lds_bytes, config.qsa_gqa_heads_per_block * 1024U);
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
        "row-major+layer-kv+raw-index";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        config.prefetch_q8_input ? "hip-gfx1151-indexed-attention-v3-prefetch8-virtual256-q8-qsa-input" : "hip-gfx1151-indexed-attention-v2";
    if (config.wave8_q8_output) variant.compiler_identity += ":wave8-q8-qsa-output-v1";
    if (config.qsa_hierarchical_selection) variant.compiler_identity += ":hierarchical-top512-v1";
    if (config.qsa_gqa_heads_per_block != 1) variant.compiler_identity += ":gqa-heads" + std::to_string(config.qsa_gqa_heads_per_block) + "-v1";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_indexed_attention_launcher(
    std::string kernel_id_value,
    HipIndexedAttentionLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP indexed-attention launcher requires id and config");
    }
    validate_config(*config);
    if (config->wave8_q8_output || config->qsa_hierarchical_selection || config->qsa_gqa_heads_per_block != 1 ||
        kernel_id_value.find(".gqa-heads2-v1") != std::string::npos || kernel_id_value.find(".gqa-heads4-v1") != std::string::npos ||
        kernel_id_value.find(qsa_hierarchical_selection_marker) != std::string::npos ||
        kernel_id_value.find(q8_projection_organization::qsa_output_wave8_marker) != std::string::npos) {
        ExecutionBindingRequest request;
        if (kernel_id_value != kernel_id(request, *config))
            throw std::invalid_argument("wave8 QSA output launcher requires its exact eager C1 kernel id");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_indexed_attention,
        config,
        &preflight_indexed_attention,
    };
}

void HipIndexedAttentionLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP indexed-attention bundle requires valid compiled execution");
    }

    if(config_.prefetch_q8_input && std::none_of(compiled.bindings.bindings.begin(),
        compiled.bindings.bindings.end(),[](const auto& binding){return binding.op==SerialOp::indexed_attention &&
            binding.kernel_id && binding.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix);}))
        throw std::invalid_argument("QSA prefetch bundle requires eligible bound step");
    ExecutionBindingRequest shape;shape.logical_rows=compiled.logical_rows;shape.request_concurrency=compiled.request_concurrency;shape.device_graph_inputs=compiled.device_graph_inputs;
    std::vector<std::string> registered;bool saw_prefetch=false;
    for(std::size_t i=0;i<compiled.execution.steps.size();++i){
        if(compiled.execution.steps[i].op!=SerialOp::indexed_attention)continue;
        const auto& binding=compiled.bindings.bindings[i];
        if(binding.host_or_runtime_step || !binding.kernel_id || binding.kernel_id->empty())throw std::invalid_argument("HIP indexed-attention step has no kernel binding");
        const bool prefetch=binding.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix);
        if(prefetch!=qsa_input_prefetch::selected(config_.prefetch_q8_input,config_.prefetch_eligible_layers,compiled.execution.steps[i].layer))throw std::invalid_argument("HIP indexed QSA prefetch binding/config mismatch");
        auto* selected=prefetch?&config_:&fallback_config_;
        if(*binding.kernel_id!=kernel_id(shape,*selected))throw std::invalid_argument("HIP indexed-attention bundle permits only its base/prefetch pair");
        if(std::find(registered.begin(),registered.end(),*binding.kernel_id)==registered.end()){registry.add_kernel_launcher(make_hip_indexed_attention_launcher(*binding.kernel_id,selected));registered.push_back(*binding.kernel_id);}
        saw_prefetch|=prefetch;
    }
    if(registered.empty())throw std::invalid_argument("compiled execution has no indexed-attention step");
    if(config_.prefetch_q8_input && !saw_prefetch)throw std::invalid_argument("QSA prefetch bundle requires eligible bound step");
}

std::vector<std::uint64_t>
HipIndexedAttentionLauncherBundle::known_window_position_readers(
    const CompiledSerialExecution& compiled,
    const ExecutionLauncherRegistry& registry) const {
    validate_config(config_);
    validate_config(fallback_config_);
    if (config_.enable_device_graph_inputs || fallback_config_.enable_device_graph_inputs ||
        compiled.device_graph_inputs || compiled.logical_rows != 1 ||
        compiled.request_concurrency != 1 || compiled.execution_identity == 0 ||
        compiled.execution.steps.size() != compiled.bindings.bindings.size()) {
        throw std::invalid_argument("QSA known-window readers require a valid eager C1 execution");
    }
    ExecutionBindingRequest shape;
    shape.logical_rows = compiled.logical_rows;
    shape.request_concurrency = compiled.request_concurrency;
    shape.device_graph_inputs = compiled.device_graph_inputs;
    std::vector<std::uint64_t> identities;
    bool saw_prefetch = false;
    for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
        const auto& step = compiled.execution.steps[i];
        const auto& binding = compiled.bindings.bindings[i];
        if (step.op != SerialOp::indexed_attention) {
            if (binding.op == SerialOp::indexed_attention)
                throw std::invalid_argument("QSA known-window binding operator differs from execution");
            continue;
        }
        if (step.id != i || binding.step_id != step.id || binding.op != step.op ||
            !step.layer || binding.host_or_runtime_step || !binding.kernel_id || binding.kernel_id->empty()) {
            throw std::invalid_argument("QSA known-window reader has invalid step binding");
        }
        const bool prefetch = qsa_input_prefetch::selected(
            config_.prefetch_q8_input, config_.prefetch_eligible_layers, step.layer);
        const auto* selected = prefetch ? &config_ : &fallback_config_;
        if (*binding.kernel_id != kernel_id(shape, *selected))
            throw std::invalid_argument("QSA known-window reader differs from bundle configuration");
        const auto* launcher = registry.find_kernel_launcher(*binding.kernel_id);
        const auto identity = implementation_identity(*selected);
        if (!launcher || launcher->key != *binding.kernel_id || identity == 0 ||
            launcher->implementation_identity != identity || launcher->function != &launch_indexed_attention ||
            launcher->preflight != &preflight_indexed_attention || launcher->user_data != selected) {
            throw std::invalid_argument("QSA known-window reader is not this bundle's exact launcher");
        }
        if (std::find(identities.begin(), identities.end(), identity) == identities.end())
            identities.push_back(identity);
        saw_prefetch |= prefetch;
    }
    if (identities.empty() || (config_.prefetch_q8_input && !saw_prefetch))
        throw std::invalid_argument("QSA known-window bundle has no admitted eligible reader");
    return identities;
}

namespace {
template <SmallWeightKind Kind>
void launch_graph_attention_state_typed(const GraphAttentionStatePlan& p,
    std::uint64_t scalar_position, hipStream_t stream) {
    const auto& r = p.request;
    GraphPositionKernelArguments dynamic;
    dynamic.context_tokens = r.context_tokens;
    dynamic.status = reinterpret_cast<std::uint32_t*>(r.status.arena_base);
    if (r.position_mode == AttentionPositionMode::device_input) {
        dynamic.position = reinterpret_cast<const std::uint64_t*>(
            r.arguments.memory.arena_base + r.arguments.position_offset);
    }
    hipLaunchKernelGGL(initialize_graph_attention_status, dim3(1), dim3(1), 0, stream,
        dynamic.status, scalar_position, dynamic);
    check_hip(hipGetLastError(), "HIP attention state status initialization");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(validate_graph_attention_sources<Kind>), dim3(32),
        dim3(kThreads), 0, stream, reinterpret_cast<const float*>(r.query.address),
        reinterpret_cast<const float*>(r.key.address), reinterpret_cast<const float*>(r.value.address),
        reinterpret_cast<const void*>(r.query_norm.address), reinterpret_cast<const void*>(r.key_norm.address),
        p.query_elements, p.kv_elements, r.head_dim, r.query_head_stride, dynamic.status);
    check_hip(hipGetLastError(), "HIP attention state finite-source validation");
    ModelContract model;
    model.head_dim = r.head_dim;
    model.partial_rotary_factor = static_cast<float>(r.rotary_dim) / static_cast<float>(r.head_dim);
    model.rms_norm_epsilon = r.epsilon;
    model.rope_theta = r.theta;
    launch_norm_rope_typed<Kind>(reinterpret_cast<float*>(r.query.address), r.query_heads,
        r.head_dim, r.query_head_stride, r.query_norm, model, scalar_position, dynamic, stream);
    launch_norm_rope_typed<Kind>(reinterpret_cast<float*>(r.key.address), r.kv_heads,
        r.head_dim, r.head_dim, r.key_norm, model, scalar_position, dynamic, stream);
    const auto width = static_cast<std::uint32_t>(p.kv_elements);
    hipLaunchKernelGGL(store_kv_kernel, dim3((width - 1U) / kThreads + 1U), dim3(kThreads),
        0, stream, reinterpret_cast<const float*>(r.key.address), reinterpret_cast<const float*>(r.value.address),
        reinterpret_cast<std::uint16_t*>(r.kv_state.address), width, scalar_position, dynamic);
    check_hip(hipGetLastError(), "HIP attention state dynamic KV write");
}
} // namespace

void launch_hip_graph_attention_state(const GraphAttentionStatePlan& plan,
    std::uint64_t scalar_control_position, void* stream_pointer) {
    validate_graph_attention_state_plan(plan);
    if (stream_pointer == nullptr) throw std::invalid_argument("attention state requires an explicitly owned stream");
    const auto stream = static_cast<hipStream_t>(stream_pointer);
    if (plan.request.position_mode == AttentionPositionMode::scalar_control) {
        if (scalar_control_position >= plan.request.context_tokens) throw std::out_of_range("scalar attention position exceeds context");
        hipStreamCaptureStatus capturing;
        check_hip(hipStreamIsCapturing(stream, &capturing), "HIP attention control capture query");
        if (capturing != hipStreamCaptureStatusNone) throw std::invalid_argument("scalar attention control cannot be captured as a dynamic plan");
    } else if (scalar_control_position != 0) {
        throw std::invalid_argument("device-input attention mode does not accept a scalar position override");
    }
    switch (small_weight_kind(plan.request.norm_ggml_type)) {
    case SmallWeightKind::f32: launch_graph_attention_state_typed<SmallWeightKind::f32>(plan, scalar_control_position, stream); return;
    case SmallWeightKind::f16: launch_graph_attention_state_typed<SmallWeightKind::f16>(plan, scalar_control_position, stream); return;
    case SmallWeightKind::bf16: launch_graph_attention_state_typed<SmallWeightKind::bf16>(plan, scalar_control_position, stream); return;
    }
    throw std::logic_error("unreachable attention norm encoding");
}

namespace {
// Shared grouping of independent heads. Each head retains the incumbent
// 256-feature product/store and 128->1 tree. Only the BF16 K load is shared.
template <std::uint32_t Heads>
__global__ void qsa_gqa_score_diagnostic_kernel(
    const float* q_gate, const std::uint16_t* kv_state, const std::uint32_t* selected,
    std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float* scores, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto flat = static_cast<std::uint32_t>(blockIdx.x);
    const auto selected_slot = flat % maximum_selected;
    if (selected_slot >= selected_count) return;
    const auto group = attention_heads / kv_heads;
    const auto blocks_per_group = (group + Heads - 1U) / Heads;
    const auto head_block = flat / maximum_selected;
    const auto kv_head = head_block / blocks_per_group;
    const auto first_head = kv_head * group + (head_block % blocks_per_group) * Heads;
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    const auto kv_width = kv_heads * head_dim;
    const auto row_base = static_cast<std::uint64_t>(selected[selected_slot]) * kv_width * 2ULL;
    float key = 0.0F;
    if (feature < head_dim)
        key = bf16_to_float(kv_state[row_base + static_cast<std::uint64_t>(kv_head) * head_dim + feature]);
    __shared__ float scratch[Heads][kThreads];
#pragma unroll
    for (std::uint32_t local = 0; local < Heads; ++local) {
        const auto head = first_head + local;
        float product = 0.0F;
        if (head < (kv_head + 1U) * group && feature < head_dim)
            product = q_gate[static_cast<std::uint64_t>(head) * head_dim * 2ULL + feature] * key;
        scratch[local][feature] = product;
    }
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) {
#pragma unroll
            for (std::uint32_t local = 0; local < Heads; ++local)
                scratch[local][feature] += scratch[local][feature + stride];
        }
        __syncthreads();
    }
    if (feature == 0) {
#pragma unroll
        for (std::uint32_t local = 0; local < Heads; ++local) {
            const auto head = first_head + local;
            if (head < (kv_head + 1U) * group) {
                if (dynamic.position.status != nullptr && !isfinite(scratch[local][0])) atomicOr(dynamic.position.status, 8U);
                scores[static_cast<std::uint64_t>(head) * maximum_selected + selected_slot] =
                    scratch[local][0] / sqrtf(static_cast<float>(head_dim));
            }
        }
    }
}

// Each independent accumulator visits slots in the incumbent chronological
// order and keeps the same score*value operand order and BF16/gate epilogue.
template <std::uint32_t Heads>
__global__ void qsa_gqa_value_diagnostic_kernel(
    const float* q_gate, const std::uint16_t* kv_state, const std::uint32_t* selected,
    const float* scores, std::uint32_t selected_count, std::uint32_t maximum_selected,
    std::uint32_t attention_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float* output, GraphSelectionArguments dynamic) {
    if (!graph_selected_count(dynamic, selected_count)) return;
    const auto group = attention_heads / kv_heads;
    const auto blocks_per_group = (group + Heads - 1U) / Heads;
    const auto head_block = static_cast<std::uint32_t>(blockIdx.x);
    const auto kv_head = head_block / blocks_per_group;
    const auto first_head = kv_head * group + (head_block % blocks_per_group) * Heads;
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    const auto kv_width = kv_heads * head_dim;
    float sums[Heads]{};
    for (std::uint32_t slot = 0; slot < selected_count; ++slot) {
        const auto row_base = static_cast<std::uint64_t>(selected[slot]) * kv_width * 2ULL;
        const auto value = bf16_to_float(kv_state[row_base + kv_width +
            static_cast<std::uint64_t>(kv_head) * head_dim + feature]);
#pragma unroll
        for (std::uint32_t local = 0; local < Heads; ++local) {
            const auto head = first_head + local;
            if (head < (kv_head + 1U) * group)
                sums[local] += scores[static_cast<std::uint64_t>(head) * maximum_selected + slot] * value;
        }
    }
#pragma unroll
    for (std::uint32_t local = 0; local < Heads; ++local) {
        const auto head = first_head + local;
        if (head < (kv_head + 1U) * group) {
            const float attn = round_bf16(sums[local]);
            const float gate = round_bf16(1.0F / (1.0F + expf(-q_gate[
                static_cast<std::uint64_t>(head) * head_dim * 2ULL + head_dim + feature])));
            output[static_cast<std::uint64_t>(head) * head_dim + feature] = round_bf16(attn * gate);
        }
    }
}

__global__ void qsa_gqa_validate_diagnostic_kernel(const float* q_gate, const std::uint16_t* kv_state,
    const std::uint32_t* selected, std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, GraphSelectionArguments dynamic) {
    __shared__ std::uint32_t admitted;
    __shared__ std::uint32_t count;
    __shared__ std::uint32_t visible;
    if (threadIdx.x == 0) {
        admitted = atomicAdd(dynamic.position.status, 0U) == 0U;
        if (admitted) {
            const auto position = *dynamic.position.position;
            if (position >= dynamic.position.context_tokens) {
                atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::invalid_position); admitted = 0U;
            } else {
                visible = static_cast<std::uint32_t>(position + 1ULL);
                count = min(visible / dynamic.ratio, dynamic.block_topk) * dynamic.ratio + visible % dynamic.ratio;
            }
        }
    }
    __syncthreads();
    if (!admitted) return;
    for (std::uint32_t index = threadIdx.x; index < attention_heads * head_dim * 2U; index += kThreads)
        if (!isfinite(q_gate[index])) atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::nonfinite_source);
    for (std::uint32_t slot = threadIdx.x; slot < count; slot += kThreads) {
        const auto token = selected[slot];
        if (token >= visible || (slot && selected[slot - 1U] >= token))
            atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::invalid_selection);
    }
    __syncthreads();
    if (threadIdx.x == 0) admitted = atomicAdd(dynamic.position.status, 0U) == 0U;
    __syncthreads();
    if (!admitted) return; // Never gather an unvalidated selected ID.
    const auto row_elements = kv_heads * head_dim * 2U;
    const auto elements = static_cast<std::uint64_t>(count) * row_elements;
    for (std::uint64_t index = threadIdx.x; index < elements; index += kThreads) {
        const auto token = selected[index / row_elements];
        const auto bits = kv_state[static_cast<std::uint64_t>(token) * row_elements + index % row_elements];
        if ((bits & 0x7f80U) == 0x7f80U)
            atomicOr(dynamic.position.status, qsa_gqa_reuse_diagnostic::nonfinite_source);
    }
}
__global__ void qsa_gqa_validate_output_diagnostic_kernel(const float* output, std::uint32_t count, std::uint32_t* status) {
    if (atomicAdd(status, 0U) != 0U) return;
    for (std::uint32_t index = threadIdx.x; index < count; index += kThreads)
        if (!isfinite(output[index])) atomicOr(status, qsa_gqa_reuse_diagnostic::invalid_arithmetic);
}
} // namespace

namespace qsa_gqa_reuse_diagnostic {
const void* score_kernel_symbol(Mode mode) {
    switch (mode) {
    case Mode::reference: return reinterpret_cast<const void*>(attention_score_kernel);
    case Mode::heads2: return reinterpret_cast<const void*>(qsa_gqa_score_diagnostic_kernel<2U>);
    case Mode::heads4: return reinterpret_cast<const void*>(qsa_gqa_score_diagnostic_kernel<4U>);
    }
    throw std::invalid_argument("invalid QSA GQA diagnostic mode");
}
const void* value_kernel_symbol(Mode mode) {
    switch (mode) {
    case Mode::reference: return reinterpret_cast<const void*>(attention_output_kernel);
    case Mode::heads2: return reinterpret_cast<const void*>(qsa_gqa_value_diagnostic_kernel<2U>);
    case Mode::heads4: return reinterpret_cast<const void*>(qsa_gqa_value_diagnostic_kernel<4U>);
    }
    throw std::invalid_argument("invalid QSA GQA diagnostic mode");
}
namespace {
void enqueue_impl(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream, bool scores_only) {
    const auto launch = validate(shape, bindings, mode);
    if (!stream || stream == hipStreamPerThread || stream == hipStreamLegacy)
        throw std::invalid_argument("QSA GQA diagnostic requires an explicitly owned stream");
    check_hip(hipPeekAtLastError(), "QSA GQA diagnostic stale error");
    const auto* query = reinterpret_cast<const float*>(bindings.q_gate.address);
    const auto* kv = reinterpret_cast<const std::uint16_t*>(bindings.kv_state.address);
    const auto* selected = reinterpret_cast<const std::uint32_t*>(bindings.selected.address);
    auto* scores = reinterpret_cast<float*>(bindings.scores.address);
    auto* output = reinterpret_cast<float*>(bindings.output.address);
    auto* status = reinterpret_cast<std::uint32_t*>(bindings.status.address);
    const GraphSelectionArguments dynamic{{reinterpret_cast<const std::uint64_t*>(bindings.position.address),
        shape.context_tokens, status}, compression, block_topk};
    hipLaunchKernelGGL(qsa_gqa_validate_diagnostic_kernel, dim3(1), dim3(kThreads), 0, stream,
        query, kv, selected, shape.query_heads, shape.kv_heads, shape.head_dim, dynamic);
    check_hip(hipGetLastError(), "QSA GQA diagnostic source validation");
#define STRIX_QSA_GQA_SCORE(kernel) \
    hipLaunchKernelGGL(kernel, dim3(launch.score_blocks), dim3(kThreads), 0, stream, \
        query, kv, selected, launch.maximum_selected, launch.maximum_selected, \
        shape.query_heads, shape.kv_heads, shape.head_dim, scores, dynamic)
    switch (mode) {
    case Mode::reference: STRIX_QSA_GQA_SCORE(attention_score_kernel); break;
    case Mode::heads2: STRIX_QSA_GQA_SCORE(HIP_KERNEL_NAME(qsa_gqa_score_diagnostic_kernel<2U>)); break;
    case Mode::heads4: STRIX_QSA_GQA_SCORE(HIP_KERNEL_NAME(qsa_gqa_score_diagnostic_kernel<4U>)); break;
    }
#undef STRIX_QSA_GQA_SCORE
    check_hip(hipGetLastError(), "QSA GQA diagnostic score");
    if (scores_only) return;
    hipLaunchKernelGGL(attention_softmax_kernel, dim3(shape.query_heads), dim3(kThreads), 0, stream,
        scores, launch.maximum_selected, launch.maximum_selected, dynamic);
    check_hip(hipGetLastError(), "QSA GQA diagnostic original softmax");
#define STRIX_QSA_GQA_VALUE(kernel) \
    hipLaunchKernelGGL(kernel, dim3(launch.value_blocks), dim3(shape.head_dim), 0, stream, \
        query, kv, selected, scores, launch.maximum_selected, launch.maximum_selected, \
        shape.query_heads, shape.kv_heads, shape.head_dim, output, dynamic)
    switch (mode) {
    case Mode::reference: STRIX_QSA_GQA_VALUE(attention_output_kernel); break;
    case Mode::heads2: STRIX_QSA_GQA_VALUE(HIP_KERNEL_NAME(qsa_gqa_value_diagnostic_kernel<2U>)); break;
    case Mode::heads4: STRIX_QSA_GQA_VALUE(HIP_KERNEL_NAME(qsa_gqa_value_diagnostic_kernel<4U>)); break;
    }
#undef STRIX_QSA_GQA_VALUE
    check_hip(hipGetLastError(), "QSA GQA diagnostic value/gate");
    hipLaunchKernelGGL(qsa_gqa_validate_output_diagnostic_kernel, dim3(1), dim3(kThreads), 0, stream,
        output, shape.query_heads * shape.head_dim, status);
    check_hip(hipGetLastError(), "QSA GQA diagnostic finite output");
}
} // namespace
void enqueue(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream) {
    enqueue_impl(shape, bindings, mode, stream, false);
}
void enqueue_scores(Shape shape, const Bindings& bindings, Mode mode, hipStream_t stream) {
    enqueue_impl(shape, bindings, mode, stream, true);
}
} // namespace qsa_gqa_reuse_diagnostic

}  // namespace strix

namespace strix {
namespace {
// Cache-only teacher rows must not publish a finite FP32 value that overflows
// its durable BF16 cast. This validator has no reduction or payload mutation.
__global__ void validate_mtp_cache_casts(const float* index,std::uint32_t index_dim,
    const float* key,const float* value,std::uint32_t kv_width,std::uint32_t* status) {
    if(atomicAdd(status,0U)!=0U)return;
    const auto count=index_dim>kv_width?index_dim:kv_width;
    for(std::uint32_t i=threadIdx.x;i<count;i+=blockDim.x) {
        bool invalid=false;
        if(i<index_dim)invalid=!isfinite(index[i]) || (float_to_bf16(index[i])&0x7f80U)==0x7f80U;
        if(i<kv_width)invalid=invalid || !isfinite(key[i]) || !isfinite(value[i]) ||
            (float_to_bf16(key[i])&0x7f80U)==0x7f80U || (float_to_bf16(value[i])&0x7f80U)==0x7f80U;
        if(invalid)atomicOr(status,8U);
    }
}
struct MtpAttentionPrepared {
    MtpDeviceView input,output,work,kv,index,status;
    WorkspacePointers workspace;
    IndexedTensors tensors;
    std::uint32_t hidden,heads,kv_heads,dim,attention_width,kv_width,q_width,index_width,index_dim;
    std::uint32_t visible,ratio,topk,blocks,selected,maximum_selected;
};
MtpAttentionPrepared prepare_mtp_attention(const MtpNativeContext& c) {
    mtp_native::context(c);const auto& m=*c.model;const auto& t=*c.tensors;
    HipIndexedAttentionLauncherConfig config;config.model=m;validate_config(config);
    const auto hidden=checked_u32(m.hidden_size,"MTP attention hidden");
    const auto heads=checked_u32(m.attention_heads,"MTP attention heads");
    const auto kv_heads=checked_u32(m.kv_heads,"MTP attention KV heads");
    const auto dim=checked_u32(m.head_dim,"MTP attention dimension");
    const auto attention_width=checked_u32(mtp_native::product(heads,dim),"MTP attention width");
    const auto kv_width=checked_u32(mtp_native::product(kv_heads,dim),"MTP KV width");
    const auto q_width=checked_u32(mtp_native::product(attention_width,2U),"MTP query/gate width");
    const auto index_width=checked_u32(mtp_native::product(m.indexer_n_heads,m.indexer_head_dim),"MTP index query width");
    const auto index_dim=checked_u32(m.indexer_head_dim,"MTP index dimension");
    const auto pair=[](MtpTensorView x) {return TensorPair{x.manifest,x.runtime};};
    IndexedTensors tensors{pair(t.query),pair(t.key),pair(t.value),pair(t.attention_output),pair(t.query_norm),pair(t.key_norm),
        pair(t.index_query),pair(t.index_key),pair(t.index_query_norm),pair(t.index_key_norm)};
    for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.out})
        if(!tensor->manifest || !tensor->runtime || tensor->manifest->ggml_type!=8U)
            throw std::invalid_argument("MTP attention profile requires original Q8 projections");
    for(const auto* tensor:{&tensors.index_q,&tensors.index_k})
        if(!tensor->manifest || !tensor->runtime || tensor->manifest->ggml_type!=30U)
            throw std::invalid_argument("MTP attention profile requires original BF16 index projections");
    for(const auto* tensor:{&tensors.q_norm,&tensors.k_norm,&tensors.index_q_norm,&tensors.index_k_norm})
        if(!tensor->manifest || !tensor->runtime || tensor->manifest->ggml_type!=0U)
            throw std::invalid_argument("MTP attention profile requires effective F32 norm gains");
    validate_matrix(tensors.q,hidden,q_width,"MTP query");validate_matrix(tensors.k,hidden,kv_width,"MTP key");
    validate_matrix(tensors.v,hidden,kv_width,"MTP value");validate_matrix(tensors.out,attention_width,hidden,"MTP output");
    validate_matrix(tensors.index_q,hidden,index_width,"MTP index query");validate_matrix(tensors.index_k,hidden,index_dim,"MTP index key");
    validate_norm(tensors.q_norm,dim,"MTP query norm");validate_norm(tensors.k_norm,dim,"MTP key norm");
    validate_norm(tensors.index_q_norm,index_dim,"MTP index query norm");validate_norm(tensors.index_k_norm,index_dim,"MTP index key norm");
    const auto& layout=c.resources->attention_workspace();
    const auto input=mtp_native::buffer(c,HipMtpBuffer::hidden0,mtp_native::product(hidden,2U));
    const auto output=mtp_native::buffer(c,HipMtpBuffer::hidden1,mtp_native::product(hidden,2U));
    const auto work=mtp_native::buffer(c,HipMtpBuffer::attention_workspace,layout.total_bytes,256U);
    const auto kv=mtp_native::buffer(c,HipMtpBuffer::kv,mtp_native::product(mtp_native::product(c.resources->context_capacity(),kv_width),4U));
    const auto index=mtp_native::buffer(c,HipMtpBuffer::raw_index,mtp_native::product(mtp_native::product(c.resources->context_capacity(),index_dim),2U));
    const auto status=mtp_native::buffer(c,HipMtpBuffer::status,8U);
    mtp_native::disjoint({input,output,work,kv,index,status});
    for(const auto* tensor:{&tensors.q,&tensors.k,&tensors.v,&tensors.out,&tensors.q_norm,&tensors.k_norm,
        &tensors.index_q,&tensors.index_k,&tensors.index_q_norm,&tensors.index_k_norm}) {
        const MtpDeviceView weights{tensor->runtime->address,tensor->runtime->bytes,tensor->runtime->allocation_generation};
        const auto alignment=tensor->manifest->ggml_type==0U?4U:2U;
        if(!weights.address || weights.address%alignment || !weights.allocation_generation || weights.bytes>UINTPTR_MAX-weights.address)
            throw std::invalid_argument("MTP attention weight range is invalid");
        mtp_native::disjoint({work,output,kv,index,status,weights});
    }
    const auto visible=checked_u32(c.position+1U,"MTP visible source rows");
    const auto ratio=checked_u32(m.indexer_compress_ratio,"MTP compression ratio");
    const auto topk=checked_u32(m.indexer_budget/m.indexer_compress_ratio,"MTP selected blocks");
    const auto blocks=visible/ratio;
    const auto selected=std::min(blocks,topk)*ratio+visible%ratio;
    if(!selected || selected>layout.maximum_selected_tokens)
        throw std::invalid_argument("MTP attention selection exceeds retained workspace");
    (void)prepare_indexed_attention_score_grid(heads,selected);
    const ExecutionBufferAddress temporary{{},work.address,work.bytes,work.allocation_generation};
    return {input,output,work,kv,index,status,workspace_pointers(temporary,layout),tensors,hidden,heads,kv_heads,dim,
        attention_width,kv_width,q_width,index_width,index_dim,visible,ratio,topk,blocks,selected,
        checked_u32(layout.maximum_selected_tokens,"MTP maximum selected rows")};
}
void require_mtp_attention_stream(const MtpNativeContext& c,void* stream) {
    mtp_native::stream(c,stream);
    if(current_hip_launch_context()) throw std::invalid_argument("MTP attention cannot nest an active native scope");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check_hip(hipStreamIsCapturing(static_cast<hipStream_t>(stream),&capture),"MTP attention capture query");
    if(capture!=hipStreamCaptureStatusNone) throw std::invalid_argument("MTP attention supports eager execution only");
}
template<WeightKind Kind> void mtp_project(const MtpDeviceView& input,const TensorPair& tensor,
    std::uint32_t width,std::uint32_t outputs,float* output,hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_to_float_kernel<Kind>),dim3(outputs),dim3(kThreads),0,stream,
        reinterpret_cast<const std::uint16_t*>(input.address),reinterpret_cast<const void*>(tensor.runtime->address),width,outputs,output);
    check_hip(hipGetLastError(),"MTP attention projection enqueue");
}
void mtp_attention_finite(const MtpAttentionPrepared& p,const float* values,std::uint32_t count,hipStream_t stream) {
    hipLaunchKernelGGL(validate_indexed_graph_values,dim3(1),dim3(kThreads),0,stream,values,count,
        reinterpret_cast<std::uint32_t*>(p.status.address));
    check_hip(hipGetLastError(),"MTP attention finite validation enqueue");
}
}
void preflight_hip_mtp_attention(const MtpNativeContext& c) {(void)prepare_mtp_attention(c);}
void preflight_hip_mtp_cache_rows(const MtpNativeContext& c) {(void)prepare_mtp_attention(c);}
void launch_hip_mtp_cache_rows(const MtpNativeContext& c,void* raw_stream) {
    const auto p=prepare_mtp_attention(c);require_mtp_attention_stream(c,raw_stream);
    const auto stream=static_cast<hipStream_t>(raw_stream);const auto& m=*c.model;const auto& w=p.workspace;
    // The original indexer stores the unnormalised BF16-rounded key. Pooled
    // index-key norm/RoPE is applied later by block_score_kernel, never here.
    mtp_project<WeightKind::bf16>(p.input,p.tensors.index_k,p.hidden,p.index_dim,w.current_index_key,stream);
    mtp_project<WeightKind::q8_0>(p.input,p.tensors.k,p.hidden,p.kv_width,w.current_k,stream);
    mtp_project<WeightKind::q8_0>(p.input,p.tensors.v,p.hidden,p.kv_width,w.current_v,stream);
    launch_norm_rope_typed<SmallWeightKind::f32>(w.current_k,p.kv_heads,p.dim,p.dim,*p.tensors.k_norm.runtime,m,c.position,{},stream);
    auto* status=reinterpret_cast<std::uint32_t*>(p.status.address);
    hipLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,
        w.current_index_key,p.index_dim,w.current_k,w.current_v,p.kv_width,status);
    check_hip(hipGetLastError(),"MTP teacher cache-cast validation enqueue");
    const GraphPositionKernelArguments guarded{nullptr,c.resources->context_capacity(),status};
    hipLaunchKernelGGL(store_raw_index_key_kernel,dim3((p.index_dim+kThreads-1U)/kThreads),dim3(kThreads),0,stream,
        w.current_index_key,reinterpret_cast<std::uint16_t*>(p.index.address),p.index_dim,c.position,guarded);
    check_hip(hipGetLastError(),"MTP teacher raw index store enqueue");
    hipLaunchKernelGGL(store_kv_kernel,dim3((p.kv_width+kThreads-1U)/kThreads),dim3(kThreads),0,stream,w.current_k,w.current_v,
        reinterpret_cast<std::uint16_t*>(p.kv.address),p.kv_width,c.position,guarded);
    check_hip(hipGetLastError(),"MTP teacher KV store enqueue");
}
void launch_hip_mtp_indexer(const MtpNativeContext& c,void* raw_stream) {
    const auto p=prepare_mtp_attention(c);require_mtp_attention_stream(c,raw_stream);
    const auto stream=static_cast<hipStream_t>(raw_stream);const auto& m=*c.model;const auto& w=p.workspace;
    mtp_project<WeightKind::bf16>(p.input,p.tensors.index_q,p.hidden,p.index_width,w.index_query,stream);
    mtp_project<WeightKind::bf16>(p.input,p.tensors.index_k,p.hidden,p.index_dim,w.current_index_key,stream);
    launch_norm_rope_typed<SmallWeightKind::f32>(w.index_query,checked_u32(m.indexer_n_heads,"MTP index heads"),p.index_dim,p.index_dim,
        *p.tensors.index_q_norm.runtime,m,c.position,{},stream);
    mtp_attention_finite(p,w.index_query,p.index_width,stream);mtp_attention_finite(p,w.current_index_key,p.index_dim,stream);
    auto* cache_status=reinterpret_cast<std::uint32_t*>(p.status.address);
    hipLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,
        w.current_index_key,p.index_dim,static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),0U,cache_status);
    check_hip(hipGetLastError(),"MTP full raw-index cache-cast validation enqueue");
    const GraphPositionKernelArguments guarded_cache{nullptr,c.resources->context_capacity(),cache_status};
    hipLaunchKernelGGL(store_raw_index_key_kernel,dim3((p.index_dim+kThreads-1U)/kThreads),dim3(kThreads),0,stream,
        w.current_index_key,reinterpret_cast<std::uint16_t*>(p.index.address),p.index_dim,c.position,guarded_cache);
    check_hip(hipGetLastError(),"MTP raw index store enqueue");
    const GraphSelectionArguments selection{{},p.ratio,p.topk};
    if(p.blocks<=p.topk) {
        hipLaunchKernelGGL(fill_visible_indices_kernel,dim3((p.visible+kThreads-1U)/kThreads),dim3(kThreads),0,stream,w.selected_indices,p.visible);
        check_hip(hipGetLastError(),"MTP full-visible index enqueue");
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_kernel<SmallWeightKind::f32>),dim3(p.blocks),dim3(p.index_dim),
            2ULL*p.index_dim*sizeof(float),stream,w.index_query,reinterpret_cast<const std::uint16_t*>(p.index.address),
            checked_u32(m.indexer_n_heads,"MTP index heads"),p.index_dim,p.ratio,reinterpret_cast<const void*>(p.tensors.index_k_norm.runtime->address),
            m.rms_norm_epsilon,checked_u32(m.rotary_dim(),"MTP rotary dimension"),m.rope_theta,w.block_scores,selection);
        check_hip(hipGetLastError(),"MTP compressed index score enqueue");
        mtp_attention_finite(p,w.block_scores,p.blocks,stream);
        hipLaunchKernelGGL(select_blocks_kernel,dim3(1),dim3(1),0,stream,w.block_scores,p.blocks,p.topk,p.ratio,p.visible,w.selected_indices,selection);
        check_hip(hipGetLastError(),"MTP compressed selection enqueue");
    }
}
void launch_hip_mtp_sparse_attention(const MtpNativeContext& c,void* raw_stream) {
    const auto p=prepare_mtp_attention(c);require_mtp_attention_stream(c,raw_stream);
    const auto stream=static_cast<hipStream_t>(raw_stream);const auto& m=*c.model;const auto& w=p.workspace;
    mtp_project<WeightKind::q8_0>(p.input,p.tensors.q,p.hidden,p.q_width,w.q_gate,stream);
    mtp_project<WeightKind::q8_0>(p.input,p.tensors.k,p.hidden,p.kv_width,w.current_k,stream);
    mtp_project<WeightKind::q8_0>(p.input,p.tensors.v,p.hidden,p.kv_width,w.current_v,stream);
    launch_norm_rope_typed<SmallWeightKind::f32>(w.q_gate,p.heads,p.dim,p.dim*2U,*p.tensors.q_norm.runtime,m,c.position,{},stream);
    launch_norm_rope_typed<SmallWeightKind::f32>(w.current_k,p.kv_heads,p.dim,p.dim,*p.tensors.k_norm.runtime,m,c.position,{},stream);
    mtp_attention_finite(p,w.q_gate,p.q_width,stream);mtp_attention_finite(p,w.current_k,p.kv_width,stream);mtp_attention_finite(p,w.current_v,p.kv_width,stream);
    auto* cache_status=reinterpret_cast<std::uint32_t*>(p.status.address);
    hipLaunchKernelGGL(validate_mtp_cache_casts,dim3(1),dim3(kThreads),0,stream,
        static_cast<const float*>(nullptr),0U,w.current_k,w.current_v,p.kv_width,cache_status);
    check_hip(hipGetLastError(),"MTP full KV cache-cast validation enqueue");
    const GraphPositionKernelArguments guarded_cache{nullptr,c.resources->context_capacity(),cache_status};
    hipLaunchKernelGGL(store_kv_kernel,dim3((p.kv_width+kThreads-1U)/kThreads),dim3(kThreads),0,stream,w.current_k,w.current_v,
        reinterpret_cast<std::uint16_t*>(p.kv.address),p.kv_width,c.position,guarded_cache);
    check_hip(hipGetLastError(),"MTP KV store enqueue");
    const GraphSelectionArguments selection{{},p.ratio,p.topk};
    hipLaunchKernelGGL(attention_score_kernel,dim3(checked_u32(static_cast<std::uint64_t>(p.heads)*p.selected,"MTP attention score grid")),
        dim3(kThreads),0,stream,w.q_gate,reinterpret_cast<const std::uint16_t*>(p.kv.address),w.selected_indices,p.selected,p.maximum_selected,
        p.heads,p.kv_heads,p.dim,w.attention_scores,selection);
    check_hip(hipGetLastError(),"MTP attention scores enqueue");
    hipLaunchKernelGGL(attention_softmax_kernel,dim3(p.heads),dim3(kThreads),0,stream,w.attention_scores,p.selected,p.maximum_selected,selection);
    check_hip(hipGetLastError(),"MTP attention softmax enqueue");
    hipLaunchKernelGGL(attention_output_kernel,dim3(p.heads),dim3(p.dim),0,stream,w.q_gate,reinterpret_cast<const std::uint16_t*>(p.kv.address),
        w.selected_indices,w.attention_scores,p.selected,p.maximum_selected,p.heads,p.kv_heads,p.dim,w.attention_output,selection);
    check_hip(hipGetLastError(),"MTP attention value enqueue");
    mtp_attention_finite(p,w.attention_output,p.attention_width,stream);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_float_to_bf16_kernel<WeightKind::q8_0>),dim3(p.hidden),dim3(kThreads),0,stream,
        w.attention_output,reinterpret_cast<const void*>(p.tensors.out.runtime->address),p.attention_width,p.hidden,
        reinterpret_cast<std::uint16_t*>(p.output.address));
    check_hip(hipGetLastError(),"MTP attention output projection enqueue");
}
} // namespace strix
