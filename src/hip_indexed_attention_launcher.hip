#include "strix/hip_indexed_attention_launcher.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
    0x5141534154544e31ULL;
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
            (1.0F +
             load_small_weight<Kind>(
                 norm_weights,
                 feature)));
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
    std::uint64_t position) {
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
    float* block_scores) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
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
    pooled[feature] =
        round_bf16(
            mean *
            inv_rms *
            (1.0F +
             load_small_weight<Kind>(
                 key_norm,
                 feature)));
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
    pooled[feature] =
        round_bf16(rotated);
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
            total +=
                fmaxf(
                    scratch[0],
                    0.0F);
        }
        __syncthreads();
    }

    if (feature == 0) {
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

__global__ void select_blocks_kernel(
    float* block_scores,
    std::uint32_t num_blocks,
    std::uint32_t select_blocks,
    std::uint32_t compress_ratio,
    std::uint32_t visible_tokens,
    std::uint32_t* selected) {
    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
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
    float* scores) {
    const auto flat =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto head =
        flat /
        selected_count;
    const auto selected_slot =
        flat %
        selected_count;
    if (head >= attention_heads) {
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
    std::uint32_t maximum_selected) {
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
    float* output) {
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
        0,
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
    float* output) {
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
        0,
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
    const ExecutionBufferAddress& output) {
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
                model, position);
        return;
    case SmallWeightKind::f16:
        launch_norm_rope_typed<
            SmallWeightKind::f16>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position);
        return;
    case SmallWeightKind::bf16:
        launch_norm_rope_typed<
            SmallWeightKind::bf16>(
                values, heads, head_dim,
                head_stride, *norm.runtime,
                model, position);
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
    std::uint32_t complete_blocks) {
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
        0,
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
        workspace.block_scores);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention block-score launch");
}

void launch_block_scores(
    const WorkspacePointers& workspace,
    const SessionStateAddress& raw_state,
    const TensorPair& k_norm,
    const ModelContract& model,
    std::uint32_t complete_blocks) {
    switch (small_weight_kind(
                k_norm.manifest->
                    ggml_type)) {
    case SmallWeightKind::f32:
        launch_block_scores_typed<
            SmallWeightKind::f32>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks);
        return;
    case SmallWeightKind::f16:
        launch_block_scores_typed<
            SmallWeightKind::f16>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks);
        return;
    case SmallWeightKind::bf16:
        launch_block_scores_typed<
            SmallWeightKind::bf16>(
                workspace, raw_state,
                k_norm, model,
                complete_blocks);
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
    };
}

void launch_indexed_attention(
    const ExecutionDispatchContext& context,
    void* user_data) {
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
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP indexed attention requires configuration");
    }

    const auto& config =
        *static_cast<
            HipIndexedAttentionLauncherConfig*>(
                user_data);
    validate_config(config);
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
            model);
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
    if (context.position >=
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

    // Indexer projections. Outputs are rounded to BF16 values represented in
    // FP32 workspace so subsequent FP32 normalization starts from model dtype.
    launch_linear_to_float(
        tensors.index_q,
        input,
        hidden,
        index_q_width,
        workspace.index_query);
    launch_linear_to_float(
        tensors.index_k,
        input,
        hidden,
        index_dim,
        workspace.current_index_key);

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
    hipLaunchKernelGGL(
        store_raw_index_key_kernel,
        dim3(raw_blocks),
        dim3(kThreads),
        0, 0,
        workspace.current_index_key,
        reinterpret_cast<
            std::uint16_t*>(
                raw_state.address),
        index_dim,
        context.position);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention raw-index state update");

    const auto visible64 =
        context.position + 1ULL;
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
    const auto selected_count =
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

    if (complete_blocks <=
        block_topk) {
        const auto blocks =
            (visible +
             kThreads - 1U) /
            kThreads;
        hipLaunchKernelGGL(
            fill_visible_indices_kernel,
            dim3(blocks),
            dim3(kThreads),
            0, 0,
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
            complete_blocks);
        hipLaunchKernelGGL(
            select_blocks_kernel,
            dim3(1),
            dim3(1),
            0, 0,
            workspace.block_scores,
            complete_blocks,
            block_topk,
            ratio,
            visible,
            workspace.selected_indices);
        check_hip(
            hipGetLastError(),
            "HIP indexed-attention top-k block selection");
    }

    // Main Q/K/V projections.
    launch_linear_to_float(
        tensors.q,
        input,
        hidden,
        q_gate_width,
        workspace.q_gate);
    launch_linear_to_float(
        tensors.k,
        input,
        hidden,
        kv_width,
        workspace.current_k);
    launch_linear_to_float(
        tensors.v,
        input,
        hidden,
        kv_width,
        workspace.current_v);

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
    hipLaunchKernelGGL(
        store_kv_kernel,
        dim3(kv_blocks),
        dim3(kThreads),
        0, 0,
        workspace.current_k,
        workspace.current_v,
        reinterpret_cast<
            std::uint16_t*>(
                kv_state.address),
        kv_width,
        context.position, GraphPositionKernelArguments{});
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention KV state update");

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
    hipLaunchKernelGGL(
        attention_score_kernel,
        dim3(
            static_cast<std::uint32_t>(
                score_blocks64)),
        dim3(kThreads),
        0, 0,
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
        workspace.attention_scores);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention score launch");

    hipLaunchKernelGGL(
        attention_softmax_kernel,
        dim3(attention_heads),
        dim3(kThreads),
        0, 0,
        workspace.attention_scores,
        selected_count,
        checked_u32(
            workspace_layout.
                maximum_selected_tokens,
            "maximum_selected_tokens"));
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention softmax launch");

    hipLaunchKernelGGL(
        attention_output_kernel,
        dim3(attention_heads),
        dim3(head_dim),
        0, 0,
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
        workspace.attention_output);
    check_hip(
        hipGetLastError(),
        "HIP indexed-attention value/gate launch");

    launch_linear_float_to_bf16(
        tensors.out,
        workspace.attention_output,
        attention_width,
        hidden,
        output);
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
            request.request_concurrency);
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
    return hash;
}

}  // namespace

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
            model).total_bytes;
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
    variant.graph_safe = false;
    variant.input_layout =
        "row-major+layer-kv+raw-index";
    variant.output_layout =
        "row-major";
    variant.compiler_identity =
        "hip-gfx1151-indexed-attention-v1";
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
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_indexed_attention,
        config,
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

    std::string registered_id;
    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::indexed_attention) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP indexed-attention step has no kernel binding");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_indexed_attention_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP indexed-attention bundle requires one selected kernel id");
        }
    }

    if (registered_id.empty()) {
        throw std::invalid_argument(
            "compiled execution has no indexed-attention step");
    }
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

}  // namespace strix
