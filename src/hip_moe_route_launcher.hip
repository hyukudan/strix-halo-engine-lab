#include "strix/hip_mtp_native_validation.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/wave_router_contract.hpp"

#include "strix/dispatch_resources.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaxExperts = 512;
constexpr std::uint32_t kMaxTopK = 16;
constexpr std::uint32_t kWaveMaxTopK = 32;
constexpr std::uint32_t kRouterWave = 32;
constexpr std::uint32_t kWaveCandidates = kMaxExperts / kRouterWave;
static_assert(kWaveCandidates <= 32);
constexpr std::uint32_t kRouterMaxRows = 8;
struct RouterRowsArgs {
    const void* hidden[kRouterMaxRows];
    float* workspace[kRouterMaxRows];
    std::uint32_t* indices[kRouterMaxRows];
    float* weights[kRouterMaxRows];
};
constexpr std::uint64_t kLauncherTag =
    0x4d4f45524f555432ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

static_assert(sizeof(Q8Block) == 34);

enum class RouterWeightKind {
    f32,
    f16,
    bf16,
    q8_0,
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

void validate_config(
    const HipMoeRouteLauncherConfig& config) {
    if (config.staged_logits && !config.parallel_projection)
        throw std::invalid_argument("staged router logits requires parallel projection");
    if (config.wave_topk)
        validate_wave_router_selection_contract(config.expert_count, config.top_k, config.parallel_projection, config.ranked_topk);
    if (config.ranked_topk && !config.parallel_projection)
        throw std::invalid_argument("ranked router top-k requires parallel projection");
    if (config.hidden_size == 0 ||
        config.expert_count == 0 ||
        config.expert_count > kMaxExperts ||
        config.top_k == 0 ||
        config.top_k > config.expert_count ||
        config.top_k > (config.wave_topk ? kWaveMaxTopK : kMaxTopK)) {
        throw std::invalid_argument(
            "HIP MoE route geometry is invalid");
    }
    if (config.activation !=
            HipActivationStorage::bf16 &&
        config.activation !=
            HipActivationStorage::fp32) {
        throw std::invalid_argument(
            "HIP MoE route activation storage is unsupported");
    }
}

RouterWeightKind weight_kind(
    std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0:
        return RouterWeightKind::f32;
    case 1:
        return RouterWeightKind::f16;
    case 8:
        return RouterWeightKind::q8_0;
    case 30:
        return RouterWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP MoE route currently supports only F32, F16, BF16 and Q8_0 router weights");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

template <typename Activation>
__device__ float load_activation(
    const Activation* values,
    std::uint64_t index);

template <>
__device__ float load_activation<float>(
    const float* values,
    std::uint64_t index) {
    return values[index];
}

template <>
__device__ float load_activation<std::uint16_t>(
    const std::uint16_t* values,
    std::uint64_t index) {
    return bf16_to_float(
        values[index]);
}

template <RouterWeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature);

template <>
__device__ float load_weight<
    RouterWeightKind::f32>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const float*>(
            weights);
    return values[
        static_cast<std::uint64_t>(
            expert) *
            hidden_size +
        feature];
}

template <>
__device__ float load_weight<
    RouterWeightKind::f16>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const __half*>(
            weights);
    return __half2float(
        values[
            static_cast<std::uint64_t>(
                expert) *
                hidden_size +
            feature]);
}

template <>
__device__ float load_weight<
    RouterWeightKind::bf16>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* values =
        static_cast<const std::uint16_t*>(
            weights);
    return bf16_to_float(
        values[
            static_cast<std::uint64_t>(
                expert) *
                hidden_size +
            feature]);
}

template <>
__device__ float load_weight<
    RouterWeightKind::q8_0>(
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        hidden_size / 32U;
    const auto& block =
        blocks[
            static_cast<std::uint64_t>(
                expert) *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

__device__ bool better(
    float left_value,
    std::uint32_t left_index,
    float right_value,
    std::uint32_t right_index) {
    return left_value > right_value ||
           (left_value == right_value &&
            left_index < right_index);
}

template <
    typename Activation,
    RouterWeightKind WeightKind>
__global__ void moe_route_kernel(
    const Activation* hidden,
    const void* weights,
    std::uint32_t hidden_size,
    std::uint32_t expert_count,
    std::uint32_t top_k,
    bool normalize_selected,
    std::uint32_t* selected_indices,
    float* selected_weights) {
    const auto row =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);

    __shared__ float logits[
        kMaxExperts];
    __shared__ float scratch[
        kThreads];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        float sum = 0.0F;
        const auto hidden_base =
            static_cast<std::uint64_t>(
                row) *
            hidden_size;
        for (std::uint32_t feature = 0;
             feature < hidden_size;
             ++feature) {
            sum +=
                load_activation(
                    hidden,
                    hidden_base +
                        feature) *
                load_weight<WeightKind>(
                    weights,
                    hidden_size,
                    expert,
                    feature);
        }
        logits[expert] = sum;
    }
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    __syncthreads();

    if (lane == 0) {
        float best_values[kMaxTopK];
        std::uint32_t best_indices[
            kMaxTopK];

        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            best_values[slot] =
                -INFINITY;
            best_indices[slot] =
                expert_count;
        }

        for (std::uint32_t expert = 0;
             expert < expert_count;
             ++expert) {
            const float probability =
                logits[expert];

            for (std::uint32_t slot = 0;
                 slot < top_k;
                 ++slot) {
                if (!better(
                        probability,
                        expert,
                        best_values[slot],
                        best_indices[slot])) {
                    continue;
                }

                for (std::uint32_t move =
                         top_k - 1U;
                     move > slot;
                     --move) {
                    best_values[move] =
                        best_values[
                            move - 1U];
                    best_indices[move] =
                        best_indices[
                            move - 1U];
                }
                best_values[slot] =
                    probability;
                best_indices[slot] =
                    expert;
                break;
            }
        }

        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

constexpr std::uint32_t kRouterExpertsPerBlock = 8;
constexpr std::uint32_t kRouterFeaturesPerTile = 256;
constexpr std::uint32_t kRouterProjectionThreads = 128;
constexpr std::uint32_t kRouterCoefficientPitch = kRouterFeaturesPerTile + 4;
constexpr std::uint32_t kRouterProjectionLdsBytes =
    (kRouterExpertsPerBlock * kRouterCoefficientPitch + kRouterFeaturesPerTile) * sizeof(float);
constexpr std::uint32_t kRouterDeclaredLdsBytes =
    (kRouterProjectionLdsBytes + 255U) & ~255U;
static_assert(kRouterDeclaredLdsBytes == 9472U);

__device__ __forceinline__ float router_dot_ascending(
    const float* input, const float* coefficients, std::uint32_t count, float sum) {
    for (std::uint32_t feature = 0; feature < count; ++feature)
        sum = __builtin_fmaf(input[feature], coefficients[feature], sum);
    return sum;
}

template <typename Activation, RouterWeightKind WeightKind>
__global__ void parallel_moe_route_logits_kernel(
    const Activation* hidden, const void* weights, std::uint32_t hidden_size,
    std::uint32_t expert_count, float* workspace) {
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto expert_base = static_cast<std::uint32_t>(blockIdx.x) * kRouterExpertsPerBlock;
    __shared__ float tile_input[kRouterFeaturesPerTile];
    __shared__ float tile_weights[kRouterExpertsPerBlock][kRouterCoefficientPitch];
    float sum = 0.0F;
    const auto hidden_base = static_cast<std::uint64_t>(row) * hidden_size;
    for (std::uint32_t tile = 0; tile < hidden_size; tile += kRouterFeaturesPerTile) {
        const auto remaining = hidden_size - tile;
        const auto count = remaining < kRouterFeaturesPerTile ? remaining : kRouterFeaturesPerTile;
        // Separate global load and LDS publication lets independent coefficient
        // loads stay in flight. The inactive expert/tail lanes still synchronize.
        float coefficients[kRouterExpertsPerBlock * kRouterFeaturesPerTile / kRouterProjectionThreads];
        #pragma unroll
        for (std::uint32_t item = 0; item < kRouterExpertsPerBlock * kRouterFeaturesPerTile / kRouterProjectionThreads; ++item) {
            const auto flat = lane + item * kRouterProjectionThreads;
            const auto expert = expert_base + flat / kRouterFeaturesPerTile;
            const auto feature = flat % kRouterFeaturesPerTile;
            coefficients[item] = expert < expert_count && feature < count
                ? load_weight<WeightKind>(weights, hidden_size, expert, tile + feature) : 0.0F;
        }
        for (std::uint32_t feature = lane; feature < count; feature += kRouterProjectionThreads)
            tile_input[feature] = load_activation(hidden, hidden_base + tile + feature);
        #pragma unroll
        for (std::uint32_t item = 0; item < kRouterExpertsPerBlock * kRouterFeaturesPerTile / kRouterProjectionThreads; ++item) {
            const auto flat = lane + item * kRouterProjectionThreads;
            tile_weights[flat / kRouterFeaturesPerTile][flat % kRouterFeaturesPerTile] = coefficients[item];
        }
        __syncthreads();
        if (lane < kRouterExpertsPerBlock && expert_base + lane < expert_count)
            sum = router_dot_ascending(tile_input, tile_weights[lane], count, sum);
        __syncthreads();
    }
    if (lane < kRouterExpertsPerBlock && expert_base + lane < expert_count)
        workspace[static_cast<std::uint64_t>(row) * expert_count + expert_base + lane] = sum;
}

// Full-row staging changes data movement only. One lane in each of two
// waves retains the exact ascending fused accumulation chain used above.
template <typename Activation>
__global__ void staged_moe_route_logits_kernel(const Activation* hidden, const void* weights,
    float* workspace) {
    constexpr std::uint32_t width = 2560;
    constexpr std::uint32_t experts = 512;
    constexpr std::uint32_t threads = 256;
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto expert = static_cast<std::uint32_t>(blockIdx.x) * 2U;
    __shared__ float input[width];
    __shared__ float coefficients[2][width];
    for (std::uint32_t feature = lane; feature < width; feature += threads)
        input[feature] = load_activation(hidden, static_cast<std::uint64_t>(row) * width + feature);
    // Eligibility proves every row starts at a real 16-byte-aligned address.
    const auto* packed = reinterpret_cast<const float4*>(weights);
    float4 values[5];
    #pragma unroll
    for (std::uint32_t item = 0; item < 5; ++item)
        values[item] = packed[(static_cast<std::uint64_t>(expert) * width) / 4U + lane + item * threads];
    #pragma unroll
    for (std::uint32_t item = 0; item < 5; ++item) {
        const auto flat = lane + item * threads;
        const auto group = flat / (width / 4U);
        const auto feature = (flat % (width / 4U)) * 4U;
        coefficients[group][feature] = values[item].x;
        coefficients[group][feature + 1U] = values[item].y;
        coefficients[group][feature + 2U] = values[item].z;
        coefficients[group][feature + 3U] = values[item].w;
    }
    __syncthreads();
    if (lane == 0U || lane == 32U) {
        const auto group = lane / 32U;
        const float sum = router_dot_ascending(input, coefficients[group], width, 0.0F);
        workspace[static_cast<std::uint64_t>(row) * experts + expert + group] = sum;
    }
}

__device__ __forceinline__ void parallel_moe_route_select_row(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    constexpr std::uint32_t row = 0; // Callers pass this row's own buffers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[kMaxExperts];
    __shared__ float scratch[kThreads];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    __syncthreads();

    if (lane == 0) {
        float best_values[kMaxTopK];
        std::uint32_t best_indices[
            kMaxTopK];

        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            best_values[slot] =
                -INFINITY;
            best_indices[slot] =
                expert_count;
        }

        for (std::uint32_t expert = 0;
             expert < expert_count;
             ++expert) {
            const float probability =
                logits[expert];

            for (std::uint32_t slot = 0;
                 slot < top_k;
                 ++slot) {
                if (!better(
                        probability,
                        expert,
                        best_values[slot],
                        best_indices[slot])) {
                    continue;
                }

                for (std::uint32_t move =
                         top_k - 1U;
                     move > slot;
                     --move) {
                    best_values[move] =
                        best_values[
                            move - 1U];
                    best_indices[move] =
                        best_indices[
                            move - 1U];
                }
                best_values[slot] =
                    probability;
                best_indices[slot] =
                    expert;
                break;
            }
        }

        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

__device__ __forceinline__ void ranked_parallel_moe_route_select_row(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    constexpr std::uint32_t row = 0; // Callers pass this row's own buffers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[kMaxExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[kMaxTopK];
    __shared__ std::uint32_t best_indices[kMaxTopK];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    // Separate slots do not overwrite the denominator scratch still read by waves.
    if (lane < top_k) {
        best_values[lane] = -INFINITY;
        best_indices[lane] = expert_count;
    }
    __syncthreads();
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x) {
        const float probability = logits[expert];
        // NaN compares better than nothing: explicitly exclude it before any write.
        if (probability == probability) {
            std::uint32_t rank = 0;
            for (std::uint32_t other = 0; other < expert_count; ++other) {
                if (better(logits[other], other, probability, expert)) ++rank;
                if (rank >= top_k) break;
            }
            // Non-NaN value/index pairs have a strict total order, so selected ranks
            // are unique. No barrier or early return exists inside this owner loop.
            if (rank < top_k) {
                best_values[rank] = probability;
                best_indices[rank] = expert;
            }
        }
    }
    __syncthreads();
    if (lane == 0) {
        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

__device__ __forceinline__ void wave_parallel_moe_route_select_row(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    constexpr std::uint32_t row = 0; // Callers pass this row's own buffers.
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float logits[kMaxExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[kWaveMaxTopK];
    __shared__ std::uint32_t best_indices[kWaveMaxTopK];
    for (std::uint32_t expert = lane; expert < expert_count; expert += blockDim.x)
        logits[expert] = workspace[static_cast<std::uint64_t>(row) * expert_count + expert];
    __syncthreads();

    float local_max =
        -INFINITY;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        local_max =
            fmaxf(
                local_max,
                logits[expert]);
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] =
                fmaxf(
                    scratch[lane],
                    scratch[
                        lane + stride]);
        }
        __syncthreads();
    }
    const float maximum =
        scratch[0];
    // Every wave must retain the maximum before lane zero reuses scratch[0]
    // for its contribution to the denominator reduction.
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        const float probability =
            expf(
                logits[expert] -
                maximum);
        logits[expert] =
            probability;
        local_sum += probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (std::uint32_t stride =
             blockDim.x / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            scratch[lane] +=
                scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator =
        scratch[0];

    for (std::uint32_t expert = lane;
         expert < expert_count;
         expert += blockDim.x) {
        logits[expert] /=
            denominator;
    }
    // Publish all divided probabilities before wave zero reads other waves' stores.
    __syncthreads();
    if (lane < kRouterWave) {
        float values[kWaveCandidates];
        std::uint32_t available = 0;
#pragma unroll
        for (std::uint32_t i = 0; i < kWaveCandidates; ++i) {
            const auto expert = lane + kRouterWave * i;
            values[i] = expert < expert_count ? logits[expert] : 0.0F;
            if (expert < expert_count && values[i] == values[i]) available |= 1U << i;
        }
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
            bool has = false;
            float best = -INFINITY;
            std::uint32_t index = expert_count;
#pragma unroll
            for (std::uint32_t i = 0; i < kWaveCandidates; ++i) {
                if ((available & (1U << i)) != 0) {
                    const auto expert = lane + kRouterWave * i;
                    if (!has || better(values[i], expert, best, index)) {
                        has = true; best = values[i]; index = expert;
                    }
                }
            }
#pragma unroll
            for (std::uint32_t mask = kRouterWave / 2U; mask > 0; mask >>= 1U) {
                const float other = __shfl_xor(best, static_cast<int>(mask), kRouterWave);
                const auto other_index = __shfl_xor(index, static_cast<int>(mask), kRouterWave);
                const int other_has = __shfl_xor(static_cast<int>(has), static_cast<int>(mask), kRouterWave);
                if (other_has != 0 && (!has || better(other, other_index, best, index))) {
                    has = true; best = other; index = other_index;
                }
            }
            if (lane == 0) { best_values[slot] = has ? best : -INFINITY; best_indices[slot] = has ? index : expert_count; }
            if (has && index % kRouterWave == lane) available &= ~(1U << (index / kRouterWave));
        }
    }
    // All eight waves participate; no early return may bypass this barrier.
    __syncthreads();
    if (lane == 0) {
        float selected_sum = 0.0F;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_sum +=
                best_values[slot];
        }
        if (!normalize_selected) {
            selected_sum = 1.0F;
        }

        const auto output_base =
            static_cast<std::uint64_t>(
                row) *
            top_k;
        for (std::uint32_t slot = 0;
             slot < top_k;
             ++slot) {
            selected_indices[
                output_base + slot] =
                best_indices[slot];
            selected_weights[
                output_base + slot] =
                best_values[slot] /
                selected_sum;
        }
    }
}

// Serial rows offset one buffer; window rows bring their own resolved
// buffers. Both forms run the same per-row selection body.
__global__ void parallel_moe_route_select_kernel(const float* workspace, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    parallel_moe_route_select_row(workspace + row * expert_count, expert_count, top_k, normalize_selected,
        selected_indices + row * top_k, selected_weights + row * top_k);
}
__global__ void ranked_parallel_moe_route_select_kernel(const float* workspace, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    ranked_parallel_moe_route_select_row(workspace + row * expert_count, expert_count, top_k, normalize_selected,
        selected_indices + row * top_k, selected_weights + row * top_k);
}
__global__ void wave_parallel_moe_route_select_kernel(const float* workspace, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    wave_parallel_moe_route_select_row(workspace + row * expert_count, expert_count, top_k, normalize_selected,
        selected_indices + row * top_k, selected_weights + row * top_k);
}
enum class RouterSelect { sorted, ranked, wave };
template <RouterSelect Mode>
__global__ void rows_moe_route_select_kernel(RouterRowsArgs rows, std::uint32_t expert_count,
    std::uint32_t top_k, bool normalize_selected) {
    const auto r = blockIdx.x;
    if constexpr (Mode == RouterSelect::wave)
        wave_parallel_moe_route_select_row(rows.workspace[r], expert_count, top_k, normalize_selected, rows.indices[r], rows.weights[r]);
    else if constexpr (Mode == RouterSelect::ranked)
        ranked_parallel_moe_route_select_row(rows.workspace[r], expert_count, top_k, normalize_selected, rows.indices[r], rows.weights[r]);
    else
        parallel_moe_route_select_row(rows.workspace[r], expert_count, top_k, normalize_selected, rows.indices[r], rows.weights[r]);
}

// Window rows share each staged weight tile; lane (expert, row) keeps the
// serial chain: fused, ascending from feature zero, carried across tiles.
constexpr std::uint32_t kRowsExpertsPerBlock = 2;
constexpr std::uint32_t kRowsTile = 512;
constexpr std::uint32_t kRowsPitch = kRowsTile + 4;
// Same fused ascending chain as router_dot_ascending; vector LDS reads are
// issued ahead of the dependent FMAs (the chain is LDS-latency bound).
__device__ __forceinline__ float router_dot_tile(const float* input, const float* coefficients, float sum) {
    const auto* x = reinterpret_cast<const float4*>(input);
    const auto* c = reinterpret_cast<const float4*>(coefficients);
    #pragma unroll
    for (std::uint32_t q = 0; q < kRowsTile / 4U; ++q) {
        const auto a = x[q], b = c[q];
        sum = __builtin_fmaf(a.x, b.x, sum); sum = __builtin_fmaf(a.y, b.y, sum);
        sum = __builtin_fmaf(a.z, b.z, sum); sum = __builtin_fmaf(a.w, b.w, sum);
    }
    return sum;
}
// F32 weights: one wave per expert, lane = row. The expert's coefficients are
// wave-uniform scalar loads and each lane streams its own row, so the chain
// never waits on LDS. Per (row, expert) it is the serial chain: fused,
// ascending from feature zero.
constexpr std::uint32_t kWaveRowsChunk = 64;
template <typename Activation, std::uint32_t Rows>
__global__ void __launch_bounds__(256) wave_rows_moe_route_logits_kernel(RouterRowsArgs rows,
    const float* __restrict__ weights, std::uint32_t hidden_size, std::uint32_t expert_count) {
    constexpr std::uint32_t per_vector = 16U / sizeof(Activation);
    const auto wave = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32U)));
    const auto lane = static_cast<std::uint32_t>(threadIdx.x % 32U);
    const auto expert = static_cast<std::uint32_t>(blockIdx.x) * 8U + wave;
    if (expert >= expert_count) return;
    const float* __restrict__ coefficients = weights + static_cast<std::uint64_t>(expert) * hidden_size;
    const auto* hidden = static_cast<const Activation*>(rows.hidden[lane < Rows ? lane : 0U]);
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < hidden_size; feature += kWaveRowsChunk) {
        float input[kWaveRowsChunk];
        #pragma unroll
        for (std::uint32_t v = 0; v < kWaveRowsChunk / per_vector; ++v) {
            const auto packed = *reinterpret_cast<const uint4*>(hidden + feature + v * per_vector);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (std::uint32_t k = 0; k < per_vector; ++k) {
                if constexpr (sizeof(Activation) == 2U)
                    input[v * per_vector + k] = bf16_to_float(static_cast<std::uint16_t>(k % 2U ? words[k / 2U] >> 16U : words[k / 2U] & 0xffffU));
                else input[v * per_vector + k] = __uint_as_float(words[k]);
            }
        }
        #pragma unroll
        for (std::uint32_t k = 0; k < kWaveRowsChunk; ++k)
            sum = __builtin_fmaf(input[k], coefficients[feature + k], sum);
    }
    if (lane < Rows) rows.workspace[lane][expert] = sum;
}
// F32 weights, every lane busy: a wave covers 32 / Rows experts x Rows rows and
// lane (expert, row) runs that pair's serial chain (fused, ascending from
// feature zero) on its own coefficients. The one-expert-per-wave form above
// left 32 - Rows lanes idle and needed 8x more waves for the same chains.
template <typename Activation, std::uint32_t Rows>
__global__ void __launch_bounds__(256) lanes_rows_moe_route_logits_kernel(RouterRowsArgs rows,
    const float* __restrict__ weights, std::uint32_t hidden_size, std::uint32_t expert_count) {
    static_assert(Rows == 2U || Rows == 4U || Rows == 8U);
    constexpr std::uint32_t per_vector = 16U / sizeof(Activation);
    constexpr std::uint32_t experts_per_wave = 32U / Rows;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x % 32U);
    const auto wave = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x / 32U);
    const auto expert = wave * experts_per_wave + lane / Rows;
    const auto row = lane % Rows;
    if (expert >= expert_count) return;
    const auto* __restrict__ coefficients = reinterpret_cast<const float4*>(weights + static_cast<std::uint64_t>(expert) * hidden_size);
    const auto* hidden = static_cast<const Activation*>(rows.hidden[row]);
    float sum = 0.0F;
    for (std::uint32_t feature = 0; feature < hidden_size; feature += kWaveRowsChunk) {
        float input[kWaveRowsChunk];
        #pragma unroll
        for (std::uint32_t v = 0; v < kWaveRowsChunk / per_vector; ++v) {
            const auto packed = *reinterpret_cast<const uint4*>(hidden + feature + v * per_vector);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (std::uint32_t k = 0; k < per_vector; ++k) {
                if constexpr (sizeof(Activation) == 2U)
                    input[v * per_vector + k] = bf16_to_float(static_cast<std::uint16_t>(k % 2U ? words[k / 2U] >> 16U : words[k / 2U] & 0xffffU));
                else input[v * per_vector + k] = __uint_as_float(words[k]);
            }
        }
        #pragma unroll
        for (std::uint32_t q = 0; q < kWaveRowsChunk / 4U; ++q) {
            const auto c = coefficients[(feature >> 2U) + q];
            sum = __builtin_fmaf(input[4U * q + 0U], c.x, sum);
            sum = __builtin_fmaf(input[4U * q + 1U], c.y, sum);
            sum = __builtin_fmaf(input[4U * q + 2U], c.z, sum);
            sum = __builtin_fmaf(input[4U * q + 3U], c.w, sum);
        }
    }
    rows.workspace[row][expert] = sum;
}
template <typename Activation, RouterWeightKind WeightKind, std::uint32_t Rows>
__global__ void __launch_bounds__(256) rows_moe_route_logits_kernel(RouterRowsArgs rows, const void* weights,
    std::uint32_t hidden_size, std::uint32_t expert_count) {
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t weight_items = kRowsExpertsPerBlock * kRowsTile / threads;
    constexpr std::uint32_t input_items = Rows * kRowsTile / threads;
    static_assert(kRowsTile % threads == 0);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    const auto expert_base = static_cast<std::uint32_t>(blockIdx.x) * kRowsExpertsPerBlock;
    __shared__ __align__(16) float coefficients[kRowsExpertsPerBlock][kRowsTile];
    __shared__ __align__(16) float input[Rows][kRowsPitch];
    float w[weight_items], x[input_items];
    const Activation* hidden[Rows];
    #pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) hidden[r] = static_cast<const Activation*>(rows.hidden[r]);
    const auto fetch = [&](std::uint32_t tile) {
        const auto count = hidden_size - tile < kRowsTile ? hidden_size - tile : kRowsTile;
        #pragma unroll
        for (std::uint32_t item = 0; item < weight_items; ++item) {
            const auto flat = lane + item * threads;
            const auto expert = expert_base + flat / kRowsTile;
            const auto feature = flat % kRowsTile;
            w[item] = expert < expert_count && feature < count
                ? load_weight<WeightKind>(weights, hidden_size, expert, tile + feature) : 0.0F;
        }
        // Row and feature are compile-time per item: no per-load pointer fetch.
        #pragma unroll
        for (std::uint32_t item = 0; item < input_items; ++item) {
            const auto feature = lane + (item % (kRowsTile / threads)) * threads;
            x[item] = feature < count ? load_activation(hidden[item / (kRowsTile / threads)],
                static_cast<std::uint64_t>(tile) + feature) : 0.0F;
        }
    };
    const auto chain_expert = lane / 32U, chain_row = lane % 32U;
    const bool chain = chain_expert < kRowsExpertsPerBlock && chain_row < Rows &&
        expert_base + chain_expert < expert_count;
    float sum = 0.0F;
    fetch(0);
    for (std::uint32_t tile = 0; tile < hidden_size; tile += kRowsTile) {
        const auto count = hidden_size - tile < kRowsTile ? hidden_size - tile : kRowsTile;
        __syncthreads();
        #pragma unroll
        for (std::uint32_t item = 0; item < weight_items; ++item) {
            const auto flat = lane + item * threads;
            coefficients[flat / kRowsTile][flat % kRowsTile] = w[item];
        }
        #pragma unroll
        for (std::uint32_t item = 0; item < input_items; ++item)
            input[item / (kRowsTile / threads)][lane + (item % (kRowsTile / threads)) * threads] = x[item];
        __syncthreads();
        if (tile + kRowsTile < hidden_size) fetch(tile + kRowsTile);
        if (chain) sum = count == kRowsTile ? router_dot_tile(input[chain_row], coefficients[chain_expert], sum) :
            router_dot_ascending(input[chain_row], coefficients[chain_expert], count, sum);
    }
    if (chain) rows.workspace[chain_row][expert_base + chain_expert] = sum;
}

std::uint64_t required_router_weight_bytes(
    RouterWeightKind kind,
    std::uint32_t hidden_size,
    std::uint32_t expert_count) {
    switch (kind) {
    case RouterWeightKind::f32:
        return
            static_cast<std::uint64_t>(
                hidden_size) *
            expert_count *
            4ULL;
    case RouterWeightKind::f16:
    case RouterWeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                hidden_size) *
            expert_count *
            2ULL;
    case RouterWeightKind::q8_0:
        if (hidden_size % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 router hidden size must be divisible by 32");
        }
        return
            static_cast<std::uint64_t>(
                hidden_size / 32U) *
            expert_count *
            sizeof(Q8Block);
    }
    throw std::logic_error(
        "unreachable router weight kind");
}

template <
    typename Activation,
    RouterWeightKind WeightKind>
void launch_default_typed(
    const ExecutionDispatchContext& context,
    const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices,
    const ExecutionBufferAddress& weights) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_route_kernel<
                Activation,
                WeightKind>),
        dim3(context.logical_rows),
        dim3(kThreads),
        0,
        hip_execution_stream(),
        reinterpret_cast<
            const Activation*>(
                hidden.address),
        reinterpret_cast<
            const void*>(
                weight.address),
        config.hidden_size,
        config.expert_count,
        config.top_k,
        config.normalize_selected,
        reinterpret_cast<
            std::uint32_t*>(
                indices.address),
        reinterpret_cast<float*>(
            weights.address));
    check_hip(
        hipGetLastError(),
        "HIP MoE route kernel launch");
}

ExecutionBufferAddress prepare_parallel_router_workspace(
    const ExecutionDispatchContext& context, const HipMoeRouteLauncherConfig& config) {
    if (context.logical_rows == 0 || context.logical_rows > 65535U)
        throw std::invalid_argument("parallel router row grid exceeds HIP launch limits");
    const auto workspace = resolve_dispatch_temporary(context, 0);
    const auto* workspace_allocation = find_arena_allocation(*context.arena, workspace.use);
    if (workspace_allocation == nullptr || workspace_allocation->first_step != context.step_id ||
        workspace_allocation->last_step != context.step_id)
        throw std::invalid_argument("parallel router workspace does not belong to this step");
    const auto required = static_cast<std::uint64_t>(context.logical_rows) * config.expert_count * sizeof(float);
    if (workspace.use.buffer.buffer_class != ExecutionBufferClass::router_workspace ||
        workspace.address % 16U != 0 || workspace.bytes < required || workspace.allocation_generation == 0)
        throw std::invalid_argument("parallel router requires admitted aligned FP32 logits workspace");
    return workspace;
}

template <typename Activation, RouterWeightKind WeightKind>
void launch_typed(
    const ExecutionDispatchContext& context, const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden, const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices, const ExecutionBufferAddress& weights) {
    if (!config.parallel_projection) {
        launch_default_typed<Activation, WeightKind>(context, config, hidden, weight, indices, weights);
        return;
    }
    const auto workspace = prepare_parallel_router_workspace(context, config);
    const bool staged = config.staged_logits && WeightKind == RouterWeightKind::f32 &&
        staged_router_logits_eligible(config.hidden_size, config.expert_count, 0, weight.address);
    if (staged) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(staged_moe_route_logits_kernel<Activation>),
            dim3(256U, context.logical_rows), dim3(256U), 0, hip_execution_stream(),
            reinterpret_cast<const Activation*>(hidden.address), reinterpret_cast<const void*>(weight.address),
            reinterpret_cast<float*>(workspace.address));
    } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<Activation, WeightKind>),
        dim3((config.expert_count + kRouterExpertsPerBlock - 1U) / kRouterExpertsPerBlock, context.logical_rows),
        dim3(kRouterProjectionThreads), 0, hip_execution_stream(),
        reinterpret_cast<const Activation*>(hidden.address), reinterpret_cast<const void*>(weight.address),
        config.hidden_size, config.expert_count, reinterpret_cast<float*>(workspace.address));
    }
    check_hip(hipGetLastError(), "HIP parallel MoE route logits launch");
    if (config.wave_topk) {
        hipLaunchKernelGGL(wave_parallel_moe_route_select_kernel, dim3(context.logical_rows), dim3(kThreads),
            0, hip_execution_stream(), reinterpret_cast<const float*>(workspace.address),
            config.expert_count, config.top_k, config.normalize_selected,
            reinterpret_cast<std::uint32_t*>(indices.address), reinterpret_cast<float*>(weights.address));
    } else if (config.ranked_topk) {
        hipLaunchKernelGGL(ranked_parallel_moe_route_select_kernel, dim3(context.logical_rows), dim3(kThreads),
            0, hip_execution_stream(), reinterpret_cast<const float*>(workspace.address),
            config.expert_count, config.top_k, config.normalize_selected,
            reinterpret_cast<std::uint32_t*>(indices.address), reinterpret_cast<float*>(weights.address));
    } else {
    hipLaunchKernelGGL(parallel_moe_route_select_kernel, dim3(context.logical_rows), dim3(kThreads),
        0, hip_execution_stream(), reinterpret_cast<const float*>(workspace.address),
        config.expert_count, config.top_k, config.normalize_selected,
        reinterpret_cast<std::uint32_t*>(indices.address), reinterpret_cast<float*>(weights.address));
    }
    check_hip(hipGetLastError(), "HIP parallel MoE route selection launch");
}

template <typename Activation>
void launch_by_weight(
    RouterWeightKind kind,
    const ExecutionDispatchContext& context,
    const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden,
    const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices,
    const ExecutionBufferAddress& weights) {
    switch (kind) {
    case RouterWeightKind::f32:
        launch_typed<
            Activation,
            RouterWeightKind::f32>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    case RouterWeightKind::f16:
        launch_typed<
            Activation,
            RouterWeightKind::f16>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    case RouterWeightKind::bf16:
        launch_typed<
            Activation,
            RouterWeightKind::bf16>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    case RouterWeightKind::q8_0:
        launch_typed<
            Activation,
            RouterWeightKind::q8_0>(
                context,
                config,
                hidden,
                weight,
                indices,
                weights);
        return;
    }
    throw std::logic_error(
        "unreachable router weight kind");
}

struct PreparedRouterLaunch {
    const HipMoeRouteLauncherConfig* config;
    ExecutionBufferAddress hidden, indices, route_weights;
    const RuntimeTensorBinding* runtime_tensor;
    RouterWeightKind kind;
};
PreparedRouterLaunch prepare_router_launch(const ExecutionDispatchContext& context, void* user_data) {
    if (context.op !=
        SerialOp::moe_route) {
        throw std::invalid_argument(
            "HIP MoE route launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 2) {
        throw std::invalid_argument(
            "HIP MoE route expects one read and two write buffers");
    }
    if (user_data == nullptr) {
        throw std::invalid_argument(
            "HIP MoE route launcher requires configuration");
    }

    const auto& config =
        *static_cast<
            HipMoeRouteLauncherConfig*>(
                user_data);
    validate_config(config);
    if (context.temporaries.size() != (config.parallel_projection ? 1U : 0U))
        throw std::invalid_argument("router temporary IO differs from launcher topology");
    if (config.parallel_projection &&
        context.temporaries.front().buffer.buffer_class != ExecutionBufferClass::router_workspace)
        throw std::invalid_argument("parallel router temporary must use router_workspace");

    const auto hidden =
        resolve_dispatch_read(
            context, 0);
    const auto indices =
        resolve_dispatch_write(
            context, 0);
    const auto route_weights =
        resolve_dispatch_write(
            context, 1);

    const auto& access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (access.tensor_indices.size() != 1U) {
        throw std::invalid_argument(
            "HIP MoE route requires exactly one router tensor");
    }
    if (context.resources == nullptr ||
        context.resources->model_manifest == nullptr) {
        throw std::invalid_argument(
            "HIP MoE route requires model manifest resources");
    }

    if (config.wave_topk && (!config.wave_manifest_admission ||
        context.resources->model_manifest != &config.wave_manifest_admission->manifest()))
        throw std::invalid_argument("wave router requires the exact owned admitted manifest");
    const auto tensor_index =
        access.tensor_indices.front();
    if (tensor_index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "HIP MoE route tensor index exceeds manifest");
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
            TensorRole::moe_router ||
        manifest_tensor.dimensions.size() < 2 ||
        manifest_tensor.dimensions[0] !=
            config.hidden_size ||
        manifest_tensor.dimensions[1] !=
            config.expert_count) {
        throw std::invalid_argument(
            "HIP MoE route tensor geometry does not match router contract");
    }
    if (!runtime_tensor_gpu_visible(runtime_tensor.storage) ||
        runtime_tensor.address == 0) {
        throw std::invalid_argument(
            "HIP MoE route requires resident or repacked GPU-visible router weights");
    }

    if (config.wave_topk && (hidden.address % (config.activation == HipActivationStorage::bf16 ? 2U : 4U) != 0 ||
        indices.address % 4U != 0 || route_weights.address % 4U != 0))
        throw std::invalid_argument("wave router requires aligned runtime inputs and outputs");
    const auto kind =
        weight_kind(
            manifest_tensor.ggml_type);
    const auto required_weights =
        required_router_weight_bytes(
            kind,
            config.hidden_size,
            config.expert_count);
    if (runtime_tensor.bytes <
        required_weights) {
        throw std::invalid_argument(
            "HIP MoE route runtime tensor is smaller than required packed weights");
    }

    const auto activation_bytes =
        config.activation ==
                HipActivationStorage::bf16
            ? 2ULL
            : 4ULL;
    const auto hidden_required =
        static_cast<std::uint64_t>(
            context.logical_rows) *
        config.hidden_size *
        activation_bytes;
    const auto indices_required =
        static_cast<std::uint64_t>(
            context.logical_rows) *
        config.top_k *
        sizeof(std::uint32_t);
    const auto route_weights_required =
        static_cast<std::uint64_t>(
            context.logical_rows) *
        config.top_k *
        sizeof(float);

    if (hidden.bytes <
            hidden_required ||
        indices.bytes <
            indices_required ||
        route_weights.bytes <
            route_weights_required) {
        throw std::invalid_argument(
            "HIP MoE route scratch buffers are undersized");
    }

    return {&config, hidden, indices, route_weights, &runtime_tensor, kind};
}
void preflight_moe_route(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_router_launch(context, user_data);
    if (prepared.config->parallel_projection)
        (void)prepare_parallel_router_workspace(context, *prepared.config);
}

void launch_moe_route(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_router_launch(context, user_data);
    const auto& config = *prepared.config;
    const auto& hidden = prepared.hidden;
    const auto& indices = prepared.indices;
    const auto& route_weights = prepared.route_weights;
    const auto& runtime_tensor = *prepared.runtime_tensor;
    const auto kind = prepared.kind;
    switch (config.activation) {
    case HipActivationStorage::bf16:
        launch_by_weight<
            std::uint16_t>(
                kind,
                context,
                config,
                hidden,
                runtime_tensor,
                indices,
                route_weights);
        return;
    case HipActivationStorage::fp32:
        launch_by_weight<float>(
            kind,
            context,
            config,
            hidden,
            runtime_tensor,
            indices,
            route_weights);
        return;
    }
    throw std::logic_error(
        "unreachable router activation storage");
}

std::string route_kernel_id(
    const ExecutionBindingRequest& request,
    const HipMoeRouteLauncherConfig& config) {
    return
        "hip.moe_route." +
        std::string(
            activation_name(
                config.activation)) +
        ".h" +
        std::to_string(
            config.hidden_size) +
        ".e" +
        std::to_string(
            config.expert_count) +
        ".k" +
        std::to_string(
            config.top_k) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.staged_logits ? ".staged_logits" : "") + (config.wave_topk ? ".wave.parallel" : config.ranked_topk ? ".ranked.parallel" :
            config.parallel_projection ? ".parallel" : "");
}

std::uint64_t launcher_identity(
    const HipMoeRouteLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(hash, config.hidden_size);
    hash_u64(hash, config.expert_count);
    hash_u64(hash, config.top_k);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    hash_u64(
        hash,
        config.normalize_selected
            ? 1ULL
            : 0ULL);
    if (config.parallel_projection) hash_u64(hash, 0x524f555445525057ULL);
    if (config.staged_logits) hash_u64(hash, 0x5354474c4f474954ULL);
    if (config.ranked_topk) hash_u64(hash, 0x52414e4b544f504bULL);
    if (config.wave_topk) hash_u64(hash, 0x57415645544f504bULL);
    return hash;
}

struct PreparedRouterRows {
    RouterRowsArgs args{};
    std::uint32_t rows = 0;
    const RuntimeTensorBinding* tensor = nullptr;
    RouterWeightKind kind = RouterWeightKind::f32;
    bool wave_logits = false;
};
// Every row passes the complete serial admission; rows add only the window
// shape (one step, consecutive positions, one weight) and cross-row aliasing.
PreparedRouterRows prepare_router_rows(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    if (rows.size() != 2U && rows.size() != 4U && rows.size() != 8U)
        throw std::invalid_argument("known-window router rows count must be 2, 4 or 8");
    if (!config.parallel_projection)
        throw std::invalid_argument("known-window router rows require the parallel router topology");
    PreparedRouterRows out;
    out.rows = static_cast<std::uint32_t>(rows.size());
    struct Range { std::uintptr_t address; std::uint64_t bytes; };
    std::array<Range, 3U * kRouterMaxRows> writes{};
    std::array<Range, kRouterMaxRows> reads{};
    const auto& first = rows.front();
    auto* user = const_cast<HipMoeRouteLauncherConfig*>(&config);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        if (row.logical_rows != 1U || row.request_concurrency != 1U ||
            row.execution_identity != first.execution_identity || row.step_id != first.step_id ||
            row.layer != first.layer || row.kernel_id != first.kernel_id || row.position != first.position + r)
            throw std::invalid_argument("known-window router rows identity/position differs");
        const auto prepared = prepare_router_launch(row, user);
        const auto workspace = prepare_parallel_router_workspace(row, config);
        if (r && (prepared.runtime_tensor->address != out.tensor->address || prepared.kind != out.kind))
            throw std::invalid_argument("known-window router rows selected weights differ");
        out.tensor = prepared.runtime_tensor; out.kind = prepared.kind;
        out.args.hidden[r] = reinterpret_cast<const void*>(prepared.hidden.address);
        out.args.workspace[r] = reinterpret_cast<float*>(workspace.address);
        out.args.indices[r] = reinterpret_cast<std::uint32_t*>(prepared.indices.address);
        out.args.weights[r] = reinterpret_cast<float*>(prepared.route_weights.address);
        reads[r] = {prepared.hidden.address, static_cast<std::uint64_t>(config.hidden_size) *
            (config.activation == HipActivationStorage::bf16 ? 2U : 4U)};
        writes[3U * r] = {prepared.indices.address, config.top_k * 4ULL};
        writes[3U * r + 1U] = {prepared.route_weights.address, config.top_k * 4ULL};
        writes[3U * r + 2U] = {workspace.address, config.expert_count * 4ULL};
    }
    const auto per_vector = config.activation == HipActivationStorage::bf16 ? 8U : 4U;
    out.wave_logits = out.kind == RouterWeightKind::f32 && config.hidden_size % kWaveRowsChunk == 0U &&
        out.tensor->address % 4U == 0U && kWaveRowsChunk % per_vector == 0U &&
        std::all_of(reads.begin(), reads.begin() + rows.size(), [](Range r) { return r.address % 16U == 0U; });
    const auto overlap = [](Range a, Range b) { return a.address < b.address + b.bytes && b.address < a.address + a.bytes; };
    const Range weight{out.tensor->address, out.tensor->bytes};
    for (std::size_t w = 0; w < 3U * rows.size(); ++w) {
        if (overlap(writes[w], weight)) throw std::invalid_argument("known-window router rows output aliases weights");
        for (std::size_t r = 0; r < rows.size(); ++r)
            if (overlap(writes[w], reads[r])) throw std::invalid_argument("known-window router rows output aliases an input row");
        for (std::size_t o = 0; o < w; ++o)
            if (overlap(writes[w], writes[o])) throw std::invalid_argument("known-window router rows outputs alias");
    }
    return out;
}

template <typename Activation, RouterWeightKind WeightKind>
void launch_router_rows_typed(const PreparedRouterRows& p, const HipMoeRouteLauncherConfig& config) {
    const dim3 grid((config.expert_count + kRowsExpertsPerBlock - 1U) / kRowsExpertsPerBlock);
    const auto* weights = reinterpret_cast<const void*>(p.tensor->address);
    const auto stream = hip_execution_stream();
    if (WeightKind == RouterWeightKind::f32 && p.wave_logits) {
        const auto* f32 = static_cast<const float*>(weights);
        // 8 waves per block; each wave holds 32 / rows experts.
        const auto blocks = [&](std::uint32_t rows) { return dim3((config.expert_count * rows + 255U) / 256U); };
        const bool aligned = config.hidden_size % kWaveRowsChunk == 0U;
        if (!aligned) {
            const dim3 waves((config.expert_count + 7U) / 8U);
            if (p.rows == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation, 2U>),
                waves, dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
            else if (p.rows == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation, 4U>),
                waves, dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(wave_rows_moe_route_logits_kernel<Activation, 8U>),
                waves, dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
        } else if (p.rows == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, 2U>),
            blocks(2U), dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
        else if (p.rows == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, 4U>),
            blocks(4U), dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(lanes_rows_moe_route_logits_kernel<Activation, 8U>),
            blocks(8U), dim3(256U), 0, stream, p.args, f32, config.hidden_size, config.expert_count);
    } else if (p.rows == 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_logits_kernel<Activation, WeightKind, 2U>),
        grid, dim3(256U), 0, stream, p.args, weights, config.hidden_size, config.expert_count);
    else if (p.rows == 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_logits_kernel<Activation, WeightKind, 4U>),
        grid, dim3(256U), 0, stream, p.args, weights, config.hidden_size, config.expert_count);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_logits_kernel<Activation, WeightKind, 8U>),
        grid, dim3(256U), 0, stream, p.args, weights, config.hidden_size, config.expert_count);
    check_hip(hipGetLastError(), "known-window router rows logits launch");
    if (config.wave_topk) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::wave>),
        dim3(p.rows), dim3(kThreads), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected);
    else if (config.ranked_topk) hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::ranked>),
        dim3(p.rows), dim3(kThreads), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(rows_moe_route_select_kernel<RouterSelect::sorted>),
        dim3(p.rows), dim3(kThreads), 0, stream, p.args, config.expert_count, config.top_k, config.normalize_selected);
    check_hip(hipGetLastError(), "known-window router rows selection launch");
}
template <typename Activation>
void launch_router_rows_by_weight(const PreparedRouterRows& p, const HipMoeRouteLauncherConfig& config) {
    switch (p.kind) {
    case RouterWeightKind::f32: launch_router_rows_typed<Activation, RouterWeightKind::f32>(p, config); return;
    case RouterWeightKind::f16: launch_router_rows_typed<Activation, RouterWeightKind::f16>(p, config); return;
    case RouterWeightKind::bf16: launch_router_rows_typed<Activation, RouterWeightKind::bf16>(p, config); return;
    case RouterWeightKind::q8_0: launch_router_rows_typed<Activation, RouterWeightKind::q8_0>(p, config); return;
    }
    throw std::logic_error("unreachable router weight kind");
}

}  // namespace

void moe_route_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    static_cast<void>(prepare_router_rows(rows, config));
}
void moe_route_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config) {
    const auto prepared = prepare_router_rows(rows, config);
    if (config.activation == HipActivationStorage::bf16) launch_router_rows_by_weight<std::uint16_t>(prepared, config);
    else launch_router_rows_by_weight<float>(prepared, config);
}

void register_hip_moe_route_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeRouteLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.staged_logits != request.staged_moe_router_logits)
        throw std::invalid_argument("staged router variant mode differs from binding request");
    if (config.wave_topk != request.wave_moe_router_topk)
        throw std::invalid_argument("wave router variant mode differs from binding request");
    if (config.ranked_topk != request.ranked_moe_router_topk)
        throw std::invalid_argument("ranked router variant mode differs from binding request");
    if (config.parallel_projection != request.parallel_moe_router)
        throw std::invalid_argument("router variant mode differs from binding request");
    if (config.hidden_size !=
            model.hidden_size ||
        config.expert_count !=
            model.num_experts ||
        config.top_k !=
            model.num_experts_per_token) {
        throw std::invalid_argument(
            "HIP MoE route config does not match model contract");
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "HIP MoE route variant requires non-zero rows and concurrency");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP MoE route may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        route_kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::moe_route;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::moe_route,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        config.staged_logits ? staged_router_logits_lds_bytes :
        config.parallel_projection ? kRouterDeclaredLdsBytes :
        (kMaxExperts + kThreads) * sizeof(float);
    variant.numerics.input_precision =
        activation_name(
            config.activation);
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "u32+fp32";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    variant.graph_safe = true;
    variant.input_layout =
        "row-major+ggml-router";
    variant.output_layout =
        "topk-indices+fp32-weights";
    variant.compiler_identity = config.staged_logits ? "hip-gfx1151-moe-route-v7-staged-logits" : config.wave_topk ? "hip-gfx1151-moe-route-v6-wave-parallel" : config.ranked_topk ? "hip-gfx1151-moe-route-v5-ranked-parallel" : config.parallel_projection
        ? "hip-gfx1151-moe-route-v4-parallel" : "hip-gfx1151-moe-route-v2";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_moe_route_launcher(
    std::string kernel_id,
    HipMoeRouteLauncherConfig* config) {
    if (kernel_id.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP MoE route launcher requires id and config");
    }
    validate_config(*config);
    if(config->wave_topk && !config->wave_manifest_admission)
        throw std::invalid_argument("wave router launcher requires owned manifest preflight admission");
    return ExecutionLauncher{
        std::move(kernel_id),
        launcher_identity(*config),
        &launch_moe_route,
        config,
        &preflight_moe_route,
    };
}

void HipMoeRouteLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP MoE route bundle requires valid compiled execution");
    }

    // Check the entire router registration contract before publishing launchers.
    if (compiled.io.steps.size() != compiled.execution.steps.size() ||
        compiled.bindings.bindings.size() != compiled.execution.steps.size())
        throw std::invalid_argument("router launcher preflight requires complete compiled IO and bindings");
    if (compiled.io.router_workspace_slots != (config_.parallel_projection ? 1U : 0U))
        throw std::invalid_argument("router launcher mode differs from compiled workspace slots");
    for (std::size_t index = 0; index < compiled.execution.steps.size(); ++index) {
        if (compiled.execution.steps[index].op != SerialOp::moe_route) continue;
        const auto& binding = compiled.bindings.bindings[index];
        const auto& io = compiled.io.steps[index];
        if (!binding.kernel_id.has_value() ||
            ((binding.kernel_id->find(".staged_logits") != std::string::npos) != config_.staged_logits ||
             binding.kernel_id->ends_with(".parallel") != config_.parallel_projection ||
             binding.kernel_id->ends_with(".ranked.parallel") != config_.ranked_topk ||
             binding.kernel_id->ends_with(".wave.parallel") != config_.wave_topk))
            throw std::invalid_argument("router launcher mode differs from compiled kernel binding");
        if (io.step_id != compiled.execution.steps[index].id || io.op != SerialOp::moe_route ||
            io.temporaries.size() != (config_.parallel_projection ? 1U : 0U) ||
            (config_.parallel_projection && io.temporaries.front().buffer.buffer_class != ExecutionBufferClass::router_workspace))
            throw std::invalid_argument("router launcher mode differs from compiled temporary IO");
    }

    std::string registered_id;
    for (std::size_t index = 0;
         index <
            compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        if (step.op !=
            SerialOp::moe_route) {
            continue;
        }

        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP MoE route step has no kernel binding");
        }

        if (registered_id.empty()) {
            registered_id =
                *binding.kernel_id;
            registry.add_kernel_launcher(
                make_hip_moe_route_launcher(
                    registered_id,
                    &config_));
        } else if (registered_id !=
                   *binding.kernel_id) {
            throw std::invalid_argument(
                "HIP MoE route bundle currently requires one route kernel id for all serial layers");
        }
    }
}

void preflight_hip_mtp_route(const MtpNativeContext& c) {
    mtp_native::context(c);
    const auto h=c.model->hidden_size,e=c.model->num_experts,k=c.model->num_experts_per_token;
    if(!h || h>std::numeric_limits<std::uint32_t>::max() || !e || e>kMaxExperts || !k || k>e || k>kWaveMaxTopK)
        throw std::invalid_argument("MTP native router geometry unsupported");
    const auto& t=c.tensors->router;
    mtp_native::tensor(t,{h,e},mtp_native::product(mtp_native::product(h,e),4));
    if(t.manifest->ggml_type!=0) throw std::invalid_argument("MTP native router requires F32 weights");
    auto input=mtp_native::buffer(c,HipMtpBuffer::hidden0,mtp_native::product(h,2));
    auto indices=mtp_native::buffer(c,HipMtpBuffer::routes,mtp_native::product(k,4));
    auto weights=mtp_native::buffer(c,HipMtpBuffer::route_weights,mtp_native::product(k,4));
    auto workspace=mtp_native::buffer(c,HipMtpBuffer::router_workspace,mtp_native::product(e,4));
    mtp_native::disjoint({input,indices,weights,workspace,
        {t.runtime->address,t.runtime->bytes,t.runtime->allocation_generation}});
}
void launch_hip_mtp_route(const MtpNativeContext& c,void* stream) {
    preflight_hip_mtp_route(c); mtp_native::stream(c,stream);
    check_hip(hipGetLastError(),"MTP router pending HIP error");
    const auto h=static_cast<std::uint32_t>(c.model->hidden_size);
    const auto e=static_cast<std::uint32_t>(c.model->num_experts);
    const auto k=static_cast<std::uint32_t>(c.model->num_experts_per_token);
    auto workspace=reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::router_workspace).address);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<std::uint16_t,RouterWeightKind::f32>),
        dim3((e+kRouterExpertsPerBlock-1)/kRouterExpertsPerBlock),dim3(kRouterProjectionThreads),0,
        reinterpret_cast<hipStream_t>(stream),
        reinterpret_cast<const std::uint16_t*>(c.resources->view(HipMtpBuffer::hidden0).address),
        reinterpret_cast<const void*>(c.tensors->router.runtime->address),h,e,workspace);
    check_hip(hipGetLastError(),"MTP native parallel route projection launch");
    hipLaunchKernelGGL(wave_parallel_moe_route_select_kernel,dim3(1),dim3(kThreads),0,
        reinterpret_cast<hipStream_t>(stream),workspace,e,k,c.model->norm_topk_probability,
        reinterpret_cast<std::uint32_t*>(c.resources->view(HipMtpBuffer::routes).address),
        reinterpret_cast<float*>(c.resources->view(HipMtpBuffer::route_weights).address));
    check_hip(hipGetLastError(),"MTP native parallel route selection launch");
}

}  // namespace strix
