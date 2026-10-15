#include "strix/hip_launch_context.hpp"
#include "strix/hip_moe_route_launcher.hpp"
#include "strix/wave_router_contract.hpp"

#include "strix/dispatch_resources.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
            for (std::uint32_t feature = 0; feature < count; ++feature)
                // Baseline gfx1151 code objects use fused FP32 accumulation
                // in the ascending dot chain; pin that operation explicitly.
                sum = __builtin_fmaf(tile_input[feature], tile_weights[lane][feature], sum);
        __syncthreads();
    }
    if (lane < kRouterExpertsPerBlock && expert_base + lane < expert_count)
        workspace[static_cast<std::uint64_t>(row) * expert_count + expert_base + lane] = sum;
}

__global__ void parallel_moe_route_select_kernel(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
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

__global__ void ranked_parallel_moe_route_select_kernel(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
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

__global__ void wave_parallel_moe_route_select_kernel(
    const float* workspace, std::uint32_t expert_count, std::uint32_t top_k,
    bool normalize_selected, std::uint32_t* selected_indices, float* selected_weights) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
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

template <typename Activation, RouterWeightKind WeightKind>
void launch_typed(
    const ExecutionDispatchContext& context, const HipMoeRouteLauncherConfig& config,
    const ExecutionBufferAddress& hidden, const RuntimeTensorBinding& weight,
    const ExecutionBufferAddress& indices, const ExecutionBufferAddress& weights) {
    if (!config.parallel_projection) {
        launch_default_typed<Activation, WeightKind>(context, config, hidden, weight, indices, weights);
        return;
    }
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
    hipLaunchKernelGGL(HIP_KERNEL_NAME(parallel_moe_route_logits_kernel<Activation, WeightKind>),
        dim3((config.expert_count + kRouterExpertsPerBlock - 1U) / kRouterExpertsPerBlock, context.logical_rows),
        dim3(kRouterProjectionThreads), 0, hip_execution_stream(),
        reinterpret_cast<const Activation*>(hidden.address), reinterpret_cast<const void*>(weight.address),
        config.hidden_size, config.expert_count, reinterpret_cast<float*>(workspace.address));
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

void launch_moe_route(
    const ExecutionDispatchContext& context,
    void* user_data) {
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
            request.request_concurrency) + (config.wave_topk ? ".wave.parallel" : config.ranked_topk ? ".ranked.parallel" :
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
    if (config.ranked_topk) hash_u64(hash, 0x52414e4b544f504bULL);
    if (config.wave_topk) hash_u64(hash, 0x57415645544f504bULL);
    return hash;
}

}  // namespace

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
    variant.compiler_identity = config.wave_topk ? "hip-gfx1151-moe-route-v6-wave-parallel" : config.ranked_topk ? "hip-gfx1151-moe-route-v5-ranked-parallel" : config.parallel_projection
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
            (binding.kernel_id->ends_with(".parallel") != config_.parallel_projection ||
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

}  // namespace strix
