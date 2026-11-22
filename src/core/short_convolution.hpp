#pragma once

#include "decode_state_rows.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/pointwise.hpp"
#include "strix/weight/formats/scalar_float.hpp"
#include <limits>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace strix::core {

struct ShortConvolution {
    std::size_t channels;
    std::size_t taps;
    // Additional per-token weights are shared by consecutive channels in each group.
    std::size_t dynamic_groups = 0;
    float residual_weight = 0.0F;
    float output_divisor = 1.0F;
};

inline bool valid_short_convolution_plan(const ShortConvolution& plan) noexcept {
    return plan.channels != 0 && plan.taps != 0 &&
           (plan.dynamic_groups == 0 || plan.channels % plan.dynamic_groups == 0) &&
           std::isfinite(plan.residual_weight) && std::isfinite(plan.output_divisor) && plan.output_divisor > 0.0F;
}

namespace causal_conv {

struct StepArgs {
    const exec::StepBlock* step;
    decode_state::SlotView history;
    ShortConvolution plan;
    const float* input;
    const float* static_weights;
    const float* dynamic_weights;
    float* pending;  // [step row][tap history row][channel], oldest first.
    float* output;
    bool round_bf16;
};

}  // namespace causal_conv

// Scalar FP32 reference for contiguous, unpadded rows. Input and output are
// [row][channel]. All writable buffers are disjoint from the other arguments.
// Static weights are [lag][channel], dynamic weights are [row][group][lag]. Lag 0
// is the current row. Dynamic weights are optional and, when present, add to the
// static weights on each tap. The projection that supplies them is a separate op.
// History is caller-owned [taps - 1][channel], initialized to zero at sequence start
// and stored oldest first. Keep a separate buffer for each stream and sequence.
inline void causal_short_convolution(
    const ShortConvolution& plan, std::span<const float> input,
    std::span<const float> static_weights, std::span<const float> dynamic_weights,
    std::span<float> history, std::span<float> output) {
    const auto channels = plan.channels;
    const auto taps = plan.taps;
    if (!valid_short_convolution_plan(plan) || input.size() % channels != 0 ||
        output.size() != input.size() || static_weights.size() % channels != 0 ||
        static_weights.size() / channels != taps || history.size() % channels != 0 ||
        history.size() / channels != taps - 1) {
        throw std::invalid_argument("short convolution: invalid shape or epilogue");
    }

    const auto rows = input.size() / channels;
    const auto groups = plan.dynamic_groups;
    if (groups == 0) {
        if (!dynamic_weights.empty()) {
            throw std::invalid_argument("short convolution: unexpected dynamic weights");
        }
    } else if (dynamic_weights.size() % taps != 0 ||
               dynamic_weights.size() / taps % groups != 0 ||
               dynamic_weights.size() / taps / groups != rows) {
        throw std::invalid_argument("short convolution: invalid dynamic weight shape");
    }
    const auto group_width = groups == 0 ? channels : channels / groups;

    for (std::size_t row = 0; row < rows; ++row) {
        const auto input_row = input.subspan(row * channels, channels);
        auto output_row = output.subspan(row * channels, channels);
        std::ranges::fill(output_row, 0.0F);
        for (std::size_t lag = 0; lag < taps; ++lag) {
            const auto sample = lag <= row
                ? input.subspan((row - lag) * channels, channels)
                : std::span<const float>(history).subspan((taps - 1 - (lag - row)) * channels, channels);
            const auto weights = static_weights.subspan(lag * channels, channels);
            for (std::size_t channel = 0; channel < channels; ++channel) {
                float weight = weights[channel];
                if (groups != 0) {
                    const auto group = channel / group_width;
                    weight += dynamic_weights[(row * groups + group) * taps + lag];
                }
                output_row[channel] += sample[channel] * weight;
            }
        }
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float residual = plan.residual_weight * input_row[channel];
            output_row[channel] = (output_row[channel] + residual) / plan.output_divisor;
        }
    }

    if (!history.empty() && !input.empty()) {
        const auto carried = std::min(history.size(), input.size());
        std::ranges::move(history.subspan(carried), history.begin());
        std::ranges::copy(input.last(carried), history.last(carried).begin());
    }
}

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace causal_conv {

struct Args {
    std::size_t rows;
    const float* input;            // [row][channel]
    const float* static_weights;   // [lag][channel], current tap first
    const float* dynamic_weights;  // [row][group][lag], or nullptr with no dynamic groups
    float* history;                // [taps - 1][channel], oldest first
    float* output;                 // [row][channel]
};

// Accumulation order follows the accessor's tap order. CUDA GDN starts with
// the first multiply, HIP GDN starts at +0, and projected convolution uses
// explicit FP32 operations even in a TU that enables contraction.
template <bool ExplicitRound, bool FirstProduct = false, class Index, class Sample, class Weight>
__device__ __forceinline__ float weighted_taps(Index taps, const Sample& sample, const Weight& weight) {
    float sum = 0.0F;
    Index tap{0};
    if constexpr (FirstProduct) {
        sum = __fmul_rn(sample(tap), weight(tap));
        ++tap;
    }
    for (; tap < taps; ++tap) {
        if constexpr (ExplicitRound) sum = __fadd_rn(sum, __fmul_rn(sample(tap), weight(tap)));
        else sum += sample(tap) * weight(tap);
    }
    return sum;
}

namespace detail {

// RoundInput: the producer left the input rows (and the dynamic taps) unrounded; every read applies the same
// scalar_float::bf16_round a separate round pass would have stored, so values, history and output are unchanged.
template <bool RoundInput>
__device__ __forceinline__ float input_value(float value) {
    if constexpr (RoundInput) return weight::scalar_float::bf16_round(value);
    return value;
}

template <bool RoundOutput>
__device__ __forceinline__ float finish_channel(const ShortConvolution& plan, float sum, float input) {
    const float residual = __fmul_rn(plan.residual_weight, input);
    const float result = __fdiv_rn(__fadd_rn(sum, residual), plan.output_divisor);
    if constexpr (RoundOutput) return weight::scalar_float::bf16_round(result);
    else return result;
}

template <bool RoundInput = false>
__device__ __forceinline__ void commit_channel(const ShortConvolution& plan, const Args& args, std::size_t channel) {
    const auto history_rows = plan.taps - 1;
    const auto carried = args.rows < history_rows ? args.rows : history_rows;
    const auto kept = history_rows - carried;
    for (std::size_t h = 0; h < kept; ++h) {
        args.history[h * plan.channels + channel] = args.history[(h + carried) * plan.channels + channel];
    }
    for (std::size_t h = kept; h < history_rows; ++h) {
        const auto row = args.rows - (history_rows - h);
        args.history[h * plan.channels + channel] = input_value<RoundInput>(args.input[row * plan.channels + channel]);
    }
}

template <bool Dynamic, bool Decode, bool RoundOutput = false, bool RoundInput = false>
__global__ void convolve(const ShortConvolution plan, const Args args, const std::size_t group_width) {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
    const std::size_t first_channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t channel_stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t channel = first_channel; channel < plan.channels; channel += channel_stride) {
        std::size_t group = 0;
        if constexpr (Dynamic) group = channel / group_width;
        for (std::size_t row = blockIdx.y; row < args.rows; row += gridDim.y) {
            const float* projected_taps = nullptr;
            if constexpr (Dynamic) projected_taps = args.dynamic_weights + (row * plan.dynamic_groups + group) * plan.taps;
            const auto sample = [&](std::size_t lag) {
                return lag <= row ? input_value<RoundInput>(args.input[(row - lag) * plan.channels + channel])
                    : args.history[(plan.taps - 1 - (lag - row)) * plan.channels + channel];
            };
            const auto weight = [&](std::size_t lag) {
                const float value = args.static_weights[lag * plan.channels + channel];
                if constexpr (Dynamic) return __fadd_rn(value, input_value<RoundInput>(projected_taps[lag]));
                return value;
            };
            const float sum = weighted_taps<true>(plan.taps, sample, weight);
            const auto index = row * plan.channels + channel;
            args.output[index] = finish_channel<RoundOutput>(plan, sum, input_value<RoundInput>(args.input[index]));
        }
        if constexpr (Decode) commit_channel<RoundInput>(plan, args, channel);
    }
}

template <bool RoundOutput, bool RoundInput = false>
__global__ void commit_prefill(const ShortConvolution plan, const Args args) {
    const std::size_t first_channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t channel_stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t channel = first_channel; channel < plan.channels; channel += channel_stride) {
        commit_channel<RoundInput>(plan, args, channel);
    }
}

}  // namespace detail

struct LaunchPlan {
    unsigned channel_blocks;
    unsigned row_blocks;
    std::size_t group_width;

    static constexpr unsigned threads = 256;
    static constexpr std::size_t grid_limit = 65535;
};

#if defined(__CUDACC__) && !defined(__HIPCC__)
namespace detail {

template <unsigned Rows, bool Dynamic, bool RoundOutput>
__global__ void convolve_step(StepArgs args, std::size_t group_width) {
    const auto& plan = args.plan;
    const auto history_rows = plan.taps - 1U;
    const auto image_size = history_rows * plan.channels;
    const std::size_t first = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t channel = first; channel < plan.channels; channel += stride) {
        unsigned completed = 0U;
        for (unsigned r = 0; r < Rows; ++r) {
            if (!exec::step_row_valid(*args.step, r)) continue;
            const auto& row = args.step->row[r];
            const auto* slot = args.history.entry(row.slot);
            if (!decode_state::accepts(slot, row.position)) continue;
            const auto parent = decode_state::parent_row(*args.step, r);
            const auto* history = reinterpret_cast<const float*>(slot->base);
            if (parent == exec::kStepNoParent) {
                if (row.parent != exec::kStepNoParent || row.position != slot->frontier) continue;
            } else {
                if ((completed & (1U << parent)) == 0U) continue;
                if (history_rows) history = args.pending + std::size_t{parent} * image_size;
            }
            const auto index = std::size_t{r} * plan.channels + channel;
            const auto sample = [&](std::size_t lag) {
                return lag == 0U ? args.input[index] : history[(history_rows - lag) * plan.channels + channel];
            };
            const auto weight = [&](std::size_t lag) {
                const float value = args.static_weights[lag * plan.channels + channel];
                if constexpr (Dynamic) {
                    const auto group = std::size_t{r} * plan.dynamic_groups + channel / group_width;
                    return __fadd_rn(value, args.dynamic_weights[group * plan.taps + lag]);
                }
                return value;
            };
            args.output[index] = finish_channel<RoundOutput>(plan, weighted_taps<true>(plan.taps, sample, weight),
                                                               args.input[index]);
            if (history_rows) {
                auto* pending = args.pending + std::size_t{r} * image_size + channel;
                for (std::size_t h = 1; h < history_rows; ++h) {
                    pending[(h - 1U) * plan.channels] = history[h * plan.channels + channel];
                }
                pending[(history_rows - 1U) * plan.channels] = args.input[index];
            }
            completed |= 1U << r;
        }
    }
}

}  // namespace detail
#endif

inline bool valid(const ShortConvolution& plan, const Args& args) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    return valid_short_convolution_plan(plan) && args.input != nullptr && args.static_weights != nullptr &&
           args.output != nullptr && (plan.taps == 1 || args.history != nullptr) &&
           (plan.dynamic_groups == 0) == (args.dynamic_weights == nullptr) &&
           args.rows <= max_elements / plan.channels && plan.taps <= max_elements / plan.channels &&
           (plan.dynamic_groups == 0 || args.rows <= max_elements / plan.taps / plan.dynamic_groups);
}

inline LaunchPlan launch_plan(const ShortConvolution& plan, const Args& args) {
    return {
        static_cast<unsigned>(std::min((plan.channels - 1) / LaunchPlan::threads + 1, LaunchPlan::grid_limit)),
        static_cast<unsigned>(std::min(args.rows, LaunchPlan::grid_limit)),
        plan.dynamic_groups == 0 ? 0 : plan.channels / plan.dynamic_groups,
    };
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
template <unsigned Rows, bool Round>
inline cudaError_t enqueue_step(const StepArgs& args, cudaStream_t stream) {
    const Args buffers{Rows, args.input, args.static_weights, args.dynamic_weights, args.pending, args.output};
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.step || !args.history.valid() || !valid(args.plan, buffers) ||
        args.plan.taps - 1U > limit / Rows / args.plan.channels) return cudaErrorInvalidValue;
    const auto grid = launch_plan(args.plan, buffers);
    if (args.plan.dynamic_groups != 0U) {
        detail::convolve_step<Rows, true, Round><<<grid.channel_blocks, grid.threads, 0, stream>>>(args, grid.group_width);
    } else {
        detail::convolve_step<Rows, false, Round><<<grid.channel_blocks, grid.threads, 0, stream>>>(args, 0U);
    }
    return cudaGetLastError();
}
#endif


template <bool RoundOutput, bool Decode, bool RoundInput = false>
inline void enqueue(const ShortConvolution& plan, const Args& args, LaunchPlan config,
                    pointwise::Stream stream) {
    const dim3 grid(config.channel_blocks, config.row_blocks);
    if (plan.dynamic_groups != 0) {
        detail::convolve<true, Decode, RoundOutput, RoundInput><<<grid, LaunchPlan::threads, 0, stream>>>(
            plan, args, config.group_width);
    } else {
        detail::convolve<false, Decode, RoundOutput, RoundInput><<<grid, LaunchPlan::threads, 0, stream>>>(
            plan, args, config.group_width);
    }
}

// Each decode thread owns its channel's history. A prefill needs a separate
// commit so all rows can read the previous history before any update.
// RoundInput = true: the conv rounds its input and dynamic-tap reads to BF16 (see detail::input_value).
template <bool RoundOutput, bool RoundInput = false>
inline pointwise::Status launch_with_store(const ShortConvolution& plan, const Args& args,
                                           pointwise::Stream stream) {
    constexpr pointwise::Status success{};
    if (args.rows == 0) return success;
    if (!valid(plan, args)) return pointwise::invalid;
    const auto config = launch_plan(plan, args);
    if (args.rows == 1) {
        enqueue<RoundOutput, true, RoundInput>(plan, args, config, stream);
        return pointwise::last_error();
    }
    enqueue<RoundOutput, false, RoundInput>(plan, args, config, stream);
    const auto status = pointwise::last_error();
    if (status != success || plan.taps == 1) return status;
    detail::commit_prefill<RoundOutput, RoundInput><<<config.channel_blocks, LaunchPlan::threads, 0, stream>>>(plan, args);
    return pointwise::last_error();
}

}  // namespace causal_conv
#endif

}  // namespace strix::core
