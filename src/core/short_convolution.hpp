#pragma once

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/pointwise.hpp"
#include <limits>
#endif
#include "strix/weight/formats/scalar_float.hpp"

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
#define STRIX_SHORT_CONV_HD __host__ __device__ __forceinline__
#else
#define STRIX_SHORT_CONV_HD inline
#endif

// Per-channel arithmetic of the causal convolution, one definition for the device kernels below and for every fused
// consumer that convolves a value it already holds (for example a producer epilogue): tap accumulation, epilogue,
// output rounding and the decode history commit. Device code uses round-to-nearest intrinsics (never contracted); the
// host forms are the plain FP32 operations.
namespace causal_conv {

namespace fp {
STRIX_SHORT_CONV_HD float add(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
STRIX_SHORT_CONV_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
STRIX_SHORT_CONV_HD float div(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
}  // namespace fp

// Accumulation order follows the accessor's tap order. CUDA GDN starts with
// the first multiply, HIP GDN starts at +0, and projected convolution uses
// explicit FP32 operations even in a TU that enables contraction.
template <bool ExplicitRound, bool FirstProduct = false, class Index, class Sample, class Weight>
STRIX_SHORT_CONV_HD float weighted_taps(Index taps, const Sample& sample, const Weight& weight) {
    float sum = 0.0F;
    Index tap{0};
    if constexpr (FirstProduct) {
        sum = fp::mul(sample(tap), weight(tap));
        ++tap;
    }
    for (; tap < taps; ++tap) {
        if constexpr (ExplicitRound) sum = fp::add(sum, fp::mul(sample(tap), weight(tap)));
        else sum += sample(tap) * weight(tap);
    }
    return sum;
}

// (taps + residual_weight * input) / output_divisor, the epilogue of every row.
STRIX_SHORT_CONV_HD float epilogue(const ShortConvolution& plan, float taps, float input) {
    const float residual = fp::mul(plan.residual_weight, input);
    return fp::div(fp::add(taps, residual), plan.output_divisor);
}

// The BF16 store rounding of a rounded convolution output.
STRIX_SHORT_CONV_HD float round_output(float value) { return weight::scalar_float::bf16_round(value); }

// History commit of one channel after `rows` input rows: keep the newest history rows, append the newest inputs.
// Element (row r, channel c) of `old`, `input` and `history` is [r * channels + c]; `old` is the history before the
// commit (it may be `history` itself: each row is read before it is overwritten), `input` the rows just convolved.
// Old and Input are pointers or any type with that subscript (a fused consumer may supply a value it holds).
template <class Index, class Old, class Input>
STRIX_SHORT_CONV_HD void commit_history(Index history_rows, Index rows, Index channels, Index channel, const Old& old,
                                        const Input& input, float* history) {
    const auto carried = rows < history_rows ? rows : history_rows;
    const auto kept = history_rows - carried;
    for (Index h = 0; h < kept; ++h) {
        history[h * channels + channel] = old[(h + carried) * channels + channel];
    }
    for (Index h = kept; h < history_rows; ++h) {
        const auto row = rows - (history_rows - h);
        history[h * channels + channel] = input[row * channels + channel];
    }
}

}  // namespace causal_conv

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

namespace detail {

// RoundInput: the producer left the input rows (and the dynamic taps) unrounded; every read applies the same
// scalar_float::bf16_round a separate round pass would have stored, so values, history and output are unchanged.
template <bool RoundInput>
__device__ __forceinline__ float input_value(float value) {
    if constexpr (RoundInput) return weight::scalar_float::bf16_round(value);
    return value;
}

// The newest input rows as the history commit reads them: BF16-rounded when the producer left them unrounded.
template <bool RoundInput>
struct RoundedRows {
    const float* rows;
    __device__ __forceinline__ float operator[](std::size_t index) const { return input_value<RoundInput>(rows[index]); }
};

template <bool RoundInput = false>
__device__ __forceinline__ void commit_channel(const ShortConvolution& plan, const Args& args, std::size_t channel) {
    commit_history(plan.taps - 1, args.rows, plan.channels, channel, args.history, RoundedRows<RoundInput>{args.input},
                   args.history);
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
            const float result = epilogue(plan, sum, input_value<RoundInput>(args.input[index]));
            if constexpr (RoundOutput) args.output[index] = round_output(result);
            else args.output[index] = result;
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

// Lanes of a multi-request decode step (strix/exec/step_lanes.hpp): lane l is ONE decode row of its own
// session, packed at row l of input / dynamic / output, with its own history at history(l). Each lane runs
// exactly the one-row decode of launch_with_store (same explicit FP32 operations in the same tap order, same
// commit), so a lane's output and history bits do not depend on the other lanes or on the width. A lane with
// history.writes(l) == false is padding: it reads a zero history and commits nothing (finite, deterministic,
// never touches session state). History is any device-copyable accessor with `T* operator()(unsigned)` and
// `bool writes(unsigned)`; it travels by value, so a captured launch replays for new slot assignments.
template <class History>
struct LaneArgs {
    unsigned lanes;
    const float* input;            // [lane][channel]
    const float* static_weights;   // [lag][channel], current tap first
    const float* dynamic_weights;  // [lane][group][lag], or nullptr with no dynamic groups
    History history;               // lane -> [taps - 1][channel], oldest first
    float* output;                 // [lane][channel]
};

namespace detail {

template <bool Dynamic, bool RoundOutput, bool RoundInput, class History>
__global__ void convolve_lanes(const ShortConvolution plan, const LaneArgs<History> args, const std::size_t group_width) {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
    const unsigned lane = blockIdx.y;
    const bool writes = args.history.writes(lane);
    float* history = args.history(lane);
    const std::size_t first_channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t channel_stride = std::size_t(gridDim.x) * blockDim.x;
    const float* input = args.input + std::size_t{lane} * plan.channels;
    for (std::size_t channel = first_channel; channel < plan.channels; channel += channel_stride) {
        std::size_t group = 0;
        if constexpr (Dynamic) group = channel / group_width;
        const float* projected_taps = nullptr;
        if constexpr (Dynamic) projected_taps = args.dynamic_weights + (std::size_t{lane} * plan.dynamic_groups + group) * plan.taps;
        // The one-row decode (row 0): lag 0 reads the input row, every other lag the committed history.
        const auto sample = [&](std::size_t lag) {
            if (lag == 0) return input_value<RoundInput>(input[channel]);
            return writes ? history[(plan.taps - 1 - lag) * plan.channels + channel] : 0.0F;
        };
        const auto weight = [&](std::size_t lag) {
            const float value = args.static_weights[lag * plan.channels + channel];
            if constexpr (Dynamic) return __fadd_rn(value, input_value<RoundInput>(projected_taps[lag]));
            return value;
        };
        const float sum = weighted_taps<true>(plan.taps, sample, weight);
        const float residual = __fmul_rn(plan.residual_weight, input_value<RoundInput>(input[channel]));
        const float result = __fdiv_rn(__fadd_rn(sum, residual), plan.output_divisor);
        const auto index = std::size_t{lane} * plan.channels + channel;
        if constexpr (RoundOutput) args.output[index] = weight::scalar_float::bf16_round(result);
        else args.output[index] = result;
        if (writes) commit_channel<RoundInput>(plan, Args{1U, input, args.static_weights, nullptr, history, nullptr}, channel);
    }
}

}  // namespace detail

template <class History>
inline bool valid_lanes(const ShortConvolution& plan, const LaneArgs<History>& args) {
    return valid_short_convolution_plan(plan) && args.lanes != 0U && args.lanes <= LaunchPlan::grid_limit &&
           args.input != nullptr && args.static_weights != nullptr && args.output != nullptr &&
           (plan.dynamic_groups == 0) == (args.dynamic_weights == nullptr) && args.input != args.output;
}

// One launch for every lane: grid (channel blocks, lanes).
template <bool RoundOutput, bool RoundInput = false, class History>
inline pointwise::Status launch_lanes(const ShortConvolution& plan, const LaneArgs<History>& args,
                                      pointwise::Stream stream) {
    if (!valid_lanes(plan, args)) return pointwise::invalid;
    const dim3 grid(static_cast<unsigned>(std::min((plan.channels - 1) / LaunchPlan::threads + 1, LaunchPlan::grid_limit)),
                    args.lanes);
    const std::size_t group_width = plan.dynamic_groups == 0 ? 0 : plan.channels / plan.dynamic_groups;
    if (plan.dynamic_groups != 0) {
        detail::convolve_lanes<true, RoundOutput, RoundInput><<<grid, LaunchPlan::threads, 0, stream>>>(
            plan, args, group_width);
    } else {
        detail::convolve_lanes<false, RoundOutput, RoundInput><<<grid, LaunchPlan::threads, 0, stream>>>(
            plan, args, group_width);
    }
    return pointwise::last_error();
}

}  // namespace causal_conv
#endif

}  // namespace strix::core
