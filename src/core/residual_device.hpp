#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "ops/norm/row_reduction.hpp"
#include "strix/attn/group_pool.hpp"
#endif

namespace strix::core::residual_device {

inline constexpr unsigned kMaxBanks = 8;
inline constexpr unsigned kMaxQueries = 8;

struct NormalizeArgs {
    const float* input{};
    float* output{};
    unsigned rows{};
    unsigned width{};
    float epsilon{};
    std::size_t input_stride{};   // Zero selects width.
    std::size_t output_stride{};
    bool round_bf16{};
};

struct MixArgs {
    const float* banks[kMaxBanks]{};   // Each bank is [rows][width].
    const float* queries{};           // [query_count][width], shared by tokens.
    float* output[kMaxQueries]{};     // Each output is [rows][width], disjoint from banks.
    unsigned rows{};
    unsigned width{};
    unsigned bank_count{};
    unsigned query_count{};
    float epsilon{};
    bool normalized{};               // Banks are already normalized as FP32.
};

struct AddArgs {
    const float* lhs{};
    const float* rhs{};
    float* output{};
    std::size_t elements{};
    bool round_bf16{};
};

struct RoundArgs {
    const float* input{};
    float* output{};
    std::size_t elements{};
};

struct ControllerArgs {
    const float* current{};
    const float* message{};
    const float* forget{};
    const float* gain{};
    float* output{};
    std::size_t elements{};
};

namespace detail {

inline constexpr std::size_t kMaxElements = std::numeric_limits<std::size_t>::max() / sizeof(float);

inline bool valid_rows(unsigned rows, unsigned width, std::size_t stride) {
    return width != 0U && stride >= width && rows <= kMaxElements / stride;
}

}  // namespace detail

inline bool fits(NormalizeArgs args) {
    if (args.rows == 0U) return true;
    return args.input && args.output && args.epsilon > 0.0F && std::isfinite(args.epsilon) &&
           detail::valid_rows(args.rows, args.width, args.input_stride ? args.input_stride : args.width) &&
           detail::valid_rows(args.rows, args.width, args.output_stride ? args.output_stride : args.width);
}

inline bool fits(const MixArgs& args) {
    if (args.rows == 0U) return true;
    if (!args.queries || args.bank_count == 0U || args.bank_count > kMaxBanks ||
        args.query_count == 0U || args.query_count > kMaxQueries ||
        !detail::valid_rows(args.rows, args.width, args.width) ||
        !detail::valid_rows(args.query_count, args.width, args.width) ||
        (!args.normalized && (!(args.epsilon > 0.0F) || !std::isfinite(args.epsilon)))) return false;
    for (unsigned bank = 0; bank < args.bank_count; ++bank)
        if (!args.banks[bank]) return false;
    for (unsigned query = 0; query < args.query_count; ++query) {
        if (!args.output[query] || args.output[query] == args.queries) return false;
        for (unsigned bank = 0; bank < args.bank_count; ++bank)
            if (args.banks[bank] == args.output[query]) return false;
        for (unsigned other = 0; other < query; ++other)
            if (args.output[other] == args.output[query]) return false;
    }
    return true;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
#if defined(__HIPCC__)
using Error = hipError_t;
using Stream = hipStream_t;
inline constexpr Error success = hipSuccess;
inline constexpr Error invalid_value = hipErrorInvalidValue;
#else
using Error = cudaError_t;
using Stream = cudaStream_t;
inline constexpr Error success = cudaSuccess;
inline constexpr Error invalid_value = cudaErrorInvalidValue;
#endif

namespace detail {

inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kGridLimit = 65535;

namespace gi = attn::group_indexer;
namespace pool = attn::group_pool;

__device__ __forceinline__ float inverse_rms(const float* input, unsigned width, float epsilon, float* scratch) {
    float sum = 0.0F;
    for (unsigned i = threadIdx.x; i < width; i += kThreads) sum = fmaf(input[i], input[i], sum);
    const float squares = ops::norm::lds_halving_sum<kThreads>(scratch, sum);
    return 1.0F / pool::sqrt_rn(squares / static_cast<float>(width) + epsilon);
}

static __global__ void normalize(NormalizeArgs args) {
    __shared__ float scratch[kThreads];
    for (std::size_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        const auto* in = args.input + row * args.input_stride;
        auto* out = args.output + row * args.output_stride;
        const float inverse = inverse_rms(in, args.width, args.epsilon, scratch);
        for (unsigned i = threadIdx.x; i < args.width; i += kThreads) {
            const float value = in[i] * inverse;
            out[i] = args.round_bf16 ? gi::bf16_round(value) : value;
        }
        __syncthreads();
    }
}

// Work on one token at a time. Cache the bank RMS once and use it for all
// queries, then mix directly into the destination without writing a temporary
// normalized bank. Captured cross-pass banks reuse their existing RMS rows.
template <bool Normalized>
static __global__ void mix(MixArgs args) {
    __shared__ float scratch[kThreads];
    __shared__ float inverse[kMaxBanks];
    __shared__ float probability[kMaxQueries][kMaxBanks];
    const float score_scale = 1.0F / pool::sqrt_rn(static_cast<float>(args.width));
    for (std::size_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        const auto offset = row * args.width;
        for (unsigned bank = 0; bank < args.bank_count; ++bank) {
            const auto* source = args.banks[bank] + offset;
            float scale = 1.0F;
            if constexpr (!Normalized) {
                scale = inverse_rms(source, args.width, args.epsilon, scratch);
                if (threadIdx.x == 0U) inverse[bank] = scale;
            }
            for (unsigned query = 0; query < args.query_count; ++query) {
                const auto* learned = args.queries + std::size_t{query} * args.width;
                float dot = 0.0F;
                for (unsigned i = threadIdx.x; i < args.width; i += kThreads) {
                    const float value = Normalized ? source[i] : source[i] * scale;
                    dot = fmaf(learned[i], value, dot);
                }
                const float score = ops::norm::lds_halving_sum<kThreads>(scratch, dot) * score_scale;
                if (threadIdx.x == 0U) probability[query][bank] = score;
            }
        }
        __syncthreads();
        if (threadIdx.x < args.query_count) {
            auto* probs = probability[threadIdx.x];
            float maximum = probs[0];
            for (unsigned bank = 1; bank < args.bank_count; ++bank) maximum = fmaxf(maximum, probs[bank]);
            float denominator = 0.0F;
            for (unsigned bank = 0; bank < args.bank_count; ++bank) {
                probs[bank] = expf(probs[bank] - maximum);
                denominator += probs[bank];
            }
            for (unsigned bank = 0; bank < args.bank_count; ++bank) probs[bank] /= denominator;
        }
        __syncthreads();
        for (unsigned query = 0; query < args.query_count; ++query) {
            auto* out = args.output[query] + offset;
            for (unsigned i = threadIdx.x; i < args.width; i += kThreads) {
                float value = 0.0F;
                for (unsigned bank = 0; bank < args.bank_count; ++bank) {
                    const float x = args.banks[bank][offset + i];
                    value = fmaf(probability[query][bank], Normalized ? x : x * inverse[bank], value);
                }
                out[i] = value;
            }
        }
        __syncthreads();
    }
}

template <class Args>
static __global__ void pointwise(Args args) {
    const auto first = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const auto step = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t i = first; i < args.elements; i += step) {
        if constexpr (std::is_same_v<Args, AddArgs>) {
            const float value = args.lhs[i] + args.rhs[i];
            args.output[i] = args.round_bf16 ? gi::bf16_round(value) : value;
        } else if constexpr (std::is_same_v<Args, RoundArgs>) {
            args.output[i] = gi::bf16_round(args.input[i]);
        } else {
            const float retained = __fmul_rn(__fsub_rn(1.0F, tanhf(args.forget[i])), args.current[i]);
            const float incoming = __fmul_rn(tanhf(args.gain[i]), args.message[i]);
            args.output[i] = __fadd_rn(retained, incoming);
        }
    }
}

inline Error last_error() {
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}

template <class Args>
inline Error launch_pointwise(Args args, Stream stream, bool pointers_valid) {
    if (args.elements == 0U) return success;
    if (!pointers_valid || args.elements > kMaxElements) return invalid_value;
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((args.elements - 1U) / kThreads + 1U, kGridLimit));
    pointwise<<<blocks, kThreads, 0, stream>>>(args);
    return last_error();
}

}  // namespace detail

// An input may alias its output only for an identical per-row layout.
inline Error launch_normalize(NormalizeArgs args, Stream stream) {
    if (args.rows == 0U) return success;
    if (!fits(args)) return invalid_value;
    if (args.input_stride == 0U) args.input_stride = args.width;
    if (args.output_stride == 0U) args.output_stride = args.width;
    if (args.input == args.output && args.input_stride != args.output_stride) return invalid_value;
    detail::normalize<<<std::min(args.rows, detail::kGridLimit), detail::kThreads, 0, stream>>>(args);
    return detail::last_error();
}

inline Error launch_mix(const MixArgs& args, Stream stream) {
    if (args.rows == 0U) return success;
    if (!fits(args)) return invalid_value;
    const auto blocks = std::min(args.rows, detail::kGridLimit);
    if (args.normalized) detail::mix<true><<<blocks, detail::kThreads, 0, stream>>>(args);
    else detail::mix<false><<<blocks, detail::kThreads, 0, stream>>>(args);
    return detail::last_error();
}

// Pointwise operations support exact in-place aliasing with any input.
inline Error launch_add(AddArgs args, Stream stream) {
    return detail::launch_pointwise(args, stream, args.lhs && args.rhs && args.output);
}

inline Error launch_round(RoundArgs args, Stream stream) {
    return detail::launch_pointwise(args, stream, args.input && args.output);
}

inline Error launch_controller(ControllerArgs args, Stream stream) {
    return detail::launch_pointwise(args, stream, args.current && args.message && args.forget && args.gain && args.output);
}
#endif

}  // namespace strix::core::residual_device
