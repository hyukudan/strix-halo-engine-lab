#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

#include "core/dense_row_plan.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/controller_mix.hpp"
#include "core/pointwise.hpp"
#include "ops/norm/norm_device.hpp"
#include "strix/attn/group_pool.hpp"
#endif

namespace strix::core::residual_mix {

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

struct NormalizePairArgs {
    NormalizeArgs first, second;
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
// Backend geometry: the decode schedule's register form (same bits as the loop form) is measured on gfx1151 only.
inline constexpr bool kMixRegisterForm = true;
#else
using Error = cudaError_t;
using Stream = cudaStream_t;
inline constexpr Error success = cudaSuccess;
inline constexpr Error invalid_value = cudaErrorInvalidValue;
inline constexpr bool kMixRegisterForm = false;
#endif

namespace detail {

inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kGridLimit = 65535;

namespace gi = attn::group_indexer;
namespace pool = attn::group_pool;

__device__ __forceinline__ float rms_inverse(float squares, unsigned width, float epsilon) {
    return 1.0F / pool::sqrt_rn(squares / static_cast<float>(width) + epsilon);
}

__device__ __forceinline__ float inverse_rms(const float* input, unsigned width, float epsilon, float* scratch) {
    const float sum = ops::norm::square_partial<kThreads>(input, width, threadIdx.x);
    const float squares = ops::norm::lds_halving_sum<kThreads>(scratch, sum);
    return rms_inverse(squares, width, epsilon);
}

struct BankNorm {
    using Scale = float;
    using ScaleFormat = bool;
    static constexpr auto reduction_order = ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    template <unsigned L>
    __device__ static float reduce(float* shared, float part) { return ops::norm::lds_halving_sum<L>(shared, part); }
    __device__ static float inv_sqrt(float value) { return 1.0F / pool::sqrt_rn(value); }
    __device__ static float finish(float value, bool round) { return round ? gi::bf16_round(value) : value; }
};

__device__ __forceinline__ void normalize_rows(NormalizeArgs args, float* scratch) {
    for (std::size_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        ops::norm::rmsnorm_row<BankNorm, kThreads, false>(
            scratch, args.input + row * args.input_stride, nullptr, nullptr, nullptr, args.round_bf16,
            args.width, args.epsilon, args.output + row * args.output_stride, 0, ops::norm::NoStore{});
        __syncthreads();
    }
}

static __global__ void normalize(NormalizeArgs args) {
    __shared__ float scratch[kThreads];
    normalize_rows(args, scratch);
}

struct NormalizePair {
    NormalizeArgs operands[2];
};

static __global__ void normalize_pair(NormalizeArgs first, NormalizeArgs second) {
    __shared__ float scratch[kThreads];
    normalize_rows(blockIdx.y == 0U ? first : second, scratch);
}

template <bool Normalized>
__device__ __forceinline__ float mix_dot_partial(const float* learned, const float* source, unsigned width,
                                                unsigned lane, float scale) {
    float dot = 0.0F;
    for (unsigned i = lane; i < width; i += kThreads) {
        const float value = Normalized ? source[i] : source[i] * scale;
        dot = fmaf(learned[i], value, dot);
    }
    return dot;
}

inline constexpr unsigned kPairedBankThreads = kThreads / 2U;
static_assert(kPairedBankThreads * kMaxBanks <= 1024U);

// Register form of the parallel (decode) schedule for width <= kThreads * kMixRegs: a lane's strided values are loaded
// before their FMA chain, so one block pays one memory round trip per chain instead of one per strided element (the
// loop form waits for each load before its FMA). The output loop hoists the bank pointers, loads a batch of elements of
// every bank before the per-query chains and shares them between queries (outputs are disjoint from banks). Every
// chain (ascending i per lane), tree and expression is the loop form's: the same bits.
inline constexpr unsigned kMixRegs = 16;
inline constexpr unsigned kMixOutBatch = 4;
inline constexpr unsigned kMixRegisterWidth = kThreads * kMixRegs;

__device__ __forceinline__ void load_strided(const float* row, unsigned width, unsigned lane, float (&v)[kMixRegs]) {
#pragma unroll
    for (unsigned k = 0; k < kMixRegs; ++k) {
        const unsigned i = lane + k * kThreads;
        v[k] = i < width ? row[i] : 0.0F;
    }
}

// square_partial<kThreads>(row, width, lane) on the loaded values.
__device__ __forceinline__ float square_chain(const float (&v)[kMixRegs], unsigned width, unsigned lane) {
    float sum = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < kMixRegs; ++k) {
        if (lane + k * kThreads < width) sum = __builtin_fmaf(v[k], v[k], sum);
    }
    return sum;
}

// mix_dot_partial<Normalized>(learned, row, width, lane, scale) on the loaded values.
template <bool Normalized>
__device__ __forceinline__ float dot_chain(const float (&learned)[kMixRegs], const float (&v)[kMixRegs], unsigned width,
                                           unsigned lane, float scale) {
    float dot = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < kMixRegs; ++k) {
        if (lane + k * kThreads < width) {
            const float value = Normalized ? v[k] : v[k] * scale;
            dot = fmaf(learned[k], value, dot);
        }
    }
    return dot;
}

// The scores of one bank slice (blockDim.x = kPairedBankThreads, virtual lanes lane and lane + kPairedBankThreads).
template <bool Normalized>
__device__ __forceinline__ void score_bank_registers(const MixArgs& args, std::size_t offset, unsigned bank,
                                                     unsigned lane, float* bank_scratch, float* inverse,
                                                     float (*probability)[kMaxBanks], float score_scale) {
    const auto* source = args.banks[bank] + offset;
    float low[kMixRegs], high[kMixRegs];
    load_strided(source, args.width, lane, low);
    load_strided(source, args.width, lane + kPairedBankThreads, high);
    float scale = 1.0F;
    if constexpr (!Normalized) {
        const float squares = ops::norm::lds_halving_sum<kPairedBankThreads>(
            bank_scratch, square_chain(low, args.width, lane) + square_chain(high, args.width, lane + kPairedBankThreads));
        scale = rms_inverse(squares, args.width, args.epsilon);
        if (lane == 0U) inverse[bank] = scale;
    }
    for (unsigned query = 0; query < args.query_count; ++query) {
        const auto* learned = args.queries + std::size_t{query} * args.width;
        float learned_low[kMixRegs], learned_high[kMixRegs];
        load_strided(learned, args.width, lane, learned_low);
        load_strided(learned, args.width, lane + kPairedBankThreads, learned_high);
        float dot = dot_chain<Normalized>(learned_low, low, args.width, lane, scale);
        const float upper = dot_chain<Normalized>(learned_high, high, args.width, lane + kPairedBankThreads, scale);
        dot = dot + upper;
        const float score = ops::norm::lds_halving_sum<kPairedBankThreads>(bank_scratch, dot) * score_scale;
        if (lane == 0U) probability[query][bank] = score;
    }
}

template <bool Normalized>
__device__ __forceinline__ void emit_registers(const MixArgs& args, std::size_t offset, unsigned output_lane,
                                               unsigned output_stride, const float* inverse,
                                               const float (*probability)[kMaxBanks]) {
    const float* rows[kMaxBanks];
#pragma unroll
    for (unsigned bank = 0; bank < kMaxBanks; ++bank)
        rows[bank] = bank < args.bank_count ? args.banks[bank] + offset : nullptr;
    for (unsigned first = output_lane; first < args.width; first += kMixOutBatch * output_stride) {
        float x[kMixOutBatch][kMaxBanks];
#pragma unroll
        for (unsigned k = 0; k < kMixOutBatch; ++k) {
            const unsigned i = first + k * output_stride;
#pragma unroll
            for (unsigned bank = 0; bank < kMaxBanks; ++bank)
                x[k][bank] = bank < args.bank_count && i < args.width ? rows[bank][i] : 0.0F;
        }
        for (unsigned query = 0; query < args.query_count; ++query) {
            auto* out = args.output[query] + offset;
#pragma unroll
            for (unsigned k = 0; k < kMixOutBatch; ++k) {
                const unsigned i = first + k * output_stride;
                if (i >= args.width) continue;
                float value = 0.0F;
#pragma unroll
                for (unsigned bank = 0; bank < kMaxBanks; ++bank) {
                    if (bank < args.bank_count)
                        value = fmaf(probability[query][bank], Normalized ? x[k][bank] : x[k][bank] * inverse[bank],
                                     value);
                }
                out[i] = value;
            }
        }
    }
}

// Work on one token at a time. Cache the bank RMS once and use it for all
// queries, then mix directly into the destination without writing a temporary
// normalized bank. Captured cross-pass banks reuse their existing RMS rows.
// The parallel schedule launches one blockDim.y slice per bank. Each x thread
// owns virtual lanes x and x+128, so their first add is stage 128 of the original
// tree. All slices enter the remaining stages and their block barriers together.
template <bool Normalized, bool ParallelBanks = false>
static __global__ void mix(MixArgs args) {
    constexpr unsigned bank_threads = ParallelBanks ? kPairedBankThreads : kThreads;
    __shared__ float scratch[ParallelBanks ? bank_threads * kMaxBanks : kThreads];
    __shared__ float inverse[kMaxBanks];
    __shared__ float probability[kMaxQueries][kMaxBanks];
    const unsigned lane = threadIdx.x;
    const unsigned first_bank = ParallelBanks ? threadIdx.y : 0U;
    const unsigned banks_per_slice = ParallelBanks ? 1U : args.bank_count;
    const unsigned output_lane = ParallelBanks ? first_bank * bank_threads + lane : lane;
    const unsigned output_stride = ParallelBanks ? blockDim.y * bank_threads : kThreads;
    auto* bank_scratch = scratch + first_bank * bank_threads;
    const float score_scale = 1.0F / pool::sqrt_rn(static_cast<float>(args.width));
    const bool registers = kMixRegisterForm && ParallelBanks && args.width <= kMixRegisterWidth;
    for (std::size_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        const auto offset = row * args.width;
        if constexpr (ParallelBanks && kMixRegisterForm) {
            if (registers)
                score_bank_registers<Normalized>(args, offset, first_bank, lane, bank_scratch, inverse, probability,
                                                 score_scale);
        }
        for (unsigned local_bank = 0; local_bank < (registers ? 0U : banks_per_slice); ++local_bank) {
            const unsigned bank = first_bank + local_bank;
            const auto* source = args.banks[bank] + offset;
            float scale = 1.0F;
            if constexpr (!Normalized) {
                if constexpr (ParallelBanks) {
                    const float low = ops::norm::square_partial<kThreads>(source, args.width, lane);
                    const float high = ops::norm::square_partial<kThreads>(source, args.width, lane + bank_threads);
                    const float squares = ops::norm::lds_halving_sum<bank_threads>(bank_scratch, low + high);
                    scale = rms_inverse(squares, args.width, args.epsilon);
                } else {
                    scale = inverse_rms(source, args.width, args.epsilon, bank_scratch);
                }
                if (lane == 0U) inverse[bank] = scale;
            }
            for (unsigned query = 0; query < args.query_count; ++query) {
                const auto* learned = args.queries + std::size_t{query} * args.width;
                float dot = mix_dot_partial<Normalized>(learned, source, args.width, lane, scale);
                if constexpr (ParallelBanks) {
                    const float high = mix_dot_partial<Normalized>(learned, source, args.width,
                                                                   lane + bank_threads, scale);
                    dot = dot + high;
                }
                const float score = ops::norm::lds_halving_sum<bank_threads>(bank_scratch, dot) * score_scale;
                if (lane == 0U) probability[query][bank] = score;
            }
        }
        __syncthreads();
        if (output_lane < args.query_count) {
            auto* probs = probability[output_lane];
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
        if constexpr (ParallelBanks && kMixRegisterForm) {
            if (registers) {
                emit_registers<Normalized>(args, offset, output_lane, output_stride, inverse, probability);
                __syncthreads();
                continue;
            }
        }
        for (unsigned query = 0; query < args.query_count; ++query) {
            auto* out = args.output[query] + offset;
            for (unsigned i = output_lane; i < args.width; i += output_stride) {
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

struct AddRound {
    const float* lhs;
    const float* rhs;
    float* output;
    __device__ void operator()(std::size_t i) const { output[i] = gi::bf16_round(__fadd_rn(lhs[i], rhs[i])); }
};

struct Round {
    const float* input;
    float* output;
    __device__ void operator()(std::size_t i) const { output[i] = gi::bf16_round(input[i]); }
};

struct Controller {
    ControllerArgs args;
    __device__ void operator()(std::size_t i) const {
        args.output[i] = controller_mix(args.current[i], args.message[i], args.forget[i], args.gain[i]);
    }
};

using pointwise::last_error;

template <bool ParallelBanks = false>
inline Error launch_mix_schedule(const MixArgs& args, Stream stream) {
    const auto blocks = std::min(args.rows, kGridLimit);
    if constexpr (ParallelBanks) {
        const dim3 threads(kPairedBankThreads, args.bank_count);
        if (args.normalized) mix<true, true><<<blocks, threads, 0, stream>>>(args);
        else mix<false, true><<<blocks, threads, 0, stream>>>(args);
    } else {
        if (args.normalized) mix<true><<<blocks, kThreads, 0, stream>>>(args);
        else mix<false><<<blocks, kThreads, 0, stream>>>(args);
    }
    return last_error();
}

template <class Transform>
inline Error launch_pointwise(std::size_t elements, Transform transform, Stream stream, bool pointers_valid) {
    if (elements == 0U) return success;
    if (!pointers_valid) return invalid_value;
    return pointwise::elements(elements, transform, stream);
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

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Two disjoint operands share a row grid. A zero stride selects the width.
inline Error launch_normalize_pair(NormalizePairArgs pair, Stream stream) {
    auto& [first, second] = pair;
    if (first.rows != second.rows || !fits(first) || !fits(second)) return invalid_value;
    if (first.rows == 0U) return success;
    for (auto* args : {&first, &second}) {
        if (args->input_stride == 0U) args->input_stride = args->width;
        if (args->output_stride == 0U) args->output_stride = args->width;
        if (args->input == args->output && args->input_stride != args->output_stride) return invalid_value;
    }
    const dim3 grid(std::min(first.rows, detail::kGridLimit), 2U);
    detail::normalize_pair<<<grid, detail::kThreads, 0, stream>>>(first, second);
    return detail::last_error();
}
#endif

inline Error launch_mix(const MixArgs& args, Stream stream) {
    if (args.rows == 0U) return success;
    if (!fits(args)) return invalid_value;
    // A short chunk scores its banks in parallel while preserving the serial reduction tree.
    if (args.rows <= 2U && args.bank_count > 1U) return detail::launch_mix_schedule<true>(args, stream);
    return detail::launch_mix_schedule(args, stream);
}

// Pointwise operations support exact in-place aliasing with any input.
inline Error launch_add(AddArgs args, Stream stream) {
    const bool valid = args.lhs && args.rhs && args.output;
    if (args.round_bf16)
        return detail::launch_pointwise(args.elements, detail::AddRound{args.lhs, args.rhs, args.output}, stream, valid);
    return detail::launch_pointwise(args.elements, pointwise::Add{args.lhs, args.rhs, args.output}, stream, valid);
}

inline Error launch_round(RoundArgs args, Stream stream) {
    return detail::launch_pointwise(args.elements, detail::Round{args.input, args.output}, stream,
                                    args.input && args.output);
}

inline Error launch_controller(ControllerArgs args, Stream stream) {
    return detail::launch_pointwise(args.elements, detail::Controller{args}, stream,
                                    args.current && args.message && args.forget && args.gain && args.output);
}

using ConstRows = std::span<const float>;
using MutableRows = std::span<float>;

inline bool bind_banks(MixArgs& args, std::span<const ConstRows> banks, std::size_t row_elements) {
    if (banks.size() > kMaxBanks) return false;
    args.bank_count = static_cast<unsigned>(banks.size());
    for (std::size_t i = 0; i < banks.size(); ++i) {
        if (banks[i].size() != row_elements) return false;
        args.banks[i] = banks[i].data();
    }
    return true;
}

inline Error launch_residual_read(std::span<const ConstRows> previous, ConstRows current, ConstRows queries,
                                  std::span<const MutableRows> outputs, unsigned width, unsigned max_rows,
                                  float epsilon, Stream stream) {
    if (width == 0U || current.empty() || current.size() % width != 0U || current.size() / width > max_rows ||
        previous.size() >= kMaxBanks || outputs.empty() || outputs.size() > kMaxQueries ||
        queries.size() != outputs.size() * std::size_t{width}) return invalid_value;
    MixArgs mix{};
    if (!bind_banks(mix, previous, current.size())) return invalid_value;
    mix.banks[mix.bank_count++] = current.data();
    mix.queries = queries.data();
    mix.rows = static_cast<unsigned>(current.size() / width);
    mix.width = width;
    mix.query_count = static_cast<unsigned>(outputs.size());
    mix.epsilon = epsilon;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].size() != current.size()) return invalid_value;
        mix.output[i] = outputs[i].data();
    }
    return launch_mix(mix, stream);
}

inline Error launch_capture_message_bank(std::span<const ConstRows> inputs, std::span<const MutableRows> normalized,
                                         unsigned width, unsigned max_rows, float epsilon, Stream stream) {
    if (inputs.empty() || inputs.size() > kMaxBanks || inputs.size() != normalized.size() || width == 0U ||
        inputs.front().empty() || inputs.front().size() % width != 0U ||
        inputs.front().size() / width > max_rows) return invalid_value;
    const auto elements = inputs.front().size();
    const auto rows = static_cast<unsigned>(elements / width);
    NormalizeArgs args[kMaxBanks]{};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != elements || normalized[i].size() != elements) return invalid_value;
        for (std::size_t j = 0; j < inputs.size(); ++j)
            if (i != j && (normalized[i].data() == inputs[j].data() ||
                           normalized[i].data() == normalized[j].data())) return invalid_value;
        args[i] = {inputs[i].data(), normalized[i].data(), rows, width, epsilon};
        if (!fits(args[i])) return invalid_value;
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto status = launch_normalize(args[i], stream);
        if (status != success) return status;
    }
    return success;
}

struct LowRankController {
    unsigned width;
    unsigned rank;
    unsigned message_count;
    float epsilon;
    bool round_operands_bf16;
    bool round_latent_bf16;

    constexpr std::size_t scratch_elements_per_row() const {
        return 4U * std::size_t{width} + rank;
    }

    constexpr std::size_t scratch_elements(unsigned rows) const {
        return rows * scratch_elements_per_row();
    }
};

struct ControllerWeights {
    ConstRows query;
    const std::uint16_t* down;
    const std::uint16_t* forget;
    const std::uint16_t* gain;
    const std::uint16_t* norm;
};

// Banks are pre-normalized FP32 rows. The final norm (and its output precision)
// is supplied by Operations::rms. Scratch is disjoint from the caller's rows.
template <class Operations>
void enqueue_low_rank_controller(LowRankController plan, ControllerWeights weights, unsigned rows, ConstRows raw,
                                 std::span<const ConstRows> bank, MutableRows output, MutableRows scratch,
                                 Operations& operations) {
    const auto elements = std::size_t{rows} * plan.width;
    if (rows == 0U || plan.width == 0U || plan.width > std::numeric_limits<unsigned>::max() / 2U ||
        plan.rank == 0U || rows > detail::kMaxElements / plan.scratch_elements_per_row() ||
        bank.empty() || bank.size() != plan.message_count || raw.size() != elements || output.size() != elements ||
        weights.query.size() != plan.width || scratch.size() < plan.scratch_elements(rows))
        throw std::invalid_argument("invalid low-rank controller buffers");

    const auto message = scratch.first(elements);
    const auto operands = scratch.subspan(elements, 2U * elements);
    // Gate outputs reuse the down input after the projection has consumed it.
    const auto forget = operands.first(elements);
    const auto gain = operands.last(elements);
    const auto conditioned = scratch.subspan(3U * elements, elements);
    const auto latent = scratch.subspan(4U * elements, std::size_t{rows} * plan.rank);
    const auto width = plan.width;
    const auto stride = 2U * width;
    const detail::NormalizePair normalize{{
        {raw.data(), operands.data(), rows, width, plan.epsilon, width, stride, plan.round_operands_bf16},
        {message.data(), operands.data() + width, rows, width, plan.epsilon, width, stride, plan.round_operands_bf16},
    }};
    MixArgs mix{};
    if (!bind_banks(mix, bank, elements) || !fits(normalize.operands[0]) || !fits(normalize.operands[1]))
        throw std::invalid_argument("invalid low-rank controller bank or normalization");
    mix.queries = weights.query.data();
    mix.output[0] = message.data();
    mix.rows = rows;
    mix.width = width;
    mix.query_count = 1U;
    mix.epsilon = plan.epsilon;
    mix.normalized = true;
    const auto stream = operations.stream();
    operations.check(launch_mix(mix, stream));
    const dim3 grid(std::min(rows, detail::kGridLimit), 2U);
    detail::normalize_pair<<<grid, detail::kThreads, 0, stream>>>(
        normalize.operands[0], normalize.operands[1]);
    operations.check(detail::last_error());

    const Bf16RowProjection down[] = {{weights.down, plan.rank, latent.data(), plan.round_latent_bf16}};
    operations.dense(operands.data(), stride, rows, down);
    const Bf16RowProjection gates[] = {{weights.forget, width, forget.data()}, {weights.gain, width, gain.data()}};
    // A backend may run the gate pair with the controller as its epilogue (conditioned straight from the two FP32
    // sums: same bits, no forget / gain image). It returns false where it does not admit the shape.
    const ControllerArgs control{raw.data(), message.data(), forget.data(), gain.data(), conditioned.data(), elements};
    bool fused = false;
    if constexpr (requires { { operations.dense_controller(latent.data(), plan.rank, rows, gates, control) } -> std::same_as<bool>; })
        fused = operations.dense_controller(latent.data(), plan.rank, rows, gates, control);
    if (!fused) {
        operations.dense(latent.data(), plan.rank, rows, gates);
        operations.check(launch_controller(control, stream));
    }
    operations.rms(conditioned.data(), weights.norm, output.data(), rows);
}
#endif

}  // namespace strix::core::residual_mix
