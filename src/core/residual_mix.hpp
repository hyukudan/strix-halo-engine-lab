#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
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

#if defined(__CUDACC__) && !defined(__HIPCC__)
namespace strix::core::residual_mix {
struct MixArgs;
}
namespace strix::cuda::mix {
[[nodiscard]] inline cudaError_t enqueue(const ::strix::core::residual_mix::MixArgs& args, cudaStream_t stream);
namespace detail {
template <bool ParallelBanks>
cudaError_t enqueue_schedule(const ::strix::core::residual_mix::MixArgs& args, cudaStream_t stream);
}
}
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

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_RESIDUAL_MIX_HD __host__ __device__
#else
#define STRIX_RESIDUAL_MIX_HD
#endif

inline constexpr unsigned kMaxQuerySets = 4;

// Rows of one learned-query tensor read against the batch's bank: queries and output are [count][width].
struct BankQuerySet {
    const float* queries{};
    float* output{};
    unsigned count{};
};

// Readouts of one token (rows = 1) whose queries all read the same bank prefix, for example every site of a pass
// against the bank captured by the previous pass. Job j = (set, index) is exactly the single-query MixArgs
// {banks, queries + index * width, output[0] = output + index * width}: the batch removes per-query launches and
// keeps the per-query arithmetic (job_args is that MixArgs).
struct BankBatchArgs {
    const float* banks[kMaxBanks]{};
    BankQuerySet sets[kMaxQuerySets]{};
    unsigned width{};
    unsigned bank_count{};
    unsigned set_count{};
    float epsilon{};
    bool normalized{};
};

struct BankJob {
    unsigned set;    // set_count when the job is out of range
    unsigned index;
};

[[nodiscard]] STRIX_RESIDUAL_MIX_HD inline unsigned job_count(const BankBatchArgs& args) {
    unsigned total = 0;
    for (unsigned set = 0; set < args.set_count && set < kMaxQuerySets; ++set) total += args.sets[set].count;
    return total;
}

// Jobs are numbered set by set in declaration order.
[[nodiscard]] STRIX_RESIDUAL_MIX_HD inline BankJob locate(const BankBatchArgs& args, unsigned job) {
    for (unsigned set = 0; set < args.set_count && set < kMaxQuerySets; ++set) {
        if (job < args.sets[set].count) return {set, job};
        job -= args.sets[set].count;
    }
    return {args.set_count, 0U};
}

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

// The single-query readout that job j of the batch performs.
[[nodiscard]] inline MixArgs job_args(const BankBatchArgs& args, unsigned job) {
    const auto where = locate(args, job);
    if (where.set >= args.set_count) throw std::out_of_range("residual bank batch job");
    const auto& set = args.sets[where.set];
    const auto offset = std::size_t{where.index} * args.width;
    MixArgs mix{};
    for (unsigned bank = 0; bank < args.bank_count && bank < kMaxBanks; ++bank) mix.banks[bank] = args.banks[bank];
    mix.queries = set.queries + offset;
    mix.output[0] = set.output + offset;
    mix.rows = 1U;
    mix.width = args.width;
    mix.bank_count = args.bank_count;
    mix.query_count = 1U;
    mix.epsilon = args.epsilon;
    mix.normalized = args.normalized;
    return mix;
}

// Every job fits as a single-query readout, and every set's output image ([count][width]) is disjoint from the
// banks, from every query image and from the other outputs: all jobs run concurrently.
inline bool fits(const BankBatchArgs& args) {
    if (args.set_count == 0U || args.set_count > kMaxQuerySets || args.bank_count == 0U ||
        args.bank_count > kMaxBanks || args.width == 0U)
        return false;
    const auto floats = [&](unsigned count) { return std::size_t{count} * args.width; };
    const auto overlaps = [](const float* a, std::size_t a_floats, const float* b, std::size_t b_floats) {
        const auto a0 = reinterpret_cast<std::uintptr_t>(a);
        const auto b0 = reinterpret_cast<std::uintptr_t>(b);
        return a0 < b0 + b_floats * sizeof(float) && b0 < a0 + a_floats * sizeof(float);
    };
    for (unsigned set = 0; set < args.set_count; ++set) {
        const auto& s = args.sets[set];
        if (!s.queries || !s.output || s.count == 0U || !detail::valid_rows(s.count, args.width, args.width))
            return false;
        for (unsigned bank = 0; bank < args.bank_count; ++bank)
            if (!args.banks[bank] || overlaps(s.output, floats(s.count), args.banks[bank], args.width)) return false;
        for (unsigned other = 0; other < args.set_count; ++other) {
            const auto& o = args.sets[other];
            if (o.queries && overlaps(s.output, floats(s.count), o.queries, floats(o.count))) return false;
            if (other < set && o.output && overlaps(s.output, floats(s.count), o.output, floats(o.count)))
                return false;
        }
    }
    const auto total = job_count(args);
    for (unsigned job = 0; job < total; ++job)
        if (!fits(job_args(args, job))) return false;
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
#endif

namespace detail {

inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kGridLimit = 65535;
inline constexpr unsigned kMixRegs = 16;
inline constexpr unsigned kMixRegisterWidth = kThreads * kMixRegs;

namespace gi = attn::group_indexer;
namespace pool = attn::group_pool;

#if defined(__HIPCC__)
__device__ __forceinline__ float rms_inverse(float squares, unsigned width, float epsilon) {
    return 1.0F / pool::sqrt_rn(squares / static_cast<float>(width) + epsilon);
}

__device__ __forceinline__ float inverse_rms(const float* input, unsigned width, float epsilon, float* scratch) {
    const float sum = ops::norm::square_partial<kThreads>(input, width, threadIdx.x);
    const float squares = ops::norm::lds_halving_sum<kThreads>(scratch, sum);
    return rms_inverse(squares, width, epsilon);
}
#endif

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

#if defined(__HIPCC__)
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

// Readout addressing of the mix body. MixArgs: `rows` rows (strided over blockIdx.x) of query_count readouts.
// BankBatchArgs: blockIdx.y is job j, read as the one-row, single-query readout job_args(args, j); the body is the same,
// so a job's bits are those of launch_mix(job_args(args, j)).
template <class Args>
struct MixJobs;

template <>
struct MixJobs<MixArgs> {
    __device__ __forceinline__ static unsigned rows(const MixArgs& args) { return args.rows; }
    __device__ __forceinline__ static unsigned queries(const MixArgs& args) { return args.query_count; }
    __device__ __forceinline__ static const float* query(const MixArgs& args, unsigned q) {
        return args.queries + std::size_t{q} * args.width;
    }
    __device__ __forceinline__ static float* output(const MixArgs& args, unsigned q) { return args.output[q]; }
};

template <>
struct MixJobs<BankBatchArgs> {
    __device__ __forceinline__ static unsigned rows(const BankBatchArgs&) { return 1U; }
    __device__ __forceinline__ static unsigned queries(const BankBatchArgs&) { return 1U; }
    __device__ __forceinline__ static const float* query(const BankBatchArgs& args, unsigned) {
        const auto job = locate(args, blockIdx.y);
        return args.sets[job.set].queries + std::size_t{job.index} * args.width;
    }
    __device__ __forceinline__ static float* output(const BankBatchArgs& args, unsigned) {
        const auto job = locate(args, blockIdx.y);
        return args.sets[job.set].output + std::size_t{job.index} * args.width;
    }
};

// Register form of the parallel (decode) schedule for width <= kThreads * kMixRegs: a lane's strided values are loaded
// before their FMA chain, so one block pays one memory round trip per chain instead of one per strided element (the
// loop form waits for each load before its FMA). The output loop hoists the bank pointers, loads a batch of elements of
// every bank before the per-query chains and shares them between queries (outputs are disjoint from banks). Every
// chain (ascending i per lane), tree and expression is the loop form's: the same bits.
inline constexpr unsigned kMixOutBatch = 4;

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
template <bool Normalized, class Args>
__device__ __forceinline__ void score_bank_registers(const Args& args, std::size_t offset, unsigned bank,
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
    for (unsigned query = 0; query < MixJobs<Args>::queries(args); ++query) {
        const auto* learned = MixJobs<Args>::query(args, query);
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

template <bool Normalized, class Args>
__device__ __forceinline__ void emit_registers(const Args& args, std::size_t offset, unsigned output_lane,
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
        for (unsigned query = 0; query < MixJobs<Args>::queries(args); ++query) {
            auto* out = MixJobs<Args>::output(args, query) + offset;
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
template <bool Normalized, bool ParallelBanks, class Args>
__device__ __forceinline__ void mix_rows(const Args& args) {
    using Jobs = MixJobs<Args>;
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
    for (std::size_t row = blockIdx.x; row < Jobs::rows(args); row += gridDim.x) {
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
            for (unsigned query = 0; query < Jobs::queries(args); ++query) {
                const auto* learned = Jobs::query(args, query);
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
        if (output_lane < Jobs::queries(args)) {
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
        for (unsigned query = 0; query < Jobs::queries(args); ++query) {
            auto* out = Jobs::output(args, query) + offset;
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

template <bool Normalized, bool ParallelBanks = false>
static __global__ void mix(MixArgs args) {
    mix_rows<Normalized, ParallelBanks>(args);
}

// One block per job (blockIdx.y), each the schedule launch_mix picks for that job's one-row readout.
template <bool Normalized, bool ParallelBanks>
static __global__ void mix_batch(BankBatchArgs args) {
    mix_rows<Normalized, ParallelBanks>(args);
}
#endif

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
#if defined(__HIPCC__)
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
#else
    return ::strix::cuda::mix::detail::enqueue_schedule<ParallelBanks>(args, stream);
#endif
}

#if defined(__HIPCC__)
// Grid (1, jobs): job j runs the block of launch_mix(job_args(args, j)) (rows 1: the paired-bank schedule when the
// bank has more than one row, else the serial one).
template <bool ParallelBanks>
inline Error launch_bank_batch_schedule(const BankBatchArgs& args, unsigned jobs, Stream stream) {
    const dim3 grid(1U, jobs);
    if constexpr (ParallelBanks) {
        const dim3 threads(kPairedBankThreads, args.bank_count);
        if (args.normalized) mix_batch<true, true><<<grid, threads, 0, stream>>>(args);
        else mix_batch<false, true><<<grid, threads, 0, stream>>>(args);
    } else {
        if (args.normalized) mix_batch<true, false><<<grid, kThreads, 0, stream>>>(args);
        else mix_batch<false, false><<<grid, kThreads, 0, stream>>>(args);
    }
    return last_error();
}
#endif

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
#if defined(__HIPCC__)
    if (args.rows == 0U) return success;
    if (!fits(args)) return invalid_value;
    // A short chunk scores its banks in parallel while preserving the serial reduction tree.
    if (args.rows <= 2U && args.bank_count > 1U) return detail::launch_mix_schedule<true>(args, stream);
    return detail::launch_mix_schedule(args, stream);
#else
    return ::strix::cuda::mix::enqueue(args, stream);
#endif
}

#if defined(__HIPCC__)
// Every job of the batch in one launch. Job j's output bits are those of launch_mix(job_args(args, j)): the same
// kernel body, schedule and per-query arithmetic, addressed by job instead of by query.
inline Error launch_bank_batch(const BankBatchArgs& args, Stream stream) {
    if (!fits(args)) return invalid_value;
    const auto jobs = job_count(args);
    if (jobs == 0U || jobs > detail::kGridLimit) return invalid_value;
    if (args.bank_count > 1U) return detail::launch_bank_batch_schedule<true>(args, jobs, stream);
    return detail::launch_bank_batch_schedule<false>(args, jobs, stream);
}
#endif

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

namespace detail {

inline bool valid_controller(const LowRankController& plan, unsigned rows) {
    return rows != 0U && plan.width != 0U && plan.width <= std::numeric_limits<unsigned>::max() / 2U &&
           plan.rank != 0U && rows <= kMaxElements / plan.scratch_elements_per_row() && plan.epsilon > 0.0F &&
           std::isfinite(plan.epsilon);
}

inline bool disjoint_rows(const float* a, std::size_t a_floats, const float* b, std::size_t b_floats) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b);
    return a0 + a_floats * sizeof(float) <= b0 || b0 + b_floats * sizeof(float) <= a0;
}

}  // namespace detail

// The controller after its message readout. `message` holds the readout of weights.query against the controller's
// bank ([rows][width]), for example one row of a launch_bank_batch output; it must be disjoint from scratch and
// output. The launches and their bits are those enqueue_low_rank_controller issues after its readout (the scratch
// layout is the same; its message image is not used).
template <class Operations>
void enqueue_low_rank_controller_tail(LowRankController plan, ControllerWeights weights, unsigned rows, ConstRows raw,
                                      ConstRows message, MutableRows output, MutableRows scratch,
                                      Operations& operations) {
    const auto elements = std::size_t{rows} * plan.width;
    if (!detail::valid_controller(plan, rows) || raw.size() != elements || output.size() != elements ||
        message.size() != elements || scratch.size() < plan.scratch_elements(rows) ||
        !detail::disjoint_rows(message.data(), elements, scratch.data() + elements,
                               plan.scratch_elements(rows) - elements) ||
        !detail::disjoint_rows(message.data(), elements, output.data(), elements))
        throw std::invalid_argument("invalid low-rank controller buffers");

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
    if (!fits(normalize.operands[0]) || !fits(normalize.operands[1]))
        throw std::invalid_argument("invalid low-rank controller normalization");
    const auto stream = operations.stream();
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

// Banks are pre-normalized FP32 rows. The final norm (and its output precision)
// is supplied by Operations::rms. Scratch is disjoint from the caller's rows.
template <class Operations>
void enqueue_low_rank_controller(LowRankController plan, ControllerWeights weights, unsigned rows, ConstRows raw,
                                 std::span<const ConstRows> bank, MutableRows output, MutableRows scratch,
                                 Operations& operations) {
    const auto elements = std::size_t{rows} * plan.width;
    if (!detail::valid_controller(plan, rows) ||
        bank.empty() || bank.size() != plan.message_count || raw.size() != elements || output.size() != elements ||
        weights.query.size() != plan.width || scratch.size() < plan.scratch_elements(rows))
        throw std::invalid_argument("invalid low-rank controller buffers");

    const auto message = scratch.first(elements);
    MixArgs mix{};
    if (!bind_banks(mix, bank, elements))
        throw std::invalid_argument("invalid low-rank controller bank or normalization");
    mix.queries = weights.query.data();
    mix.output[0] = message.data();
    mix.rows = rows;
    mix.width = plan.width;
    mix.query_count = 1U;
    mix.epsilon = plan.epsilon;
    mix.normalized = true;
    operations.check(launch_mix(mix, operations.stream()));
    enqueue_low_rank_controller_tail(plan, weights, rows, raw, ConstRows(message), output, scratch, operations);
}
#endif

}  // namespace strix::core::residual_mix

#if defined(__CUDACC__) && !defined(__HIPCC__)
// The CUDA readout consumes MixArgs and fits() from this header.
#include "cuda/core/mix_cuda.cuh"
#endif
