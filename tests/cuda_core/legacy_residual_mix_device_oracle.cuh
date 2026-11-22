#pragma once
// Frozen TEST-ONLY CUDA oracle from src/core/residual_mix.hpp before the K8c migration:
// Git blob 6a2f9b5ea664113ebbba6da454d823c6f6d2aca9 (main 38430cb98).
// The legacy readout statements, schedules and reduction order are retained. The
// old square/reduction/sqrt helpers are frozen below too, so this reference never
// dispatches into the production readout or inherits a new reduction schedule.
// Only the MixArgs ABI and validation are shared. Never include this file in a
// production target; it exists to detect regressions after the old kernel is removed.
#include "core/residual_mix.hpp"

namespace strix::tests::mix::legacy_cuda {
using core::residual_mix::MixArgs;
using core::residual_mix::kMaxBanks;
using core::residual_mix::kMaxQueries;
using core::residual_mix::fits;
using Error = cudaError_t;
using Stream = cudaStream_t;
inline constexpr Error success = cudaSuccess;
inline constexpr Error invalid_value = cudaErrorInvalidValue;
inline constexpr bool kMixRegisterForm = false;

namespace legacy_norm {
// Frozen from src/ops/norm/norm_device.hpp and row_reduction.hpp on the same tree.
template <unsigned L>
__device__ __forceinline__ float square_partial(const float* input, unsigned width, unsigned lane) {
    float sum = 0.0F;
    for (unsigned i = lane; i < width; i += L) sum = __builtin_fmaf(input[i], input[i], sum);
    return sum;
}
template <unsigned L>
__device__ __forceinline__ float lds_halving_sum(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}
}
// The device branch of the old group_pool::sqrt_rn is sqrtf, not a host std::sqrt.
__device__ __forceinline__ float legacy_sqrt(float x) { return sqrtf(x); }

namespace detail {
inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kGridLimit = 65535;
inline Error last_error() { return cudaGetLastError(); }

__device__ __forceinline__ float rms_inverse(float squares, unsigned width, float epsilon) {
    return 1.0F / legacy_sqrt(squares / static_cast<float>(width) + epsilon);
}

__device__ __forceinline__ float inverse_rms(const float* input, unsigned width, float epsilon, float* scratch) {
    const float sum = legacy_norm::square_partial<kThreads>(input, width, threadIdx.x);
    const float squares = legacy_norm::lds_halving_sum<kThreads>(scratch, sum);
    return rms_inverse(squares, width, epsilon);
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
        const float squares = legacy_norm::lds_halving_sum<kPairedBankThreads>(
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
        const float score = legacy_norm::lds_halving_sum<kPairedBankThreads>(bank_scratch, dot) * score_scale;
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
    const float score_scale = 1.0F / legacy_sqrt(static_cast<float>(args.width));
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
                    const float low = legacy_norm::square_partial<kThreads>(source, args.width, lane);
                    const float high = legacy_norm::square_partial<kThreads>(source, args.width, lane + bank_threads);
                    const float squares = legacy_norm::lds_halving_sum<bank_threads>(bank_scratch, low + high);
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
                const float score = legacy_norm::lds_halving_sum<bank_threads>(bank_scratch, dot) * score_scale;
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

} // namespace detail

inline Error launch_mix(const MixArgs& args, Stream stream) {
    if (args.rows == 0U) return success;
    if (!fits(args)) return invalid_value;
    // A short chunk scores its banks in parallel while preserving the serial reduction tree.
    if (args.rows <= 2U && args.bank_count > 1U) return detail::launch_mix_schedule<true>(args, stream);
    return detail::launch_mix_schedule(args, stream);
}

} // namespace strix::tests::mix::legacy_cuda
