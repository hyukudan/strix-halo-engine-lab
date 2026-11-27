#pragma once
// Fused low-rank controller (HIP, decode rows) in place of the six launches of enqueue_low_rank_controller (readout,
// operand-norm pair, down projection, gate pair, pointwise controller, final norm): a down launch over (kDownBlocks, rows)
// and a gate launch over (width / kGateOutputs, rows) (core/controller_block_plan.hpp). Every value is produced by the
// arithmetic of the launch it replaces, so the stored rows are the same bits:
// - readout (residual_mix::mix, normalized banks): per bank a lane chain dot = fmaf(query[i], x[i], dot) over
//   i = t + 256 k, the 256-lane LDS halving tree, score = sum * (1 / sqrt(width)); max, expf, running denominator, divide
//   in bank order; message = fmaf(p1, x1, fmaf(p0, x0, 0)), FP32 unrounded. Where the caller has the message already
//   (a residual bank batch row) the readout is skipped and the message rows are read;
// - operand norms (residual_mix normalize_pair): ops::norm::rmsnorm_row_regs with BankNorm on the raw row and on the
//   message, BF16-rounded, kept as FP32 operand rows in LDS;
// - down projection (dense_linear tight rows): output o on one wave, lane L takes the 16-value chunks L, L + 32, ... of
//   the 2 * width operand, each chunk's values in ascending order as acc = acc + w * x (separate multiply and add), the
//   xor butterfly 16..1, the sum rounded to BF16 (the latent);
// - gate pair (dense_linear grouped tight rows, K = 64): the same chunk body, so only lanes 0..3 of the 32-lane form
//   hold a chain and lanes 4..31 hold +0: the butterfly yields ((q0 + 0) + (q2 + 0)) + ((q1 + 0) + (q3 + 0)) (adding the
//   zero partners turns a -0 chain into +0). Here eight outputs share a wave, four lanes each, and the last two
//   butterfly levels (xor 2, xor 1) run inside each four-lane group: the same sums;
// - controller (core::controller_mix) on those FP32 sums, the raw row and the message;
// - final norm: ops::norm::rmsnorm_row_regs with Bf16ScaleSqrtDivTraits (what rmsnorm_strided_kernel runs per row).
// Every down block recomputes its row's message and operand rows (the same values) and stores eight latents (wave w of
// block b: latent 8b + w); block 0 stores the readout message (banks form) and zeroes the row's ticket. Each gate block
// stores the controller values of its kGateOutputs positions; the last gate block of the row to take the ticket
// (device-scope fences around it) reads the whole controller row and runs the final norm. Which block that is changes
// no value.
// Compile with -ffp-contract=off like the launches it replaces; the multiply-add steps below are explicit anyway.
#if defined(__HIPCC__)

#include "core/controller_block_plan.hpp"
#include "core/dense_linear.hpp"
#include "core/residual_mix.hpp"
#include "core/weight_decode.hpp"
#include "ops/norm/norm_device.hpp"
#include "ops/norm/rms_bf16_rows.hpp"
#include "strix/attn/group_indexer.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::core::controller_block {

namespace detail {

// 1. The message row into message_row (LDS), from the banks (readout) or the given message rows. Ends synchronized.
template <unsigned Regs>
__device__ __forceinline__ void message_row_of(const Args& a, std::size_t offset, float* message_row, float* scratch_a,
                                               float* scratch_b) {
    constexpr unsigned width = kThreads * Regs;
    const unsigned t = threadIdx.x;
    if (a.bank0 != nullptr) {
        const float* const bank0 = a.bank0 + offset;
        const float* const bank1 = a.bank1 + offset;
        float x0[Regs], x1[Regs], dot0 = 0.0F, dot1 = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < Regs; ++k) {
            const unsigned i = t + k * kThreads;
            x0[k] = bank0[i];
            x1[k] = bank1[i];
        }
#pragma unroll
        for (unsigned k = 0; k < Regs; ++k) {
            const float learned = a.query[t + k * kThreads];
            dot0 = fmaf(learned, x0[k], dot0);
            dot1 = fmaf(learned, x1[k], dot1);
        }
        // Two lds_halving_sum<kThreads> trees side by side: s[l] = s[l] + s[l + h], h = 128 .. 1.
        scratch_a[t] = dot0;
        scratch_b[t] = dot1;
        __syncthreads();
        for (unsigned h = kThreads / 2U; h > 0U; h >>= 1U) {
            if (t < h) {
                scratch_a[t] = scratch_a[t] + scratch_a[t + h];
                scratch_b[t] = scratch_b[t] + scratch_b[t + h];
            }
            __syncthreads();
        }
        const float sum0 = scratch_a[0], sum1 = scratch_b[0];
        __syncthreads();
        const float score_scale = 1.0F / ::strix::attn::group_pool::sqrt_rn(static_cast<float>(width));
        float p0 = sum0 * score_scale, p1 = sum1 * score_scale;
        const float maximum = fmaxf(p0, p1);
        p0 = expf(p0 - maximum);
        float denominator = 0.0F;
        denominator += p0;
        p1 = expf(p1 - maximum);
        denominator += p1;
        p0 /= denominator;
        p1 /= denominator;
#pragma unroll
        for (unsigned k = 0; k < Regs; ++k) {
            float value = 0.0F;
            value = fmaf(p0, x0[k], value);
            value = fmaf(p1, x1[k], value);
            message_row[t + k * kThreads] = value;
        }
    } else {
        const float* const message = a.message + offset;
#pragma unroll
        for (unsigned k = 0; k < Regs; ++k) message_row[t + k * kThreads] = message[t + k * kThreads];
    }
    __syncthreads();
}

// 2. Operand norms (BF16-rounded FP32 rows) of the raw row and the message row into stage[0, 2 width). Ends synchronized.
template <unsigned Regs>
__device__ __forceinline__ void operand_rows_of(const Args& a, const float* raw, const float* message_row, float* stage,
                                                float* scratch_a) {
    constexpr unsigned width = kThreads * Regs;
    using BankNorm = residual_mix::detail::BankNorm;
    ops::norm::rmsnorm_row_regs<BankNorm, kThreads, false, Regs>(scratch_a, raw, nullptr, nullptr, nullptr, true, width,
                                                                a.operand_epsilon, stage, 0, ops::norm::NoStore{});
    ops::norm::rmsnorm_row_regs<BankNorm, kThreads, false, Regs>(scratch_a, message_row, nullptr, nullptr, nullptr, true,
                                                                width, a.operand_epsilon, stage + width, 0,
                                                                ops::norm::NoStore{});
    __syncthreads();

}

// One latent's lane chain of the down projection: chunks c = lane, lane + 32, ... of the 2 * width operand, each chunk's
// values in ascending order as acc = acc + w * x.
template <unsigned Inputs>
__device__ __forceinline__ float down_lane_chain(const std::uint8_t* row, const float* stage, unsigned lane) {
    using Chunk = weight_decode::Chunk<weight_decode::Format::bf16>;
    static_assert(Inputs % (16U * 32U) == 0U);
    float acc = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < Inputs / (16U * 32U); ++j) {
        const unsigned c = lane + 32U * j;
        const Chunk weights(row, c * 16U);
        const float* const x = stage + c * 16U;
#pragma unroll
        for (unsigned e = 0; e < 16U; ++e) acc = __fadd_rn(acc, __fmul_rn(weights.at(e), x[e]));
    }
    return acc;
}

// One gate output n of the pair on a four-lane group (member = lane & 3 holds chunk 16 * member of K = 64): every lane of
// the group ends with the two FP32 sums.
__device__ __forceinline__ void gate_pair_sums(const Args& a, unsigned n, unsigned member, const float (&lat)[16],
                                               float& forget, float& gain) {
    using Chunk = weight_decode::Chunk<weight_decode::Format::bf16>;
    const Chunk forget_chunk(reinterpret_cast<const std::uint8_t*>(a.forget) + std::size_t{n} * kRank * 2U, 16U * member);
    const Chunk gain_chunk(reinterpret_cast<const std::uint8_t*>(a.gain) + std::size_t{n} * kRank * 2U, 16U * member);
    forget = 0.0F;
    gain = 0.0F;
#pragma unroll
    for (unsigned e = 0; e < 16U; ++e) {
        forget = __fadd_rn(forget, __fmul_rn(forget_chunk.at(e), lat[e]));
        gain = __fadd_rn(gain, __fmul_rn(gain_chunk.at(e), lat[e]));
    }
    // The 32-lane butterfly's levels 16, 8, 4 add the +0 of the idle lanes (one add of +0 equals the three).
    forget = __fadd_rn(forget, 0.0F);
    gain = __fadd_rn(gain, 0.0F);
    forget = forget + __shfl_xor(forget, 2, 32);
    gain = gain + __shfl_xor(gain, 2, 32);
    forget = forget + __shfl_xor(forget, 1, 32);
    gain = gain + __shfl_xor(gain, 1, 32);
}

// The scratch rows (core/controller_block_plan.hpp scratch_elements_for).
struct SplitRows {
    float* controlled;  // [rows][width] controller values (input of the final norm)
    float* message;     // [rows][width] readout message (banks form)
    float* latent;      // [rows][kRank]
    unsigned* ticket;   // [rows]
};
__device__ __forceinline__ SplitRows split_rows(const Args& a) {
    const std::size_t rows = std::size_t{a.rows} * a.width;
    float* const latent = a.scratch + 2U * rows;
    return {a.scratch, a.scratch + rows, latent, reinterpret_cast<unsigned*>(latent + std::size_t{a.rows} * kRank)};
}

// Launch 1 over (kDownBlocks, rows): the row's message and operand rows (recomputed by every block), then one
// latent per wave.
template <unsigned Regs>
__global__ __launch_bounds__(kThreads) void controller_down_kernel(Args a) {
    constexpr unsigned width = kThreads * Regs;
    constexpr unsigned inputs = 2U * width;  // down K
    constexpr unsigned groups = kThreads / 32U;
    static_assert(kDownBlocks * groups == kRank);

    __shared__ float message_row[width];
    __shared__ float stage[2U * width];  // operand rows [raw | message]
    __shared__ float scratch_a[kThreads], scratch_b[kThreads];

    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U, row = blockIdx.y;
    const std::size_t offset = std::size_t{row} * width;
    const SplitRows out = split_rows(a);

    message_row_of<Regs>(a, offset, message_row, scratch_a, scratch_b);
    if (blockIdx.x == 0U) {
        if (a.bank0 != nullptr) {
#pragma unroll
            for (unsigned k = 0; k < Regs; ++k) out.message[offset + t + k * kThreads] = message_row[t + k * kThreads];
        }
        if (t == 0U) out.ticket[row] = 0U;
    }
    operand_rows_of<Regs>(a, a.raw + offset, message_row, stage, scratch_a);

    const unsigned o = blockIdx.x * groups + wave;
    const auto* const weights = reinterpret_cast<const std::uint8_t*>(a.down) + std::size_t{o} * inputs * 2U;
    const float sum = dense_linear::wave_sum32(down_lane_chain<inputs>(weights, stage, lane));
    if (lane == 0U) out.latent[std::size_t{row} * kRank + o] = ::strix::attn::group_indexer::bf16_round(sum);
}

// Launch 2 over (width / kGateOutputs, rows): kGateOutputs gate pairs and their controller values per block;
// the row's last block to take the ticket runs the final norm over the whole controller row.
template <unsigned Regs>
__global__ __launch_bounds__(kThreads) void controller_gate_kernel(Args a) {
    constexpr unsigned width = kThreads * Regs;
    constexpr unsigned groups = kThreads / 32U;
    static_assert(kGateOutputs % (8U * groups) == 0U);

    __shared__ float scratch_a[kThreads];
    __shared__ bool last;

    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U, row = blockIdx.y;
    const std::size_t offset = std::size_t{row} * width;
    const SplitRows out = split_rows(a);
    const float* const raw = a.raw + offset;
    const float* const message = (a.bank0 != nullptr ? out.message : a.message) + offset;

    const unsigned member = lane & 3U, group = lane >> 2U;
    float lat[16];
#pragma unroll
    for (unsigned e = 0; e < 16U; ++e) lat[e] = out.latent[std::size_t{row} * kRank + 16U * member + e];
#pragma unroll
    for (unsigned step = 0; step < kGateOutputs / (8U * groups); ++step) {
        const unsigned n = blockIdx.x * kGateOutputs + step * 8U * groups + 8U * wave + group;
        float forget = 0.0F, gain = 0.0F;
        gate_pair_sums(a, n, member, lat, forget, gain);
        if (member == 0U) out.controlled[offset + n] = ::strix::core::controller_mix(raw[n], message[n], forget, gain);
    }
    // Publish this block's controller values (device-scope fences around the ticket): the last block of the row sees
    // every block's.
    __threadfence();
    __syncthreads();
    if (t == 0U) last = atomicAdd(out.ticket + row, 1U) == gridDim.x - 1U;
    __syncthreads();
    if (!last) return;
    __threadfence();
    ops::norm::rmsnorm_row_regs<ops::norm::Bf16ScaleSqrtDivTraits, kThreads, false, Regs>(
        scratch_a, out.controlled + offset, nullptr, nullptr, a.scale, 0, width, a.norm_epsilon, a.output + offset,
        offset, ops::norm::NoStore{});
}

}  // namespace detail

// The fused controller over a.rows rows (two launches). hipErrorInvalidValue, nothing launched, where the arguments are
// not admitted.
inline hipError_t enqueue(const Args& a, hipStream_t stream) {
    if (!admits(a)) return hipErrorInvalidValue;
    switch (row_regs(a.width)) {
    case 9U: {
        detail::controller_down_kernel<9U><<<dim3(kDownBlocks, a.rows), kThreads, 0, stream>>>(a);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) return status;
        detail::controller_gate_kernel<9U><<<dim3(a.width / kGateOutputs, a.rows), kThreads, 0, stream>>>(a);
        break;
    }
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

}  // namespace strix::core::controller_block
#endif
