#pragma once
// Block-per-row low-rank controller (HIP, decode rows): ONE launch, grid = rows, one 256-thread block per row, in place of
// the six launches of enqueue_low_rank_controller (readout, operand-norm pair, down projection, gate pair, pointwise
// controller, final norm). Every value is produced by the arithmetic of the launch it replaces, so the stored rows are
// the same bits:
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
// The message, operand, latent and gate-sum rows of the scratch the launches write are consumed by this block only and are
// not stored.
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

template <unsigned Regs>
__global__ __launch_bounds__(kThreads) void controller_block_kernel(Args a) {
    constexpr unsigned width = kThreads * Regs;
    constexpr unsigned inputs = 2U * width;  // down K
    constexpr unsigned groups = kThreads / 32U;
    using BankNorm = residual_mix::detail::BankNorm;
    using Chunk = weight_decode::Chunk<weight_decode::Format::bf16>;

    __shared__ float message_row[width];  // the message, then the conditioned row
    __shared__ float stage[2U * width];   // operand rows [raw | message], then the forget / gain sums
    __shared__ float latent[kRank];
    __shared__ float scratch_a[kThreads], scratch_b[kThreads];

    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    const std::size_t offset = std::size_t{blockIdx.x} * width;
    const float* const raw = a.raw + offset;

    // 1. The message row.
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

    // 2. Operand norms (BF16-rounded FP32 rows).
    ops::norm::rmsnorm_row_regs<BankNorm, kThreads, false, Regs>(scratch_a, raw, nullptr, nullptr, nullptr, true, width,
                                                                a.operand_epsilon, stage, 0, ops::norm::NoStore{});
    ops::norm::rmsnorm_row_regs<BankNorm, kThreads, false, Regs>(scratch_a, message_row, nullptr, nullptr, nullptr, true,
                                                                width, a.operand_epsilon, stage + width, 0,
                                                                ops::norm::NoStore{});
    __syncthreads();

    // 3. Down projection: kRank outputs, kRank / groups per wave.
    // Two outputs of a wave run side by side (independent chains; each keeps its own chunk and element order).
    static_assert(kRank % (2U * groups) == 0U);
    for (unsigned o = wave; o < kRank; o += 2U * groups) {
        const auto* const row0 = reinterpret_cast<const std::uint8_t*>(a.down) + std::size_t{o} * inputs * 2U;
        const auto* const row1 = row0 + std::size_t{groups} * inputs * 2U;
        float acc0 = 0.0F, acc1 = 0.0F;
#pragma unroll 3
        for (unsigned c = lane; c < inputs / 16U; c += 32U) {
            const Chunk weights0(row0, c * 16U);
            const Chunk weights1(row1, c * 16U);
            const float* const x = stage + c * 16U;
#pragma unroll
            for (unsigned e = 0; e < 16U; ++e) {
                acc0 = __fadd_rn(acc0, __fmul_rn(weights0.at(e), x[e]));
                acc1 = __fadd_rn(acc1, __fmul_rn(weights1.at(e), x[e]));
            }
        }
        const float sum0 = dense_linear::wave_sum32(acc0);
        const float sum1 = dense_linear::wave_sum32(acc1);
        if (lane == 0U) {
            latent[o] = ::strix::attn::group_indexer::bf16_round(sum0);
            latent[o + groups] = ::strix::attn::group_indexer::bf16_round(sum1);
        }
    }
    __syncthreads();

    // 4. Gate pair: eight outputs per wave step, four lanes (one 16-value chunk each) per output.
    {
        const unsigned member = lane & 3U, group = lane >> 2U;
        float lat[16];
#pragma unroll
        for (unsigned e = 0; e < 16U; ++e) lat[e] = latent[16U * member + e];
        for (unsigned first = 8U * wave; first < width; first += 8U * groups) {
            const unsigned n = first + group;
            const Chunk forget_chunk(reinterpret_cast<const std::uint8_t*>(a.forget) + std::size_t{n} * kRank * 2U,
                                     16U * member);
            const Chunk gain_chunk(reinterpret_cast<const std::uint8_t*>(a.gain) + std::size_t{n} * kRank * 2U,
                                   16U * member);
            float forget = 0.0F, gain = 0.0F;
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
            if (member == 0U) {
                stage[n] = forget;
                stage[width + n] = gain;
            }
        }
    }
    __syncthreads();

    // 5. Controller on the FP32 sums, in place over the message row (each thread owns its elements).
#pragma unroll
    for (unsigned k = 0; k < Regs; ++k) {
        const unsigned i = t + k * kThreads;
        message_row[i] = ::strix::core::controller_mix(raw[i], message_row[i], stage[i], stage[width + i]);
    }
    __syncthreads();

    // 6. Final norm of the conditioned row into the output row.
    ops::norm::rmsnorm_row_regs<ops::norm::Bf16ScaleSqrtDivTraits, kThreads, false, Regs>(
        scratch_a, message_row, nullptr, nullptr, a.scale, 0, width, a.norm_epsilon, a.output + offset, offset,
        ops::norm::NoStore{});
}

}  // namespace detail

// One launch over a.rows rows. hipErrorInvalidValue, nothing launched, where the shape is not admitted.
inline hipError_t enqueue(const Args& a, hipStream_t stream) {
    if (!admits(a)) return hipErrorInvalidValue;
    switch (row_regs(a.width)) {
    case 9U: detail::controller_block_kernel<9U><<<dim3(a.rows), kThreads, 0, stream>>>(a); break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

}  // namespace strix::core::controller_block
#endif
