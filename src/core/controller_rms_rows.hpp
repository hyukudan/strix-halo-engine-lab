#pragma once
// Row-owner form of the low-rank controller's gate pair (HIP, prefill rows): one CTA owns 16 whole rows, runs the
// [forget, gain] projections of its rows (K = rank <= 64, every K16 unit in registers), stores
// controller_mix(current, message, forget sum, gain sum) into the output rows and then normalizes those rows in place
// with the shared RMS row function. The conditioned rows are never stored apart from the output and never re-read
// from DRAM by a second launch: 3 activation streams per call instead of 5 (gates + controller, then the norm).
//
// Same bits as bf16_controller_kernel followed by ops::norm::enqueue_rms_bf16_rows (parallel form):
// - every gate output runs one WMMA chain from +0 over the K16 units in ascending order, its A fragment holding the
//   tile stager's pack2<bf16> of the FP32 latent (k at k % 16, low half first) and its B fragment the BF16 weight row
//   as stored, the incumbent tile's operand bits and chain (f16_tile_gemm.hpp, "all forms store the same bits");
// - the epilogue is core::controller_mix, the incumbent tile's epilogue function;
// - the norm is rmsnorm_row<Bf16ScaleSqrtDivTraits, 256, false> on the row, the function rmsnorm_strided_kernel calls
//   with one 256-lane block per row (in place is the strided kernel's admitted input == output form).
#if defined(__HIPCC__)

#include "core/controller_mix.hpp"
#include "core/f16_tile_gemm.hpp"
#include "ops/norm/rms_bf16_rows.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix::core::controller_rms_rows {

inline constexpr unsigned kRows = 16U, kThreads = 256U, kWaves = kThreads / 32U, kMaxUnits = 4U;

struct Args {
    const float* latent;          // [rows][rank] FP32 (the down projection's output)
    const std::uint16_t* forget;  // [width][rank] BF16
    const std::uint16_t* gain;    // [width][rank] BF16
    const float* current;         // [rows][width]
    const float* message;         // [rows][width]
    const std::uint16_t* scale;   // [width] BF16 norm scale (may be null: the norm kernel's unscaled form)
    float* output;                // [rows][width]: the conditioned rows, then their norm in place
    unsigned rows;
    unsigned width;
    unsigned rank;
    float epsilon;
};

namespace detail {

template <unsigned Units>
__global__ __launch_bounds__(kThreads) void controller_rms_kernel(Args a) {
    namespace tile = strix::attn_core::tile;
    using Frag = f16_tile_gemm::Frag;
    using Float4 = f16_tile_gemm::Float4;
    using Float8 = f16_tile_gemm::Float8;
    constexpr unsigned rank = 16U * Units;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, c = lane & 15U;
    const unsigned r0 = blockIdx.x * kRows;

    // A fragments of row r0 + c (both lane halves hold the same row, as the tile's LDS fragments do).
    Frag fa[Units];
    const unsigned ar = r0 + c;
#pragma unroll
    for (unsigned u = 0; u < Units; ++u) {
        if (ar < a.rows) {
            const float* src = a.latent + std::size_t{ar} * rank + 16U * u;
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const Float4 f = *reinterpret_cast<const Float4*>(src + 4U * i);
                fa[u][2U * i] = tile::pack2<tile::Operand::bf16>(f[0], f[1]);
                fa[u][2U * i + 1U] = tile::pack2<tile::Operand::bf16>(f[2], f[3]);
            }
        } else {
            fa[u] = Frag{};
        }
    }

    // Wave w owns the 16-column groups w, w + 8, ...: lane column n = 16 g + c, rows r0 + 2 j + lane / 16.
    for (unsigned n0 = 16U * wave; n0 < a.width; n0 += 16U * kWaves) {
        const unsigned n = n0 + c;
        float current[8], message[8];
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = r0 + 2U * j + (lane >> 4U);
            const std::size_t i = std::size_t{r < a.rows ? r : r0} * a.width + n;
            current[j] = a.current[i];
            message[j] = a.message[i];
        }
        Frag ff[Units], fg[Units];
#pragma unroll
        for (unsigned u = 0; u < Units; ++u) {
            ff[u] = *reinterpret_cast<const Frag*>(a.forget + std::size_t{n} * rank + 16U * u);
            fg[u] = *reinterpret_cast<const Frag*>(a.gain + std::size_t{n} * rank + 16U * u);
        }
        Float8 forget = {}, gain = {};
#pragma unroll
        for (unsigned u = 0; u < Units; ++u) {
            forget = tile::mma<tile::Operand::bf16>(__builtin_bit_cast(tile::Frag, fa[u]),
                                                    __builtin_bit_cast(tile::Frag, ff[u]), forget);
            gain = tile::mma<tile::Operand::bf16>(__builtin_bit_cast(tile::Frag, fa[u]),
                                                  __builtin_bit_cast(tile::Frag, fg[u]), gain);
        }
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = r0 + 2U * j + (lane >> 4U);
            if (r < a.rows)
                a.output[std::size_t{r} * a.width + n] =
                    ::strix::core::controller_mix(current[j], message[j], forget[j], gain[j]);
        }
    }

    // The block's conditioned rows are complete (workgroup barrier orders the global stores before the loads).
    __syncthreads();
    using Traits = ops::norm::Bf16ScaleSqrtDivTraits;
    __shared__ float s[kThreads];
    for (unsigned k = 0; k < kRows; ++k) {
        const unsigned r = r0 + k;
        if (r >= a.rows) break;  // uniform across the block
        float* row = a.output + std::size_t{r} * a.width;
        ops::norm::rmsnorm_row<Traits, kThreads, false>(s, row, nullptr, nullptr, a.scale, 0, a.width, a.epsilon, row,
                                                       std::size_t{r} * a.width, ops::norm::NoStore{});
    }
}

}  // namespace detail

// The shapes the row-owner form takes: prefill rows (decode rows keep their gemv gates), rank a multiple of 16 up to
// 64, width a multiple of 16 above 256, 16-byte aligned latent rows and weights, output either the current rows or
// disjoint from current, message and latent, and always disjoint from the operands a CTA reads after another CTA may
// have stored output rows (both gate weight panels and the norm scale). Rows of at most 256 values keep the two launches: there the separate
// norm runs its one-wave form (ops/norm/rms_bf16_rows.hpp), the same bits for finite rows but not always the same
// NaN payload as the block form this kernel shares with rmsnorm_strided_kernel.
inline bool admits(const Args& a, unsigned decode_rows) noexcept {
    const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16U == 0U; };
    const auto disjoint = [](const void* p, std::size_t p_bytes, const void* q, std::size_t q_bytes) {
        const auto p0 = reinterpret_cast<std::uintptr_t>(p), q0 = reinterpret_cast<std::uintptr_t>(q);
        return p0 + p_bytes <= q0 || q0 + q_bytes <= p0;
    };
    if (!a.latent || !a.forget || !a.gain || !a.current || !a.message || !a.output || a.rows <= decode_rows ||
        a.rank == 0U || a.rank % 16U != 0U || a.rank > 16U * kMaxUnits || a.width <= kThreads || a.width % 16U != 0U ||
        a.width > std::numeric_limits<unsigned>::max() - 256U ||
        std::size_t{a.rows} > std::numeric_limits<std::size_t>::max() / sizeof(float) / a.width ||
        !aligned(a.latent) || !aligned(a.forget) || !aligned(a.gain) || !(a.epsilon > 0.0F))
        return false;
    const std::size_t rows_bytes = std::size_t{a.rows} * a.width * sizeof(float);
    const std::size_t panel_bytes = std::size_t{a.width} * a.rank * sizeof(std::uint16_t);
    return (a.output == a.current || disjoint(a.output, rows_bytes, a.current, rows_bytes)) &&
           disjoint(a.output, rows_bytes, a.message, rows_bytes) &&
           disjoint(a.output, rows_bytes, a.latent, std::size_t{a.rows} * a.rank * sizeof(float)) &&
           disjoint(a.output, rows_bytes, a.forget, panel_bytes) && disjoint(a.output, rows_bytes, a.gain, panel_bytes) &&
           (!a.scale || disjoint(a.output, rows_bytes, a.scale, std::size_t{a.width} * sizeof(std::uint16_t)));
}

// One CTA per 16 rows, without the rounding addition that wraps for row counts near the unsigned maximum.
inline constexpr unsigned grid_blocks(unsigned rows) noexcept { return rows / kRows + (rows % kRows != 0U ? 1U : 0U); }

}  // namespace strix::core::controller_rms_rows
#endif
