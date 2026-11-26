#pragma once

#include "moe_core/microscaled_r1.hpp"
#include "moe_core/microscaled_r1_pipe_plan.hpp"

namespace strix::moe_core::microscaled_r1::pipe {

namespace body = cuda_body;
inline constexpr unsigned kWarpMask = 0xffffffffU;

template <unsigned Outputs>
struct Tile {
    unsigned codes[Outputs];
    unsigned scales[Outputs];
};

template <class Operand, unsigned Outputs>
__device__ __forceinline__ Tile<Outputs> load_tile(const typename Operand::Tile16Address (&rows)[Outputs],
                                                    unsigned base, unsigned lane) {
    using Run = CudaRun<Operand::kNvfp4 ? 1U : 2U, Operand::kBlockColumns>;
    constexpr unsigned values_per_byte = Operand::kNvfp4 ? 2U : 1U;
    Tile<Outputs> tile{};
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        tile.codes[i] = __ldg(reinterpret_cast<const unsigned*>(rows[i].codes + base / values_per_byte) + lane);
        if (lane < Run::kScales / ScaleWord::kBytes) {
            tile.scales[i] = __ldg(reinterpret_cast<const unsigned*>(rows[i].scale_code +
                                                                    base / Operand::kBlockColumns) + lane);
        }
    }
    return tile;
}

template <packed::MatrixFormat Format, unsigned Columns, unsigned Outputs, class Input>
__device__ __forceinline__ void dot_rows(const packed::PackedMatrix& matrix, unsigned col, unsigned stride,
                                        unsigned live, const Input* input, unsigned width, unsigned lane,
                                        float (&out)[Outputs]) {
    using Operand = packed::PackedOperand<Format>;
    using Run = CudaRun<Operand::kNvfp4 ? 1U : 2U, Operand::kBlockColumns>;
    constexpr unsigned values_per_byte = Operand::kNvfp4 ? 2U : 1U;
    // Preserve the incumbent decoder choice on short and unaligned rows. The vector scale path
    // requires an aligned base and pitch even when a particular output would happen to be aligned.
    if (width < Run::kColumns || matrix.columns % (Run::kWordBytes * values_per_byte) != 0U ||
        matrix.columns % (ScaleWord::kBytes * Operand::kBlockColumns) != 0U ||
        reinterpret_cast<std::uintptr_t>(matrix.codes) % Run::kWordBytes != 0U ||
        reinterpret_cast<std::uintptr_t>(matrix.block_scales) % ScaleWord::kBytes != 0U ||
        reinterpret_cast<std::uintptr_t>(input) % (2U * sizeof(Input)) != 0U) {
        body::dot_rows<Format, Columns>(matrix, col, stride, live, input, width, lane, out);
        return;
    }

    const Operand weights{matrix};
    typename Operand::Tile16Address rows[Outputs];
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        rows[i] = weights.tile16_address(slot_row(col, i, Columns, stride, live), 0U);
        out[i] = 0.0F;
    }

    const unsigned full = width / Run::kColumns * Run::kColumns;
    auto tile = load_tile<Operand>(rows, 0U, lane);
    for (unsigned base = 0; base < full; base += Run::kColumns) {
        const unsigned next_base = base + Run::kColumns;
        Tile<Outputs> next{};
        if (next_base < full) next = load_tile<Operand>(rows, next_base, lane);
#pragma unroll 1
        for (unsigned stripe = 0; stripe < Run::kPairsPerWord; ++stripe) {
            const auto x = body::aligned_input_pair(input + base + stripe * Run::kStripeColumns + 2U * lane);
            const unsigned code_lane = Run::code_lane(stripe, lane);
            const unsigned scale_index = Run::scale_lane(stripe, lane);
#pragma unroll
            for (unsigned i = 0; i < Outputs; ++i) {
                const unsigned word = __shfl_sync(kWarpMask, tile.codes[i], code_lane);
                const unsigned scales = __shfl_sync(kWarpMask, tile.scales[i], ScaleWord::lane(scale_index));
                const auto scale = static_cast<std::uint8_t>(scales >> ScaleWord::shift(scale_index));
                body::accumulate(body::decode(weights, word >> Run::code_shift(lane), scale), x, out[i]);
            }
        }
        tile = next;
    }

    for (unsigned k = full + 2U * lane; k < width; k += Run::kStripeColumns) {
        const auto x = body::input_pair(input + k);
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            const auto scale = Operand::decode_scale(rows[i].scale_code[k / Operand::kBlockColumns]);
            body::accumulate(weights.decode_pair(rows[i].codes + k / values_per_byte, scale), x, out[i]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) out[i] = detail::gemv::butterfly_sum(out[i]);
}

// MXFP8 can reach 12 CTAs/SM with a 40-register allocation. NVFP4 needs
// budget for the decode slow path, so target ten CTAs instead.
template <packed::MatrixFormat Format>
__global__ __launch_bounds__(kGateWarps * kWarp, Format == packed::MatrixFormat::mxfp8 ? 12 : 10)
void gate_up(Args args) {
    if (args.failure && *args.failure != 0U) return;
    const unsigned slot = blockIdx.y;
    const unsigned expert = args.ids[slot];
    const unsigned col = blockIdx.x * kGateWarps + threadIdx.x / kWarp;
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    float pair[2];
    dot_rows<Format, 1U>(args.matrices[expert].gate_up, col, args.intermediate, 1U,
                         args.input, args.hidden, lane, pair);
    if (lane == 0U) {
        const moe::SoftcappedSwiglu<detail::activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
        args.activation[std::size_t{slot} * args.intermediate + col] = detail::activation::activate_bf16(swiglu, pair[0], pair[1]);
    }
}

template <packed::MatrixFormat Format>
__global__ __launch_bounds__(kTop * kWarp) void down_combine(Args args) {
    if (args.failure && *args.failure != 0U) return;
    __shared__ float terms[kDownColumns][kTop];
    const unsigned slot = threadIdx.x / kWarp;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned col = blockIdx.x * kDownColumns;
    const unsigned expert = args.ids[slot];
    if (expert < args.experts) {
        const auto* input = args.activation + std::size_t{slot} * args.intermediate;
        float projected[kDownColumns];
        dot_rows<Format, kDownColumns>(args.matrices[expert].down, col, 0U, args.hidden - col,
                                       input, args.intermediate, lane, projected);
        if (lane == 0U) {
            unsigned rank = 0;
#pragma unroll
            for (unsigned i = 0; i < kTop; ++i) {
                const unsigned other = args.ids[i];
                rank += other < expert || (other == expert && i < slot);
            }
            const float weight = args.weights[slot];
#pragma unroll
            for (unsigned i = 0; i < kDownColumns; ++i)
                terms[i][rank] = detail::combine::weighted_bf16<false>(weight, projected[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x < kDownColumns && col + threadIdx.x < args.hidden) {
#pragma unroll
        for (unsigned i = 0; i < kTop; ++i) {
            if (args.ids[i] >= args.experts) return;
        }
        float result = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < kTop; ++i) result = __fadd_rn(result, terms[threadIdx.x][i]);
        args.output[col + threadIdx.x] = detail::combine::rounded_bf16<false>(result);
    }
}

template <packed::MatrixFormat Format>
inline cudaError_t launch_gate_up(const Args& args, cudaStream_t stream) {
    if (!fits<Format, kTop>(args)) return cudaErrorInvalidValue;
    const auto shape = gate_up_launch(args.intermediate);
    gate_up<Format><<<shape.template grid<dim3>(), shape.template block<dim3>(), shape.shared_bytes, stream>>>(args);
    return cudaGetLastError();
}

template <packed::MatrixFormat Format>
inline cudaError_t launch_down_combine(const Args& args, cudaStream_t stream) {
    if (!fits<Format, kTop>(args)) return cudaErrorInvalidValue;
    const auto shape = down_combine_launch(args.hidden);
    down_combine<Format><<<shape.template grid<dim3>(), shape.template block<dim3>(), shape.shared_bytes, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace strix::moe_core::microscaled_r1::pipe
