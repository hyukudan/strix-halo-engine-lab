#pragma once

#include "moe_core/device_grouped_plan.hpp"
#include "moe_core/expert_activation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__)
#include "core/nvfp4_bf16_table.hpp"
#include "strix/hip_wave32_dpp.hpp"
#elif defined(__CUDACC__)
#include "cuda/core/reduce.cuh"
#endif

namespace strix::moe_core::microscaled_r1 {

namespace packed = strix::weight::microscaled;

// One token, already routed in selection order. Matrices refer to an admitted
// layer's device projections. The optional device failure word is the route
// status. Keep all inputs, output and TopK * intermediate BF16 activations
// alive through stream completion. The intermediate buffer must be disjoint.
struct Args {
    const device_grouped::ExpertWeights* matrices;
    const unsigned* ids;
    const float* weights;
    const unsigned* failure;
    const float* input;
    std::uint16_t* activation;
    float* output;
    unsigned experts, hidden, intermediate;
    float gate_cap, up_cap;
};

template <packed::MatrixFormat Format, unsigned TopK>
inline bool fits(const Args& a) {
    static_assert(TopK > 0U && TopK <= 32U);
    constexpr auto k = packed::PackedOperand<Format>::kBlockColumns;
    constexpr auto max_columns = std::numeric_limits<unsigned>::max() / 2U;
    return a.matrices && a.ids && a.weights && a.input && a.activation && a.output &&
           a.experts >= TopK && a.hidden >= k && a.hidden <= max_columns && a.hidden % k == 0U &&
           a.intermediate >= k && a.intermediate <= max_columns && a.intermediate % k == 0U &&
           std::uint64_t{TopK} * a.intermediate <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) &&
           std::isfinite(a.gate_cap) && a.gate_cap > 0.0F && std::isfinite(a.up_cap) && a.up_cap > 0.0F;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

namespace scalar = weight::scalar_float;
namespace activation = expert_activation::detail;

inline constexpr unsigned kWarp = 32;
inline constexpr unsigned kWarps = 8;
inline constexpr unsigned kThreads = kWarp * kWarps;
#if defined(__HIPCC__)
// HIP launch bounds express the occupancy target in waves per execution unit.
inline constexpr unsigned kMinWavesPerEu = 8;
#endif
inline constexpr unsigned kGateColumns = 1;
inline constexpr unsigned kDownColumns = 1;

__device__ __forceinline__ float widen(std::uint16_t value) {
    return scalar::u2f(std::uint32_t{value} << 16U);
}

__device__ __forceinline__ float input_bf16(std::uint16_t value) { return widen(value); }
__device__ __forceinline__ float input_bf16(float value) { return activation::round_linear(value); }

__device__ __forceinline__ float warp_sum(float value) {
#if defined(__HIPCC__)
    value += hip_wave32_dpp::xor32<16>(value);
    value += hip_wave32_dpp::xor32<8>(value);
    value += hip_wave32_dpp::xor32<4>(value);
    value += hip_wave32_dpp::xor32<2>(value);
    return value + hip_wave32_dpp::xor32<1>(value);
#else
    return cuda::core::butterfly(value);
#endif
}

#if defined(__HIPCC__)
using CodeChunk = std::uint32_t __attribute__((ext_vector_type(4), may_alias));
using InputQuad = float __attribute__((ext_vector_type(4), may_alias));
using Bf16Quad = std::uint32_t __attribute__((ext_vector_type(2), may_alias));

__device__ __forceinline__ InputQuad load_inputs(const float* input) {
    const auto x = *reinterpret_cast<const InputQuad*>(input);
    return {input_bf16(x[0]), input_bf16(x[1]), input_bf16(x[2]), input_bf16(x[3])};
}

__device__ __forceinline__ InputQuad load_inputs(const std::uint16_t* input) {
    const auto x = *reinterpret_cast<const Bf16Quad*>(input);
    return {widen(static_cast<std::uint16_t>(x[0])), scalar::u2f(x[0] & 0xffff0000U),
            widen(static_cast<std::uint16_t>(x[1])), scalar::u2f(x[1] & 0xffff0000U)};
}

template <class Operand, unsigned Outputs, class Input>
__device__ __forceinline__ void stream_rows(const Operand& weights,
                                           const typename Operand::Tile16Address (&rows)[Outputs],
                                           const Input* input, unsigned width, unsigned lane, float (&out)[Outputs]) {
    constexpr unsigned codes_per_pair = Operand::kNvfp4 ? 1U : 2U;
    constexpr unsigned values_per_byte = 2U / codes_per_pair;
    constexpr unsigned values_per_word = 8U / codes_per_pair;
    constexpr unsigned chunk = 4U * values_per_word;
    constexpr unsigned scale_count = Operand::kNvfp4 ? 2U : 1U;
    constexpr unsigned words_per_scale = 4U / scale_count;
    for (unsigned k = chunk * lane; k < width; k += chunk * kWarp) {
        CodeChunk codes[Outputs];
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            codes[i] = *reinterpret_cast<const CodeChunk*>(rows[i].codes + k / values_per_byte);
        }
#pragma unroll 1
        for (unsigned block = 0; block < scale_count; ++block) {
            float scales[Outputs];
#pragma unroll
            for (unsigned i = 0; i < Outputs; ++i) {
                scales[i] = Operand::decode_scale(rows[i].scale_code[k / Operand::kBlockColumns + block]);
            }
            const auto stream_words = [&](const auto& tables) {
                // Keep the code and rounded activations live one word at a time.
#pragma unroll 1
                for (unsigned w = 0; w < words_per_scale; ++w) {
                    const unsigned word = block * words_per_scale + w;
#pragma unroll
                    for (unsigned quad = 0; quad < values_per_word / 4U; ++quad) {
                        const auto x = load_inputs(input + k + word * values_per_word + quad * 4U);
#pragma unroll
                        for (unsigned i = 0; i < Outputs; ++i) {
                            const std::uint32_t code = codes[i][0];
                            if constexpr (Operand::kNvfp4) {
                                const auto decoded = tables[i].decode_quad(code >> (quad * 16U));
                                out[i] = __fmaf_rn(decoded.template widen<0U>(), x[0], out[i]);
                                out[i] = __fmaf_rn(decoded.template widen<1U>(), x[1], out[i]);
                                out[i] = __fmaf_rn(decoded.template widen<2U>(), x[2], out[i]);
                                out[i] = __fmaf_rn(decoded.template widen<3U>(), x[3], out[i]);
                            } else {
                                const auto* bytes = reinterpret_cast<const std::uint8_t*>(&code);
#pragma unroll
                                for (unsigned p = 0; p < 2U; ++p) {
                                    const auto pair = weights.decode_pair(bytes + (quad * 2U + p) * codes_per_pair, tables[i]);
                                    out[i] = __fmaf_rn(widen(static_cast<std::uint16_t>(pair)), x[2U * p], out[i]);
                                    out[i] = __fmaf_rn(widen(static_cast<std::uint16_t>(pair >> 16U)), x[2U * p + 1U], out[i]);
                                }
                            }
                        }
                    }
#pragma unroll
                    for (unsigned i = 0; i < Outputs; ++i) codes[i] = codes[i].yzww;
                }
            };
            if constexpr (Operand::kNvfp4) {
                bool sign_symmetric = true;
#pragma unroll
                for (unsigned i = 0; i < Outputs; ++i) {
                    sign_symmetric &= core::Nvfp4Bf16Table::sign_symmetric(scales[i], weights.matrix.outer_scale);
                }
                const auto build_and_stream = [&]<class Table>() {
                    Table tables[Outputs];
#pragma unroll
                    for (unsigned i = 0; i < Outputs; ++i) tables[i] = Table::build(weights, scales[i]);
                    stream_words(tables);
                };
                if (sign_symmetric) {
                    build_and_stream.template operator()<core::Nvfp4Bf16Table>();
                } else {
                    build_and_stream.template operator()<core::Nvfp4Bf16SignedTable>();
                }
            } else {
                stream_words(scales);
            }
        }
    }
}
#endif

// Each lane accumulates its own K chunks in order, then the wave reduces
// each output separately. CUDA and unaligned operands keep pair-sized chunks.
template <packed::MatrixFormat Format, unsigned Columns, unsigned Outputs, class Input>
__device__ __forceinline__ void dot_rows(const packed::PackedMatrix& matrix, unsigned col, unsigned stride,
                                         unsigned live, const Input* input, unsigned width, unsigned lane,
                                         float (&out)[Outputs]) {
    using Operand = packed::PackedOperand<Format>;
    const Operand weights{matrix};
    typename Operand::Tile16Address rows[Outputs];
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        // Reuse the first valid row for unused columns in the final wave.
        const unsigned offset = i % Columns < live ? i % Columns : 0U;
        rows[i] = weights.tile16_address(col + offset + (i / Columns) * stride, 0);
        out[i] = 0.0F;
    }
#if defined(__HIPCC__)
    // Check the base and row pitch so every row admits a 16-byte code load.
    constexpr unsigned chunk = Operand::kNvfp4 ? 32U : 16U;
    if (matrix.columns % chunk == 0U && width % chunk == 0U &&
        reinterpret_cast<std::uintptr_t>(matrix.codes) % alignof(CodeChunk) == 0U &&
        reinterpret_cast<std::uintptr_t>(input) % (4U * sizeof(Input)) == 0U) {
        stream_rows(weights, rows, input, width, lane, out);
    } else
#endif
    {
        for (unsigned k = 2U * lane; k < width; k += 2U * kWarp) {
            const float a0 = input_bf16(input[k]);
            const float a1 = input_bf16(input[k + 1U]);
            const auto code = Operand::kNvfp4 ? k / 2U : k;
            const auto scale = k / Operand::kBlockColumns;
#pragma unroll
            for (unsigned i = 0; i < Outputs; ++i) {
                const auto pair = weights.decode_pair(rows[i].codes + code, Operand::decode_scale(rows[i].scale_code[scale]));
                out[i] = __fmaf_rn(widen(static_cast<std::uint16_t>(pair)), a0, out[i]);
                out[i] = __fmaf_rn(widen(static_cast<std::uint16_t>(pair >> 16U)), a1, out[i]);
            }
        }
    }
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) out[i] = warp_sum(out[i]);
}

// Paired gate and up GEMV share the input loads. Store only the BF16
// activation made with the same pre-activation BF16 rounding as grouped GEMM.
template <packed::MatrixFormat Format>
#if defined(__HIPCC__)
__global__ __launch_bounds__(kThreads, kMinWavesPerEu)
#else
__global__ __launch_bounds__(kThreads)
#endif
void gate_up(Args args) {
    if (args.failure && *args.failure != 0U) return;
    const unsigned slot = blockIdx.y;
    const unsigned expert = args.ids[slot];
    const unsigned col = (blockIdx.x * kWarps + threadIdx.x / kWarp) * kGateColumns;
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned live = args.intermediate - col;
    float pair[2U * kGateColumns];
    dot_rows<Format, kGateColumns>(args.matrices[expert].gate_up, col, args.intermediate, live,
                                   args.input, args.hidden, lane, pair);
    if (lane == 0U) {
        const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
#pragma unroll
        for (unsigned i = 0; i < kGateColumns; ++i) {
            if (i >= live) break;
            const float value = swiglu(activation::round_linear(pair[i]), activation::round_linear(pair[kGateColumns + i]));
            args.activation[std::size_t{slot} * args.intermediate + col + i] = scalar::bf16_encode(value);
        }
    }
}

// One CTA owns kDownColumns hidden output columns. Each warp projects down one or more
// routed experts. Store weighted terms in ascending expert order, then
// consume them with separate FP32 adds. No down-projection image is needed.
template <packed::MatrixFormat Format, unsigned TopK>
#if defined(__HIPCC__)
__global__ __launch_bounds__(kThreads, kMinWavesPerEu)
#else
__global__ __launch_bounds__(kThreads)
#endif
void down_combine(Args args) {
    if (args.failure && *args.failure != 0U) return;
    __shared__ float terms[kDownColumns][TopK];
    const unsigned col = blockIdx.x * kDownColumns;
    const unsigned live = args.hidden - col;
    const unsigned lane = threadIdx.x % kWarp;
    for (unsigned slot = threadIdx.x / kWarp; slot < TopK; slot += kWarps) {
        const unsigned expert = args.ids[slot];
        if (expert >= args.experts) continue;
        const auto* input = args.activation + std::size_t{slot} * args.intermediate;
        float projected[kDownColumns];
        dot_rows<Format, kDownColumns>(args.matrices[expert].down, col, 0U, live,
                                       input, args.intermediate, lane, projected);
        if (lane == 0U) {
            unsigned rank = 0;
#pragma unroll
            for (unsigned i = 0; i < TopK; ++i) {
                const auto other = args.ids[i];
                rank += other < expert || (other == expert && i < slot);
            }
#pragma unroll
            for (unsigned i = 0; i < kDownColumns; ++i) {
                terms[i][rank] = __fmul_rn(args.weights[slot], activation::round_linear(projected[i]));
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
#pragma unroll
        for (unsigned slot = 0; slot < TopK; ++slot) {
            if (args.ids[slot] >= args.experts) return;
        }
#pragma unroll
        for (unsigned i = 0; i < kDownColumns; ++i) {
            if (i >= live) break;
            float result = 0.0F;
#pragma unroll
            for (unsigned slot = 0; slot < TopK; ++slot) result = __fadd_rn(result, terms[i][slot]);
            args.output[col + i] = activation::round_linear(result);
        }
    }
}

} // namespace detail

// Decode-only plan. The caller admits a wave32 HIP device or CUDA sm_80+
// and supplies prevalidated weights and a checked route for the token.
template <packed::MatrixFormat Format, unsigned TopK>
#if defined(__HIPCC__)
inline hipError_t launch(const Args& args, hipStream_t stream) {
    if (!fits<Format, TopK>(args)) return hipErrorInvalidValue;
#else
inline cudaError_t launch(const Args& args, cudaStream_t stream) {
    if (!fits<Format, TopK>(args)) return cudaErrorInvalidValue;
#endif
    constexpr auto threads = detail::kThreads;
    const auto columns = (args.intermediate - 1U) / (detail::kWarps * detail::kGateColumns) + 1U;
    const auto down_blocks = (args.hidden - 1U) / detail::kDownColumns + 1U;
    detail::gate_up<Format><<<dim3(columns, TopK), threads, 0, stream>>>(args);
#if defined(__HIPCC__)
    if (const auto status = hipGetLastError(); status != hipSuccess) return status;
#else
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
#endif
    detail::down_combine<Format, TopK><<<down_blocks, threads, 0, stream>>>(args);
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}
#endif

} // namespace strix::moe_core::microscaled_r1
