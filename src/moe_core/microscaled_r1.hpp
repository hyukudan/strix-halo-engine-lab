#pragma once

#include "core/gemv_rows.hpp"
#include "moe_core/device_grouped_plan.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/grouped_rows.hpp"
#include "strix/exec/descriptor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__)
#include "core/nvfp4_bf16_table.hpp"
#elif defined(__CUDACC__) && !defined(STRIX_MOE_R1_SCALAR_DOT)
// CUDA runs the packed one-row body (same lane chunks, FMA order, BF16 rounds and butterfly; fixture
// microscaled_r1_cuda_fixture). Fixture translation units define STRIX_MOE_R1_SCALAR_DOT to keep the scalar
// body below as their reference.
#include "cuda/core/moe_microscaled_r1.cuh"
#define STRIX_MOE_R1_CUDA_PACKED_DOT 1
#endif

// Geometry helpers run on the host (launch grids) and in the kernels (column and row maps).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_HD __host__ __device__
#else
#define STRIX_R1_HD
#endif

namespace strix::moe_core::microscaled_r1 {

namespace packed = strix::weight::microscaled;

// One token, already routed in selection order. Matrices refer to an admitted
// layer's device projections. The optional device failure words are the route
// status, one per routed row (kMaxRows). Keep all inputs, output and TopK * intermediate BF16 activations
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

struct StepArgs {
    const exec::StepBlock* step;
    // The matrix directory and geometry are shared. Other pointers are
    // contiguous row-major images, including one optional failure per row.
    Args batch;
};

inline constexpr unsigned kStepTopK = 8U;

// CUDA host launch thunk. The SDK-neutral stream handle keeps shared recipe
// argument packing in ordinary C++ even when the body is compiled by nvcc.
void enqueue_cuda_step(const StepArgs&, packed::MatrixFormat, unsigned rows, bool gate_up, void* stream);

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

// Several routed rows in one launch (decode verify windows, short chunks): row r reads input + r * hidden, ids and
// weights + r * TopK, activation + r * TopK * intermediate and writes output + r * hidden. Every row runs the one-row
// body on its own TopK assignments (same lane chunks, FMA order, BF16 rounds and ascending-expert adds), so row r's
// output bits equal a one-row launch on row r. Experts shared by two rows are read once per row (no dedup): the one-row
// body beats the grouped tile kernel per routed expert while distinct experts / assignments stays high (verify widths).
inline constexpr unsigned kMaxRows = 8U;

// Element offsets of routed row `row` in a rows launch. gate_up runs assignment a = row * TopK + slot (grid y) and
// writes activation row a; down_combine (grid y = row) reads activation rows activation + slot * intermediate.
struct RowOffsets {
    std::size_t input, route, activation, output;
};
template <unsigned TopK>
STRIX_R1_HD constexpr RowOffsets row_offsets(unsigned row, unsigned hidden, unsigned intermediate) {
    return {std::size_t{row} * hidden, std::size_t{row} * TopK, std::size_t{row} * TopK * intermediate,
            std::size_t{row} * hidden};
}

template <packed::MatrixFormat Format, unsigned TopK>
inline bool fits_rows(const Args& a, unsigned rows) {
    return rows >= 1U && rows <= kMaxRows && fits<Format, TopK>(a) &&
           std::uint64_t{rows} * TopK * a.intermediate <=
               std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) &&
           std::uint64_t{rows} * a.hidden <= std::numeric_limits<unsigned>::max();
}

// Launch geometry. A wave owns `columns` consecutive outputs and streams their rows together (gate_up: the gate and
// up rows of each column; down_combine: one row per column per routed expert). Each row keeps its own accumulator,
// lane->K mapping and butterfly, so the column count changes no output bit. It sets how many rows share one wave's
// dependent load chain (ids -> matrices -> codes -> scales): on gfx1151 one row per wave left that chain carrying
// 1.3 KB (gate) / 0.65 KB (down) and the kernels at 61 % / 53 % of the byte roof.
// sm_120 (no-model bench cuda_core_moe_r1_geometry_bench, synthetic top-8 2304 / 1152 experts, rows 1): one column
// per warp left gate_up / down_combine at 880 / 655-700 GB/s (NVFP4); two / four columns are 10 % / 14-19 % faster
// per launch, with the same bits.
struct Geometry {
    unsigned gate_columns, down_columns;
};
inline constexpr unsigned kWavesPerBlock = 8;
inline constexpr Geometry kHipGeometry{2U, 4U};
inline constexpr Geometry kCudaGeometry{2U, 4U};
#if defined(__HIPCC__)
inline constexpr Geometry kGeometry = kHipGeometry;
#else
inline constexpr Geometry kGeometry = kCudaGeometry;
#endif

STRIX_R1_HD constexpr unsigned gate_blocks(unsigned intermediate, unsigned columns) {
    return (intermediate - 1U) / (kWavesPerBlock * columns) + 1U;
}
STRIX_R1_HD constexpr unsigned gate_first_column(unsigned block, unsigned wave, unsigned columns) {
    return (block * kWavesPerBlock + wave) * columns;
}
STRIX_R1_HD constexpr unsigned down_blocks(unsigned hidden, unsigned columns) { return (hidden - 1U) / columns + 1U; }
// Weight row read by output slot i of a wave whose first column is col; `live` columns remain in the matrix.
// Slots past the live columns re-read the first valid row (their results are never stored).
STRIX_R1_HD constexpr unsigned slot_row(unsigned col, unsigned i, unsigned columns, unsigned stride, unsigned live) {
    const unsigned offset = i % columns < live ? i % columns : 0U;
    return col + offset + (i / columns) * stride;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

namespace scalar = weight::scalar_float;
namespace activation = expert_activation::detail;
namespace combine = grouped_rows::detail;
namespace gemv = core::gemv_rows;

inline constexpr unsigned kWarp = gemv::wave_size;
inline constexpr unsigned kWarps = kWavesPerBlock;
inline constexpr unsigned kThreads = kWarp * kWarps;
#if defined(__HIPCC__)
// HIP launch bounds express the occupancy target in waves per execution unit.
inline constexpr unsigned kMinWavesPerEu = 8;
#endif
inline constexpr unsigned kGateColumns = kGeometry.gate_columns;
inline constexpr unsigned kDownColumns = kGeometry.down_columns;

__device__ __forceinline__ float widen(std::uint16_t value) {
    return scalar::u2f(std::uint32_t{value} << 16U);
}

__device__ __forceinline__ float input_bf16(std::uint16_t value) { return widen(value); }
__device__ __forceinline__ float input_bf16(float value) { return activation::round_linear(value); }

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

// One lane chunk of each row: the 16 code bytes and the chunk's block scale codes, loaded together before any decode.
template <class Operand, unsigned Outputs>
struct LaneChunk {
    static constexpr unsigned kScales = Operand::kNvfp4 ? 2U : 1U;
    CodeChunk codes[Outputs];
    std::uint32_t scales[Outputs];  // block b's scale code in byte b (shifted out, never indexed at run time)
};

template <class Operand, unsigned Outputs>
__device__ __forceinline__ void load_chunk(const typename Operand::Tile16Address (&rows)[Outputs], unsigned k,
                                           LaneChunk<Operand, Outputs>& chunk) {
    constexpr unsigned values_per_byte = Operand::kNvfp4 ? 2U : 1U;
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        chunk.codes[i] = *reinterpret_cast<const CodeChunk*>(rows[i].codes + k / values_per_byte);
        chunk.scales[i] = 0U;
#pragma unroll
        for (unsigned block = 0; block < LaneChunk<Operand, Outputs>::kScales; ++block) {
            chunk.scales[i] |= std::uint32_t{rows[i].scale_code[k / Operand::kBlockColumns + block]} << (8U * block);
        }
    }
}

// Decode one loaded chunk (values k .. k + chunk - 1) of each row into its accumulator, in ascending K.
template <class Operand, unsigned Outputs, class Input>
__device__ __forceinline__ void decode_chunk(const Operand& weights, LaneChunk<Operand, Outputs>& chunk,
                                             const Input* input, unsigned k, float (&out)[Outputs]) {
    constexpr unsigned codes_per_pair = Operand::kNvfp4 ? 1U : 2U;
    constexpr unsigned values_per_word = 8U / codes_per_pair;
    constexpr unsigned scale_count = LaneChunk<Operand, Outputs>::kScales;
    constexpr unsigned words_per_scale = 4U / scale_count;
    auto& codes = chunk.codes;
#pragma unroll 1
    for (unsigned block = 0; block < scale_count; ++block) {
        float scales[Outputs];
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            scales[i] = Operand::decode_scale(static_cast<std::uint8_t>(chunk.scales[i] >> (8U * block)));
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

// Lane l owns chunks l, l + 32, l + 64, ... of every row (the lane_chunks partition) and accumulates them in that
// order. The iterations where every lane owns a chunk run for all rows together. The partial last iteration owns
// `rem` chunks per row (lanes 0 .. rem - 1); instead of the whole wave decoding all rows for those few lanes, the
// tails of 32 / rem rows run in one pass: lane L takes row-lane (row L / rem, lane L % rem), receives that row-lane's
// accumulator by a shuffle, adds its last chunk with the same decode, and the sum is shuffled back. Every row-lane
// keeps its own FMA chain in ascending K and the butterfly is unchanged, so the bits are those of the plain loop.
template <class Operand, unsigned Outputs, class Input>
__device__ __forceinline__ void stream_rows(const Operand& weights,
                                           const typename Operand::Tile16Address (&rows)[Outputs],
                                           const Input* input, unsigned width, unsigned lane, float (&out)[Outputs]) {
    constexpr unsigned chunk_values = Operand::kNvfp4 ? 32U : 16U;
    constexpr unsigned span = chunk_values * kWarp;
    const unsigned full = width / span * span;
    for (unsigned k = chunk_values * lane; k < full; k += span) {
        LaneChunk<Operand, Outputs> chunk;
        load_chunk<Operand>(rows, k, chunk);
        decode_chunk(weights, chunk, input, k, out);
    }
    const unsigned rem = (width - full) / chunk_values;
    if (rem == 0U) return;
    const unsigned rows_per_pass = kWarp / rem;
    const unsigned sub = lane / rem, owner = lane % rem;
    for (unsigned first = 0; first < Outputs; first += rows_per_pass) {
        const unsigned row = first + sub;
        float acc = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            const float moved = __shfl(out[i], static_cast<int>(owner), static_cast<int>(kWarp));
            if (row == i) acc = moved;
        }
        if (sub < rows_per_pass && row < Outputs) {
            typename Operand::Tile16Address address[1] = {rows[0]};
#pragma unroll
            for (unsigned i = 1; i < Outputs; ++i) {
                if (row == i) address[0] = rows[i];
            }
            const unsigned k = full + chunk_values * owner;
            LaneChunk<Operand, 1U> chunk;
            float one[1] = {acc};
            load_chunk<Operand>(address, k, chunk);
            decode_chunk(weights, chunk, input, k, one);
            acc = one[0];
        }
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            const unsigned source = (i - first) * rem + owner;
            const float moved = __shfl(acc, static_cast<int>(source % kWarp), static_cast<int>(kWarp));
            if (i >= first && i < first + rows_per_pass && lane < rem) out[i] = moved;
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
#if defined(STRIX_MOE_R1_CUDA_PACKED_DOT)
    cuda_body::dot_rows<Format, Columns>(matrix, col, stride, live, input, width, lane, out);
#else
    using Operand = packed::PackedOperand<Format>;
    const Operand weights{matrix};
    typename Operand::Tile16Address rows[Outputs];
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        // Reuse the first valid row for unused columns in the final wave.
        rows[i] = weights.tile16_address(slot_row(col, i, Columns, stride, live), 0);
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
        gemv::lane_chunks<2U>(width, lane, [&](unsigned k) {
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
        });
    }
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) out[i] = gemv::butterfly_sum(out[i]);
#endif
}

// Paired gate and up GEMV share the input loads. Store only the BF16
// activation made with the same pre-activation BF16 rounding as grouped GEMM.
// blockIdx.y = row * TopK + slot (kMaxRows comment): the routed assignment, which also indexes ids and activation.
// GateColumns: launch geometry (Geometry comment); the default is the backend geometry.
template <packed::MatrixFormat Format, unsigned TopK, unsigned GateColumns = kGateColumns>
#if defined(__HIPCC__)
__global__ __launch_bounds__(kThreads, kMinWavesPerEu)
#else
__global__ __launch_bounds__(kThreads)
#endif
void gate_up(Args args) {
    const unsigned assignment = blockIdx.y;
    if (args.failure && args.failure[assignment / TopK] != 0U) return;
    const unsigned expert = args.ids[assignment];
    const unsigned col = gate_first_column(blockIdx.x, threadIdx.x / kWarp, GateColumns);
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned live = args.intermediate - col;
    const auto at = row_offsets<TopK>(assignment / TopK, args.hidden, args.intermediate);
    const float* input = args.input + at.input;
    auto* activation_row = args.activation + at.activation + std::size_t{assignment % TopK} * args.intermediate;
    float pair[2U * GateColumns];
    dot_rows<Format, GateColumns>(args.matrices[expert].gate_up, col, args.intermediate, live,
                                  input, args.hidden, lane, pair);
    if (lane == 0U) {
        const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
#pragma unroll
        for (unsigned i = 0; i < GateColumns; ++i) {
            if (i >= live) break;
            activation_row[col + i] =
                activation::activate_bf16(swiglu, pair[i], pair[GateColumns + i]);
        }
    }
}

// One CTA owns kDownColumns hidden output columns. Each warp projects down one or more
// routed experts. Store weighted terms in ascending expert order, then
// consume them with separate FP32 adds. No down-projection image is needed.
// blockIdx.y = routed row: the row's ids, weights, activations and output (kMaxRows comment).
template <packed::MatrixFormat Format, unsigned TopK, unsigned DownColumns = kDownColumns>
#if defined(__HIPCC__)
__global__ __launch_bounds__(kThreads, kMinWavesPerEu)
#else
__global__ __launch_bounds__(kThreads)
#endif
void down_combine(Args args) {
    if (args.failure && args.failure[blockIdx.y] != 0U) return;
    const auto at = row_offsets<TopK>(blockIdx.y, args.hidden, args.intermediate);
    args.ids += at.route;
    args.weights += at.route;
    args.activation += at.activation;
    args.output += at.output;
    __shared__ float terms[DownColumns][TopK];
    const unsigned col = blockIdx.x * DownColumns;
    const unsigned live = args.hidden - col;
    const unsigned lane = threadIdx.x % kWarp;
    for (unsigned slot = threadIdx.x / kWarp; slot < TopK; slot += kWarps) {
        const unsigned expert = args.ids[slot];
        if (expert >= args.experts) continue;
        const auto* input = args.activation + std::size_t{slot} * args.intermediate;
        float projected[DownColumns];
        dot_rows<Format, DownColumns>(args.matrices[expert].down, col, 0U, live,
                                      input, args.intermediate, lane, projected);
        if (lane == 0U) {
            unsigned rank = 0;
#pragma unroll
            for (unsigned i = 0; i < TopK; ++i) {
                const auto other = args.ids[i];
                rank += other < expert || (other == expert && i < slot);
            }
#pragma unroll
            for (unsigned i = 0; i < DownColumns; ++i) {
                terms[i][rank] = combine::weighted_bf16<false>(args.weights[slot], projected[i]);
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
        for (unsigned i = 0; i < DownColumns; ++i) {
            if (i >= live) break;
            float result = 0.0F;
#pragma unroll
            for (unsigned slot = 0; slot < TopK; ++slot) result = __fadd_rn(result, terms[i][slot]);
            args.output[col + i] = combine::rounded_bf16<false>(result);
        }
    }
}

} // namespace detail

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Individual nodes use the same packed kernels and geometry as the combined
// launch. The caller orders gate-up before the down projection.
template <packed::MatrixFormat Format, unsigned TopK>
inline cudaError_t launch_gate_up(const Args& args, cudaStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows)) return cudaErrorInvalidValue;
    const auto columns = gate_blocks(args.intermediate, detail::kGateColumns);
    detail::gate_up<Format, TopK><<<dim3(columns, rows * TopK), detail::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

template <packed::MatrixFormat Format, unsigned TopK>
inline cudaError_t launch_down_combine(const Args& args, cudaStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows)) return cudaErrorInvalidValue;
    const auto blocks = down_blocks(args.hidden, detail::kDownColumns);
    detail::down_combine<Format, TopK><<<dim3(blocks, rows), detail::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

#if defined(STRIX_MOE_R1_CUDA_PACKED_DOT)
namespace detail {

// The replay-step kernels keep one gate / one down column per warp (the geometry their row bodies are written for),
// independent of the decode default geometry above; both geometries give the same bits.
inline constexpr unsigned kStepGateColumns = 1U;
inline constexpr unsigned kStepDownColumns = 1U;

template <unsigned Rows>
__device__ __forceinline__ unsigned eligible_rows(const StepArgs& args) {
    unsigned mask = 0U;
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        if (exec::step_row_valid(*args.step, r) && (!args.batch.failure || args.batch.failure[r] == 0U)) {
            mask |= 1U << r;
        }
    }
    return mask;
}

// Each pass shares a code/scale read for rows selecting the same expert at
// this route slot. Different experts, including duplicate selections at
// different route slots, retain independent terms.
template <unsigned Rows, unsigned TopK>
__device__ __forceinline__ unsigned matching_rows(const Args& a, unsigned mask, unsigned slot, unsigned expert) {
    unsigned matches = 0U;
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        if ((mask & (1U << r)) && a.ids[r * TopK + slot] == expert) matches |= 1U << r;
    }
    return matches;
}

template <unsigned Rows, packed::MatrixFormat Format, unsigned TopK>
__global__ __launch_bounds__(kThreads) void gate_up_step(StepArgs args) {
    const auto& a = args.batch;
    const unsigned slot = blockIdx.y;
    const unsigned col = gate_first_column(blockIdx.x, threadIdx.x / kWarp, kStepGateColumns);
    if (col >= a.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    unsigned remaining = eligible_rows<Rows>(args);
    const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{a.gate_cap, a.up_cap};
    while (remaining != 0U) {
        const auto lead = static_cast<unsigned>(__ffs(static_cast<int>(remaining)) - 1);
        const unsigned expert = a.ids[lead * TopK + slot];
        const unsigned active = matching_rows<Rows, TopK>(a, remaining, slot, expert);
        remaining &= ~active;
        if (expert >= a.experts) continue;
        float pair[Rows][2];
        cuda_body::dot_step_rows<Format, 1, Rows>(a.matrices[expert].gate_up, col, a.intermediate, 1U,
            a.input, a.hidden, active, a.hidden, lane, pair);
        if (lane != 0U) continue;
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            if (active & (1U << r)) {
                a.activation[(std::size_t{r} * TopK + slot) * a.intermediate + col] =
                    activation::activate_bf16(swiglu, pair[r][0], pair[r][1]);
            }
        }
    }
}

template <unsigned Rows, packed::MatrixFormat Format, unsigned TopK>
__global__ __launch_bounds__(kThreads) void down_combine_step(StepArgs args) {
    const auto& a = args.batch;
    const unsigned col = blockIdx.x;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned eligible = eligible_rows<Rows>(args);
    __shared__ float terms[Rows][TopK];
    for (unsigned slot = threadIdx.x / kWarp; slot < TopK; slot += kWarps) {
        unsigned remaining = eligible;
        while (remaining != 0U) {
            const auto lead = static_cast<unsigned>(__ffs(static_cast<int>(remaining)) - 1);
            const unsigned expert = a.ids[lead * TopK + slot];
            const unsigned active = matching_rows<Rows, TopK>(a, remaining, slot, expert);
            remaining &= ~active;
            if (expert >= a.experts) continue;
            float projected[Rows][1];
            cuda_body::dot_step_rows<Format, 1, Rows>(a.matrices[expert].down, col, 0U, 1U,
                a.activation + std::size_t{slot} * a.intermediate, std::size_t{TopK} * a.intermediate,
                active, a.intermediate, lane, projected);
            if (lane != 0U) continue;
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                if ((active & (1U << r)) == 0U) continue;
                const auto* ids = a.ids + r * TopK;
                unsigned rank = 0U;
#pragma unroll
                for (unsigned i = 0; i < TopK; ++i) {
                    rank += ids[i] < expert || (ids[i] == expert && i < slot);
                }
                terms[r][rank] = combine::weighted_bf16<false>(a.weights[r * TopK + slot], projected[r][0]);
            }
        }
    }
    __syncthreads();
    const unsigned row = threadIdx.x;
    if (row >= Rows || (eligible & (1U << row)) == 0U) return;
#pragma unroll
    for (unsigned slot = 0; slot < TopK; ++slot) {
        if (a.ids[row * TopK + slot] >= a.experts) return;
    }
    float result = 0.0F;
#pragma unroll
    for (unsigned slot = 0; slot < TopK; ++slot) result = __fadd_rn(result, terms[row][slot]);
    a.output[std::size_t{row} * a.hidden + col] = combine::rounded_bf16<false>(result);
}

}  // namespace detail

template <unsigned Rows, packed::MatrixFormat Format, unsigned TopK, bool GateUp>
inline cudaError_t enqueue_step(const StepArgs& args, cudaStream_t stream) {
    static_assert(Rows <= 16U && session::is_step_replay_width(Rows));
    const auto& a = args.batch;
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
    if (!args.step || !fits<Format, TopK>(a) || a.intermediate > limit / Rows / TopK) return cudaErrorInvalidValue;
    if constexpr (GateUp) {
        const auto blocks = gate_blocks(a.intermediate, detail::kStepGateColumns);
        detail::gate_up_step<Rows, Format, TopK><<<dim3(blocks, TopK), detail::kThreads, 0, stream>>>(args);
    } else {
        const auto blocks = down_blocks(a.hidden, detail::kStepDownColumns);
        detail::down_combine_step<Rows, Format, TopK><<<blocks, detail::kThreads, 0, stream>>>(args);
    }
    return cudaGetLastError();
}
#endif
#endif

// Decode plan for 1 .. kMaxRows routed rows. The caller admits a wave32 HIP device or CUDA sm_80+
// and supplies prevalidated weights and a checked route for every row.
template <packed::MatrixFormat Format, unsigned TopK>
#if defined(__HIPCC__)
inline hipError_t launch(const Args& args, hipStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows)) return hipErrorInvalidValue;
#else
inline cudaError_t launch(const Args& args, cudaStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows)) return cudaErrorInvalidValue;
#endif
    constexpr auto threads = detail::kThreads;
    const auto columns = gate_blocks(args.intermediate, detail::kGateColumns);
    const auto down_grid = down_blocks(args.hidden, detail::kDownColumns);
    detail::gate_up<Format, TopK><<<dim3(columns, rows * TopK), threads, 0, stream>>>(args);
#if defined(__HIPCC__)
    if (const auto status = hipGetLastError(); status != hipSuccess) return status;
#else
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
#endif
    detail::down_combine<Format, TopK><<<dim3(down_grid, rows), threads, 0, stream>>>(args);
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}
#endif

} // namespace strix::moe_core::microscaled_r1
