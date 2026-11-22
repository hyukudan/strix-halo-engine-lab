#pragma once

#include "core/gemv_rows.hpp"
#include "moe_core/device_grouped_plan.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/grouped_rows.hpp"
#include "moe_core/r1_route.hpp"
#include "moe_core/r1_tail.hpp"
#include "strix/exec/descriptor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__HIPCC__)
#include "core/nvfp4_bf16_table.hpp"
#elif defined(__CUDACC__) && !defined(STRIX_MOE_R1_SCALAR_DOT)
// CUDA runs the packed one-row body (same lane chunks, FMA order, BF16 rounds and butterfly; fixture
// microscaled_r1_cuda_fixture). Fixture translation units define STRIX_MOE_R1_SCALAR_DOT to keep the scalar
// body below as their reference.
#include "cuda/core/moe_microscaled_r1.cuh"
#define STRIX_MOE_R1_CUDA_PACKED_DOT 1
#endif
#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "cuda/core/common.cuh"
#endif

// Geometry helpers run on the host (launch grids) and in the kernels (column and row maps).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_R1_HD __host__ __device__
#else
#define STRIX_R1_HD
#endif

namespace strix::moe_core::microscaled_r1 {

namespace packed = strix::weight::microscaled;

// Entry schedule of an R1 kernel (a trait of the one R1 family, not a second kernel).
// EntryComplete: launched after its predecessor completed; the kernel has no griddepcontrol (incumbent instructions).
// EntryPdl (CUDA): launched as a programmatic dependent. Every CTA waits before its first read and triggers right after:
// route ids, weights, failure words and activations are all predecessor output, and every expert weight address
// depends on the route, so nothing is read before the wait (cuda/core/common.cuh contract; bits unchanged).
struct EntryComplete {};
struct EntryPdl {};

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

// Tail legality against an R1 launch: the tail's own plan rules (r1_tail::admits) plus range checks against the
// launch's buffers. No tail store may touch any byte of the stored MoE output rows (the exact incumbent rows, written
// by other CTAs), of the input rows or, with top_k != 0, of the activation rows every CTA reads. No tail input may
// overlap the MoE output: the tail stages its inputs before the combined store, where the separate nodes would read
// the stored MoE row.
inline bool admits_tail_ranges(const Args& args, const r1_tail::Footprint& tail, unsigned rows, unsigned top_k) {
    const auto row_elements = std::uint64_t{rows} * args.hidden;
    const auto output = r1_tail::range_of(args.output, static_cast<std::size_t>(row_elements), sizeof(float));
    const auto input = r1_tail::range_of(args.input, static_cast<std::size_t>(row_elements), sizeof(float));
    const auto activation_elements = std::uint64_t{rows} * top_k * args.intermediate;
    const auto activation = r1_tail::range_of(args.activation, static_cast<std::size_t>(activation_elements),
                                              sizeof(std::uint16_t));
    if (!output.valid || !input.valid || !activation.valid) return false;
    for (const auto& write : tail.writes)
        if (r1_tail::overlap(write, output) || r1_tail::overlap(write, input) || r1_tail::overlap(write, activation))
            return false;
    for (const auto& read : tail.reads)
        if (r1_tail::overlap(read, output)) return false;
    return true;
}

template <class Tail>
inline bool admits_tail(const Args& args, const Tail& tail, unsigned rows, unsigned top_k = 0U) {
    if (rows == 0U || rows > kMaxRows || !r1_tail::admits(tail, rows, args.hidden)) return false;
    return admits_tail_ranges(args, r1_tail::footprint(tail, rows, args.hidden), rows, top_k);
}

// Inline route legality (r1_route.hpp) for a launch of `rows` routed rows with TopK picks: the policy routes exactly
// this launch (experts, top_k, checked status), the route publishes into the Args route buffers, and nothing the route
// writes overlaps anything a CTA of the launch reads (logits, bias, input, the other route buffers) or the activation
// rows it writes: CTAs read logits and write activations while one CTA per row publishes.
template <unsigned TopK, class Policy, class Math, class Traits>
inline bool admits_route(const Args& args, const r1_route::Inline<Policy, Math, Traits>& route, unsigned rows) {
    constexpr const moe::RoutePolicy& policy = Policy::value;
    if (moe::validate(policy) != nullptr || policy.top_k != TopK || policy.experts != args.experts ||
        !Traits::checked_status || rows == 0U || rows > kMaxRows)
        return false;
    if (!route.logits || !route.bias || route.ids != args.ids || route.weights != args.weights ||
        route.failure != args.failure || !route.failure)
        return false;
    const std::size_t assignments = std::size_t{rows} * TopK;
    const r1_tail::Range writes[] = {
        r1_tail::range_of(route.ids, assignments, sizeof(*route.ids)),
        r1_tail::range_of(route.weights, assignments, sizeof(*route.weights)),
        r1_tail::range_of(route.failure, rows, sizeof(*route.failure)),
    };
    const r1_tail::Range others[] = {
        r1_tail::range_of(route.logits, std::size_t{rows} * args.experts, sizeof(float)),
        r1_tail::range_of(route.bias, args.experts, sizeof(float)),
        r1_tail::range_of(args.input, std::size_t{rows} * args.hidden, sizeof(float)),
        r1_tail::range_of(args.activation, assignments * args.intermediate, sizeof(std::uint16_t)),
    };
    for (const auto& w : writes) {
        if (!w.valid) return false;
        for (const auto& o : others)
            if (!o.valid || r1_tail::overlap(w, o)) return false;
    }
    return !r1_tail::overlap(writes[0], writes[1]) && !r1_tail::overlap(writes[0], writes[2]) &&
           !r1_tail::overlap(writes[1], writes[2]);
}

// Expert-union launches (gate_up_union / down_union): rows <= MaxRows <= r1_route::kMaxUnionRows routed rows laid out
// as a rows launch (row_offsets), the experts as a shared presence mask (r1_route::kMaxUnionExperts), the assignments
// of one expert in passes of at most MaxRows inputs.
template <packed::MatrixFormat Format, unsigned TopK, unsigned MaxRows>
inline bool fits_union(const Args& a, unsigned rows) {
    static_assert(MaxRows >= 1U && MaxRows <= r1_route::kMaxUnionRows && MaxRows * TopK <= 256U);
    return rows >= 1U && rows <= MaxRows && a.experts <= r1_route::kMaxUnionExperts && fits<Format, TopK>(a) &&
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

// Expert-union launches (CUDA): one pass carries up to MaxRows inputs per weight row, so the per-input accumulators
// grow with MaxRows; wide passes take fewer columns per warp to stay in registers (the columns change no bit).
template <unsigned MaxRows>
inline constexpr Geometry kUnionGeometry =
    MaxRows <= 4U ? kCudaGeometry : (MaxRows <= 8U ? Geometry{1U, 2U} : Geometry{1U, 1U});

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

template <class Entry>
__device__ __forceinline__ void enter() {
#if defined(__CUDACC__) && !defined(__HIPCC__)
    if constexpr (std::is_same_v<Entry, EntryPdl>) {
        (void)cuda_core::pdl_wait();
        cuda_core::pdl_trigger();
    }
#else
    static_assert(!std::is_same_v<Entry, EntryPdl>, "the PDL entry schedule is a CUDA schedule");
#endif
}

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

// dot_rows for `count` <= Inputs inputs against the same weight rows; out[n] is bitwise dot_rows on inputs[n]. The
// CUDA packed body reads and decodes each weight word once for all inputs (the expert union); the other bodies run
// dot_rows per input (same bits, no read sharing).
template <packed::MatrixFormat Format, unsigned Columns, unsigned Inputs, unsigned Outputs, class Input>
__device__ __forceinline__ void dot_inputs(const packed::PackedMatrix& matrix, unsigned col, unsigned stride,
                                           unsigned live, const Input* const (&inputs)[Inputs], unsigned count,
                                           unsigned width, unsigned lane, float (&out)[Inputs][Outputs]) {
#if defined(STRIX_MOE_R1_CUDA_PACKED_DOT)
    cuda_body::dot_inputs<Format, Columns>(matrix, col, stride, live, inputs, count, width, lane, out);
#else
#pragma unroll
    for (unsigned n = 0; n < Inputs; ++n) {
        if (n < count) dot_rows<Format, Columns>(matrix, col, stride, live, inputs[n], width, lane, out[n]);
    }
#endif
}

// Inline route of the CTA's own row (wave 0) into shared ids / weights / status (r1_route.hpp).
template <class Policy, class Math, class Traits>
__device__ __forceinline__ void route_inline(const r1_route::Inline<Policy, Math, Traits>& route, std::uint32_t* ids,
                                             float* weights, unsigned* status, unsigned rows, unsigned row) {
#if defined(__CUDACC__) && !defined(__HIPCC__)
    r1_route::route_rows(route, ids, weights, status, rows, r1_route::CtaRow{row, rows});
#else
    static_assert(sizeof(Policy) == 0U, "the inline R1 route is a CUDA route source");
    (void)route, (void)ids, (void)weights, (void)status, (void)rows, (void)row;
#endif
}

// Inline route of every row of a union launch: wave w routes row w (rows <= kWarps).
template <class Policy, class Math, class Traits>
__device__ __forceinline__ void route_union(const r1_route::Inline<Policy, Math, Traits>& route, std::uint32_t* ids,
                                            float* weights, unsigned* status, unsigned rows) {
#if defined(__CUDACC__) && !defined(__HIPCC__)
    r1_route::route_rows(route, ids, weights, status, rows, r1_route::WaveRows<kWarps>{});
#else
    static_assert(sizeof(Policy) == 0U, "the inline R1 route is a CUDA route source");
    (void)route, (void)ids, (void)weights, (void)status, (void)rows;
#endif
}

// Publish routed row `row` (TopK ids and weights, the status word) from shared memory to the route buffers.
template <unsigned TopK, class Policy, class Math, class Traits>
__device__ __forceinline__ void publish_row(const r1_route::Inline<Policy, Math, Traits>& route,
                                            const std::uint32_t* ids, const float* weights, const unsigned* status,
                                            unsigned row) {
    for (unsigned k = threadIdx.x; k < TopK; k += kThreads) {
        route.ids[row * TopK + k] = ids[row * TopK + k];
        route.weights[row * TopK + k] = weights[row * TopK + k];
    }
    if (threadIdx.x == 0U) route.failure[row] = status[row];
}

// Paired gate and up GEMV share the input loads. Store only the BF16
// activation made with the same pre-activation BF16 rounding as grouped GEMM.
// blockIdx.y = row * TopK + slot (kMaxRows comment): the routed assignment, which also indexes ids and activation.
// GateColumns: launch geometry (Geometry comment); the default is the backend geometry.
// Route (r1_route.hpp, at most one): an Inline source. The CTA routes its own row before reading any weight (wave 0, the
// checked route body), takes its expert from that route, and CTA x == 0 of slot 0 publishes the row's route (ids,
// weights, status) to the Args buffers for down_combine. The empty pack is the incumbent kernel (route from Args).
template <packed::MatrixFormat Format, unsigned TopK, unsigned GateColumns = kGateColumns, class Entry = EntryComplete,
          class... Route>
#if defined(__HIPCC__)
__global__ __launch_bounds__(kThreads, kMinWavesPerEu)
#else
__global__ __launch_bounds__(kThreads)
#endif
void gate_up(Args args, Route... route) {
    static_assert(sizeof...(Route) <= 1U);
    enter<Entry>();
    const unsigned assignment = blockIdx.y;
    unsigned expert;
    if constexpr (sizeof...(Route) == 1U) {
        const unsigned row = assignment / TopK, rows = gridDim.y / TopK;
        __shared__ std::uint32_t ids[kMaxRows * TopK];
        __shared__ float weights[kMaxRows * TopK];
        __shared__ unsigned status[kMaxRows];
        (route_inline(route, ids, weights, status, rows, row), ...);
        __syncthreads();
        if (blockIdx.x == 0U && assignment % TopK == 0U) (publish_row<TopK>(route, ids, weights, status, row), ...);
        if (status[row] != 0U) return;
        expert = ids[assignment];
    } else {
        if (args.failure && args.failure[assignment / TopK] != 0U) return;
        expert = args.ids[assignment];
    }
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
// Tail (r1_tail.hpp, at most one): a per-channel consumer of the stored combined value, staged by thread 0 before the
// barrier and finished after the store. The empty pack is the incumbent kernel (signature (Args), same machine code).
template <packed::MatrixFormat Format, unsigned TopK, unsigned DownColumns = kDownColumns, class Entry = EntryComplete,
          class... Tail>
#if defined(__HIPCC__)
__global__ __launch_bounds__(kThreads, kMinWavesPerEu)
#else
__global__ __launch_bounds__(kThreads)
#endif
void down_combine(Args args, Tail... tail) {
    static_assert(sizeof...(Tail) <= 1U);
    constexpr bool kTail = sizeof...(Tail) == 1U;
    enter<Entry>();
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
    [[maybe_unused]] r1_tail::StagedOf<DownColumns, Tail...> staged{};
    if constexpr (kTail) {
        if (threadIdx.x == 0U) staged = (tail.template stage<DownColumns>(blockIdx.y, col, live), ...);
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
            const float combined = combine::rounded_bf16<false>(result);
            args.output[col + i] = combined;
            if constexpr (kTail) (tail.template finish<DownColumns>(staged, i, blockIdx.y, col + i, combined), ...);
        }
    }
}


#if defined(__CUDACC__) && !defined(__HIPCC__)
// ---- Expert union of a routed rows launch (r1_route.hpp; CUDA bodies) ----
// Shared plan: the route of every row, the live rows (valid step row, zero status) and the presence mask of the
// distinct experts. Group g of the union is the g-th distinct expert in ascending id (r1_route::expert_union).
template <unsigned TopK, unsigned MaxRows>
struct UnionPlan {
    static constexpr unsigned kAssignments = MaxRows * TopK;
    static constexpr unsigned kAssignmentWords = (kAssignments + 31U) / 32U;
    static constexpr unsigned kExpertWords = r1_route::kMaxUnionExperts / 32U;
    std::uint32_t ids[kAssignments];
    float weights[kAssignments];
    unsigned status[MaxRows];
    unsigned present[kExpertWords];
    unsigned group[kAssignmentWords];
    unsigned live;
    unsigned expert;
};

template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ void union_clear(UnionPlan<TopK, MaxRows>& plan, unsigned experts) {
    using Plan = UnionPlan<TopK, MaxRows>;
    for (unsigned t = threadIdx.x; t < Plan::kExpertWords; t += kThreads) plan.present[t] = 0U;
    if (threadIdx.x < Plan::kAssignmentWords) plan.group[threadIdx.x] = 0U;
    if (threadIdx.x == 0U) plan.expert = experts;
}

// Route of every row from the Args route buffers (written by a route launch or published by an inline-routed gate_up).
template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ void union_load(UnionPlan<TopK, MaxRows>& plan, const Args& args, unsigned rows) {
    for (unsigned t = threadIdx.x; t < rows * TopK; t += kThreads) {
        plan.ids[t] = args.ids[t];
        plan.weights[t] = args.weights[t];
    }
    if (threadIdx.x < rows) plan.status[threadIdx.x] = args.failure ? args.failure[threadIdx.x] : 0U;
}

// Wave 0, once the route is in shared memory: live rows = launched, valid in the step block (when bound), status 0.
template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ void union_live(UnionPlan<TopK, MaxRows>& plan, unsigned rows, const exec::StepBlock* step) {
    if (threadIdx.x >= kWarp) return;
    const unsigned r = threadIdx.x;
    const bool live = r < rows && (step == nullptr || exec::step_row_valid(*step, r)) && plan.status[r] == 0U;
    const unsigned mask = __ballot_sync(0xffffffffU, live);
    if (r == 0U) plan.live = mask;
}

template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ bool union_member(const UnionPlan<TopK, MaxRows>& plan, unsigned a, unsigned experts) {
    return (plan.live >> (a / TopK) & 1U) != 0U && plan.ids[a] < experts;
}

template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ void union_mark(UnionPlan<TopK, MaxRows>& plan, unsigned rows, unsigned experts) {
    for (unsigned a = threadIdx.x; a < rows * TopK; a += kThreads) {
        if (union_member(plan, a, experts)) atomicOr(&plan.present[plan.ids[a] / 32U], 1U << (plan.ids[a] % 32U));
    }
}

__device__ __forceinline__ unsigned popcount(unsigned bits) { return static_cast<unsigned>(__popc(bits)); }

// Group of expert e: the number of distinct experts below it.
template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ unsigned union_rank(const UnionPlan<TopK, MaxRows>& plan, unsigned e) {
    unsigned rank = popcount(plan.present[e / 32U] & ((1U << (e % 32U)) - 1U));
    for (unsigned w = 0; w < e / 32U; ++w) rank += popcount(plan.present[w]);
    return rank;
}

template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ unsigned union_groups(const UnionPlan<TopK, MaxRows>& plan, unsigned experts) {
    unsigned groups = 0U;
    for (unsigned w = 0; w < (experts + 31U) / 32U; ++w) groups += popcount(plan.present[w]);
    return groups;
}

// Expert of group g (g < union_groups).
template <unsigned TopK, unsigned MaxRows>
__device__ __forceinline__ unsigned union_expert(const UnionPlan<TopK, MaxRows>& plan, unsigned g) {
    for (unsigned w = 0;; ++w) {
        unsigned bits = plan.present[w];
        const unsigned count = popcount(bits);
        if (g < count) {
            for (unsigned i = 0; i < g; ++i) bits &= bits - 1U;
            return w * 32U + static_cast<unsigned>(__ffs(static_cast<int>(bits)) - 1);
        }
        g -= count;
    }
}

// Next pass of at most MaxRows assignments from the mask words, ascending (taken[n] for n < count; static indices, so
// the pass stays in registers). Returns the count taken.
template <unsigned Words, unsigned MaxRows>
__device__ __forceinline__ unsigned union_take(unsigned (&mask)[Words], unsigned (&taken)[MaxRows]) {
    unsigned count = 0U;
#pragma unroll
    for (unsigned n = 0; n < MaxRows; ++n) {
        bool found = false;
#pragma unroll
        for (unsigned w = 0; w < Words; ++w) {
            if (!found && mask[w] != 0U) {
                taken[n] = w * 32U + static_cast<unsigned>(__ffs(static_cast<int>(mask[w])) - 1);
                mask[w] &= mask[w] - 1U;
                found = true;
            }
        }
        if (!found) taken[n] = 0U;
        count += found ? 1U : 0U;
    }
    return count;
}

// gate_up over the expert union of `rows` <= MaxRows routed rows: blockIdx.y = union group g (the g-th distinct expert
// of the live assignments, ascending id); CTAs past the union exit. Each warp streams its columns of that expert once
// for all assignments that selected it and stores each assignment's activation row exactly as gate_up does (same dot
// per input, same rounding), so every activation equals the rows launch. Route: an Inline source; waves w < rows route
// row w (MaxRows <= kWarps) and CTA (0, 0) publishes every row's route. step (may be null): rows invalid in the step block
// take no part (their activations are not written).
template <packed::MatrixFormat Format, unsigned TopK, unsigned MaxRows,
          unsigned GateColumns = kUnionGeometry<MaxRows>.gate_columns, class Entry = EntryComplete, class... Route>
__global__ __launch_bounds__(kThreads) void gate_up_union(Args args, unsigned rows, const exec::StepBlock* step,
                                                           Route... route) {
    static_assert(sizeof...(Route) <= 1U && (sizeof...(Route) == 0U || MaxRows <= kWarps));
    using Plan = UnionPlan<TopK, MaxRows>;
    enter<Entry>();
    __shared__ Plan plan;
    union_clear(plan, args.experts);
    if constexpr (sizeof...(Route) == 1U) {
        (route_union(route, plan.ids, plan.weights, plan.status, rows), ...);
    } else {
        union_load(plan, args, rows);
    }
    __syncthreads();
    if constexpr (sizeof...(Route) == 1U) {
        if (blockIdx.x == 0U && blockIdx.y == 0U) {
            for (unsigned row = 0; row < rows; ++row) (publish_row<TopK>(route, plan.ids, plan.weights, plan.status, row), ...);
        }
    }
    union_live(plan, rows, step);
    __syncthreads();
    union_mark(plan, rows, args.experts);
    __syncthreads();
    for (unsigned a = threadIdx.x; a < rows * TopK; a += kThreads) {
        if (union_member(plan, a, args.experts) && union_rank(plan, plan.ids[a]) == blockIdx.y) {
            atomicOr(&plan.group[a / 32U], 1U << (a % 32U));
            plan.expert = plan.ids[a];
        }
    }
    __syncthreads();
    const unsigned expert = plan.expert;
    const unsigned col = gate_first_column(blockIdx.x, threadIdx.x / kWarp, GateColumns);
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned live = args.intermediate - col;
    unsigned mask[Plan::kAssignmentWords];
#pragma unroll
    for (unsigned w = 0; w < Plan::kAssignmentWords; ++w) mask[w] = plan.group[w];
    const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
    for (;;) {
        unsigned taken[MaxRows];
        const unsigned count = union_take(mask, taken);
        if (count == 0U) break;
        const float* inputs[MaxRows];
#pragma unroll
        for (unsigned n = 0; n < MaxRows; ++n) {
            inputs[n] = args.input + row_offsets<TopK>(n < count ? taken[n] / TopK : 0U, args.hidden,
                                                       args.intermediate).input;
        }
        float pair[MaxRows][2U * GateColumns];
        dot_inputs<Format, GateColumns>(args.matrices[expert].gate_up, col, args.intermediate, live, inputs, count,
                                        args.hidden, lane, pair);
        if (lane == 0U) {
#pragma unroll
            for (unsigned n = 0; n < MaxRows; ++n) {
                if (n >= count) break;
                auto* activation_row = args.activation + std::size_t{taken[n]} * args.intermediate;
#pragma unroll
                for (unsigned i = 0; i < GateColumns; ++i) {
                    if (i >= live) break;
                    activation_row[col + i] = activation::activate_bf16(swiglu, pair[n][i], pair[n][GateColumns + i]);
                }
            }
        }
    }
}

// down_combine over the expert union of `rows` <= MaxRows routed rows: one CTA owns DownColumns hidden columns of every
// row. Its warps take the union groups in turn and stream each expert's columns once for all assignments that selected
// it; lane 0 stores each assignment's weighted term at the rank down_combine gives it (ascending expert, slot on ties).
// Thread r then adds row r's terms in that order and stores the combined value: every live row's output equals the rows
// launch (and the one-row launch) bit for bit. Rows that are not live, or route an invalid id, are not stored.
template <packed::MatrixFormat Format, unsigned TopK, unsigned MaxRows,
          unsigned DownColumns = kUnionGeometry<MaxRows>.down_columns, class Entry = EntryComplete>
__global__ __launch_bounds__(kThreads) void down_union(Args args, unsigned rows, const exec::StepBlock* step) {
    static_assert(MaxRows <= kWarp);
    using Plan = UnionPlan<TopK, MaxRows>;
    enter<Entry>();
    __shared__ Plan plan;
    __shared__ float terms[MaxRows][DownColumns][TopK];
    union_clear(plan, args.experts);
    union_load(plan, args, rows);
    __syncthreads();
    union_live(plan, rows, step);
    __syncthreads();
    union_mark(plan, rows, args.experts);
    __syncthreads();
    const unsigned col = blockIdx.x * DownColumns;
    const unsigned live = args.hidden - col;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned groups = union_groups(plan, args.experts);
    for (unsigned g = threadIdx.x / kWarp; g < groups; g += kWarps) {
        const unsigned expert = union_expert(plan, g);
        unsigned mask[Plan::kAssignmentWords];
#pragma unroll
        for (unsigned w = 0; w < Plan::kAssignmentWords; ++w) {
            const unsigned a = w * kWarp + lane;
            mask[w] = __ballot_sync(0xffffffffU,
                                    a < rows * TopK && union_member(plan, a, args.experts) && plan.ids[a] == expert);
        }
        for (;;) {
            unsigned taken[MaxRows];
            const unsigned count = union_take(mask, taken);
            if (count == 0U) break;
            const std::uint16_t* inputs[MaxRows];
#pragma unroll
            for (unsigned n = 0; n < MaxRows; ++n) {
                inputs[n] = args.activation + std::size_t{n < count ? taken[n] : 0U} * args.intermediate;
            }
            float projected[MaxRows][DownColumns];
            dot_inputs<Format, DownColumns>(args.matrices[expert].down, col, 0U, live, inputs, count,
                                            args.intermediate, lane, projected);
            if (lane == 0U) {
#pragma unroll
                for (unsigned n = 0; n < MaxRows; ++n) {
                    if (n >= count) break;
                    const unsigned row = taken[n] / TopK, slot = taken[n] % TopK;
                    const auto* ids = plan.ids + row * TopK;
                    unsigned rank = 0;
#pragma unroll
                    for (unsigned i = 0; i < TopK; ++i) rank += ids[i] < expert || (ids[i] == expert && i < slot);
#pragma unroll
                    for (unsigned i = 0; i < DownColumns; ++i) {
                        terms[row][i][rank] = combine::weighted_bf16<false>(plan.weights[taken[n]], projected[n][i]);
                    }
                }
            }
        }
    }
    __syncthreads();
    const unsigned row = threadIdx.x;
    if (row >= rows || (plan.live >> row & 1U) == 0U) return;
#pragma unroll
    for (unsigned slot = 0; slot < TopK; ++slot) {
        if (plan.ids[row * TopK + slot] >= args.experts) return;
    }
    float* output = args.output + row_offsets<TopK>(row, args.hidden, args.intermediate).output;
#pragma unroll
    for (unsigned i = 0; i < DownColumns; ++i) {
        if (i >= live) break;
        float result = 0.0F;
#pragma unroll
        for (unsigned slot = 0; slot < TopK; ++slot) result = __fadd_rn(result, terms[row][i][slot]);
        output[col + i] = combine::rounded_bf16<false>(result);
    }
}
#endif

} // namespace detail

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Individual nodes use the same packed kernels and geometry as the combined
// launch. The caller orders gate-up before the down projection.
template <packed::MatrixFormat Format, unsigned TopK, class Entry = EntryComplete>
inline cudaError_t launch_gate_up(const Args& args, cudaStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows)) return cudaErrorInvalidValue;
    const auto columns = gate_blocks(args.intermediate, detail::kGateColumns);
    detail::gate_up<Format, TopK, detail::kGateColumns, Entry><<<dim3(columns, rows * TopK), detail::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

template <packed::MatrixFormat Format, unsigned TopK, class Entry = EntryComplete>
inline cudaError_t launch_down_combine(const Args& args, cudaStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows)) return cudaErrorInvalidValue;
    const auto blocks = down_blocks(args.hidden, detail::kDownColumns);
    detail::down_combine<Format, TopK, detail::kDownColumns, Entry><<<dim3(blocks, rows), detail::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

// down_combine with a tail (r1_tail.hpp) that consumes the stored combined row inside the same launch.
template <packed::MatrixFormat Format, unsigned TopK, class Entry = EntryComplete, class Tail>
inline cudaError_t launch_down_combine_tail(const Args& args, const Tail& tail, cudaStream_t stream,
                                            unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows) || !admits_tail(args, tail, rows, TopK)) return cudaErrorInvalidValue;
    const auto blocks = down_blocks(args.hidden, detail::kDownColumns);
    detail::down_combine<Format, TopK, detail::kDownColumns, Entry, Tail><<<dim3(blocks, rows), detail::kThreads, 0, stream>>>(args, tail);
    return cudaGetLastError();
}

// gate_up that routes its rows itself (r1_route::Inline): no checked route launch runs before it, and the route it
// publishes is the one down_combine reads.
template <packed::MatrixFormat Format, unsigned TopK, class Entry = EntryComplete, class Route>
inline cudaError_t launch_gate_up_routed(const Args& args, const Route& route, cudaStream_t stream,
                                         unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows) || !admits_route<TopK>(args, route, rows)) return cudaErrorInvalidValue;
    const auto columns = gate_blocks(args.intermediate, detail::kGateColumns);
    detail::gate_up<Format, TopK, detail::kGateColumns, Entry, Route>
        <<<dim3(columns, rows * TopK), detail::kThreads, 0, stream>>>(args, route);
    return cudaGetLastError();
}

// Expert-union rows launches (decode verify windows): gate_up / down_combine of `rows` <= MaxRows routed rows that read
// each distinct expert once. step: optional step block; rows it marks invalid take no part. The gate grid has one
// group per possible distinct expert (r1_route::union_groups); groups past the union exit at once.
template <packed::MatrixFormat Format, unsigned TopK, unsigned MaxRows, class Entry = EntryComplete>
inline cudaError_t launch_gate_up_union(const Args& args, unsigned rows, cudaStream_t stream,
                                        const exec::StepBlock* step = nullptr) {
    if (!fits_union<Format, TopK, MaxRows>(args, rows)) return cudaErrorInvalidValue;
    constexpr auto geometry = kUnionGeometry<MaxRows>;
    const auto columns = gate_blocks(args.intermediate, geometry.gate_columns);
    const auto groups = r1_route::union_groups(rows, TopK, args.experts);
    detail::gate_up_union<Format, TopK, MaxRows, geometry.gate_columns, Entry>
        <<<dim3(columns, groups), detail::kThreads, 0, stream>>>(args, rows, step);
    return cudaGetLastError();
}

template <packed::MatrixFormat Format, unsigned TopK, unsigned MaxRows, class Entry = EntryComplete, class Route>
inline cudaError_t launch_gate_up_union_routed(const Args& args, const Route& route, unsigned rows,
                                               cudaStream_t stream, const exec::StepBlock* step = nullptr) {
    if (!fits_union<Format, TopK, MaxRows>(args, rows) || !admits_route<TopK>(args, route, rows))
        return cudaErrorInvalidValue;
    constexpr auto geometry = kUnionGeometry<MaxRows>;
    const auto columns = gate_blocks(args.intermediate, geometry.gate_columns);
    const auto groups = r1_route::union_groups(rows, TopK, args.experts);
    detail::gate_up_union<Format, TopK, MaxRows, geometry.gate_columns, Entry, Route>
        <<<dim3(columns, groups), detail::kThreads, 0, stream>>>(args, rows, step, route);
    return cudaGetLastError();
}

template <packed::MatrixFormat Format, unsigned TopK, unsigned MaxRows, class Entry = EntryComplete>
inline cudaError_t launch_down_union(const Args& args, unsigned rows, cudaStream_t stream,
                                     const exec::StepBlock* step = nullptr) {
    if (!fits_union<Format, TopK, MaxRows>(args, rows)) return cudaErrorInvalidValue;
    constexpr auto geometry = kUnionGeometry<MaxRows>;
    const auto blocks = down_blocks(args.hidden, geometry.down_columns);
    detail::down_union<Format, TopK, MaxRows, geometry.down_columns, Entry>
        <<<blocks, detail::kThreads, 0, stream>>>(args, rows, step);
    return cudaGetLastError();
}
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

// The same plan with a tail folded into the down launch (r1_tail.hpp): two launches instead of two plus the tail nodes.
template <packed::MatrixFormat Format, unsigned TopK, class Tail>
#if defined(__HIPCC__)
inline hipError_t launch_tail(const Args& args, const Tail& tail, hipStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows) || !admits_tail(args, tail, rows, TopK)) return hipErrorInvalidValue;
#else
inline cudaError_t launch_tail(const Args& args, const Tail& tail, cudaStream_t stream, unsigned rows = 1U) {
    if (!fits_rows<Format, TopK>(args, rows) || !admits_tail(args, tail, rows, TopK)) return cudaErrorInvalidValue;
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
    detail::down_combine<Format, TopK, detail::kDownColumns, EntryComplete, Tail><<<dim3(down_grid, rows), threads, 0, stream>>>(args, tail);
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}
#endif

} // namespace strix::moe_core::microscaled_r1
