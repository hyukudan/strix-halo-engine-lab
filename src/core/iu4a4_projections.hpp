#pragma once
// A4W4 trait of a projection call (operand class of the shared A4W4 dense family core/iu4a4_dense.hpp, contract
// dense-iu4a4-rot-v1; a lossy class: an engine selects it only through a FAST / TURBO lever behind its gate).
//
// The projections of one call that read the same input rows (e.g. an attention block's Q / K / V / gate projections of
// the hidden rows) run as ONE quantise pass of the input + ONE GEMM over one concatenated rotated weight image, every
// projection's 16-column tiles stored into its own row-major output (Layout, core/iu4a4_projection_layout.hpp). One
// launch over the whole call keeps the grid wide: separate launches of 1,024..3,072 outputs fill too few CTAs at
// prefill heights.
//
// Pieces (all of them policies of the shared family's kernels, no kernel of their own):
//   Bf16Segments  load-time weight source of the family's requantise kernel: image column c = row c - first_col[s] of
//                 segment s's BF16 matrix (pitched rows), requantised by iu4a4_rot::requantise_values.
//   SplitCols     Cols policy of the family's GEMM: column tile i -> the owning segment's output.
//   RoundBf16Store  Store policy: the GEMM's output value (acc * base) rounded to BF16 (nearest even, NaN stays NaN)
//                 and stored widened to FP32, for calls whose projections publish BF16-valued outputs.
//   Bf16SigmoidGate  map2 functor of the family's quantiser: the gated product bf16(bf16(x) * sigmoid(bf16(g)))
//                 (core/row_transforms.hpp Bf16SigmoidProduct, same operations), so a gated input is quantised without
//                 storing the FP32 product.
#include "core/iu4a4_dense.hpp"
#include "core/iu4a4_projection_layout.hpp"
#include "strix/attn/group_indexer.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::core::iu4a4_projections {
namespace ad = iu4a4_dense;
namespace rot = iu4a4_rot;
static_assert(kBlockInputs == rot::kBlock && kMaxInputs == rot::kMaxInputs, "layout constants mirror the contract");

struct Bf16Segments {
    const std::uint16_t* weights[kMaxSegments]{};  // segment s: row r of its matrix at weights[s] + r * pitch[s]
    std::size_t pitch[kMaxSegments]{};            // u16 elements between rows (>= layout.in)
    Layout layout{};
    __device__ __forceinline__ float requantise(unsigned col, unsigned b, unsigned n, std::uint8_t* dst) const {
        const unsigned s = layout.segment(col);
        const std::uint16_t* row = weights[s] + std::size_t(col - layout.first_col[s]) * pitch[s] + std::size_t(b) * rot::kBlock;
        const auto value = [row](unsigned i) { return __builtin_bit_cast(float, std::uint32_t{row[i]} << 16U); };
        return rot::requantise_values(value, b, col, n, dst);
    }
};

struct SplitCols {
    float* y[kMaxSegments]{};  // segment s output [rows][layout.outputs(s)]
    Layout layout{};
    __device__ __forceinline__ f16_dense_tall::ColumnTile tile(unsigned i) const {
        const unsigned c = i * kTileColumns, s = layout.segment(c);
        const bool live = c < layout.n;
        return {nullptr, live ? y[s] + (c - layout.first_col[s]) : y[0], layout.outputs(s), live};
    }
};

struct RoundBf16Store {
    using Tile = f16_dense_tall::RawStore::Tile;
    __device__ __forceinline__ Tile open(unsigned i, const f16_dense_tall::ColumnTile& ct, unsigned c) const {
        return f16_dense_tall::RawStore{}.open(i, ct, c);
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float v) const {
        t.y[static_cast<std::size_t>(r) * t.stride + t.c] = attn::group_indexer::bf16_round(v);
    }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

struct Bf16SigmoidGate {
    __device__ __forceinline__ static float apply(float input, float logit) {
        const float gate = attn::group_indexer::bf16_round(logit);
        const float sigmoid = __fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-gate)));
        return attn::group_indexer::bf16_round(__fmul_rn(attn::group_indexer::bf16_round(input), sigmoid));
    }
};

// A call's rotated weight image (built once at load by enqueue_build_image) and its layout.
struct Image {
    const std::uint8_t* codes = nullptr;  // layout.weight_code_bytes()
    const float* step = nullptr;          // layout.weight_step_bytes()
    Layout layout{};
};

// The quantised input of a call: Act buffers sized for capacity_rows rows of `in` inputs (16-byte aligned).
struct Workspace {
    std::uint8_t* codes = nullptr;
    std::uint32_t* u = nullptr;
    float* base = nullptr;
    unsigned capacity_rows = 0U, in = 0U;
};
[[nodiscard]] inline std::size_t workspace_code_bytes(unsigned rows, unsigned in) { return ad::act_code_bytes(rows, in); }
[[nodiscard]] inline std::size_t workspace_u_bytes(unsigned rows, unsigned in) { return ad::act_u_bytes(rows, in); }
[[nodiscard]] inline std::size_t workspace_base_bytes(unsigned rows) { return ad::act_base_bytes(rows); }

namespace detail {
inline bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16U == 0U; }
}  // namespace detail

// Load time: the image of `src.layout` into codes / step (16-byte aligned device buffers of the layout's byte sizes).
inline hipError_t enqueue_build_image(const Bf16Segments& src, std::uint8_t* codes, float* step, hipStream_t stream) {
    const Layout& l = src.layout;
    if (!l.admitted() || !ad::supported(l.in, l.n) || !codes || !step || !detail::aligned16(codes) || !detail::aligned16(step))
        return hipErrorInvalidValue;
    for (unsigned s = 0; s < l.count; ++s)
        if (!src.weights[s] || src.pitch[s] < l.in) return hipErrorInvalidValue;
    ad::enqueue_requantise_source(src, l.in, l.n, codes, step, stream);
    return hipGetLastError();
}

// Whether a call of `rows` rows of image.layout fits the workspace and the family's limits (asked before enqueueing).
[[nodiscard]] inline bool admits(const Image& image, unsigned rows, const Workspace& ws, bool gated) {
    const Layout& l = image.layout;
    return l.admitted() && ad::supported(l.in, l.n) && (!gated || ad::map2_supported(l.in)) && rows > 0U &&
           rows <= ws.capacity_rows && ws.in == l.in && image.codes && image.step && ws.codes && ws.u && ws.base &&
           detail::aligned16(image.codes) && detail::aligned16(image.step) && detail::aligned16(ws.codes) &&
           detail::aligned16(ws.u) && detail::aligned16(ws.base);
}

// One call: y_s[r][o] for every segment s, r < rows, from input rows x0 (row stride x_stride floats, a multiple of 4,
// 16-byte aligned). Gate (non-null x1, same stride): the quantised value is Bf16SigmoidGate(x0, x1). outputs[s] is
// segment s's output [rows][layout.outputs(s)]; outputs must not overlap the input, the workspace or each other.
inline hipError_t enqueue_call(const float* x0, const float* x1, unsigned x_stride, unsigned rows, const Image& image,
                               std::span<float* const> outputs, bool round_bf16, const Workspace& ws, hipStream_t stream) {
    const Layout& l = image.layout;
    if (!x0 || !detail::aligned16(x0) || (x1 && !detail::aligned16(x1)) || x_stride < l.in || x_stride % 4U != 0U ||
        !admits(image, rows, ws, x1 != nullptr) || outputs.size() != l.count)
        return hipErrorInvalidValue;
    SplitCols cols{};
    cols.layout = l;
    for (unsigned s = 0; s < l.count; ++s) {
        if (!outputs[s]) return hipErrorInvalidValue;
        cols.y[s] = outputs[s];
    }
    const ad::Act act{ws.codes, ws.u, ws.base, rows, l.in};
    if (x1) ad::enqueue_quantise_map2<Bf16SigmoidGate>(x0, x1, x_stride, act, stream);
    else ad::enqueue_quantise(x0, x_stride, act, stream);
    const ad::Weights w{image.codes, image.step, l.in, l.n};
    if (round_bf16) ad::enqueue_gemm(w, act, cols, stream, RoundBf16Store{});
    else ad::enqueue_gemm(w, act, cols, stream);
    return hipGetLastError();
}

}  // namespace strix::core::iu4a4_projections
