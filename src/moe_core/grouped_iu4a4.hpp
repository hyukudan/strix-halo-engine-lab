#pragma once
// Native HIP launch binding for the SDK-free grouped IU4A4 contract.
#if defined(__HIPCC__)
#include "core/iu4a4_dense.hpp"
#include "moe_core/expert_activation.hpp"
#endif
#include "moe_core/grouped_iu4a4_contract.hpp"

namespace strix::moe_core::grouped_iu4a4 {

#if defined(__HIPCC__)
namespace dense = strix::core::iu4a4_dense;

// GroupCta source of a TileRows plan. Tile t: weights / rows of tiles[t]; its first assignment row is
// (index_tiles[t].input - index_base) / index_tiles[t].input_row (the gate/up tiles locate the assignment rows of both
// projections: the down tiles are the same rows, row for row); image row = map ? map[a] : a. Outputs go to
// out + a * out_row (assignment order: the destination derives from the admitted base, pitch and assignment interval;
// the tile's own output fields are not read). admit(t) is the bounds rule: the weights are an expert slot of the image
// (base + e * stride, e < experts), the index row is a whole row of the base, the rows equal the index tile's, are at
// most max_rows and stay inside index_rows. own(t) is the same rule on the tile's own input address and pitch (the
// interval the planner gave that tile). The validation launch applies admit() to every active gate/up and down tile and
// requires each down tile to name its gate/up tile's expert and, through own(), its assignment interval, before any
// compute launch; bind(t) applies admit() again and binds no rows (stores nothing) where it fails. active(t) reads the
// pass verdict word, which no launch of the compute phase writes, so every wave of a CTA sees one value.
struct TileSource {
    const TileRows* tiles{};
    const TileRows* index_tiles{};
    const std::uint16_t* index_base{};
    const std::uint32_t* map{};
    const unsigned* tile_count{};
    const unsigned* verdict{};
    const std::uint8_t* codes_from{};
    std::size_t codes_stride{};
    ExpertImage image{};
    unsigned in{}, n{};
    float* out{};
    std::size_t out_row{};
    unsigned index_rows{}, max_rows{};
    struct Bound {
        std::size_t expert, first;
        unsigned rows;
        bool ok;
    };
    __device__ __forceinline__ bool active(unsigned t) const {
        return (!verdict || *verdict == 0U) && (!tile_count || t < *tile_count);
    }
    __device__ __forceinline__ Bound admit(unsigned t) const { return admit_rows(tiles[t], index_tiles[t]); }
    __device__ __forceinline__ Bound own(unsigned t) const { return admit_rows(tiles[t], tiles[t]); }
    __device__ __forceinline__ Bound admit_rows(const TileRows tile, const TileRows index) const {
        const auto wa = reinterpret_cast<std::uintptr_t>(tile.weights.codes), wb = reinterpret_cast<std::uintptr_t>(codes_from);
        const auto ia = reinterpret_cast<std::uintptr_t>(index.input), ib = reinterpret_cast<std::uintptr_t>(index_base);
        const std::size_t e = wa >= wb ? (wa - wb) / codes_stride : ~std::size_t{0};
        const std::size_t at = ia >= ib ? (ia - ib) / sizeof(std::uint16_t) : ~std::size_t{0};
        const std::size_t first = index.input_row != 0U ? at / index.input_row : ~std::size_t{0};
        const bool ok = wa >= wb && (wa - wb) % codes_stride == 0U && e < image.experts && ia >= ib &&
                        (ia - ib) % sizeof(std::uint16_t) == 0U && index.input_row != 0U && at % index.input_row == 0U &&
                        tile.rows == index.rows && tile.rows <= max_rows && first <= index_rows &&
                        tile.rows <= index_rows - first;
        return {e, first, tile.rows, ok};
    }
    __device__ __forceinline__ dense::GroupCta bind(unsigned t) const {
        const Bound b = admit(t);
        if (!b.ok) return {};  // no rows: the CTA leaves before its first barrier
        const dense::Weights w{image.codes + b.expert * image.code_bytes, image.steps + b.expert * image.step_floats, in, n};
        return {w, map, static_cast<unsigned>(b.first), b.rows, out + b.first * out_row, out_row, nullptr};
    }
};

namespace detail {
// Validation launch of a pass (one CTA, stream-ordered before every compute launch; Pass::fault). A planner failure is
// copied into the verdict word. Otherwise kFaultBounds is set when the device count is zero or past the plan's tiles,
// an active gate/up or down tile fails admit(), a down tile names another expert than its gate/up tile or (own(): its
// own input address and pitch) another assignment interval, the active tiles do not cover [0, assignments) exactly once
// (row sum and pairwise disjoint intervals; admit() keeps each inside), or a map entry names a token past token_rows.
__global__ __launch_bounds__(1024) void validate_kernel(TileSource gate, TileSource down, const unsigned* failure,
                                                        unsigned tiles, unsigned assignments, unsigned token_rows, unsigned* verdict) {
    __shared__ unsigned s_first[kMaxPassTiles], s_rows[kMaxPassTiles];
    __shared__ unsigned s_bad;
    __shared__ unsigned long long s_sum;
    const unsigned tid = threadIdx.x;
    const unsigned planned = failure ? *failure : 0U;  // one value for the CTA (no launch of the pass writes it yet)
    if (planned != 0U) {
        if (tid == 0U) atomicOr(verdict, planned);
        return;
    }
    const unsigned count = gate.tile_count ? *gate.tile_count : tiles;
    const unsigned live = count < tiles ? count : tiles;
    if (tid == 0U) s_bad = count == 0U || count > tiles ? 1U : 0U, s_sum = 0ULL;
    __syncthreads();
    for (unsigned t = tid; t < live; t += blockDim.x) {
        const TileSource::Bound g = gate.admit(t), d = down.admit(t), o = down.own(t);
        const bool ok = g.ok && d.ok && o.ok && d.expert == g.expert && o.first == g.first;
        if (!ok) atomicOr(&s_bad, 1U);
        s_first[t] = ok ? static_cast<unsigned>(g.first) : 0U;
        s_rows[t] = ok ? g.rows : 0U;
        atomicAdd(&s_sum, static_cast<unsigned long long>(ok ? g.rows : 0U));
    }
    for (unsigned a = tid; a < assignments; a += blockDim.x)
        if (gate.map[a] >= token_rows) atomicOr(&s_bad, 1U);
    __syncthreads();
    if (s_sum != assignments) {
        if (tid == 0U) atomicOr(verdict, kFaultBounds);
        return;
    }
    for (unsigned t = tid; t < live; t += blockDim.x) {
        const unsigned f = s_first[t], r = s_rows[t];
        for (unsigned u = t + 1U; r != 0U && u < live; ++u)
            if (s_rows[u] != 0U && f < s_first[u] + s_rows[u] && s_first[u] < f + r) atomicOr(&s_bad, 1U);
    }
    __syncthreads();
    if (tid == 0U && s_bad != 0U) atomicOr(verdict, kFaultBounds);
}
}  // namespace detail

// Paired epilogue: the incumbent expert activation (moe::SoftcappedSwiglu, BF16 result) stored as FP32.
struct SwigluOut {
    static constexpr bool kPaired = true;
    float gate_cap, up_cap;
    __device__ __forceinline__ float operator()(float gate, float up) const {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
        const moe::SoftcappedSwiglu<expert_activation::detail::DeviceMath> swiglu{gate_cap, up_cap};
        return weight::scalar_float::u2f(std::uint32_t{expert_activation::detail::activate_bf16(swiglu, gate, up)} << 16U);
    }
};

// Load time: expert matrix -> rotated image slot (codes, steps) on stream s.
template <packed::MatrixFormat Format>
inline void enqueue_requantise_expert(const packed::PackedMatrix& matrix, unsigned pair_rows, unsigned in, unsigned n,
                                      std::uint8_t* codes, float* steps, hipStream_t s,
                                      rot::WeightRule rule = rot::WeightRule::amax) {
    dense::enqueue_requantise_source(ExpertSource<Format>{matrix, pair_rows, rule}, in, n, codes, steps, s);
}

static_assert(kFaultBounds == dense::kBoundsFault, "one fault bit for the class and its core");

// The pass (see Pass) on stream s: five launches, each submitted only after the previous one was accepted. Refuses
// (hipErrorInvalidValue, nothing enqueued) what refusal() names and grids the grouped launch cannot take. Order:
// validation (verdict word, see Pass::fault), token image, gate/up (paired SwiGLU -> h), h image, down (-> y). The
// compute launches only read the verdict; when it is set none of them stores (both activation images included), so a
// refused pass leaves the workspace and the outputs as they were and no row a tile did not write is read.
inline hipError_t launch(const Pass& p, hipStream_t s) {
    static_assert(kMaxRowInputs == dense::detail::kCtaInputs, "the verdict-gated images are the one-CTA-per-row form");
    const unsigned gate_width = 2U * p.intermediate;
    if (refusal(p) != nullptr || !dense::grouped_supported(p.hidden, gate_width, true, p.tiles, p.max_tile_rows) ||
        !dense::grouped_supported(p.intermediate, p.hidden, false, p.tiles, p.max_tile_rows))
        return hipErrorInvalidValue;
    const auto carve = [](std::uint8_t*& at, unsigned rows, unsigned in) {
        dense::Act a{};
        a.rows = rows;
        a.in = in;
        a.codes = at;
        at += align256(dense::act_code_bytes(rows, in));
        a.u = reinterpret_cast<std::uint32_t*>(at);
        at += align256(dense::act_u_bytes(rows, in));
        a.base = reinterpret_cast<float*>(at);
        at += align256(dense::act_base_bytes(rows));
        return a;
    };
    std::uint8_t* at = p.act;
    const dense::Act tokens = carve(at, p.rows, p.hidden), inner = carve(at, p.assignments, p.intermediate);
    const TileSource gate{p.gate_up, p.gate_up, p.index_base, p.assignment_rows, p.tile_count, p.fault,
                          p.images->gate_up_codes, p.images->gate_up_stride, p.images->gate_up, p.hidden, gate_width,
                          p.h, p.intermediate, p.assignments, p.max_tile_rows};
    const TileSource down{p.down, p.gate_up, p.index_base, nullptr, p.tile_count, p.fault,
                          p.images->down_codes, p.images->down_stride, p.images->down, p.intermediate, p.hidden,
                          p.y, p.hidden, p.assignments, p.max_tile_rows};
    STRIX_HIP_LAUNCH_GGL(Rdna3Layout, detail::validate_kernel, dim3(1), dim3(1024), 0, s, gate, down, p.failure, p.tiles,
                         p.assignments, p.rows, p.fault);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_quantise_rows_gated(p.fault, p.tokens, p.hidden, tokens, s);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_grouped_gemm(gate, p.tiles, p.max_tile_rows, gate_width, tokens, s, SwigluOut{p.gate_cap, p.up_cap});
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_quantise_rows_gated(p.fault, p.h, p.intermediate, inner, s);
    if (const auto e = hipGetLastError(); e != hipSuccess) return e;
    dense::enqueue_grouped_gemm(down, p.tiles, p.max_tile_rows, p.hidden, inner, s);
    return hipGetLastError();
}
#endif

}  // namespace strix::moe_core::grouped_iu4a4

