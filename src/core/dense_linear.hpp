#pragma once
// Dense decode projection binding of the shared small-row sweep (gemv_rows_direct.inc) to the shared
// BF16 / Q8_0 chunk decoders (weight_decode.hpp). out[r][o] = sum_k W[o][k] * x[r][k], rows 1..16.
// Epilogue kinds: store, or SiLU(gate) * up over two matrices that share every activation read.
// Home: shared (core): no model names; qwen35 binds it, Gemma4/Qwen4 can later.
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix::core {
// Row sweeps of a dense projection. rows <= 16 (decode / verify / draft windows): ONE call of the real count; the bound
// kernel pads to its 1/2/4/8/16 bucket inside, reads pad rows only inside the caller's 16-row scratch and stores only
// rows < count (dense_linear::kernel `nrows`, dense_linear_iu4 `nrows`). Per-row results are bucket-independent, so the
// stored rows are bitwise those of the power-of-two split; the split re-streamed every weight per sweep (Strix 27B IU4
// verify rows 3 / 6 / 7 / 12 = 129 / 133 / 191 / 144 ms vs rows 4 / 8 / 16 = 78 / 85 / 132 ms, hip_dense_dec4).
// rows > 16 (exact wide prefill windows): 16-row sweeps, then the tail split in powers of two so no call rounds a
// logical tail up into storage past the window.
template <class F>
void dense_row_sweeps(unsigned rows, F&& run) {
    if (rows == 0U) return;
    if (rows <= 16U) return static_cast<void>(run(0U, rows));
    unsigned offset = 0;
    for (unsigned tile = 16; tile; tile /= 2)
        while (rows - offset >= tile) {
            run(offset, tile);
            offset += tile;
        }
}

// Non-owning execution scratch view; its backend context owns the allocations.
struct DenseWideScratch {
    std::int8_t* codes = nullptr;
    float *deltas = nullptr, *g = nullptr, *u = nullptr;
};

// Host admission for embedding/dense inputs, shared by backend adapters.
inline std::uint32_t parse_u32(std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned 32-bit decimal integer");
    return value;
}
inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {
    std::size_t result = 1;
    for (auto n : factors) {
        if (n && result > std::numeric_limits<std::size_t>::max() / n)
            throw std::invalid_argument("allocation size overflow");
        result *= n;
    }
    return result;
}
inline void require_capacity(std::size_t pos, std::size_t rows, std::size_t capacity) {
    if (pos > capacity || rows > capacity - pos)
        throw std::invalid_argument("max_positions exceeded");
}
template <class Token>
void validate_token_ids(const Token* ids, std::size_t count, std::uint32_t vocab) {
    if (count && !ids) throw std::invalid_argument("null token IDs");
    for (std::size_t i = 0; i < count; ++i)
        if (static_cast<std::uint64_t>(ids[i]) >= vocab)
            throw std::invalid_argument("token ID outside vocabulary");
}
} // namespace strix::core

#include "strix/decode_projection_group.hpp"

namespace strix::core::dense_linear {
// One launch for up to kTightGroupMax projections of one input (decode rows), indexed by the shared projection-group
// table (strix/decode_projection_group.hpp; the backend-neutral rows plan emits the same index per launch). Block b of
// the launch belongs to the projection p with first[p] <= b < first[p + 1] (decode::locate) and runs p's
// single-projection body as its block b - first[p], so every stored value is computed by the same instructions as in
// p's own launch (bitwise equal).
inline constexpr unsigned kTightGroupMax = 6U;  // 6 = the 3-column-role attention prepare (q, relative, sink, gate, k, v) of one layer
using TightBlocks = decode::GroupIndex<kTightGroupMax>;

// Member p gets ceil(outputs[p] / block_outputs) blocks. outputs[p] > 0 for p < count, 1 <= count <= kTightGroupMax.
// Returns false (plan untouched) on an empty member, a count outside 1..kTightGroupMax or a block total past 32 bits.
constexpr bool plan_tight_blocks(const unsigned* outputs, unsigned count, unsigned block_outputs, TightBlocks& plan) {
    if (count == 0U || count > kTightGroupMax || block_outputs == 0U) return false;
    TightBlocks next{};
    for (unsigned p = 0; p < count; ++p) {
        if (outputs[p] == 0U) return false;
        const unsigned blocks = outputs[p] / block_outputs + (outputs[p] % block_outputs != 0U ? 1U : 0U);
        if (blocks > 0xffffffffU - next.first[p]) return false;
        next.first[p + 1U] = next.first[p] + blocks;
    }
    next.count = count;
    plan = next;
    return true;
}
} // namespace strix::core::dense_linear

#if defined(__HIPCC__)
#include "gemv_rows.hpp"
#include "strix/attn/group_indexer.hpp"
#include "weight_decode.hpp"

namespace strix::core::dense_linear {

enum class Epilogue : std::uint8_t { store, silu_mul, store_tight_rows, store_tight_rows_bf16, store_mapped };

struct View {
    const std::uint8_t* data;
    unsigned inputs, outputs;
    std::size_t row_bytes;
};

__device__ __forceinline__ float wave_sum32(float v) {
    // Fixed butterfly (xor 16,8,4,2,1): identical bits for every schedule.
#pragma unroll
    for (unsigned m = 16U; m > 0U; m >>= 1U)
        v += __shfl_xor(v, int(m), 32);
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0F + expf(-x)); }

// 4 waves per block, T outputs per wave. grid = ceil(outputs / (4 * T)). Computes the Rows bucket (x must hold Rows
// rows for the original epilogues). The tight-row variant aliases unused reads to row 0.
// Every variant stores only rows < nrows. store_mapped (Rows 1) runs output o of the launch on weight row ids[o]
// (ids == nullptr: row o) with the same per-output body; it stores the FP32 sum at out[o].
template <unsigned Rows, unsigned T, weight_decode::Format Format, Epilogue Ep>
__device__ __forceinline__ void body(View w, View up, const float* x, float* out, unsigned nrows, unsigned block,
                                     const std::uint32_t* ids = nullptr) {
    const unsigned first = (block * 4U + threadIdx.x / 32U) * T, lane = threadIdx.x % 32U;
    constexpr bool tight = Ep == Epilogue::store_tight_rows || Ep == Epilogue::store_tight_rows_bf16;
    constexpr unsigned Streams = Ep == Epilogue::silu_mul ? 2U : 1U;
    const auto inputs = w.inputs, outputs = w.outputs;
#define GEMV_ROWS_WEIGHT_ROW(stream, o)                                                                              \
    (Ep == Epilogue::store_mapped ? w.data + std::size_t(ids ? ids[o] : (o)) * w.row_bytes                          \
     : (stream)                   ? up.data + std::size_t(o) * up.row_bytes                                         \
                                  : w.data + std::size_t(o) * w.row_bytes)
#define GEMV_ROWS_CHUNK weight_decode::Chunk<Format>
#define GEMV_ROWS_LOAD_CHUNK(row, start) weight_decode::Chunk<Format>(row, start)
#define GEMV_ROWS_LANE_SUM(partial) wave_sum32(partial)
#define GEMV_ROWS_INPUT_ROW(row) (tight && (row) != 0U && (row) >= nrows ? 0U : (row))
#define GEMV_ROWS_STORE(row, o, sa, sb)                                                                              \
    if (lane == 0 && (o) < outputs && (row) < nrows)                                                               \
    out[std::size_t(row) * outputs + (o)] = (Ep == Epilogue::silu_mul ? silu(sa) * (sb) : \
        Ep == Epilogue::store_tight_rows_bf16 ? attn::group_indexer::bf16_round(sa) : (sa))
    if constexpr (Rows == 1U && T == 1U) {
        // One row, one output per wave: two chunk loads in flight per lane (cold K3072 x 2304 BF16: 84 -> 73 us).
#define GEMV_ROWS_K_UNROLL 2
#include "gemv_rows_direct.inc"
#undef GEMV_ROWS_K_UNROLL
    } else {
#include "gemv_rows_direct.inc"
    }
#undef GEMV_ROWS_WEIGHT_ROW
#undef GEMV_ROWS_CHUNK
#undef GEMV_ROWS_LOAD_CHUNK
#undef GEMV_ROWS_LANE_SUM
#undef GEMV_ROWS_INPUT_ROW
#undef GEMV_ROWS_STORE
}

template <unsigned Rows, unsigned T, weight_decode::Format Format, Epilogue Ep>
__global__ void __launch_bounds__(128) kernel(View w, View up, const float* x, float* out, unsigned nrows) {
    body<Rows, T, Format, Ep>(w, up, x, out, nrows, blockIdx.x);
}

// Mapped decode rows (Rows 1, one output per wave): the launch computes *count outputs, output o on weight row ids[o],
// each with the incumbent tight-row body (same lane split and order, so the same bits as the full launch's row ids[o]).
// *count > capacity selects ALL w.outputs rows in order (ids unused). The grid is fixed by the caller and the blocks
// loop over the blocks the count needs, so no host read is needed between producer and launch (graph-capture safe).
// out[o] receives the FP32 sum; it holds max(capacity, w.outputs) floats.
struct MappedRows {
    const std::uint32_t* ids;
    const std::uint32_t* count;
    unsigned capacity;
};

template <weight_decode::Format Format>
__global__ void __launch_bounds__(128) mapped_rows_kernel(View w, const float* x, float* out, MappedRows m) {
    const unsigned count = *m.count;
    const bool all = count > m.capacity;
    if (!all) w.outputs = count;
    const unsigned blocks = (w.outputs + 3U) / 4U;
    for (unsigned block = blockIdx.x; block < blocks; block += gridDim.x)
        body<1U, 1U, Format, Epilogue::store_mapped>(w, w, x, out, 1U, block, all ? nullptr : m.ids);
}

// Projections of one input x in one launch (TightBlocks). Single-matrix epilogues only.
struct TightGroup {
    View w[kTightGroupMax];
    float* out[kTightGroupMax];
    TightBlocks blocks;
};

template <unsigned Rows, unsigned T, weight_decode::Format Format, Epilogue Ep>
__global__ void __launch_bounds__(128) grouped_kernel(TightGroup g, const float* x, unsigned nrows) {
    static_assert(Ep != Epilogue::silu_mul, "grouped projections store one matrix each");
    unsigned projection = 0U;
    std::uint32_t block = 0U;
    decode::locate(g.blocks.first, g.blocks.count, blockIdx.x, projection, block);
    body<Rows, T, Format, Ep>(g.w[projection], g.w[projection], x, g.out[projection], nrows, block);
}

// Outputs per wave. A decode row (Rows 1) runs one output per wave: twice the waves in flight, and every output keeps
// its own chunk / element order, so the bits match any other tile (cold BF16, T2 -> T1: K2304 x 3072 69 -> 63 us).
template <unsigned Rows>
inline constexpr unsigned kTightTile = Rows == 1U ? 1U : Rows <= 8U ? 2U : 4U;

template <unsigned Rows, bool RoundBf16 = false>
inline void launch_bf16_tight_rows(View w, const float* input, float* output, unsigned rows, hipStream_t stream) {
    constexpr unsigned tile = kTightTile<Rows>;
    constexpr unsigned block_outputs = 4U * tile;
    constexpr auto epilogue = RoundBf16 ? Epilogue::store_tight_rows_bf16 : Epilogue::store_tight_rows;
    kernel<Rows, tile, weight_decode::Format::bf16, epilogue>
        <<<(w.outputs + block_outputs - 1U) / block_outputs, 128, 0, stream>>>(w, w, input, output, rows);
}

template <unsigned Rows, bool RoundBf16 = false>
inline hipError_t launch_bf16_tight_rows_group(const View* w, float* const* output, unsigned count,
                                               const float* input, unsigned rows, hipStream_t stream) {
    constexpr unsigned tile = kTightTile<Rows>;
    constexpr auto epilogue = RoundBf16 ? Epilogue::store_tight_rows_bf16 : Epilogue::store_tight_rows;
    if (count == 0U || count > kTightGroupMax) return hipErrorInvalidValue;
    TightGroup group{};
    unsigned outputs[kTightGroupMax]{};
    for (unsigned p = 0; p < count; ++p) {
        group.w[p] = w[p];
        group.out[p] = output[p];
        outputs[p] = w[p].outputs;
    }
    if (!plan_tight_blocks(outputs, count, 4U * tile, group.blocks)) return hipErrorInvalidValue;
    grouped_kernel<Rows, tile, weight_decode::Format::bf16, epilogue>
        <<<group.blocks.first[count], 128, 0, stream>>>(group, input, rows);
    return hipGetLastError();
}

// The caller validates the tight view and the live row count (1..16).
template <bool RoundBf16 = false>
inline hipError_t enqueue_bf16_tight_rows(View w, const float* input, float* output, unsigned rows, hipStream_t stream) {
    if (rows <= 1U) launch_bf16_tight_rows<1, RoundBf16>(w, input, output, rows, stream);
    else if (rows <= 2U) launch_bf16_tight_rows<2, RoundBf16>(w, input, output, rows, stream);
    else if (rows <= 4U) launch_bf16_tight_rows<4, RoundBf16>(w, input, output, rows, stream);
    else if (rows <= 8U) launch_bf16_tight_rows<8, RoundBf16>(w, input, output, rows, stream);
    else launch_bf16_tight_rows<16, RoundBf16>(w, input, output, rows, stream);
    return hipGetLastError();
}

// Same contract per view as enqueue_bf16_tight_rows; all views read the same input. One launch.
template <bool RoundBf16 = false>
inline hipError_t enqueue_bf16_tight_rows_group(const View* w, float* const* output, unsigned count,
                                                const float* input, unsigned rows, hipStream_t stream) {
    if (rows <= 1U) return launch_bf16_tight_rows_group<1, RoundBf16>(w, output, count, input, rows, stream);
    if (rows <= 2U) return launch_bf16_tight_rows_group<2, RoundBf16>(w, output, count, input, rows, stream);
    if (rows <= 4U) return launch_bf16_tight_rows_group<4, RoundBf16>(w, output, count, input, rows, stream);
    if (rows <= 8U) return launch_bf16_tight_rows_group<8, RoundBf16>(w, output, count, input, rows, stream);
    return launch_bf16_tight_rows_group<16, RoundBf16>(w, output, count, input, rows, stream);
}

} // namespace strix::core::dense_linear
#endif
