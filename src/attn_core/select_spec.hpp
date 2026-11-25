#pragma once

#include <cstddef>
#include <cstdint>

namespace strix::cuda::core::select {

enum class KeyOrder : std::uint8_t { ordered, canonical_float, raw_float, eligible_float };

// A pick can emit a group ID, expand groups into tokens with a partial tail,
// or reserve the newest (possibly incomplete) block. Keys and IDs are in
// chronological order. A join supplies global IDs in that same order.
struct PickSpec {
    unsigned top_k;
    unsigned expansion = 1;
    KeyOrder order = KeyOrder::canonical_float;
    bool force_newest = false;
};

enum class Pool : std::uint8_t { none, source, user };

struct PickRow {
    const float* scores = nullptr;
    const std::uint32_t* keys = nullptr;
    const std::uint32_t* ids = nullptr;
    unsigned groups = 0, tail = 0;
    std::uint32_t* selected = nullptr;
    std::uint32_t* selected_keys = nullptr; // Optional ordered keys, one per selected group.
    std::uint32_t* count = nullptr;
    float* mask_scores = nullptr;
    Pool pool = Pool::none;
    unsigned pool_width = 0, pool_top = 0;
    std::uint8_t* candidates = nullptr;
    std::uint32_t* pool_keys = nullptr;
};

// Query codes contain four signed, doubled E2M1 values per word. Cache entries
// contain 64 packed E2M1 bytes then four E8M0 exponent bytes.
struct PackedQuery {
    int codes[32][32];
    float scales[32][4];
    float weights[32];
};

struct PackedFp8Query {
    alignas(4) std::uint8_t codes[32][128];
    float weights[32];
};

struct ScoreRow {
    const float* query = nullptr;
    const float* keys = nullptr;
    const float* weights = nullptr;
    const PackedQuery* packed_query = nullptr;
    const std::uint8_t* packed_keys = nullptr;
    const PackedFp8Query* fp8_query = nullptr;
    const float* key_scales = nullptr;
    const std::uint16_t* bf16_keys = nullptr;
    unsigned visible = 0;
    float* scores = nullptr;
    std::uint32_t* ordered = nullptr;
    unsigned* nonfinite = nullptr;
};

// These are numerical forms. A different scheduling tile may use the same
// form only when it preserves the declared reduction and each write guard.
enum class ScoreForm : std::uint8_t {
    relu_tree, weighted_fma, weighted_bf16, weighted_bf16x3, weighted_fp4, weighted_fp8, block_max, block_max_tiled,
};

struct ScoreSpec {
    unsigned top_k;
    unsigned heads;
    unsigned dim;
    unsigned block_width = 1;
};

// All strides count BF16 elements. The caller owns the cache extents and
// keeps the sequence, layer and position indices within those extents.
struct StridedScoreCache {
    const std::uint16_t* base;
    std::size_t sequence_stride, layer_stride, row_stride;

#if defined(__CUDACC__) || defined(__HIPCC__)
    __host__ __device__
#endif
    constexpr std::size_t offset(unsigned sequence, unsigned layer, std::size_t position) const {
        return sequence * sequence_stride + layer * layer_stride + position * row_stride;
    }
};

struct ScoreSequence {
    unsigned first_row, first_position;
};

struct ScoreTile {
    unsigned first_row, rows;
};

inline constexpr unsigned strided_score_width = 128;
inline constexpr unsigned strided_score_tile_rows = 32;
inline constexpr unsigned strided_score_tile_lanes = 8;
inline constexpr unsigned strided_score_warps = 4;

// Passed by value at launch. The device row map, sequence origins, and tile
// list can be built once per row group and reused across layers. Rows in a
// sequence are consecutive from its origin. Tiled launches require a tile
// partition of those rows, with each tile wholly within one sequence.
struct StridedScoreArgs {
    StridedScoreCache cache;
    const unsigned* row_sequence;
    const ScoreSequence* sequences;
    const float* query; // [row][head][128], 16-byte aligned for the tiled form
    float* scores;    // [row][head][max_groups]
    unsigned layer, row_count, max_groups; // max_groups bounds all visible blocks
    const ScoreTile* tiles = nullptr;
    unsigned tile_count = 0;
};

namespace detail {

struct StridedScoreGrid {
    unsigned x = 0, y = 0, z = 0, threads = 0;
};

template <ScoreForm Form>
inline bool pack_strided_score_grid(const StridedScoreArgs& args, ScoreSpec spec, StridedScoreGrid& grid) {
    static_assert(Form == ScoreForm::block_max || Form == ScoreForm::block_max_tiled);
    grid = {};
    if (args.row_count == 0 || args.max_groups <= spec.top_k) return true;
    if (!args.cache.base || !args.row_sequence || !args.sequences || !args.query || !args.scores ||
        args.cache.row_stride < strided_score_width || spec.dim != strided_score_width ||
        spec.block_width != strided_score_width || spec.heads == 0) return false;

    constexpr unsigned max_grid_x = 2147483647U, max_grid_yz = 65535U;
    if constexpr (Form == ScoreForm::block_max_tiled) {
        if (!args.tiles || args.tile_count == 0 || args.tile_count > args.row_count || args.tile_count > max_grid_x ||
            spec.heads > max_grid_yz || args.max_groups > max_grid_yz ||
            reinterpret_cast<std::uintptr_t>(args.query) % 16U != 0) return false;
        grid = {args.tile_count, spec.heads, args.max_groups, strided_score_tile_rows * strided_score_tile_lanes};
    } else {
        const auto items_per_row = std::uint64_t{spec.heads} * args.max_groups;
        if (args.row_count > std::uint64_t{max_grid_x} * strided_score_warps / items_per_row) return false;
        const auto blocks = (args.row_count * items_per_row + strided_score_warps - 1U) / strided_score_warps;
        grid = {static_cast<unsigned>(blocks), 1, 1, 32U * strided_score_warps};
    }
    return true;
}

} // namespace detail

} // namespace strix::cuda::core::select
