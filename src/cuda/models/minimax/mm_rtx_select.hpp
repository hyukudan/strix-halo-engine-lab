#pragma once

#include "attn_core/select_spec.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::minimax::rtxpf::cuda {

inline constexpr unsigned kIdxHeads = 4, kBlock = 128, kTopK = 16;

// The K/V and index-key caches belong to one sequence. Positions need not
// start at zero when the producer is processing a layer-cut block.
struct SeqView {
    std::uint32_t row0, rows, cap;
    std::uint16_t *k, *v, *ik;
    std::uint32_t pos0 = 0;
};

namespace select = ::strix::cuda::core::select;
inline constexpr select::PickSpec kMsaPick{kTopK, 1, select::KeyOrder::canonical_float, true};
inline constexpr select::ScoreSpec kMsaScore{kTopK, kIdxHeads, kBlock, kBlock};

// Only index keys use uniform sequence slots. K/V keep their own capacities.
inline constexpr select::StridedScoreCache msa_index_cache(const std::uint16_t* base,
                                                            std::size_t capacity, unsigned layers) {
    constexpr auto width = select::strided_score_width;
    const auto layer_stride = capacity * width;
    return {base, layers * layer_stride, layer_stride, width};
}

inline std::vector<select::ScoreSequence> msa_score_sequences(std::span<const SeqView> seqs) {
    std::vector<select::ScoreSequence> origins;
    origins.reserve(seqs.size());
    for (const auto& seq : seqs) origins.push_back({seq.row0, seq.pos0});
    return origins;
}

inline std::vector<select::ScoreTile> msa_score_tiles(std::span<const SeqView> seqs) {
    std::vector<select::ScoreTile> tiles;
    for (const auto& seq : seqs) {
        for (unsigned r = 0; r < seq.rows;) {
            const auto count = std::min(select::strided_score_tile_rows, seq.rows - r);
            tiles.push_back({seq.row0 + r, count});
            r += count;
        }
    }
    return tiles;
}

// One descriptor per (row, index head), matching the score and attend strides.
// The buffers are device addresses. The host only assembles their addresses.
inline std::vector<select::PickRow> msa_pick_rows(std::span<const SeqView> seqs, std::size_t rows,
                                                 unsigned max_blocks, const float* scores,
                                                 std::uint32_t* selected, std::uint32_t* counts) {
    std::vector<select::PickRow> picks(rows * kIdxHeads);
    for (const auto& seq : seqs) {
        for (std::uint32_t r = 0; r < seq.rows; ++r) {
            const auto groups = static_cast<unsigned>((std::uint64_t{seq.pos0} + r + kBlock) / kBlock);
            const auto first = (std::size_t{seq.row0} + r) * kIdxHeads;
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                const auto i = first + h;
                picks[i].scores = scores + i * max_blocks;
                picks[i].groups = groups;
                picks[i].selected = selected + i * kTopK;
                picks[i].count = counts + i;
            }
        }
    }
    return picks;
}

}  // namespace strix::minimax::rtxpf::cuda
