#pragma once

// Backend-neutral geometry of a wide multi-stream hyper-connection (HC) projection / output pair: every constant
// the tile kernels derive from the model's hidden size, stream count and low rank. The values come from the model
// description (strix/model/bind.hpp binds them from models/<model>.toml); no model number is written here.
//
//   width       Streams * Hidden: the flattened stream width the projection reduces over;
//   stages      width / 32: staged steps of 32 K values;
//   stage_pitch halves per staged LDS row of 32 K values (the 80-byte row that keeps the 16-byte loads conflict-free);
//   rank_blocks Rank / 32: Q8_0 blocks of one up-weight row;
//   down / up   FP16 pack sizes (halves) of the projection (Rank x width) and output (Streams x Hidden x Rank) weights.

#include <cstdint>

namespace strix::core::hc {

template <unsigned Hidden, unsigned Streams, unsigned Rank>
struct WideGeometry {
    static constexpr unsigned hidden = Hidden;
    static constexpr unsigned streams = Streams;
    static constexpr unsigned rank = Rank;
    static constexpr unsigned k_step = 32U;
    static constexpr unsigned width = Streams * Hidden;
    static constexpr unsigned stages = width / k_step;
    static constexpr unsigned stage_pitch = k_step + 8U;
    static constexpr unsigned rank_blocks = Rank / k_step;
    static constexpr std::uint64_t down_pack_halves = std::uint64_t{Rank} * width;
    static constexpr std::uint64_t up_pack_halves = std::uint64_t{Streams} * Hidden * Rank;
    static_assert(Hidden != 0U && Streams != 0U && Rank != 0U);
    static_assert(width % k_step == 0U && Rank % k_step == 0U, "the staged K step divides the width and the rank");
    static_assert(std::uint64_t{Streams} * Hidden <= 0xffffffffU, "the flattened width fits 32 bits");
};

// Byte geometry of a lane-chain weight tile (core/hc_lane_chain.hpp): one lane per weight row of Rank codes stored
// as Rank / 32 blocks; a wave stages 32 consecutive rows. Q8_0 blocks are 34 bytes (FP16 scale, 32 codes); the
// Q6P32 copy stores the same codes and scales in 26 bytes per block.
template <bool Q6P32>
struct LaneChainTile {
    static constexpr std::uint64_t block_bytes = Q6P32 ? 26U : 34U;
    static constexpr std::uint32_t codes_per_block = 32U;
    // Bytes of one weight row: (Rank / 32) blocks (the block count is formed in 32 bits, as the launchers form it).
    static constexpr std::uint64_t row_bytes(std::uint32_t rank) noexcept {
        return static_cast<std::uint64_t>(rank / codes_per_block) * block_bytes;
    }
    // Bytes of one staged wave tile of 32 rows: block_bytes * Rank (a multiple of 16 bytes when Rank % 32 == 0).
    static constexpr std::uint64_t tile_bytes(std::uint32_t rank) noexcept { return block_bytes * rank; }
};

}  // namespace strix::core::hc
