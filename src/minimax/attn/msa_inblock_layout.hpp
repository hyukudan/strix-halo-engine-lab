#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define MMSA_INBLOCK_HD __host__ __device__
#else
#define MMSA_INBLOCK_HD
#endif

namespace strix::minimax::msa::fused::inblock {

// Independent of HIP and numerical helpers so the indexing contract is host-testable.
template <unsigned HeadDim, unsigned MaxChunks, unsigned Group, unsigned Tile>
struct Layout {
    static constexpr unsigned kTileBytes = 4 * Tile * HeadDim * sizeof(std::uint16_t);
    static constexpr unsigned kStatsOffset = kTileBytes + MaxChunks * sizeof(std::uint32_t);
    static constexpr unsigned kStatsCount = Group * 2 * MaxChunks;
    static constexpr unsigned kSharedBytes = kStatsOffset + kStatsCount * sizeof(float);
    static_assert(HeadDim % 32 == 0 && kStatsOffset % alignof(float) == 0);

    // Consecutive lanes read consecutive chunks during the ordered combine.
    MMSA_INBLOCK_HD static constexpr unsigned stats_index(unsigned head, unsigned field, unsigned chunk) {
        return (head * 2 + field) * MaxChunks + chunk;
    }

    // Each lane reads its own writes in the producer's reserved numerator storage.
    MMSA_INBLOCK_HD static constexpr std::size_t numerator_index(std::size_t row_head, unsigned chunk, unsigned lane, unsigned dim) {
        return (row_head * MaxChunks + chunk) * (2 + HeadDim) + 2 + lane * (HeadDim / 32) + dim;
    }
};

}  // namespace strix::minimax::msa::fused::inblock

#undef MMSA_INBLOCK_HD
