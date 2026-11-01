#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace strix::glm::rtxpf::producer {
inline constexpr unsigned max_capacity=131072,max_row_tile=32768;
// Zero is the capacity-scaled candidate; explicit numeric tiles remain fixed.
inline unsigned effective_row_tile(unsigned capacity,unsigned requested){return requested?requested:(capacity>=32768?2048:256);}
inline bool valid_schedule(unsigned capacity,unsigned tile){return capacity>=4&&capacity<=max_capacity&&capacity%4==0&&tile>=32&&tile<=std::min(capacity,max_row_tile)&&tile%32==0;}
// Default prefill tile (tierstream service). The tile must not eat the expert
// arena: candidates start at min(capacity, default_max_row_tile) (8192 = one piece
// at 8K, 4 at 32K) rounded down to 32 and halve (rounded down to 32) to 32, each a
// valid schedule; the first one is used that the whole-device plan admits AND whose
// activation + workspace growth over tile 2048 is at most 5% of the expert arena
// planned at tile 2048 (tiles <= 2048 have no growth). 0 ends the list.
inline constexpr unsigned default_max_row_tile=8192,baseline_row_tile=2048,tile_arena_share_percent=5;
// MTP prefill runs in pieces of at most 2048 rows whatever the trunk tile (its per-row arithmetic does not
// depend on the piece: per-row router, GEMM K order and absolute DSA positions; pieces stay 4-aligned).
inline constexpr unsigned mtp_max_row_tile=2048;
inline unsigned mtp_row_tile(unsigned tile){return tile<mtp_max_row_tile?tile:mtp_max_row_tile;}
inline unsigned first_tile_candidate(unsigned capacity,unsigned cap_tile=max_row_tile){return std::min({capacity,max_row_tile,cap_tile})/32*32;}
inline unsigned next_tile_candidate(unsigned tile){return tile>32?std::max(32u,tile/2/32*32):0u;}
inline bool tile_growth_within(std::uint64_t growth,std::uint64_t baseline_arena){return growth*100<=std::uint64_t(tile_arena_share_percent)*baseline_arena;}
template<class Fits> unsigned largest_fitting_tile(unsigned capacity,Fits fits,unsigned cap_tile=max_row_tile){
    for(unsigned t=first_tile_candidate(capacity,cap_tile);t;t=next_tile_candidate(t))if(fits(t))return t;return 0;
}
struct RowTile {unsigned first,rows,bucket;};
inline RowTile row_tile(unsigned first,unsigned prefix,unsigned tile){auto n=std::min(tile,prefix-first);return {first,n,n==tile?0u:1u};}
// Persistent residuals + final hidden only. Working rows are not capacity rows.
inline std::uint64_t activation_bytes(unsigned capacity,unsigned tile){return std::uint64_t(capacity)*4096*4*5+std::uint64_t(tile)*4096*4*14+std::uint64_t(tile)*24*4+std::uint64_t(capacity+1)*4+std::uint64_t(tile)*64+4;}
}
