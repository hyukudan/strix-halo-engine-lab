#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace strix::glm::rtxpf::producer {
inline constexpr unsigned max_capacity=131072,max_row_tile=32768;
// Zero is the capacity-scaled candidate; explicit numeric tiles remain fixed.
inline unsigned effective_row_tile(unsigned capacity,unsigned requested){return requested?requested:(capacity>=32768?2048:256);}
inline bool valid_schedule(unsigned capacity,unsigned tile){return capacity>=4&&capacity<=max_capacity&&capacity%4==0&&tile>=32&&tile<=std::min(capacity,max_row_tile)&&tile%32==0;}
struct RowTile {unsigned first,rows,bucket;};
inline RowTile row_tile(unsigned first,unsigned prefix,unsigned tile){auto n=std::min(tile,prefix-first);return {first,n,n==tile?0u:1u};}
// Persistent residuals + final hidden only. Working rows are not capacity rows.
inline std::uint64_t activation_bytes(unsigned capacity,unsigned tile){return std::uint64_t(capacity)*4096*4*5+std::uint64_t(tile)*4096*4*14+std::uint64_t(tile)*24*4+std::uint64_t(capacity+1)*4+std::uint64_t(tile)*64+4;}
}
