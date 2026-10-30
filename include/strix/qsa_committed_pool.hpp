#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strix {
struct MtpQsaPoolPlan {
    std::uint64_t capacity=0, blocks=0, entry_stride=0, key_bytes=0, scratch_bytes=0, metadata_bytes=16384, reserved_bytes=0;
    std::uint32_t entries=13, maximum_writer_rows=9, provisional_blocks=3, maxima_stride=0;
};
inline MtpQsaPoolPlan mtp_qsa_pool_plan(std::uint64_t capacity, std::uint32_t writer_rows=9,
    std::uint64_t selector_scratch=0) {
    if (!capacity || capacity>UINT32_MAX || !writer_rows || writer_rows>9)
        throw std::invalid_argument("MTP QSA pool capacity/writer span is not admitted");
    MtpQsaPoolPlan p; p.capacity=capacity; p.blocks=capacity/4; p.maximum_writer_rows=writer_rows;
    p.provisional_blocks=(3U+writer_rows+3U)/4U;
    p.entry_stride=p.blocks*128U*2U; // Always 256 aligned; no sixteen-entry rounding.
    p.key_bytes=p.entry_stride*p.entries;
    p.maxima_stride=static_cast<std::uint32_t>(std::max<std::uint64_t>((p.blocks+3)/4,16384));
    p.scratch_bytes=std::max(selector_scratch,std::uint64_t{8}*p.maxima_stride*4);
    if (p.scratch_bytes>UINT64_MAX-p.key_bytes-p.metadata_bytes) throw std::overflow_error("MTP QSA pool reservation overflow");
    p.reserved_bytes=p.key_bytes+p.scratch_bytes+p.metadata_bytes;
    return p;
}
inline bool mtp_qsa_pool_fits(const MtpQsaPoolPlan& p,std::uint64_t budget,std::uint64_t windows,
    std::uint64_t captures,std::uint64_t checkpoint) noexcept {
    for (const auto bytes:{p.reserved_bytes,windows,captures,checkpoint}) {
        if (bytes>budget) return false;
        budget-=bytes;
    }
    return true;
}
// Exclusive-end frontiers. Proposals never certify raw bytes; replay does.
struct QsaCommittedFrontier {
    std::uint64_t raw_end=0;
    std::uint32_t committed=0, provisional=0;
    std::uint64_t transaction=0, invalidations=0, rebuilds=0, promotions=0;
    void invalidate(std::uint64_t first) noexcept {
        committed=std::min(committed,static_cast<std::uint32_t>(first/4));
        provisional=committed; raw_end=std::min(raw_end,first); ++invalidations; ++transaction;
    }
    void before_rewrite(std::uint64_t first,std::uint32_t rows,const MtpQsaPoolPlan& p,bool certified_prefix) {
        if (!rows || rows>p.maximum_writer_rows || first>p.capacity || rows>p.capacity-first)
            throw std::invalid_argument("MTP QSA writer exceeds admitted span/capacity");
        invalidate(first);
        if (certified_prefix) raw_end=first;
        if (first+rows>raw_end+p.maximum_writer_rows)
            throw std::invalid_argument("MTP QSA speculative suffix exceeds admission");
    }
    void built(std::uint32_t end,const MtpQsaPoolPlan& p) {
        if (end>p.blocks || end>raw_end/4+p.provisional_blocks)
            throw std::invalid_argument("MTP QSA build exceeds admitted suffix");
        if (end>provisional) {rebuilds+=end-provisional; provisional=end;}
        committed=std::max(committed,std::min(end,static_cast<std::uint32_t>(raw_end/4)));
    }
    void commit(std::uint64_t end) noexcept {
        const auto next=std::min(provisional,static_cast<std::uint32_t>(end/4));
        if (next>committed) promotions+=next-committed;
        committed=next; provisional=next; raw_end=end;
    }
};
} // namespace strix
