#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::glm::rtxpf::producer {
// Source bytes stay in source order in caller-owned pinned memory. The scatter
// map sends each tensor directly to its original packed device offset.
struct SourceRange {std::uint64_t source, destination, bytes;};
struct DirectRead {std::uint64_t source, buffer, bytes;};
struct DirectSlice {std::uint64_t buffer, destination, bytes;};
struct CoalescedPlan {
    static constexpr std::uint64_t alignment=65536, minimum=8ULL<<20, target=32ULL<<20, maximum=target+minimum;
    std::vector<DirectRead> reads;
    std::vector<DirectSlice> slices;
    std::uint64_t bytes=0;
    static std::uint64_t round(std::uint64_t x){
        if(x>std::numeric_limits<std::uint64_t>::max()-alignment)throw std::runtime_error("plan overflow");
        return (x+alignment-1)/alignment*alignment;
    }
    CoalescedPlan()=default;
    CoalescedPlan(std::vector<SourceRange> ranges,std::uint64_t file_bytes){
        if(file_bytes>UINT64_MAX-4095)throw std::runtime_error("file alignment overflow");
        const auto file_end=(file_bytes+4095)/4096*4096;
        std::sort(ranges.begin(),ranges.end(),[](auto a,auto b){return a.source<b.source;});
        std::vector<std::pair<std::uint64_t,std::uint64_t>> extents;
        for(auto r:ranges){
            if(!r.bytes||r.source>file_bytes||r.bytes>file_bytes-r.source||r.destination>UINT64_MAX-r.bytes)
                throw std::runtime_error("plan range");
            auto lo=r.source/alignment*alignment,hi=std::min(round(r.source+r.bytes),file_end);
            if(hi-lo<minimum){hi=std::min(file_end,lo+minimum);
                if(hi-lo<minimum)lo=hi>=minimum?(hi-minimum)/alignment*alignment:0;}
            if(!extents.empty()&&lo<=extents.back().second)extents.back().second=std::max(hi,extents.back().second);
            else extents.emplace_back(lo,hi);
        }
        for(auto [lo,hi]:extents){
            const auto buffer=bytes;
            for(auto off=lo;off<hi;){auto n=std::min(target,hi-off);if(hi-off>target&&hi-off<maximum)n=hi-off;
                reads.push_back({off,bytes,n});bytes+=n;off+=n;}
            for(auto r:ranges)if(r.source>=lo&&r.source+r.bytes<=hi)slices.push_back({buffer+r.source-lo,r.destination,r.bytes});
        }
    }
};
}
