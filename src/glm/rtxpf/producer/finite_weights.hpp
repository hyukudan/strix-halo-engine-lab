#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include "strix/serve/json.hpp"

namespace strix::glm::rtxpf::producer {
// Startup-only scalar validation, including tensors unaligned in a shard and
// scalars spanning a direct-read boundary. Integer packing is never interpreted.
class FiniteWeights {
    unsigned width_=0,mask_=0,used_=0;
    std::array<unsigned char,4> tail_{};
    static void need(bool ok){if(!ok)throw std::runtime_error("nonfinite source weight");}
    void scalar(const void* p)const {
        if(width_==2){std::uint16_t bits;std::memcpy(&bits,p,2);need((bits&mask_)!=mask_);}
        else {std::uint32_t bits;std::memcpy(&bits,p,4);need((bits&0x7f800000u)!=0x7f800000u);}
    }
public:
    explicit FiniteWeights(const std::string& dtype){
        if(dtype=="F16"){width_=2;mask_=0x7c00;}
        else if(dtype=="BF16"){width_=2;mask_=0x7f80;}
        else if(dtype=="F32")width_=4;
    }
    void append(const void* data,std::size_t n){
        if(!width_)return;
        auto p=static_cast<const unsigned char*>(data);
        while(used_&&n){tail_[used_++]=*p++;--n;if(used_==width_){scalar(tail_.data());used_=0;}}
        while(n>=width_){scalar(p);p+=width_;n-=width_;}
        while(n){tail_[used_++]=*p++;--n;}
    }
    void finish()const {if(used_)throw std::runtime_error("partial floating source weight");}
};
// Cache-mode startup gate. The cache builder's float validation is not proven
// by a manifest SHA, so every F16/BF16/F32 tensor of a held blob is scanned at
// its manifest offset/dtype before startup succeeds. Pieces arrive in
// ascending, contiguous order with arbitrary splits; all bytes must be seen.
struct FiniteRange {std::uint64_t offset,bytes;std::string dtype;};
inline std::vector<FiniteRange> blob_finite_ranges(const strix::serve::Json& manifest,const std::string& blob){
    auto field=[](const strix::serve::Json& j,const char* key)->const strix::serve::Json&{auto* v=j.find(key);if(!v)throw std::runtime_error("missing manifest field");return *v;};
    auto integer=[](const strix::serve::Json& j){auto n=j.as_int();if(n<0)throw std::runtime_error("negative cache integer");return std::uint64_t(n);};
    std::vector<FiniteRange> ranges;
    for(const auto& [name,t]:field(manifest,"tensors").as_object()){
        if(field(t,"blob").as_string()!=blob)continue;
        const auto& dtype=field(t,"dtype").as_string();
        if(dtype=="F16"||dtype=="BF16"||dtype=="F32")ranges.push_back({integer(field(t,"offset")),integer(field(t,"bytes")),dtype});
    }
    return ranges;
}
class BlobFiniteScan {
    std::vector<FiniteRange> ranges_;
    std::vector<FiniteWeights> scans_;
    std::uint64_t bytes_,next_=0;
    std::size_t first_=0;
public:
    BlobFiniteScan(std::vector<FiniteRange> ranges,std::uint64_t bytes):ranges_(std::move(ranges)),bytes_(bytes){
        std::sort(ranges_.begin(),ranges_.end(),[](const FiniteRange& a,const FiniteRange& b){return a.offset<b.offset;});
        std::uint64_t end=0;
        for(const auto& r:ranges_){
            if(r.offset<end||r.offset>bytes_||r.bytes>bytes_-r.offset)throw std::runtime_error("finite scan tensor bounds");
            end=r.offset+r.bytes;scans_.emplace_back(r.dtype);
        }
    }
    void append(std::uint64_t offset,const void* data,std::size_t n){
        if(offset!=next_||n>bytes_-offset)throw std::runtime_error("finite scan out of order");
        auto p=static_cast<const unsigned char*>(data);const auto end=offset+n;
        while(first_<ranges_.size()&&ranges_[first_].offset+ranges_[first_].bytes<=offset)++first_;
        for(auto i=first_;i<ranges_.size()&&ranges_[i].offset<end;++i){
            auto lo=std::max(ranges_[i].offset,offset),hi=std::min(ranges_[i].offset+ranges_[i].bytes,end);
            if(lo<hi)scans_[i].append(p+(lo-offset),std::size_t(hi-lo));
        }
        next_=end;
    }
    void finish()const {
        if(next_!=bytes_)throw std::runtime_error("finite scan incomplete");
        for(const auto& s:scans_)s.finish();
    }
};
}
