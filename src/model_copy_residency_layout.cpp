#include "strix/hip_model_residency.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace strix {
std::vector<CopyResidencyRange> build_copy_residency_ranges(const ModelManifest& manifest,const HipModelResidencyPolicy& policy) {
    const auto add=[](std::uint64_t a,std::uint64_t b){if(b>std::numeric_limits<std::uint64_t>::max()-a)throw std::overflow_error("HIP model residency range overflow");return a+b;};
    std::vector<CopyResidencyRange> ranges;
    for(const auto& tensor:manifest.tensors){
        if(tensor.placement.kind==PlacementKind::remote || (tensor.residency==ResidencyPolicy::file_backed_candidate && !policy.promote_file_backed_candidates))continue;
        ranges.push_back({tensor.absolute_offset,add(tensor.absolute_offset,tensor.byte_size)});
    }
    std::sort(ranges.begin(),ranges.end(),[](const auto& a,const auto& b){return a.begin!=b.begin?a.begin<b.begin:a.end<b.end;});
    std::vector<CopyResidencyRange> merged;
    for(const auto range:ranges){if(!merged.empty() && range.begin<=add(merged.back().end,policy.coalesce_gap_bytes))merged.back().end=std::max(merged.back().end,range.end);else merged.push_back(range);}
    return merged;
}
std::uint64_t copy_residency_tensor_offset(const ManifestTensor& tensor,const std::vector<CopyResidencyRange>& ranges){
    for(const auto range:ranges)if(tensor.absolute_offset>=range.begin && tensor.absolute_offset<=range.end && tensor.byte_size<=range.end-tensor.absolute_offset)return tensor.absolute_offset-range.begin;
    throw std::invalid_argument("tensor has no copied residency range");
}
}
